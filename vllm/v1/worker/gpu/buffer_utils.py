# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)

# Default round-robin depth for the UVA buffer pools. Must be >= the number of
# concurrent in-flight steps (engine batch_queue_size).
_DEFAULT_MAX_CONCURRENCY = 2


def set_default_max_concurrency(n: int) -> None:
    global _DEFAULT_MAX_CONCURRENCY
    _DEFAULT_MAX_CONCURRENCY = max(2, n)


def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # pin_memory() is no-op if the memory is already pinned.
    pinned = x.pin_memory()
    return out.copy_(pinned, non_blocking=True)


class UvaBuffer:
    """一块 pinned memory，cpu / np / uva 三者是同一块物理内存的三个视图：
       cpu -- CPU 端 tensor
       np  -- NumPy view，CPU 侧直接写
       uva -- GPU 可直读的 view（UVA，不经显式 cudaMemcpy）"""
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)


class UvaBufferPool:
    """UvaBuffer 池，round-robin 轮转，支持并发写时不互相覆盖。
       copy_to_uva: CPU -> pinned（返回 GPU 可直读的 uva view）
       copy_to_gpu:   CPU -> pinned -> GPU（显式搬上显存）"""
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        # 返回 uva view，GPU 可直接读，无需显式 cudaMemcpy
        return buf.uva[:n]

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class UvaBackedTensor:
    """
    CPU 侧有一个"源"tensor（self.cpu / self.np），通过 UvaBufferPool
       把数据搬到 pinned memory 后返回 uva view 给 GPU 使用。

    Unified Virtual Addressing——CUDA 提供的能力：
        CPU pinned memory 和 GPU 显存共享同一个地址空间，
        GPU kernel 可以用同一个指针直接读写 CPU 端的 pinned memory，
        无需显式 cudaMemcpy，硬件通过 PCIe DMA 自动完成搬运。

    UvaBuffer 就是一块CPU 和 GPU 都能直接读写的共享 pinned memory——CPU 端用 NumPy 写，
    GPU 端用 UVA tensor 读，省掉中间的显式 cudaMemcpy，适合频繁的小数据从 CPU→GPU 传输场景（如 block table 更新、sampling 参数同步）。
    """
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        self.dtype = dtype

        # Source of truth: CPU 端的权威数据
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)

        # CPU -> UVA(CPU)
        self.gpu = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        # CPU -> UVA(CPU)
        # GPU直读
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


class StagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int | None = None,
        uva_instead_of_gpu: bool = False,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        # max_num_reqs
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            # size 是完整的，不是只有num_rows
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            # UVA pinned CPU内存，GPU可以直读
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        # 所有requests的contents都会flatten放入write_contents
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        # [ max_concurrency, num_rows ]
        # gpu可以直读的uva缓存区，最多允许 num_rows个请求并发
        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        """暂存一次写入到 CPU 列表，不触发 GPU 操作。
           index: 目标行号（如 request index）
           start: 行内起始列
           x:     写入数据
           多次 stage_write 的数据在 write_contents 里扁平拼接，
           write_cu_lens 用累积长度标记每条边界。"""
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        """将 CPU 侧暂存的所有写入一次批量应用到 GPU tensor。
           每条 staged write 对应一个 kernel program，做：
             gpu[row][start : start+len(content)] = content
           例: stage_write(3, 0, [101,202,303]) + stage_write(5, 2, [42,99])
             → write_indices = [3,5]   write_starts = [0,2]
               write_contents = [101,202,303,42,99]   write_cu_lens = [3,5]
             → pid=0: gpu[3][0:3]=[101,202,303]   pid=1: gpu[5][2:4]=[42,99]"""
        n = len(self._staged_write_indices)
        if n == 0:
            return

        # CPU -> CPU(UVA)
        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        # CPU -> GPU
        # 所有requests的contents都会flatten放入write_contents
        write_contents = async_tensor_h2d(
            self._staged_write_contents, device=self.device, dtype=self.dtype
        )

        # Write diffs to the GPU buffer
        #  n 个thread block并行执行
        _apply_write_kernel[(n,)](
            self.gpu, # GPU or UVA
            self.gpu.stride(0),
            indices_uva, # UVA
            starts_uva, # UVA
            write_contents, # GPU
            cu_lens_uva, # UVA
            None,
            BLOCK_SIZE=1024,
            MULTI_GROUP=False,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


class FusedStagedWriter:
    """Applies the staged writes of several `StagedWriteTensor`s at once."""

    def __init__(
        self, device: torch.device, max_writes: int, max_concurrency: int | None = None
    ):
        # 4 个 UvaBufferPool 全部用 int32：group/indices/starts/cu_lens 都是整型路由元数据，
        # 只承载"写哪个 tensor、哪一行、从哪开始、写多长"，不占 GPU 显存
        new_pool = partial(
            UvaBufferPool, dtype=torch.int32, max_concurrency=max_concurrency
        )
        # 每条写入归属的 tensor 编号（group 标签）
        self.group_ids = new_pool(max_writes)
        # 目标行号（如 request index）
        self.indices = new_pool(max_writes)
        # 行内起始列
        self.starts = new_pool(max_writes)
        # 全局累积长度，标记每条写入在 contents 中的边界
        self.cu_lens = new_pool(max_writes)
        self.device = device

    def apply(
        self,
        tensors: Sequence[StagedWriteTensor],
        output_ptrs: torch.Tensor,
        output_strides: torch.Tensor,
    ) -> None:
        """Apply and clear the staged writes of `tensors` with one kernel."""
        # 先在所有 tensor 的 staged writes 在 CPU 侧聚合成 5 个全局列表
        group_ids: list[int] = []
        indices: list[int] = []
        starts: list[int] = []
        contents: list[int | float] = []
        cu_lens: list[int] = []

        # 遍历每个 tensor，用其在列表中的下标作为 group_id
        for group_id, t in enumerate(tensors):
            n = len(t._staged_write_indices)
            if n == 0:
                # 该 tensor 没有待写内容，跳过
                continue

            # 给这 n 条写入统一打上 group_id 标签，kernel 据此路由到对应输出 tensor
            group_ids.extend([group_id] * n)
            indices.extend(t._staged_write_indices)
            starts.extend(t._staged_write_starts)
            # contents 是跨 group 全局拼接的，所以 cu_lens 也要整体平移 content_base，
            # 拼成连续的全局累积长度，保证 kernel 用 cu_lens[pid-1]/[pid] 取边界正确
            content_base = len(contents)
            contents.extend(t._staged_write_contents)
            cu_lens.extend(content_base + cu_len for cu_len in t._staged_write_cu_lens)

        if not group_ids:
            # 没有任何写入，直接返回，避免启动空 kernel
            return

        # 元数据走 UVA：CPU 写进 pinned 缓冲，GPU kernel 直接读，省去显式 H2D 拷贝
        group_ids_uva = self.group_ids.copy_to_uva(group_ids)
        indices_uva = self.indices.copy_to_uva(indices)
        starts_uva = self.starts.copy_to_uva(starts)
        cu_lens_uva = self.cu_lens.copy_to_uva(cu_lens)
        # 内容数据量大且 kernel 内连续读，直接异步搬上 GPU（non_blocking 重叠传输）
        contents_gpu = async_tensor_h2d(contents, device=self.device, dtype=torch.int32)

        # 单次 kernel 启动，MULTI_GROUP=True：
        # output_ptrs/output_strides 是 [num_groups] 指针数组，kernel 内按 group_id 解引用到各自 tensor
        _apply_write_kernel[(len(group_ids),)](
            output_ptrs,
            output_strides,
            indices_uva,
            starts_uva,
            contents_gpu,
            cu_lens_uva,
            group_ids_uva,
            BLOCK_SIZE=1024,
            # 与 StagedWriteTensor.apply_write 的区别：跨多个 tensor 融合、靠 group_id 路由
            MULTI_GROUP=True,
        )
        # 一批写完后统一清空所有 tensor 的暂存，便于下一轮 stage
        for t in tensors:
            t.clear_staged_writes()


@triton.jit
def _apply_write_kernel(
    output_ptr,  # MULTI_GROUP: ptr-to-ptrs [num_groups]; else: data ptr
    output_stride,  # MULTI_GROUP: ptr-to-strides [num_groups]; else: row stride
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    write_group_ids_ptr,  # [num_writes], used only when MULTI_GROUP
    BLOCK_SIZE: tl.constexpr,
    MULTI_GROUP: tl.constexpr,
):
    """
    Triton kernel：将多条分散写入批量写入目标 tensor。

    每个 program (pid) 负责一条写入，将 write_contents 中对应区间的内容
    写入 output 的指定行和列偏移位置。支持单组和多组模式：
    单组模式下所有写入目标同一 tensor；多组模式下每条写入可指向不同的
    输出 tensor（如多个 KV cache group）。

    Args:
        output_ptr: 单组模式下为目标数据指针；多组模式下为指向各组数据指针的指针数组
        output_stride: 单组模式下为行 stride；多组模式下为指向各组行 stride 的指针
        write_indices_ptr: 各条写入的目标行号（UVA pinned 内存）
        write_starts_ptr: 各条写入的行内起始列偏移（UVA pinned 内存）
        write_contents_ptr: 扁平拼接的全部写入内容（GPU 显存）
        write_cu_lens_ptr: 累积长度数组，标记每条写入内容在 write_contents 中的边界（UVA pinned 内存）
        write_group_ids_ptr: 各条写入所属的组 ID，仅多组模式下使用
        BLOCK_SIZE: tl.constexpr，每个线程块处理的元素数
        MULTI_GROUP: tl.constexpr，是否启用多组模式
    """
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    if MULTI_GROUP:
        # Each write targets a different output tensor (KV cache group);
        # resolve its base pointer and row stride per write.
        group_id = tl.load(write_group_ids_ptr + pid)
        row_ptr = _load_ptr(output_ptr + group_id, tl.int32)
        row_stride = tl.load(output_stride + group_id)
    else:
        row_ptr = output_ptr
        row_stride = output_stride
    row_ptr += row_idx * row_stride + start_idx

    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < content_len
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(row_ptr + block, content, mask=mask)


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)
