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
        # CPU -> UVA(CPU) -> GPU
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
            # UVA 是什么？范围不频繁的选择UVA？
            # size 是完整的，不是只有num_rows
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        # [ max_concurrency, num_rows ]
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
        write_contents = async_tensor_h2d(
            self._staged_write_contents, device=self.device, dtype=self.dtype
        )

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
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
        new_pool = partial(
            UvaBufferPool, dtype=torch.int32, max_concurrency=max_concurrency
        )
        self.group_ids = new_pool(max_writes)
        self.indices = new_pool(max_writes)
        self.starts = new_pool(max_writes)
        self.cu_lens = new_pool(max_writes)
        self.device = device

    def apply(
        self,
        tensors: Sequence[StagedWriteTensor],
        output_ptrs: torch.Tensor,
        output_strides: torch.Tensor,
    ) -> None:
        """Apply and clear the staged writes of `tensors` with one kernel."""
        group_ids: list[int] = []
        indices: list[int] = []
        starts: list[int] = []
        contents: list[int | float] = []
        cu_lens: list[int] = []

        for group_id, t in enumerate(tensors):
            n = len(t._staged_write_indices)
            if n == 0:
                continue

            group_ids.extend([group_id] * n)
            indices.extend(t._staged_write_indices)
            starts.extend(t._staged_write_starts)
            content_base = len(contents)
            contents.extend(t._staged_write_contents)
            cu_lens.extend(content_base + cu_len for cu_len in t._staged_write_cu_lens)

        if not group_ids:
            return

        group_ids_uva = self.group_ids.copy_to_uva(group_ids)
        indices_uva = self.indices.copy_to_uva(indices)
        starts_uva = self.starts.copy_to_uva(starts)
        cu_lens_uva = self.cu_lens.copy_to_uva(cu_lens)
        contents_gpu = async_tensor_h2d(contents, device=self.device, dtype=torch.int32)

        _apply_write_kernel[(len(group_ids),)](
            output_ptrs,
            output_strides,
            indices_uva,
            starts_uva,
            contents_gpu,
            cu_lens_uva,
            group_ids_uva,
            BLOCK_SIZE=1024,
            MULTI_GROUP=True,
        )
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
