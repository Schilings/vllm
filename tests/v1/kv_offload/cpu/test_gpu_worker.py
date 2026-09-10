# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
import time
import uuid

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

# 测试参数：GPU/CPU block 数量、GPU 页大小（字节）、block_size_factor
# （1 个 CPU block 对应多少连续的 GPU 子块，用于对齐粒度）
NUM_GPU_BLOCKS = [64]
NUM_CPU_BLOCKS = [256]
GPU_PAGE_SIZES = [512, 1024]
BLOCK_SIZE_FACTORS = [1, 3]
NUM_TENSORS = [4]
SEEDS = [0]
DEVICE_TYPE = current_platform.device_type
DEVICES = [f"{DEVICE_TYPE}:0"]
NUM_MAPPINGS = [3]
NUM_MAPPINGS_PER_GROUP = [2]


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("block_size_factor", BLOCK_SIZE_FACTORS)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("num_tensors", NUM_TENSORS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_shared_memory", [False, True])
@torch.inference_mode()
def test_transfer(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings: int,
    gpu_page_size_bytes: int,
    block_size_factor: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    num_tensors: int,
    seed: int,
    device: str,
    use_shared_memory: bool,
) -> None:
    # 单一 KV cache group / 单 tensor 场景下的基本搬移正确性测试。
    # gpu_to_cpu=False 时测 CPU->GPU（load），否则测 GPU->CPU（store）。
    set_random_seed(seed)

    # build CanonicalKVCacheTensor list: one per tensor
    # 构造 num_tensors 个独立的 GPU 端张量，每个形状 (num_gpu_blocks, gpu_page_size_bytes)
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for i in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    # one group containing all tensors, one data ref per tensor
    # 把全部 tensor 归到一个 group 里，每个 tensor 一个 data ref（描述其在 GPU 上的页大小）
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(num_tensors)
        ]
    ]

    kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors,
        group_data_refs=kv_cache_groups_data_refs,
    )

    mmap_region: SharedOffloadRegion | None = None
    if use_shared_memory:
        # ⚠️ use_shared_memory=True 时，CPU 端存储用共享内存 mmap 区域而非普通 tensor，
        # 用以测试 SharedOffloadRegion 路径（跨进程可见的 CPU 卸载区）。
        cpu_page_size = round_up(
            gpu_page_size_bytes * num_tensors * block_size_factor,
            SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT,
        )
        mmap_region = SharedOffloadRegion(
            instance_id=str(uuid.uuid4()),
            num_blocks=num_cpu_blocks,
            rank=0,
            kv_bytes_per_block=cpu_page_size,
            cpu_page_size=cpu_page_size,
        )

    # 构造 worker：持有 GPU 端张量、block_size_factor、CPU block 数与可选的共享内存区
    worker = CPUOffloadingWorker(
        kv_caches=kv_caches,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
        mmap_region=mmap_region,
    )

    # select block mappings
    # 随机挑选若干 GPU block 与 CPU block 作为本次搬移的源/目的
    gpu_blocks = random.sample(range(num_gpu_blocks), num_mappings * block_size_factor)
    cpu_blocks = random.sample(range(num_cpu_blocks), num_mappings)

    # expand cpu blocks to gpu-page granularity for uniform comparison:
    # each cpu block maps to block_size_factor consecutive sub-blocks
    # 把 CPU block 按 block_size_factor 展开成 GPU 子块粒度，便于逐子块比对
    cpu_blocks_expanded = [
        cpu_block * block_size_factor + j
        for cpu_block in cpu_blocks
        for j in range(block_size_factor)
    ]

    # maybe skip some GPU blocks to test reading/writing from the middle of a CPU block
    # ⚠️ 故意跳过开头的若干子块，制造"非对齐"搬移，验证从 CPU block 中间读写的逻辑
    # ⚠️ 第一个 GPU block → 第 0 个 CPU block 的第 skip（=factor-1）号槽位；
    # 第 0 个 CPU block 的前 factor-1 个槽位因源端跳过而保持原值；
    # 其余 GPU 子块顺序填满后续 CPU block。
    # GPU 实际写入的槽位数 == GPU 子块数，与 CPU block 总容量之差就是故意留空的头部偏移。
    blocks_to_skip = block_size_factor - 1
    if blocks_to_skip > 0:
        gpu_blocks = gpu_blocks[blocks_to_skip:]
        cpu_blocks_expanded = cpu_blocks_expanded[blocks_to_skip:]

    # set transfer direction
    if gpu_to_cpu:
        # GPU->CPU：store 路径，handler 为 _store_handler
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            # ⚠️ 想象一个 group：逻辑 GPU 子块 [B0, B1, B2, B3, B4, B5]（factor=3，对应 2 个 CPU block）。
            # 其中 B0 和 B4 是 null：
            #   剔除后 src_block_ids = [B1, B2, B3, B5]，num_group_blocks=4。
            #   start_gpu_block_idx = 1（第一个非 null 是 B1，绝对下标 1）。
            #   block_indices = 1。
            # worker 写 CPU：第 0 个 CPU block 从槽位 1 % 3 = 1 开始 → [空, B1, B2]；第 1 个 CPU block [B3, B5, 空]。
            # block_indices=1 保证了 B1 不会被写到 CPU 第 0 槽（那样就错位了），而是对齐到它在逻辑序列里本该在的槽位。
            # 那些 null 子块（B0、B4）对应的 CPU 槽位保持原值（padding），未来 load 回来时同样按 block_indices 跳过这些槽。
            #
            # ⚠️ 一句话总结
            #   src_block_ids：剔除 null 后的有效 GPU 子块扁平列表（物理已干净）。
            #   group_sizes：每个 group 的有效子块数。
            #   block_indices[i]：第 i 个 group 第一个有效 GPU 子块的逻辑起始下标，用途是让 worker 在**目的端（CPU）**按 block_idx % factor 偏移对齐写入，补偿"源端剔除了 null 子块"造成的位置错位。它不是源端再 skip 一次，而是目的端对齐声明。
            # ⚠️ 诶等等，gpu_blocks已经是skip之后的结果了，那这是skip了两次吗
            # 两个 skip 作用在不同层面
            # skip ①：gpu_blocks = gpu_blocks[blocks_to_skip:]（L140）
            # 这是测试自己在 Python 层面把源 block 列表掐头。结果：gpu_blocks 现在是"已跳过前 skip 个"的剩余 GPU 子块列表，长度 = m*f - (f-1)。
            # skip ②：block_indices=(blocks_to_skip,)（L157）
            # 这是传给 GPULoadStoreSpec 的元数据，告诉 worker：源端第一个 block 在"逻辑连续序列"里的起始下标是 blocks_to_skip。
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        # 建立 CPU 子块 -> GPU 子块 的映射，供后续校验
        dst_to_src = dict(zip(cpu_blocks_expanded, gpu_blocks))
        num_dst_sub_blocks = num_gpu_blocks
    else:
        # CPU->GPU：load 路径，handler 为 _load_handler
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        # ⚠️ 同理，源 cpu_blocks（长度 3）被 block_size_factor=factor 展开成 9 个槽位。
        # 实际被读取的只有 7 个（cpu_blocks_expanded 跳过头部 skip 后剩 7），和目的 gpu_blocks（7 子块）成对。
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=(len(gpu_blocks),), block_indices=(blocks_to_skip,)
        )
        # 建立 GPU 子块 -> CPU 子块 的映射，供后续校验
        dst_to_src = dict(zip(gpu_blocks, cpu_blocks_expanded))
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    # 搬移前把源/目的张量填成随机值，确保搬移确实发生且不被零值掩盖
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    # clone src and dst tensors before transfer
    # 保留搬移前的快照，用于比对"源不变"和"目的被正确覆盖"
    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    # call transfer function via public API
    # 通过公开 API 提交搬移任务（job_id=1）
    start_time = time.time()
    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    # 提交后内部 transfer 队列里应当只有这一个 job
    assert {x.job_id for x in handler._transfers} == {1}

    # wait for transfer to complete
    # 轮询 get_finished()，最多等 10 秒；校验 job 成功、size 与时间合理性
    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            # ⚠️ 期望搬移字节数 = GPU 子块数 * 所有 tensor 页大小之和
            assert finished[0].transfer_size == (
                len(gpu_blocks)
                * sum([x.page_size_bytes for x in handler.kv_cache_groups_data_refs[0]])
            )
            assert finished[0].transfer_time > 0
            assert finished[0].transfer_time < (time.time() - start_time)
            break
        time.sleep(0.1)

    # verify src tensors did not change
    # 源张量在搬移后必须原样不变
    # ⚠️ KV offloading 的 store/load 是单向拷贝——从源读出、写入目的，源只是"被读"的角色。
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity.
    # ⚠️ 逐 tensor、逐子块校验目的张量：被映射到的源子块内容应一致，未映射处应保持原值
    for src_tensor, dst_tensor, orig_dst_tensor in zip(
        handler.src_tensors,
        handler.dst_tensors,
        orig_dst_tensors,
    ):
        # view both GPU and CPU tensors as (n, gpu_page_size_bytes) for comparison.
        # 统一 reshape 成 (n, gpu_page_size_bytes) 视图，按子块索引比对
        src_view = src_tensor.reshape(-1, gpu_page_size_bytes)
        dst_view = dst_tensor.reshape(-1, gpu_page_size_bytes)
        orig_dst_view = orig_dst_tensor.reshape(-1, gpu_page_size_bytes)
        for dst_sub_block in range(num_dst_sub_blocks):
            src_sub_block = dst_to_src.get(dst_sub_block)
            if src_sub_block is not None:
                # ⚠️ 该目的子块有对应源子块：应从源取期望值
                expected = src_view[src_sub_block]
            else:
                # 无映射：应保持搬移前的原值
                expected = orig_dst_view[dst_sub_block]
            torch.testing.assert_close(dst_view[dst_sub_block].cpu(), expected.cpu())

    # Drop loop-variable refs so mmap_obj has no exported buffers at cleanup.
    # 释放循环变量引用，避免 mmap 区域在 cleanup 时仍被 Python 持有
    del orig_tensor, tensor, src_tensor, dst_tensor, orig_dst_tensor
    del src_view, dst_view, orig_dst_view, expected

    worker.shutdown()
    if mmap_region:
        mmap_region.cleanup()


@pytest.mark.parametrize("gpu_to_cpu", [True, False])
@pytest.mark.parametrize("num_mappings_per_group", NUM_MAPPINGS_PER_GROUP)
@pytest.mark.parametrize("gpu_page_size_bytes", GPU_PAGE_SIZES)
@pytest.mark.parametrize("block_size_factor", BLOCK_SIZE_FACTORS)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_cpu_blocks", NUM_CPU_BLOCKS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_transfer_multi_group(
    default_vllm_config,
    gpu_to_cpu: bool,
    num_mappings_per_group: int,
    gpu_page_size_bytes: int,
    block_size_factor: int,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    seed: int,
    device: str,
) -> None:
    """Test transfers with three KV cache groups:
    - Group 0: aligned transfer with num_mappings_per_group blocks
    - Group 1: zero blocks (empty group)
    - Group 2: unaligned CPU->GPU transfer (logical_offset=block_size_factor-1,
      causing the implementation to skip source sub-blocks) with
      num_mappings_per_group blocks
    """
    # 多 group 场景测试：验证单 job 内混合"对齐/空/非对齐"三类 group 时，
    # worker 能按 group_sizes / block_indices 正确分别搬移与跳过。
    set_random_seed(seed)

    # 3 groups, each with 2 tensors
    # 构造 3 个 group，每 group 2 个 tensor，共 6 个 GPU 端张量
    num_groups = 3
    tensors_per_group = 2
    num_tensors = num_groups * tensors_per_group
    kv_cache_tensors: list[CanonicalKVCacheTensor] = []
    for _ in range(num_tensors):
        gpu_tensor = torch.zeros(
            (num_gpu_blocks, gpu_page_size_bytes),
            dtype=torch.int8,
            device=device,
        )
        kv_cache_tensors.append(
            CanonicalKVCacheTensor(
                tensor=gpu_tensor,
                page_size_bytes=gpu_page_size_bytes,
            )
        )

    # 把 6 个 tensor 按编号分配到 3 个 group，每个 group 持有 2 个连续 tensor 的 data ref
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]] = [
        [
            CanonicalKVCacheRef(
                tensor_idx=g * tensors_per_group + i,
                page_size_bytes=gpu_page_size_bytes,
            )
            for i in range(tensors_per_group)
        ]
        for g in range(num_groups)
    ]

    canonical_kv_caches = CanonicalKVCaches(
        tensors=kv_cache_tensors, group_data_refs=kv_cache_groups_data_refs
    )

    # 多 group 场景不测共享内存，仅用普通 CPU 张量 worker
    worker = CPUOffloadingWorker(
        kv_caches=canonical_kv_caches,
        block_size_factor=block_size_factor,
        num_cpu_blocks=num_cpu_blocks,
    )

    # group 0: aligned, group 1: empty, group 2: unaligned on CPU->GPU
    # ⚠️  三个 group 的 CPU block 数：[正常, 空, 正常]
    group_sizes_in_cpu_blocks = [num_mappings_per_group, 0, num_mappings_per_group]

    # 各类 group 的总量：CPU block 数 = 各 group 之和，GPU 子块数 = CPU block * factor
    total_cpu_blocks = sum(group_sizes_in_cpu_blocks)
    total_gpu_blocks_needed = total_cpu_blocks * block_size_factor
    gpu_blocks_all = random.sample(range(num_gpu_blocks), total_gpu_blocks_needed)
    cpu_blocks_all = random.sample(range(num_cpu_blocks), total_cpu_blocks)

    # split gpu/cpu blocks per group
    # ⚠️ 按各 group 的 CPU block 数把随机块切分到每个 group（GPU 侧需乘 factor）
    gpu_blocks_per_group: list[list[int]] = []
    cpu_blocks_per_group: list[list[int]] = []
    gpu_offset = 0
    cpu_offset = 0
    for size in group_sizes_in_cpu_blocks:
        gpu_count = size * block_size_factor
        gpu_blocks_per_group.append(gpu_blocks_all[gpu_offset : gpu_offset + gpu_count])
        cpu_blocks_per_group.append(cpu_blocks_all[cpu_offset : cpu_offset + size])
        gpu_offset += gpu_count
        cpu_offset += size

    # expand cpu blocks to gpu-page granularity
    # ⚠️ 每个 group 的 CPU block 展开成 GPU 子块粒度列表
    cpu_blocks_expanded_per_group = [
        [
            cpu_block * block_size_factor + j
            for cpu_block in cpu_blocks
            for j in range(block_size_factor)
        ]
        for cpu_blocks in cpu_blocks_per_group
    ]

    # skip sub-blocks from group 2 to test unaligned transfers.
    # ⚠️ 对 group 2 两端各跳过 factor-1 个子块，模拟"非对齐"CPU->GPU 搬移（中间段）
    sub_blocks_to_skip = block_size_factor - 1  # e.g. 2 when block_size_factor=3
    if sub_blocks_to_skip > 0:
        gpu_blocks_per_group[2] = gpu_blocks_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]
        cpu_blocks_expanded_per_group[2] = cpu_blocks_expanded_per_group[2][
            sub_blocks_to_skip:-sub_blocks_to_skip
        ]

    # build flat gpu_blocks list and group_sizes in GPU blocks
    # ⚠️ 展平成单一 GPU block 列表，并记录每个 group 占用的 GPU 子块数
    gpu_blocks: list[int] = []
    group_sizes: list[int] = []
    for gpu_blks in gpu_blocks_per_group:
        gpu_blocks.extend(gpu_blks)
        group_sizes.append(len(gpu_blks))

    # build flat cpu_blocks list
    # 同样展平 CPU block 列表
    cpu_blocks = []
    for cpu_blks in cpu_blocks_per_group:
        cpu_blocks.extend(cpu_blks)

    # block_indices: only relevant for unaligned transfers
    # ⚠️ block_indices 指定每个 group 在源端跳过的子块数；仅 group 2 非 0
    block_indices: list[int] = [0, 0, sub_blocks_to_skip]

    if gpu_to_cpu:
        # GPU->CPU：store 路径
        handler = worker._store_handler
        src_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        dst_spec = CPULoadStoreSpec(cpu_blocks)
        # per-group mapping: cpu sub-block -> gpu sub-block
        # 每个 group 单独建立 目的(CPU子块)->源(GPU子块) 映射
        dst_to_src_per_group = [
            dict(zip(expanded, gpu_blks))
            for expanded, gpu_blks in zip(
                cpu_blocks_expanded_per_group, gpu_blocks_per_group
            )
        ]
        num_dst_sub_blocks = num_cpu_blocks * block_size_factor
    else:
        # CPU->GPU：load 路径
        handler = worker._load_handler
        src_spec = CPULoadStoreSpec(cpu_blocks)
        dst_spec = GPULoadStoreSpec(
            gpu_blocks, group_sizes=group_sizes, block_indices=block_indices
        )
        # per-group mapping: gpu sub-block -> cpu sub-block
        # 每个 group 单独建立 目的(GPU子块)->源(CPU子块) 映射
        dst_to_src_per_group = [
            dict(zip(gpu_blks, expanded))
            for gpu_blks, expanded in zip(
                gpu_blocks_per_group, cpu_blocks_expanded_per_group
            )
        ]
        num_dst_sub_blocks = num_gpu_blocks

    # randomize src and dst tensors before transfer
    # 搬移前随机化源/目的张量
    for tensor in handler.src_tensors:
        tensor.random_()
    for tensor in handler.dst_tensors:
        tensor.random_()

    # 保留快照用于源不变与目的正确性校验
    orig_src_tensors = [x.clone() for x in handler.src_tensors]
    orig_dst_tensors = [x.clone() for x in handler.dst_tensors]

    # 通过公开 API 提交单 job（job_id=1），覆盖多 group 的混合搬移
    if gpu_to_cpu:
        assert worker.submit_store(1, src_spec, dst_spec)
    else:
        assert worker.submit_load(1, src_spec, dst_spec)
    assert {x.job_id for x in handler._transfers} == {1}

    # 轮询等待完成，校验 job 成功且总字节数 = 各 group 子块数 * 各 group tensor 页大小之和
    end_time = time.time() + 10
    while time.time() < end_time:
        finished = worker.get_finished()
        if finished:
            assert finished[0].job_id == 1
            assert finished[0].success
            expected_bytes = sum(
                group_size * sum([x.page_size_bytes for x in data_refs])
                for group_size, data_refs in zip(
                    group_sizes, handler.kv_cache_groups_data_refs
                )
            )
            assert finished[0].transfer_size == expected_bytes
            break
        time.sleep(0.1)

    # verify src tensors did not change
    # 源张量在搬移后必须原样不变
    for orig_tensor, tensor in zip(orig_src_tensors, handler.src_tensors):
        assert torch.equal(orig_tensor, tensor)

    # verify dst tensors at gpu-page granularity
    # 逐 group、逐 tensor、逐子块校验：有映射处取源值，空 group/无映射处保持原值
    for group_idx, dst_to_src in enumerate(dst_to_src_per_group):
        group_tensor_offset = group_idx * tensors_per_group
        for tensor_idx in range(tensors_per_group):
            src_tensor = handler.src_tensors[group_tensor_offset + tensor_idx]
            dst_tensor = handler.dst_tensors[group_tensor_offset + tensor_idx]
            orig_dst_tensor = orig_dst_tensors[group_tensor_offset + tensor_idx]
            src_view = src_tensor.view(-1, gpu_page_size_bytes)
            dst_view = dst_tensor.view(-1, gpu_page_size_bytes)
            orig_dst_view = orig_dst_tensor.view(-1, gpu_page_size_bytes)
            for dst_sub_block in range(num_dst_sub_blocks):
                src_sub_block = dst_to_src.get(dst_sub_block)
                if src_sub_block is not None:
                    expected = src_view[src_sub_block]
                else:
                    expected = orig_dst_view[dst_sub_block]
                torch.testing.assert_close(
                    dst_view[dst_sub_block].cpu(), expected.cpu()
                )

    worker.shutdown()
