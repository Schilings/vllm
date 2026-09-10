# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter,
    StagedWriteTensor,
    UvaBackedTensor,
    _load_ptr,
)


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int],
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ):
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups

        # ⚠️ 按照不同的kernel block size，实际进一步拆分block size
        self.blocks_per_kv_block = [
            bs // kbs for bs, kbs in zip(block_sizes, kernel_block_sizes)
        ]

        # 每组一份block table
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            # 按照不同的kernel block size，实际进一步拆分block size
            max_num_blocks = max_num_blocks_per_group[i] * self.blocks_per_kv_block[i]
            # ⚠️ GPU
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)

        # ⚠️ UVA, num_kv_cache_groups x [max_num_reqs, 1 ]
        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )
        # ⚠️ GPU，多个kv group的以下信息打包一起用，不用分开传输到gpu
        # group_ids, indices, starts, cu_lens
        # 4 个 num_kv_cache_groups x [max_num_reqs, 1 ]
        self.fused_writer: FusedStagedWriter | None = None
        if self.num_kv_cache_groups > 1:
            # Only the multi-group path uses the fused writer.
            self.fused_writer = FusedStagedWriter(
                self.device, self.num_kv_cache_groups * self.max_num_reqs
            )

        # ⚠️ GPU：前向输入用的 block table，按 batch_idx 密集排布。
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        #
        # 与上面 self.block_tables 的区别（两份 tensor 用途不同，并非冗余复制）：
        #   self.block_tables  —— 按 req_idx 索引的【权威存储】
        #       · req_idx 是调度器给每个请求分配的持久槽位（persistent batch 中稳定不变）；
        #       · 通过 append_block_ids→stage_write 增量更新，apply_staged_writes 写进 b.gpu；
        #       · 数据随请求生命周期累积，跨 step 保留，是 block id 的"真相来源"。
        #   self.input_block_tables (此处) —— 按 batch_idx 索引的【前向就绪视图】
        #       · 模型 forward / attention kernel 需要当前这批请求按顺序 0..num_reqs-1 密集排布；
        #       · 由 gather_block_tables() 根据 idx_mapping(batch_idx→req_idx) 把
        #         block_tables[req_idx] 的整行 gather 到 input_block_tables[batch_idx]，
        #         多余行清零（见 _gather_block_tables_kernel）；
        #       · 每步内容被重新 gather 覆盖，但【tensor 地址固定】，以便 CUDA graph capture
        #         复用同一块显存（get_dummy_block_tables 返回的就是它，地址不能变）。
        # 因此 torch.zeros_like(b.gpu) 只是预先分配目标缓冲，真正填值发生在每步的 gather，
        # 而不是初始化时把 block_tables "复制" 过来。
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        # ⚠️ GPU： num_kv_cache_groups x [max_num_batched_tokens, 1 ]
        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        # Called at init and after a CuMem kv_cache wake-up. The ptr tensors
        # cache raw data_ptr() values that go stale once the underlying tensors
        # are reallocated on wake; block_sizes_tensor needs re-populating
        # because its storage lives under the kv_cache pool tag and comes back
        # with undefined contents.
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            # 该req目前多少block
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            # 该req新增的blocks
            block_ids = new_block_ids[i]
            # 该组的block_size被kernel_block_size进一步整除
            bpk = self.blocks_per_kv_block[i]
            # 所以要翻bpk倍，实际对于 bpk * num_blocks 个block
            if bpk > 1:
                block_ids = [b * bpk + k for b in block_ids for k in range(bpk)]
            # 增量，还未写入
            self.block_tables[i].stage_write(req_index, start, block_ids)
            self.num_blocks.np[i, req_index] = start + len(block_ids)

    def apply_staged_writes(self) -> None:
        if self.num_kv_cache_groups == 1:
            # Single group: write directly, skipping the per-write group lookup.
            self.block_tables[0].apply_write()
        else:
            # Multiple groups: apply all block tables with one fused kernel.
            assert self.fused_writer is not None
            self.fused_writer.apply(
                self.block_tables, self.block_table_ptrs, self.block_table_strides
            )
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, ...]:
        # idx_mapping: [batch_size] 的映射 batch_idx -> req_idx，即"当前前向批里第 batch_idx
        #   个槽对应 persistent batch 里的哪个 req_idx"。kernel 据此把 block_tables[req_idx]
        #   的整行 gather 到 input_block_tables[batch_idx]。
        # num_reqs_padded: CUDA graph 下固定的"填充后请求数"，保证每次 launch 的 grid 形状一致，
        #   使 capture 到的计算图可稳定 replay。
        num_reqs = idx_mapping.shape[0]
        # grid = (num_kv_cache_groups, num_reqs_padded)：
        #   program_id(0) = kv cache group，program_id(1) = batch_idx（含 padding 行）。
        # 用 num_reqs_padded 而不是 num_reqs 启动，是为了把"清零 padding 行"融合进同一个 kernel
        # （见 _gather_block_tables_kernel 中 batch_idx >= num_reqs 的分支），省掉一次额外的清零 kernel。
        # kernel 入参（顺序与 _gather_block_tables_kernel 签名一致）：
        #   1) idx_mapping: 源行索引 -> 目标行索引 的映射（batch_idx -> req_idx）
        #   2) block_table_ptrs: 源，各 group 的 block_tables.gpu 指针数组（req_idx 序）
        #   3) input_block_table_ptrs: 目标，各 group 的 input_block_tables 指针数组（batch_idx 序）
        #   4) block_table_strides: 各 group 的行 stride（= 该 group 的 max_num_blocks）
        #   5) num_blocks.gpu: [num_kv_cache_groups, max_num_reqs] 每 req 的实际有效 block 数
        #   6) num_blocks.gpu.stride(0): num_blocks 的行 stride（在 group 间跳转用）
        #   7) num_reqs: 实际请求数，kernel 据此区分真实行与 padding 行
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs_padded)](
            idx_mapping,
            self.block_table_ptrs,
            self.input_block_table_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            num_reqs,
            BLOCK_SIZE=1024,  # type: ignore
        )
        # 返回每个 group 的 input_block_tables，裁剪到 num_reqs_padded 行。
        # 返回的是 persistent tensor 的视图（地址不变），不分配新内存——这是 CUDA graph 能复用
        # 同一地址的前提；内容已于上面的 kernel 中被重新 gather 覆盖。
        return tuple(bt[:num_reqs_padded] for bt in self.input_block_tables)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            self.max_num_batched_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.slot_mappings,
            self.slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return self.slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    max_num_blocks = stride  # stride equals max_num_blocks for this group.
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
