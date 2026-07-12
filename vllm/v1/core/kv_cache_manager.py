# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """🧩 调度层与 KVCacheManager 之间的接口数据类，隐藏内部 block 结构。

    🧬 设计要点：
    ┌──────────────────────────────────────────────────────────────────┐
    │ ╔══════════════════ KVCacheBlocks 数据布局 ═══════════════════╗ │
    │ ║                                                              ║ │
    │ ║  blocks[i][j] = i-th KV cache group 的 j-th block            ║ │
    │ ║                                                              ║ │
    │ ║  外层 tuple[group][block] — 不按 [block][group] 排的原因:     ║ │
    │ ║  若将来各 group 有不同 block_size，则 block 数不相等，        ║ │
    │ ║  按 [block][group] 排会变成参差不齐的、不适合 vectorize 变换  ║ │
    │ ║                                                              ║ │
    │ ║  ⚠️ 空序列用空 tuple 表示（非空 list），                       ║ │
    │ ║     预构造的空 KVCacheBlocks 在 KVCacheManager 中复用避免 GC  ║ │
    │ ╚══════════════════════════════════════════════════════════════╝ │
    └──────────────────────────────────────────────────────────────────┘
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """`blocks[i][j]` 即第 i 个 KV cache group 的第 j 个 block。
    不使用 block 作为外层维度的原因：这假设所有 kv_cache_groups 有相同 block 数，
    虽然在现阶段成立，但将来允许不同 block_size 时会被打破。
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    """🧩 KV 缓存管理门面 — 调度层与 KV 缓存子系统之间的统一入口。

    ╔══════════════════ 🧩 对外接口清单 ═══════════════════╗
    ║                                                      ║
    ║  前缀与分配（调度步核心路径）:                         ║
    ║  📖 get_computed_blocks()   前缀缓存命中查找           ║
    ║  ✍️ allocate_slots()        为请求分配 KV slot        ║
    ║  🗑️ free()                  释放请求的所有 block      ║
    ║                                                      ║
    ║  辅助接口:                                            ║
    ║  📖 get_blocks/get_block_ids() 查询已分配 block       ║
    ║  ✍️ cache_blocks()          将 block 写入前缀缓存哈希  ║
    ║  🗑️ remove_skipped_blocks() 释放滑动窗口外的 block    ║
    ║  🗑️ evict_blocks()         主动驱逐前缀缓存           ║
    ║  🔄 reset_prefix_cache()    重置前缀缓存（RLHF 用）    ║
    ║  📖 take_events()           拉取 KV 缓存事件           ║
    ║  📖 new_step_starts()       新调度步通知              ║
    ╚══════════════════════════════════════════════════════╝

    ╔══════════════ 🔗 宏观交互调用链 ═════════════════════╗
    ║                                                      ║
    ║  Scheduler.schedule()                                ║
    ║  │                                                   ║
    ║  ├─① get_computed_blocks(request)                    ║
    ║  │   └→ HybridKVCacheCoordinator.find_longest_cache… ║
    ║  │      ├─ FullAttentionManager   (左扫前缀)         ║
    ║  │      └─ SlidingWindowManager   (右扫连续窗口)     ║
    ║  │        → 取交集 → (KVCacheBlocks, hit_len)        ║
    ║  │                                                   ║
    ║  ├─② allocate_slots(request, …)                      ║
    ║  │   ├─ Stage1: remove_skipped_blocks → 窗口外淘汰   ║
    ║  │   ├─ Stage2: get_num_blocks_to_allocate 容量检查  ║
    ║  │   ├─ Stage3: allocate_new_computed_blocks 前缀块  ║
    ║  │   ├─ Stage4: allocate_new_blocks 新 block 分配    ║
    ║  │   └─ Stage5: cache_blocks → 写入前缀哈希          ║
    ║  │                                                   ║
    ║  └─③ free(finished_requests) 释放完成请求            ║
    ║                                                      ║
    ║  ┌── 混合同步屏障 ─────────────────────────────────┐ ║
    ║  │                                                │ ║
    ║  │ Full 层需要 seq 全部 block 的 K/V               │ ║
    ║  │ SWA  层只需要 window_size 内的 block            │ ║
    ║  │ → 前缀命中取交集保证同一 forward 步的一致性     │ ║
    ║  │ → allocate_slots 时 Full/SW 分别算所需 block 数 │ ║
    ║  └────────────────────────────────────────────────┘ ║
    ╚══════════════════════════════════════════════════════╝

    🧬 设计要点：
    - 自身不包含业务逻辑，全部委托给 self.coordinator（三选一）:
      HybridKVCacheCoordinator（≥2 种注意力类型）
      UnitaryKVCacheCoordinator（1 种注意力类型）
      KVCacheCoordinatorNoPrefixCache（禁用前缀缓存）
    - watermark: 为防止新请求频繁被抢占，调度 WAITING/PREEMPTED 请求时
      保留一部分 free block 作为缓冲
    - empty_kv_cache_blocks: 预构造的空 KVCacheBlocks，复用避免 GC
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        max_in_flight_tokens: int | None = None,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
        watermark: float = 0.0,
    ) -> None:
        self.max_model_len = max_model_len
        # When unset, fall back to `max_model_len` so the recycling-aware cap
        # collapses to the prior (uncapped) admission behavior. The scheduler
        # always supplies the real value at runtime.
        if max_in_flight_tokens is None:
            max_in_flight_tokens = max_model_len

        self.enable_caching = enable_caching
        self.enable_kv_cache_events = enable_kv_cache_events
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Watermark: minimum number of KV cache blocks to keep free when
        # admitting waiting/preempted requests, to avoid frequent preemptions.
        assert watermark >= 0.0, "watermark must be non-negative"
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
        self.kv_cache_event_metadata = tuple(
            (
                get_kv_cache_spec_kind(group.kv_cache_spec).value,
                get_kv_cache_spec_sliding_window(group.kv_cache_spec),
            )
            for group in kv_cache_config.kv_cache_groups
        )

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """📖 前缀缓存命中查找 — 调用链第①步。

        🔗 调用链定位：
        Scheduler._schedule_request()
          └─→ KVCacheManager.get_computed_blocks()
               └─→ HybridKVCacheCoordinator.find_longest_cache_hit()
                    ├─→ FullAttentionManager  (左扫：逐 block 查哈希表，首次 miss 停止)
                    └─→ SlidingWindowManager  (右扫：找连续 sliding_window_contiguous_blocks)
                    → 取交集 return (KVCacheBlocks, num_hit_tokens)

        ⚙️ 行为：
        - 前缀缓存关闭或请求标记 skip_reading_prefix_cache → 返回 (空, 0)
        - max_cache_hit_length = request.num_tokens - 1
          ※ 减 1 因为即使全命中也要重算最后一个 token 拿 logits
        - 调用 coordinator.find_longest_cache_hit → 拿到各 group 的命中 block 列表和交集长度
        - 命中 > 0 且开启 kv_cache_events → 发送 BlockStored 事件
        - 开启 log_stats → 记录前缀缓存统计

        📥 request: 包含 block_hashes（递归哈希链）的请求
        📤 (KVCacheBlocks, int): 命中的 block 列表 + 命中 token 数
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # When kv_cache_report_mode is "full", emit BlockStored events
        # for the reused prefix cache blocks so that external consumers
        # (e.g. gateway) can learn about them.
        if (
            num_new_computed_tokens > 0
            and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"
        ):
            for group_idx, group_blocks in enumerate(computed_blocks):
                num_blocks = len(group_blocks)
                if num_blocks > 0:
                    group = self.kv_cache_config.kv_cache_groups[group_idx]
                    block_size = group.kv_cache_spec.block_size
                    self.block_pool.emit_cached_block_events(
                        request,
                        num_blocks,
                        block_size,
                        group_idx,
                    )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        """✍️ KV slot 分配 — 调用链第②步，调度核心。

        🔗 调用链定位：
        Scheduler._schedule_request()
          └─→ KVCacheManager.allocate_slots()
               ├─ Stage1: coordinator.remove_skipped_blocks()
               │           → SWA 层释放滑动窗口外的旧 block
               ├─ Stage2: coordinator.get_num_blocks_to_allocate()
               │           → 各 group 分别计算需要的 block 数（Full 需全序列，
               │              SWA 只需窗口内）
               │           → 容量不够返回 None
               ├─ Stage3: coordinator.allocate_new_computed_blocks()
               │           → 将前缀缓存命中的 block 追加到 request
               ├─ Stage4: coordinator.allocate_new_blocks()
               │           → 从 BlockPool 分配物理 block
               └─ Stage5: coordinator.cache_blocks()
                          → 将 block hash 写入前缀缓存哈希表

        ⚙️ Block 布局：
        ┌──────────────────────────────────────────────────────────┐
        │ < comp > | < new_comp > | < ext_comp > | < new > | lkhd  │
        └──────────────────────────────────────────────────────────┘
                              │<─────── 需要分配 ────────>│
                              │<── 需要写入前缀哈希 ──>│（取 verified 部分）

        ⚙️ 三阶段分配：
        - Stage1：释放 comp 中不必要的 block 并检查容量
        - Stage2：处理前缀 token (comp+new_comp+ext_comp)，释放窗口外 block
        - Stage3：为新 token (new+lookahead) 分配 block

        ⚠️ 混合同步：Full 和 SWA 层各自按自己的 block_size 和 sliding_window
           计算所需 block 数，coordinator 汇总判断容量是否足够。

        📥 num_new_tokens: 新增要计算的 token 数（含 draft token）
        📥 num_new_computed_tokens: 前缀缓存命中新增的 token 数
        📥 new_computed_blocks: 前缀缓存命中的 block 列表
        📥 num_lookahead_tokens: 推测解码 lookahead token 数（EAGLE）
        📥 num_external_computed_tokens: 外部缓存（P/D 场景）命中 token
        📥 delay_cache_blocks: True → 跳过缓存写入（P/D 等待远程接收）
        📥 full_sequence_must_fit: True → 防空转：完整序列放不下就不分配
        📥 reserved_blocks: 留给其他 in-flight 序列的缓冲 block 数
        📤 KVCacheBlocks | None: 新分配的 block 列表（None = 分配失败）
        """

        # ① 入口校验：异步 KV 加载时可能没有新 token 要算，但 ext_comp 仍要分配 slot
        # 如果新 token 和外部缓存 token 都为 0，说明调用有误
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks

        # ② Stage1: full_sequence_must_fit 的门槛检查（防止 chunked prefill 过度接纳）
        #   如果完整序列放不下，直接返回 None，不等 chunked 分批才失败
        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                new_computed_blocks=new_computed_block_list,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_local_computed_tokens=num_local_computed_tokens,
                num_tokens_main_model=full_num_tokens,
                apply_admission_cap=True,  # 启用 SWA/ChunkedLocal 的回收感知容量上限
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None

        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens, self.max_model_len
        )

        # ③ Stage2: 释放滑动窗口外的旧 block（SWA 层环形复用机制的核心）
        #   为什么在分配前做？先清出空间 → 减少后续分配的 evict 压力
        #   为什么用 (total - inflight)？in-flight 步骤的 attention 窗口仍要读旧的 block，
        #   且被拒绝的 spec token 可能回滚，所以不能释放 inflight 范围内的 block
        self.coordinator.remove_skipped_blocks(
            request.request_id,
            max(0, total_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        # ④ Stage2 续: 计算需要分配的 block 数量
        #   get_num_blocks_to_allocate 会遍历所有 single_type_managers:
        #     FullAttentionManager:   ceil(num_tokens / block_size)
        #     SlidingWindowManager:   ceil(min(num_tokens, sliding_window) / block_size)
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        # ⑤ 容量检查：free_blocks - reserved >= required + watermark
        #   reserved_blocks — 留给其他 in-flight 序列完成的缓冲
        #   watermark_blocks — 防止 WAITING/PREEMPTED 请求因接纳过多而频繁抢占
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

        # ⑥ Stage3: 将前缀缓存命中的 block 追加到 request 的 block 列表
        #   为什么先追前缀再分新 block？避免先分新 block 但前缀 block 后又分配不了的尴尬
        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        # ⑦ Stage4: 从 BlockPool 分配全新的物理 block
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # ⑧ P/D 场景：如果要从远端 recv，延迟缓存写入
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # ⑨ Stage5: 将新分配的 block hash 写入前缀缓存哈希表
        #   取 min(new, request.num_tokens) 而不是直接用 new 的原因：
        #   new 包含 draft token（可能被拒），只缓存 verified 部分
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """🗑️ 释放请求的所有 block — 调用链第③步。

        ⚙️ 🔗 Scheduler → KVCacheManager.free() → coordinator.free()
          → 遍历 single_type_managers → free(request_id)
            → block_pool.free_block() 逐个释放（ref_count -= 1，归 0 则归还 free pool）

        ⚠️ 释放顺序：逆序释放（tail block 先释放），这样前缀缓存引用的 block 最后释放，
           确保其他共享这些 block 的请求不会受影响。
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for R-SWA gap eviction.
        """
        self.coordinator.remove_skipped_blocks(
            request_id, processed_computed_tokens, num_prompt_tokens
        )

    def pop_blocks_for_free(self, request: Request) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without
        returning them to the block pool. The caller must eventually free
        them in reverse order (so that tail blocks are evicted first).

        Args:
            request: The request to pop the blocks for.

        Returns:
            The request's blocks in allocation order.
        """
        return self.coordinator.pop_blocks_for_free(request.request_id)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        events = self.block_pool.take_events()
        for event in events:
            if not isinstance(event, BlockStored):
                continue
            if event.group_idx is None:
                continue
            if event.group_idx < 0 or event.group_idx >= len(
                self.kv_cache_event_metadata
            ):
                logger.warning(
                    "Group index `%s` not in KV cache metadata", event.group_idx
                )
                continue
            # Annotate here so BlockPool can keep emitting structural cache
            # events without owning semantic KV cache spec metadata.
            kind, sliding_window = self.kv_cache_event_metadata[event.group_idx]
            event.kv_cache_spec_kind = kind
            event.kv_cache_spec_sliding_window = sliding_window
        return events

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def get_block_ids_for_computed_tokens(
        self,
        request_id: str,
        num_computed_tokens: int,
    ) -> tuple[list[int], ...]:
        """Get block ids covering the request's computed tokens."""
        block_ids = self.get_block_ids(request_id)
        clipped_block_ids: list[list[int]] = []
        for group, ids in zip(self.kv_cache_config.kv_cache_groups, block_ids):
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec) or isinstance(
                spec, (CrossAttentionSpec, EncoderOnlyAttentionSpec)
            ):
                clipped_block_ids.append(ids)
                continue

            num_valid_blocks = cdiv(num_computed_tokens, spec.block_size)
            clipped_block_ids.append(ids[:num_valid_blocks])
        return tuple(clipped_block_ids)

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending copies and return their retained endpoints."""
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks

    def new_step_starts(self) -> None:
        """Notify the coordinator that a new step is starting."""
        self.coordinator.new_step_starts()
