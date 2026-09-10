# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _TransferMetricName,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_blocks: int | None
    # Number of this group's offloaded blocks per full-attention alignment
    # segment. Used to skip storing SWA blocks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_block_count: int | None = None
    # True for EAGLE/MTP draft-model attention groups. The trailing block
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False


def get_sliding_window_size_in_blocks(
    kv_cache_spec: KVCacheSpec, offloaded_block_size: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, offloaded_block_size)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def resolve_mamba_align_size(spec: "OffloadingSpec") -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be rounded
    down to a multiple of the offloaded block size. Asserts that all such
    groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, gpu_block_size in enumerate(spec.gpu_block_size):
        kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            offload_block_size = gpu_block_size * spec.block_size_factor
            assert mamba_align_size is None or mamba_align_size == offload_block_size
            mamba_align_size = offload_block_size
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int
    offload_prompt_only: bool

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the offloaded_block_size of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_offloaded_block_sizes: set[int] = set()
        for idx, gpu_block_size in enumerate(spec.gpu_block_size):
            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_blocks(
                kv_spec, gpu_block_size * spec.block_size_factor
            )
            if sw is None:
                full_attn_offloaded_block_sizes.add(
                    gpu_block_size * spec.block_size_factor
                )

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_offloaded_block_sizes) == 1:
            alignment_tokens = full_attn_offloaded_block_sizes.pop()

        def _alignment_block_count(
            offloaded_block_size: int,
            sliding_window_size_in_blocks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_blocks is None:
                return None
            if alignment_tokens <= offloaded_block_size:
                return None
            per_segment = alignment_tokens // offloaded_block_size
            if sliding_window_size_in_blocks >= per_segment:
                return None
            return per_segment

        eagle_groups = {
            idx
            for idx, g in enumerate(spec.kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }

        use_eagle = (
            spec.vllm_config.speculative_config is not None
            and spec.vllm_config.speculative_config.use_eagle()
        )
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(spec.kv_cache_config.kv_cache_groups)))

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing block of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                    sliding_window_size_in_blocks=(
                        sw := get_sliding_window_size_in_blocks(
                            spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            gpu_block_size * spec.block_size_factor,
                        )
                    ),
                    alignment_block_count=_alignment_block_count(
                        gpu_block_size * spec.block_size_factor, sw
                    ),
                    kv_event_group_spec=get_offloading_event_group_spec(
                        spec.kv_cache_config.kv_cache_groups[idx]
                    ),
                    is_eagle_group=idx in eagle_groups,
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
            offload_prompt_only=spec.offload_prompt_only,
        )


@dataclass
class RequestGroupState:
    # ⚠️
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    #
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    #
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    # local prefix caching的命中长度
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    # 记录请求的当前进行中的传输任务
    transfer_jobs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        # ⚠️
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        """把请求新增的 block_hashes 增量转成每个 offloaded block 一个 offload key。

        粒度换算：block_hashes 以 hash_block_size 个 token 为一块（细粒度），而一个
        offloaded block 跨 offloaded_block_size 个 token（粗粒度）。二者关系为
        hash_block_size_factor = offloaded_block_size // hash_block_size，即 f 个
        block_hash 合成 1 个 offloaded block，并取其“最后一个”哈希作为该 block 的代表键。

        本方法可被反复调用，只为尚未处理的新 offloaded block 补 key（幂等增量）。
        """
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            f = group_config.hash_block_size_factor
            # 已处理的 offloaded block 数 n = len(offload_keys)。
            # 起点 f*n + f - 1 = (n+1)*f - 1 恰为 offloaded block n 的最后一个哈希下标；
            # 之后以 f 为步长跳到每个后续 offloaded block 的末尾哈希，直到 block_hashes 末尾。
            # 设 f=2，block_hashes = [h0,h1,h2,h3,h4,h5]（6 个哈希 → 3 个 offloaded block）。
            # 若 offload_keys 已有 1 项（n=1，block 0 已处理）：
            # start = 2·1+2-1 = 3，step 2 → 下标 3, 5 → 哈希 h3, h5（即 block 1、block 2 的代表哈希）
            # 追加 make_offload_key(h3, g)、make_offload_key(h5, g)
            # 之后 offload_keys 长度变为 3，与 6//2 对齐。
            for req_block_hash in islice(
                self.req.block_hashes,
                f * len(group_state.offload_keys) + f - 1,
                None,
                f,
            ):
                # 用「块内容哈希 + group 下标」组成全局唯一 key，供后续 lookup/store。
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


def _create_req_context(req: Request) -> ReqContext:
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=req.kv_transfer_params,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
    ):
        #
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats: OffloadingConnectorStats | None = None

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(spec)

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            # 查看存储的cpu blocks是否含有
            match self.manager.lookup(key, req_context):
                case LookupResult.HIT:
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                # 中断
                case LookupResult.MISS:
                    break
        # 如果有一块还没传输完成，就返回None，甚至不是返回0
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            match self.manager.lookup(keys[idx], req_context):
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0

            # 一旦匹配到window大小，可以直接返回
            if consecutive_hits == sliding_window_size:
                # 返回的是 idx + sliding_window_size 表示前面所有block都被hit了
                return idx + sliding_window_size if not defer_lookup else None

        # 如果有一块还没传输完成，就返回None，甚至不是返回0
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            # ⚠️ 减1
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing block
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        # 循环：找到多个kv group都能接受的最大hit length
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            # hybird attention，多个kv group需要prefix caching
            # 每个group进行prefix caching
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                # 进行prefix caching，肯定得先有hash值 ---> offload_keys
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys)
                    >= req_status.req.num_tokens // offloaded_block_size
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to block-aligned boundary for this group
                # max_hit_size_tokens 需要取 多个groups的最小值
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                # sliding window size 向上取整 几个cpu block
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                # For eagle groups, query one extra block that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_blocks is not None:
                    query_max = min(
                        max_hit_size_tokens + offloaded_block_size,
                        len(offload_keys) * offloaded_block_size,
                    )

                # 需要进行prefix caching匹配的范围
                num_blocks = min(
                    cdiv(query_max, offloaded_block_size), len(offload_keys)
                )
                # 前面一部分是已经在gpu上存在的了，不需要做cpu的prefix caching
                start_block_idx = num_computed_tokens // offloaded_block_size
                # 只对后面部分进行cpu的prefix caching
                offload_keys = offload_keys[start_block_idx:num_blocks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_blocks: int | None
                if sliding_window_size_in_blocks is None:
                    # FullAttention越长越好
                    # 如果有一块还没传输完成，就返回None，甚至不是返回0
                    num_hit_blocks = self._maximal_prefix_lookup(
                        offload_keys, req_status.req_context
                    )
                else:
                    # SlidingWindowAttention要么短连续要么达到 sliding_window_size_in_blocks 个cpu block
                    # 如果有一块还没传输完成，就返回None，甚至不是返回0
                    # 返回的是 idx + sliding_window_size
                    # 表示前面offload_keys[:idx + sliding_window_size]所有block都被hit了
                    required_window = sliding_window_size_in_blocks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_blocks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if num_hit_blocks == 0:
                    return 0

                # 如果有一块还没传输完成，就返回None，甚至不是返回0
                if num_hit_blocks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_blocks -= 1
                        eagle_verified.add(group_idx)

                    # max_hit_size_tokens 需要取 多个groups的最小值
                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        # 这里把前面gpu的prefix cache也算上了
                        offloaded_block_size * (start_block_idx + num_hit_blocks),
                    )

                # max_hit_size_tokens 是 gpu+cpu 的prefix cache总长度
                # 减去gpu的就是cpu的 --> new_num_hit_tokens
                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens

                # 一个block都hit不满
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                # 小于上一次hit，说明hit的范围变小了，需要重新进行prefix caching
                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        # _blocks_being_loaded 跟踪「当前正在从 CPU 加载到 GPU 的 block 集合」，
        # 防止对同一个 prefix-cached block 发起重复并发加载，导致 CUDA stream 竞态
        if self._blocks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                offloaded_block_size = group_config.offloaded_block_size
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                # 只看hit中的cpu部分的block hash
                offload_keys = offload_keys[start_block_idx:num_blocks]
                # 如果是swa，只看hit中的尾部window内的block
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]

                # 如果任何一个已经在 _blocks_being_loaded 里，就延迟这个请求，返回 None
                # 说明有某个block正在从cpu加载到gpu，被这个请求hit中了，
                # 为什么防止重复load，取消这个请求的cpu load
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        # 最终返回的还是 cpu上命中的token数
        return num_hit_tokens

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        """1️⃣ Scheduler: 新请求加入调度（``add_request``）时调用。
        把 request 注册进 ``OffloadingConnectorScheduler``，使其后续在
        ``get_num_new_matched_tokens`` 中能查到 offload tier 上的外部 KV 命中。
        """
        # Context 1
        req_context = _create_req_context(request)
        # Context 2
        offloading_context = self.manager.on_new_request(req_context)
        # State
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        # 记录state
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        """2️⃣ Scheduler: ``schedule()`` 评估每个待调度请求时调用。

        返回 ``(num_external_tokens, load_kv_async)``：从 offload tier 还能加载多少token 的 KV。
        返回 ``None`` 表示连接器还没算完，Scheduler 会把这个请求推迟到下一步再查。
        结果用于计算 ``num_external_computed_tokens``。
        """
        # 准备进行ext prefix caching，先清空
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        # 该请求有
        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        # 进行prefix caching，肯定得先有hash值 ---> offload_keys
        # cpu block size更大，从gpu block size中取出尾部hash代表cpu block的hash
        req_status.update_offload_keys()
        # 在这之前的local prefix caching命中长度
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            # ext prefix cache hit长度
            # 使用cpu block hash ---> offload_keys
            # 类似gpu prefix caching，使用hash在进行混合注意力的前缀匹配
            num_hit_tokens = self._lookup(req_status)

        # 记录prefix cache总长度（local + external）
        req_status.update_num_hit_blocks(num_computed_tokens + (num_hit_tokens or 0))

        # 将命中的cpu block都标记lru最近使用，延迟被删
        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        """"""
        """3️⃣ Scheduler: 给请求分配 KV block（allocate/append slots）之后调用。
        转发给 ``OffloadingConnectorScheduler.update_state_after_alloc``，
        记录哪些block 将接收外部加载的 KV，并据此决定是否触发一次 load。
        """
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        # ⚠️ 记录当前request哪些block hash，需要从cpu load
        keys_to_load: list[OffloadKey] = []
        #
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            # 记录当前调度该request分配的新block id
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys

            # ⚠️ prefix cache要存放在gpu block数，前一半在gpu，后一半需要从cpu load
            num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            # prefix cache要存放在gpu block数，前一半在gpu，后一半需要从cpu load
            # ⚠️ 遍历找到前一半在gpu的block边界，前一半在gpu的block数
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * gpu_block_size
            )
            # ⚠️ 后一半需要从cpu load的block数
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_blocks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_blocks
                    * self.config.block_size_factor
                )

            # cpu block size不一样，看看cache拆分cpu block有多少个
            num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
            assert len(offload_keys) >= num_blocks
            if num_pending_gpu_blocks:
                start_block_idx = (
                    num_locally_computed_gpu_blocks // self.config.block_size_factor
                )
                # ⚠️ 确定需要从cpu load的block hash
                keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            # ⚠️ 目标是 后半部分的空的gpu blocks
            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            # ⚠️ 后一半的gpu blocks数：每个group有几个gpu block需要从cpu reload之后存放
            group_sizes.append(num_pending_gpu_blocks)
            # ⚠️ 前一半的gpu blocks数
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_block_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_block_idx = num_blocks

        # ⚠️ 来源是 后半部分对应的cpu blocks
        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        # ⚠️ 提交传输任务
        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        # ⚠️ 同一个Request的传输任务，有且同时只能有一个，不管是是load还是store
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        # ⚠️ 记录哪些cpu block正在load中
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        # ⚠️ 遍历所有调度的请求 (请求id，调度新分配的blocks，是否从被抢占状态恢复的)
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            # ⚠️ 更新 cpu block hash
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_block_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            block_size_factor = self.config.block_size_factor
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_block_idx * block_size_factor
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        # 本函数：遍历本步被调度的所有请求，为每个请求构造"把 GPU KV 存到 CPU offload tier"的 store 任务。
        # 输出 store_jobs 会被 build_connector_meta 收集并下发给 worker 执行。
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}

        # ⚠️ 只遍历本步实际被调度的请求（有 num_scheduled_tokens 才算这一步行进过）
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            # 未被 offloading 跟踪的请求（如未启用 offload 的请求）直接跳过
            if req_status is None:
                continue
            req = req_status.req

            # ⚠️ 计算本步结束后该请求已算出的 token 数，并夹到真实总 token 数以内
            # （异步调度下 num_computed_tokens 可能短暂超前于实际已算 token，需 min 兜底）
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
            # 用户可通过 kv_transfer_params["max_offload_tokens"] 限制最多 offload 多少 token
            max_offload_tokens = req_status.max_offload_tokens
            if max_offload_tokens is not None:
                num_offloadable_tokens = min(num_offloadable_tokens, max_offload_tokens)

            # Skip decode-phase blocks: clamp to the prompt length so only
            # prefill (prompt) blocks become eligible for store. next_stored_idx
            # never advances past this boundary, so decode blocks are never
            # queued in this or any later step.
            # offload_prompt_only=True 时，只 offload prefill（prompt）阶段的 block，
            # decode 阶段的 block 永不入队（next_stored_idx 不越过 prompt 边界）。
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req.num_prompt_tokens
                )

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            # 第一遍：确定本步"有哪些 offloaded block 的 key 需要被 store"。
            # 仅做 key 筛选（不构造 block_id 列表），因为还要问 manager 这些 key 是否真值得存。
            # ⚠️ 需要确定哪些cpu bloch hash用来进行store
            # 每个 kv group都不一样
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                # ⚠️ 本 group 能 offload 的 block 数 = 可 offload token 数 // 该 group 的 offloaded_block_size（CPU 粒度）
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                # EAGLE/MTP draft 组的最后一个 block 是易变的（无稳定 hash），排除掉
                if group_config.is_eagle_group:
                    num_blocks = max(0, num_blocks - 1)

                # 记录了store的进度
                # ⚠️  next_stored_block_idx 记录"上次已 offload 到第几个 offloaded block"，本次从它继续（增量、幂等）
                start_block_idx = group_state.next_stored_block_idx
                # 本次没有新 block 需要 offload，跳过该 group
                if num_blocks <= start_block_idx:
                    continue

                # ⚠️ 取出 [start_block_idx, num_blocks) 区间内的 offload key（每个 offloaded block 一个 key）
                # ⚠️ CPU block hash！！
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]

                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                # ⚠️ 每个 offloaded block 由 factor 个 GPU 子块组成；这里按"代表子块"取样——
                # ⚠️ 取每个 offloaded block 内的最后一个 GPU 子块（下标 = start*block_size_factor + block_size_factor - 1，步长 block_size_factor）
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)

                # SWA 对齐优化相关：本 group 一个 alignment segment 含多少 offloaded block；tail 为滑动窗口大小（block 数）
                alignment_block_count = group_config.alignment_block_count
                tail = group_config.sliding_window_size_in_blocks

                # ⚠️
                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    # 代表子块为 null（block_id==0）说明该 offloaded block 整体是 skip/null，直接跳过不存
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%.
                    # 滑窗组优化：只有每个对齐段末尾 tail 个 block 才可能被 load 命中，其余段内靠前的 block 永不命中，跳过不存
                    if alignment_block_count is not None:
                        assert tail is not None
                        abs_block_idx = start_block_idx + key_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    # ⚠️ 需要确定哪些cpu bloch hash用来进行store
                    # 通过全部过滤的 key 进入"候选 store key"列表
                    new_offload_keys.append(offload_key)

            # 没有任何候选 key，推进进度指针后跳过本请求（不构造 job）
            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            # ⚠️ 需要确定哪些cpu bloch hash用来进行store
            # 把候选 key 交给 manager（如 LMCache/本地池），由其决定真正要存的 key 集合（去重、容量、是否已存在等）
            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            # manager 返回的 keys_to_store 可能比候选少（例如已存在/被策略拒绝）；为空则同样推进进度跳过
            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            # ⚠️ 刷新这些 key 的 LRU（标记为最近使用，避免被淘汰）
            self._touch(req_status)

            # 转成集合便于 O(1) 查找
            keys_to_store = set(store_output.keys_to_store)

            # ⚠️ 第二遍：基于 manager 最终认可的 keys_to_store，逐 group 构造真实的 src_spec（GPU 子块列表）
            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                # 滑窗组 vs 全注意力组的区别仅在于"block 被跟踪淘汰的时机"（见 _block_id_to_pending_jobs 逻辑）
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                # ⚠️ cpu block
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                # ⚠️ cpu block
                start_block_idx = group_state.next_stored_block_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                # ⚠️ cpu block --> gpu block
                # 记录本 group 第一个"有效（非 null）GPU 子块"的绝对下标，用于 block_indices
                start_gpu_block_idx: int | None = None
                # 遍历本 group 本次要 offload 的 offloaded block 区间
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    # manager 不认可的 key 跳过（不搬该 block）
                    if offload_key not in keys_to_store:
                        continue

                    offloaded_block_idx = start_block_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, offloaded_block_idx, offload_key
                    )

                    # ⚠️ 把 offloaded block 下标换算成它在 GPU 子块序列里的起点下标
                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    # 展开该 offloaded block 内的 factor 个 GPU 子块，逐个处理
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        # null 子块（padding/skip）跳过：不加入 src、不占目的端槽位
                        if block_id == 0:
                            continue

                        # ⚠️记录第一个有效子块的下标（仅首遇时），即本 group 的 block_indices
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        # 有效 GPU 子块加入源列表（物理已剔除 null，干净列表）
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        # 按组类型分别记录，供后续"被抢占/回收时 flush 该 job"使用
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                # 本 group 实际有效子块数
                group_sizes.append(num_group_blocks)
                # ⚠️ block_indices[i] = 第 i 个 group 的第一个有效 GPU 子块相对于"该 group 逻辑连续 GPU 子块序列起点"的偏移。
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

                # 若本 group 没有任何有效子块，退化为 0（worker 端按 block_idx%factor 对齐也不会出错）
                block_indices.append(start_gpu_block_idx or 0)
                # 推进该 group 的 offload 进度指针到 num_blocks，保证下次从新位置增量构造
                group_state.next_stored_block_idx = num_blocks

            # 组装源 spec：GPU 子块列表 + 各 group 子块数 + 各 group 首个有效子块偏移
            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            # 目的 spec 由 manager 提供（已含 CPU block 布局/分配），store 路径下 worker 据此写入
            dst_spec = store_output.store_spec

            # 生成本次 job 的唯一 id
            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            # 同一请求同时只能有一个方向在飞；若有在飞任务则必为 store（load 与 store 互斥，详见 update_state_after_alloc 注）
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            # 滑窗 block 可能在请求结束前就被回收（滑动窗口推进），注册到 _block_id_to_pending_jobs，
            # 一旦该 block_id 被重新分配即触发 flush，确保 store 在其被覆盖前完成
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            # 非滑窗 block 在请求运行期由 ref_cnt 保护，只在请求结束时才登记以保护 store 完成
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            # 记录 job 供 build_connector_meta 下发
            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """"""
        """4️⃣ Scheduler: ``schedule()`` 末尾调用，构造下发 worker 的 metadata。
        构造 ``OffloadingConnectorMetadata`` 并挂到 ``scheduler_output.kv_connector_metadata``，下发给 worker。
        注意：此调用会**重置**连接器 scheduler 侧的状态（每个 step 一次）。
        """

        # ⚠️
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )
        # ⚠️
        self.manager.on_schedule_end(schedule_end_context)

        # Flush jobs for preempted requests.
        # ⚠️ 处理被抢占的请求：阻塞确保其store任务已经完成
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        # ⚠️
        meta = OffloadingConnectorMetadata(
            # 此次调度需要进行的 load 任务
            load_jobs=self._current_batch_load_jobs,
            # ⚠️
            store_jobs=self._build_store_jobs(scheduler_output),
            #
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        return bool(self._jobs) or self.manager.has_pending_work()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            if self._connector_stats is None:
                self._connector_stats = transfer_stats
            else:
                self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.non_sliding_window_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if not req_status.transfer_jobs and req_status.req.is_finished():
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self._connector_stats
        self._connector_stats = None

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            return False, None

        self.manager.on_request_finished(req_status.req_context)

        if not req_status.transfer_jobs:
            # No in-flight jobs: no later complete_store()/complete_load() calls
            # need this request's state.
            del self._req_status[request.request_id]
            return False, None

        # In-flight jobs remain after the request stopped. Their completion may
        # still call manager.complete_store()/complete_load(), so keep req_status.
        # Pending stores outlive the request's block ownership; register them so
        # future reuse of those blocks triggers a flush.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored blocks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from block 0
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_block_idx = 0

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
