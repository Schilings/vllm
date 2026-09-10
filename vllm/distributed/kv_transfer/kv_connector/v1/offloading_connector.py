# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request


class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):
    """KV offload 的总入口（V1 连接器）。

    本类是一个**薄分发器**：根据 ``role`` 把调用转发给
    - ``SCHEDULER`` 角色 -> :class:`OffloadingConnectorScheduler` （运行在调度进程）
    - ``WORKER``   角色 -> :class:`OffloadingConnectorWorker` （运行在每个 worker 进程）

    Scheduler 持有 SCHEDULER 角色的实例（``self.connector``，见
    ``vllm/v1/core/sched/scheduler.py``）。每个方法的 docstring 注明了它是被
    Scheduler 还是 worker 在什么时机调用、以及相对调用顺序，便于按调度主循环顺序阅读。
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """返回 ``True`` 让调度器优先分配“跨层”block（一个 block 容纳所有层的 KV），
        加速 offload 时的 KV 传输；对应 worker 侧的 ``register_cross_layers_kv_cache``。
        """
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    def shutdown(self) -> None:
        """⑨ 生命周期: 进程关闭时调用，清理 worker / scheduler 两侧资源。"""
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()

    # 5️⃣ Worker 侧（WORKER 角色）：收到 scheduler 下发的 metadata 后，在 forward 前后被调用。
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """5️⃣ Worker: 初始化时注册各层 KV cache 张量（仅一次）。

        转发给 ``OffloadingConnectorWorker.register_kv_caches``。
        仅当连接器未使用跨层 block（``prefer_cross_layer_blocks`` 为 False）时走这里。
        """
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        """⑤ Worker: 初始化时注册跨层 KV cache 张量（仅一次）。

        转发给 ``OffloadingConnectorWorker.register_cross_layers_kv_cache``。
        本类 ``prefer_cross_layer_blocks`` 返回 True，因此走的是这条路而非
        ``register_kv_caches``。
        """
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        """5️⃣ Worker: 请求被抢占或 block 即将被淘汰/覆盖前调用。
        转发给 ``OffloadingConnectorWorker.handle_preemptions``，用于在 block 被
        覆盖前保全尚未完成的异步 save 数据。``kv_connector_metadata`` 必须是
        ``OffloadingConnectorMetadata``。
        """
        assert self.connector_worker is not None
        assert isinstance(kv_connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """5️⃣ Worker: forward 前调用，异步把外部 KV 载入分页 buffer。
        转发给 ``OffloadingConnectorWorker.start_kv_transfers``，使用当前绑定好的
        ``OffloadingConnectorMetadata``（由 scheduler 的 ``build_connector_meta`` 下发）。
        """
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # OffloadingConnector 采用整步异步加载，不在每层等待，故为空实现。
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        # 落盘（store）被推迟到 get_finished() 统一排队，故每层 save 为空实现。
        pass

    def wait_for_save(self):
        # Store deferral is handled in get_finished(), which always runs even
        # when wait_for_save() is skipped (e.g. kv_connector_no_forward).
        pass

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """5️⃣ Worker: 回报完成异步传输的 req id，并把 store 任务排队到下一步。
        转发给 ``OffloadingConnectorWorker``：
        1. ``prepare_store_kv`` 先把本步需要落盘的 KV 排进下一步的
           ``start_kv_transfers``（放在这里而非 ``wait_for_save``，保证即便
           ``wait_for_save`` 被跳过也能排队）。
        2. ``get_finished`` 返回已完成异步收/发的 req id，供 scheduler 跟踪。
        """
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)

        # Defer store jobs to the next step's start_kv_transfers. Done here
        # (rather than wait_for_save) so stores are queued even on steps where
        # wait_for_save is skipped.
        #
        self.connector_worker.prepare_store_kv(self._connector_metadata)
        #
        return self.connector_worker.get_finished(finished_req_ids)

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """5️⃣ Worker: 把 worker 侧状态回传给 scheduler（对应 ⑥ ``update_connector_output``）。

        转发给 ``OffloadingConnectorWorker.build_connector_worker_meta``；若当前是
        scheduler 角色实例（无 worker）则返回 ``None``。
        """
        if self.connector_worker is not None:
            return self.connector_worker.build_connector_worker_meta()
        return None

    # ①–④、⑥–⑨ Scheduler 侧（SCHEDULER 角色）：由 scheduler 主循环按序调用。
    def on_new_request(self, request: "Request") -> None:
        """1️⃣ Scheduler: 新请求加入调度（``add_request``）时调用。

        把 request 注册进 ``OffloadingConnectorScheduler``，使其后续在
        ``get_num_new_matched_tokens`` 中能查到 offload tier 上的外部 KV 命中。
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """2️⃣ Scheduler: ``schedule()`` 评估每个待调度请求时调用。

        返回 ``(num_external_tokens, load_kv_async)``：从 offload tier 还能加载多少
        token 的 KV。返回 ``None`` 表示连接器还没算完，Scheduler 会把这个请求推迟到
        下一步再查。结果用于计算 ``num_external_computed_tokens``。
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """3️⃣ Scheduler: 给请求分配 KV block（allocate/append slots）之后调用。

        转发给 ``OffloadingConnectorScheduler.update_state_after_alloc``，记录哪些
        block 将接收外部加载的 KV，并据此决定是否触发一次 load。
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """4️⃣ Scheduler: ``schedule()`` 末尾调用，构造下发 worker 的 metadata。

        构造 ``OffloadingConnectorMetadata`` 并挂到
        ``scheduler_output.kv_connector_metadata``，下发给 worker。注意：此调用会
        **重置**连接器 scheduler 侧的状态（每个 step 一次）。
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def has_pending_push_work(self) -> bool:
        """⑧ Scheduler: ``has_requests()`` 中调用，保持在途 push 写。

        即便没有“活”请求，只要连接器还有在途的 push 写（异步落盘），也返回 ``True``
        以保持引擎主循环继续 step。
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_pending_push_work()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """⑥ Scheduler: 回收 worker 输出后调用，回灌聚合后的 metadata。

        转发给 ``OffloadingConnectorScheduler.update_connector_output``：
        ``finished_recving`` -> 下一步可调度该请求；``finished_sending`` -> 释放 block。
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """⑦ 非 HMA 路径: 请求结束时调用。

        转发给 ``OffloadingConnectorScheduler.request_finished``。返回 ``True`` 表示
        连接器接管了 block 的异步释放，Scheduler 暂不解绑，直到 ``get_finished`` 回报该
        req id。本类支持 HMA，正常走 ``request_finished_all_groups``。
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """⑦ HMA 路径（本类走这里）: 请求结束时调用。

        转发给 ``OffloadingConnectorScheduler.request_finished``。返回 ``True`` 表示
        连接器接管了 block 的异步释放，Scheduler 暂不解绑，直到 ``get_finished`` 回报该
        req id。
        """
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    def take_events(self) -> Iterable[KVCacheEvent]:
        """⑨ Scheduler: 抽取自上次调用以来新增的 KV cache 事件。"""
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """⑨ 声明本连接器要求的 KV cache 布局为 ``"HND"``。"""
        return "HND"

    # ---- ⑨ Scheduler: 重置连接器内部缓存 ----
    def reset_cache(self) -> bool | None:
        """⑨ Scheduler: 重置连接器内部缓存（如 prefix cache 命中表）。"""
        assert self.connector_scheduler is not None
        self.connector_scheduler.reset_cache()
        return True

    # ---- ⑨ Scheduler: 取统计信息 ----
    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """⑨ Scheduler: 取上一统计周期的 KV 连接器统计信息。"""
        if self.connector_scheduler is not None:
            return self.connector_scheduler.get_stats()
        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        """⑨ 构造 ``OffloadingConnectorStats``（用于反序列化历史统计 data）。"""
        return (
            OffloadingConnectorStats(data=data)
            if data is not None
            else OffloadingConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        """⑨ 构造 offload 专用的 Prometheus 指标集合 ``OffloadPromMetrics``。"""
        return OffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )
