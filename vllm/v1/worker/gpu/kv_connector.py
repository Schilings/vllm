# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        pass

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass


class ActiveKVConnector(KVConnector):
    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ):
        self.vllm_config = vllm_config
        # 取出全局唯一的 KV transfer group 实例（Scheduler 侧与 Worker 侧共享同一注册表）。
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        # TODO: support cross_layers_kv_cache
        # (see https://github.com/vllm-project/vllm/pull/27743)
        # 当前 V2 只走逐层 dict 注册，尚未接入跨层/uniform 布局的批量传输优化。
        self.kv_connector.register_kv_caches(kv_caches_dict)
        # 设置宿主↔设备之间搬运 KV block 的底层拷贝算子（如 copy_kv_blocks）。
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._disabled = False

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        # 每个 forward 前调用：处理抢占、绑定本次调度的 metadata，并启动 KV load。
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        # 请求被抢占/block 即将被覆盖前，保全尚未完成的异步 save。
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        # 把本步的 connector metadata 绑定到当前上下文，供后续 load/save 使用。
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)

        # TODO: sort out KV Connectors' use of forward_context
        # 启动 KV load：按 metadata 把命中前缀的 KV 从卸载层（CPU 等）搬回 GPU。
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())

    def post_forward(
        self, finished_req_ids: set[str], wait_for_save: bool = True
    ) -> KVConnectorOutput | None:
        # 每个 forward 后调用：等待异步 save 完成并收集本轮 connector 的输出状态。
        if self._disabled:
            return None

        output = KVConnectorOutput()
        if wait_for_save:
            # 阻塞等待本步所有异步 KV save（offload）落地。
            self.kv_connector.wait_for_save()
        # 分别收集：已发完/已收完的请求、load 出错的 block、统计信息、
        # KV cache 事件（供 prefix cache 等消费）、以及 worker 侧元数据。
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        # 清空本步绑定的 metadata，避免影响后续调度步。
        self.kv_connector.clear_connector_metadata()
        return output

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        # 当本步没有真正的模型 forward（例如某些调度步只需推进 connector 状态）时，
        # 仍然跑一遍 connector 的 pre/post 钩子，仅产出 KV connector 输出。
        if self._disabled:
            return EMPTY_MODEL_RUNNER_OUTPUT

        self.pre_forward(scheduler_output)
        finished_req_ids = scheduler_output.finished_req_ids
        # 这里不等待 save（wait_for_save=False），因为本步不走正常 forward 流程。
        kv_connector_output = self.post_forward(finished_req_ids, wait_for_save=False)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    def set_disabled(self, disabled: bool) -> None:
        # 禁用/启用 connector：禁用时把全局 agent 置空，确保逐层 connector 钩子不再被调用。
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR
    # ⚠️ 
    return ActiveKVConnector(vllm_config, kv_caches_dict)
