# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    # AsyncScheduler 只是 Scheduler 的薄封装：它自身仍是同步对象（schedule() 同步返回），
    # 真正的"调度与执行重叠"发生在上层 EngineCore/executor。这里只重写两个 Hook 来维护
    # 在 overlap 场景下不自相矛盾的状态（占位符记账 + 丢弃过期在途帧）。
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ⚠️ 复用的只读占位符列表，用于投机解码（spec decoding）的 draft token 占位，默认全 -1。
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        # 流水线并行大小，用于 V2 runner 计算 decode 可调度步号（PP microbatching）。
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # 每步 schedule() 之后调用。先执行父类逻辑（更新 running/waiting、分配块等），
        # 再为处于 overlap 中的请求补充占位符计数。
        super()._update_after_schedule(scheduler_output)

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Use the latest num of scheduled draft tokens in next step as placeholder.
        # ⚠️
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule

        # 遍历本步所有被调度的请求。
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            # prefill 的分块（chunk）阶段不需要占位符，跳过。
            if request.is_prefill_chunk:
                continue

            # 若请求使用结构化输出且其仍有占位符，本步标记为"有待定结构化输出 token"。
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))

            # ⚠️ 累加占位符数量：提前为"已经排了但 GPU 还没产出真实 token"的位置记账，
            # 主 token（num_sampled_tokens_per_step） + 投机 draft token
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            # ⚠️
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # 计算该请求下一轮可被调度做 decode 的步号（当前步 + PP 大小），
                # 用于流水线并行的 microbatch 对齐。
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # 模型输出回填时调用（每个请求一次）。先处理"被 force-preempt 的过期在途帧"。
        if request.async_tokens_to_discard > 0:
            # 在 reset_prefix_cache 等场景被强制抢占后，会有一批已经发出、但已过期的
            # 异步输出帧。每调用一次丢弃一帧，直到计数器耗尽，期间不真正消费任何 token。
            request.async_tokens_to_discard -= 1
            return [], False

        # 记录更新前的状态，用于下面判断该请求是否还处在 RUNNING（抢占的请求要跳过 cache）。
        status_before_update = request.status

        # ⚠️ 调用父类逻辑：把真实采样的 token 写入请求，返回（可能被 truncated 的 token, 是否停止）。
        # ⚠️ Scheduler：简单的将新tokens追加到req.output_ids
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # ⚠️ 真实产出了 token，扣减之前累加的占位符计数。断言保证不出现负数（防止下溢）。
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # 把新 token 对应的 KV 块标记为已缓存。被抢占（状态不再 RUNNING）的请求跳过，
        # 因为它的块可能已被回收/重分配。
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
