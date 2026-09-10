# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.tasks import GenerationTask
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class ModelSpecificAttnMetadata:
    """Base class for model-specific attention metadata.

    多态注入点:让具体模型(交叉注意力、Mamba 等)在不改主干
    build_attn_metadata 的前提下,往 attention metadata 里塞自己专属的字段。
    主干调用方见 attn_utils.build_attn_metadata。"""

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        # 注入到所有层共用的 CommonAttentionMetadata(按 KV cache group 区分)。
        # 例:EncoderDecoder 注入 encoder_seq_lens;Mamba 注入 is_prefilling。
        return {}

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        # 注入到每个 AttentionGroup 的专属 metadata builder(按 builder 类型区分)。
        # 例:Mamba 仅对 Mamba2/GDN builder 注入 num_accepted_tokens。
        return {}


class ModelState(ABC):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype

        self.supports_mm_inputs = encoder_cache is not None
        if encoder_cache is not None:
            self.encoder_cache = encoder_cache
            self.encoder_runner = EncoderRunner(
                model=self.model,
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                encoder_cache=encoder_cache,
                dtype=self.dtype,
                device=self.device,
            )

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        from vllm.model_executor.models.interfaces import (
            supports_realtime,
            supports_transcription,
        )
        from vllm.model_executor.models.interfaces_base import is_text_generation_model

        supported_tasks = list[GenerationTask]()
        if is_text_generation_model(self.model):
            supported_tasks.append("generate")
        if supports_transcription(self.model):
            if self.model.supports_transcription_only:
                return ("transcription",)
            supported_tasks.append("transcription")
        if supports_realtime(self.model):
            supported_tasks.append("realtime")
        return tuple(supported_tasks)

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        # 请求加入 batch 时的钩子。默认无操作;Default 用来初始化 RoPE 位置,
        # Mamba 用来 seed running state block 索引。
        return None

    def remove_request(self, req_id: str) -> None:
        # 请求离开 batch 时的钩子(默认无操作)。
        return None

    def apply_staged_writes(self) -> None:
        # 把"暂存写"刷入实际状态(如 RoPE 的位置更新)。在需要惰性更新状态时调用。
        return None

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Hook run on real batches before the forward pass (after block tables
        are gathered). Used by mamba "align" prefix caching to pre-copy state
        across block boundaries. No-op by default."""
        # 在 forward 前、block tables 已 gather 之后运行。Mamba 在此把循环状态
        # 跨块边界预拷贝(align 语义)。返回 None = 默认 no-op。
        return None

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        # forward 之后的钩子,用于记录本步接受数 / 保存非注意力状态。
        # Mamba 在此 scatter num_accepted_tokens 并做 align 后处理。默认 no-op。
        return None

    @abstractmethod
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        raise NotImplementedError

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor | None:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        return None

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Gather cached multimodal embeddings."""
        return self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            input_batch.prefill_len_np,
            input_batch.num_computed_tokens_np,
            draft_lookahead=draft_lookahead,
        )

    @abstractmethod
    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        # 构造 attention metadata 的核心钩子。子类选 padded/未 padded 形状,算
        # max_seq_len,最后调用 build_attn_metadata 产出 {layer_name: metadata}。是
        # 连接 ModelState 与 AttentionGroup/metadata builder 的主入口。
        raise NotImplementedError

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        """Wrap or replace the default sampler.

        Called after model loading with the already-constructed base
        ``Sampler``.  Return ``None`` to keep the defaults, or
        ``(sampler, rejection_sampler | None)`` to override.
        """
        # 允许模型替换默认采样器(如 Medusa/EAGLE 自定义采样路径)。返回 None 保持默认。
        return None

    num_new_sampled_tokens_per_step: int = 1
    """New tokens sampled on each decode step 
    (excluding accepted draft tokens, a.k.a num bonus tokens)."""
