# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    compute_mm_prefix_ranges,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.rope import get_rope_state
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class DefaultModelState(ModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)

        # 默认只管理 rope state？
        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        # Pruner is used for multimodal embedding pruning (EVS).
        self.mm_pruner = maybe_create_mm_pruner(
            self.model_config, model, self.rope_state, encoder_cache
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        # 新请求入 batch:初始化 RoPE 的 prefill 位置(若有 rope_state)。
        if self.rope_state is not None:
            assert new_req_data.prefill_token_ids is not None
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=new_req_data.mm_features,
            )

    def apply_staged_writes(self) -> None:
        # 把 RoPE 暂存的位置更新刷入实际状态(与 get_mm_embeddings 的 EVS 重算联动)。
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        return self.encoder_runner.inputs_embeds[:num_tokens]

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        # 1) 准备并(按需)执行多模态 encoder,产出视觉等 embedding。
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            # Execute the multimodal encoder.
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            # Cache the encoder outputs by mm_hash
            # 按 mm_hash 缓存 encoder 输出,相同输入可复用,避免重复编码。
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        # 2) 取出缓存的多模态 embedding(基类从 encoder_cache 按位置 gather)。
        mm_embeds, is_mm_embed = super().gather_mm_embeddings(input_batch)
        if self.mm_pruner is not None and mm_embeds:
            # EVS: recompute mrope positions for pruned media.
            # EVS(可跳过视觉 token)场景下,剪枝后需要重算 mRoPE 位置。
            mm_embeds = self.mm_pruner.recompute(mm_embeds, input_batch, req_states)
            # We must flush the staged rope updates for prepare_inputs() to pick up.
            # 位置变了,必须先把暂存的 RoPE 更新刷入,prepare_inputs 才能读到新位置。
            self.apply_staged_writes()

        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        # input_batch.input_ids may be padded for CUDA graphs.
        # 3) 把文本 embedding 与多模态 embedding 合并成 inputs_embeds。
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]
        inputs_embeds = self.encoder_runner.get_inputs_embeds(
            input_ids_unpadded, mm_embeds, is_mm_embed
        )
        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_mm_embed = super().gather_mm_embeddings(
            input_batch, draft_lookahead
        )
        if self.mm_pruner is not None:
            # EVS: strip the appended mrope-position channels.
            mm_embeds = self.mm_pruner.strip(mm_embeds)
        return mm_embeds, is_mm_embed

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        # 把非注意力状态(本类主要是 RoPE 位置)覆盖式注入模型输入。
        if self.rope_state is None:
            return {}  # Common case (1D positions). 常见情况用 1D positions,无需特殊处理。

        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return {"positions": positions}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = {}
        if self.supports_mm_inputs:
            inputs_embeds = self.encoder_runner.inputs_embeds[:num_tokens]
            model_inputs["inputs_embeds"] = inputs_embeds
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

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
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            # FULL 图模式下用 padded 形状(与捕获时一致,保证重放形状严格匹配)。
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            # 分段图 / eager 用未 padded 的真实形状。
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            # 捕获阶段用最坏情况 max_seq_len(=max_model_len),保证任意重放都合法。
            max_seq_len = self.max_model_len
        else:
            # 正式服务时，不用capture
            # 实际服务时取真实上界,缓冲区刚好够用,省显存。
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if (
            self.supports_mm_inputs
            and self.encoder_cache is not None
            and self.model_config.is_mm_prefix_lm
        ):
            # 多模态 prefix-LM:计算请求内各文档/媒体段的范围,供注意力区分处理。
            req_doc_ranges = compute_mm_prefix_ranges(
                req_ids=input_batch.req_ids,
                mm_features=self.encoder_cache.mm_features,
                sliding_window=self.model_config.get_sliding_window(),
            )
        # ⚠️ 调用 build_attn_metadata 产出 {layer_name: metadata}
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            mm_req_doc_ranges=req_doc_ranges,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        return attn_metadata
