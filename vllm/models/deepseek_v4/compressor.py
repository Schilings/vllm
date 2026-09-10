# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        # state_dim: 单个 token 的压缩状态总维度 = kv_state_dim + score_state_dim，
        # 由调用方传入（DeepseekCompressor 处为 2*coff*head_dim）。fp32 存储。
        self.state_dim = state_dim
        # 状态缓存统一用 fp32（压缩中间态需要高精度累加，避免误差累积）。
        self.dtype = dtype
        self.prefix = prefix
        # kv_cache 在 __init__ 时为空占位张量；真正分配与"注入"发生在引擎启动时：
        #   1) 各层 get_kv_cache_spec() 描述形状/对齐，vLLM 从一块大 kv_raw_tensor
        #      用 torch.as_strided 切出每层的 kv_cache 视图（见 attn_utils.py 的
        #      _reshape_attention_kv_cache）。
        #   2) bind_kv_cache()（worker/utils.py）执行
        #      forward_context[layer_name].kv_cache = kv_cache，直接改写本对象的
        #      .kv_cache 属性——forward_context 正是上面注册的 static_forward_context，
        #      故"注入"本质是一次普通属性赋值，覆盖掉占位空张量。
        #   3) forward 时 DeepseekCompressor 通过 self._static_forward_context[
        #      self.k_cache_prefix] 反查取回这个已填充的 kv_cache 写入压缩结果。
        self.kv_cache = torch.tensor([])
        # 把本层注册进 static_forward_context：forward 时通过 prefix 反查拿到
        # 真实分配好的 kv_cache（见 DeepseekCompressor.forward 里
        # self._static_forward_context[self.k_cache_prefix]）。
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        # coff(co-compress factor): C4 时 overlap=True → coff=2；C128 时 coff=1。
        # sliding_window = coff * compress_ratio：状态缓存需要覆盖的"未压缩 token
        # 滑动窗口"长度。C4 → 2*4=8，C128 → 1*128=128。即每个压缩块依赖其前
        # sliding_window 个原始 token 的局部状态。
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        # SlidingWindowMLASpec：压缩状态缓存本质是"滑窗 + 单向量(无 K/V 分离)"的
        # MLA 布局。head_size=self.state_dim(=kv_state_dim+score_state_dim)，
        # num_kv_heads=1(压缩后已降秩为单头向量)，sliding_window 限制只可见最近
        # sliding_window 个原始 token 的状态。alignment 同主 KV cache：fp8_ds_mla
        # 用 576B 对齐，plain 用 512B。
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        # compress_ratio: 压缩比，C4=4 / C128=128。决定每几个原始 token 压成 1 个状态。
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        # head_dim: 压缩后 KV 的隐维度。HCA/CSA 用 512，indexer 复用 compressor 时用 128。
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        # k_cache_prefix: 主 KV cache 层在 static_forward_context 中的注册名，forward
        # 时据此取出真实 kv_cache 张量写入压缩结果。
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        # RoPE 段维度（DeepSeek-V4 中 qk_rope_head_dim，如 64）；nope_head_dim 为
        # 非位置编码段 = head_dim - rope_head_dim（如 512-64=448）。
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        # overlap: 是否"重叠压缩"。C4(CSA) 需要重叠窗口 → True；C128(HCA) → False。
        self.overlap = compress_ratio == 4
        # coff: co-compress factor，C4 时 2，C128 时 1。用于放大状态/投影维度。
        self.coff = 1 + self.overlap

        # APE(Absolute Position Embedding)状态：形状 [compress_ratio, coff*head_dim]，
        # 在 save_partial_states 时叠加到 kv/score 上注入绝对位置信息。fp32、不训练。
        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        # disable_tp=True：fused_wkv_wgate 退化为 ReplicatedLinear（权重全复制、
        # 不按 TP 切分）。原因：压缩投影在各 TP rank 上需要完整 hidden→coff*head_dim
        # 的输出，且后续 compressor 状态是 per-token 局部计算，无需跨 rank 分片。
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            # 两路合并输出：kv 投影 [coff*head_dim] + gate/score 投影 [coff*head_dim]
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        # RMSNorm 作用于压缩后的 head_dim 向量（压缩→norm→RoPE 流水线的一步）。
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        # 持有 CompressorStateCache 子模块：state_dim = 2*coff*head_dim，即
        # kv_state(coff*head_dim) + score_state(coff*head_dim) 拼接。score 是
        # Lightning Indexer 路由用的稀疏分数，与 kv 一起被压缩缓存。
        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # 缓存 static_forward_context 引用：__init__ 时才能拿到 vllm_config，
        # forward 时已不可用，故提前存引用供 forward 反查 kv_cache。
        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        # 量化 + 写 cache 的布局参数，分两条路径（对应 HCA/CSA 的 head_dim=512 与
        # indexer 的 head_dim=128）。这些参数决定 compress_norm_rope_store 内核里
        # 每个 token 在 KV cache 中占多少字节、scale 怎么排。
        if self.head_dim == 512:
            # HCA/CSA 主 KV cache 路径（head_dim=512，nope=448 + rope=64）。
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            # 单 token KV cache 行字节数（fp8）：NoPE 段 448 + RoPE 段 64*2(cos/sin
            # 各存一份 fp8) = 576；这里 token_stride 即每行 fp8 字节数。
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            # scale 维：NoPE 按 64 分组 → 448//64=7 个 scale，+1 为 rope 段对齐 pad，
            # 即 7 real + 1 pad。
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            # indexer 压缩 KV 路径（head_dim=128）。
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                # MXFP4 每元素 4 bit，故 fp4 行字节数 = 128//2 = 64。
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                # 默认 fp8 路径：quant_block=128 恰等于 head_dim，整 head 一组量化，
                # 单 head 1 个 fp32 scale（4 字节）。token_stride = 128(fp8 字节)。
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # kv_score: fused_wkv_wgate 投影输出，形状 [num_tokens, 2*coff*head_dim]
        #   前半 coff*head_dim 为 kv，后半 coff*head_dim 为 score(路由权重)。
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # positions: 每个 token 的绝对序列位置，形状 [num_tokens]
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # 沿特征维切成 kv 与 score，各 [num_tokens, coff*head_dim]。
        # 输入 bf16(GEMM 输出)，下游状态缓存用 fp32 高精度。
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # 取当前 forward 的 attention metadata；dummy profiling run 时
        # attn_metadata 不是 dict，直接 return 跳过实际计算。
        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        # 取出本 compressor 层对应的 CompressorMetadata（含 slot_mapping/block_table）。
        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        # token_to_req_indices: [num_tokens] 每 token → (req_id, 局部索引)。
        token_to_req_indices = state_metadata.token_to_req_indices
        # slot_mapping: [num_tokens] 每 token 在 state_cache 的物理槽位。
        slot_mapping = state_metadata.slot_mapping
        # num_actual: 本步有效 token 数（剔除 padding）。
        num_actual = slot_mapping.shape[0]
        # block_table: [num_reqs, num_blocks] 逻辑块→物理块映射（与 KV cache 共享）。
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # ⚠️ state_cache: 真实分配的 paged 张量，形状 [num_blocks, block_size, 2*state_dim]
        #   其中 state_dim = coff*head_dim（kv 与 score 同维）。
        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache

        # state_width = 单路宽度(=total//2)；前 [..,:state_width] 是 kv_state，
        # 后 [..,state_width:] 是 score_state。
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        # PDL(Programmatic Dependent Launch)在 ROCm/XPU 不可用；CUDA 上显式关闭，
        # 原因见下方 NOTE（避免 RAW 竞态）。
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # 1️⃣第一步：把 kv/score（叠加 APE 绝对位置偏置）写入 state_cache。
        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and the compress kernels
        # below depend on preceding kernel outputs (kv/score from the cublas
        # GEMM; state_cache from this kernel) but neither emits/waits on PDL
        # grid dependency primitives, so launch_pdl=True caused a
        # read-after-write race and non-deterministic output.
        save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # 2️⃣第二步：融合压缩内核 compress → RMSNorm → RoPE → FP8 quant → 写主 KV cache。
        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        # 从 static_forward_context 反查主 KV cache 层，拿到真实 kv_cache 张量写入。
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # 写 cache 方式判定：
        # - store_full_kv: head_dim=512 且 cache 非 uint8(paged) → 写连续 bf16 / 非
        #   paged fp8 整行（仅 cutedsl 用）。
        # - store_full_fp8: cache 是 e4m3fn 整行 fp8（非 block-scaled）。
        # - fp8_scale: 整行 fp8 的全局 scale（仅 store_full_fp8 时非 None）。
        # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
        # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
        store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
        store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        fp8_scale = (
            getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
            if store_full_fp8
            else None
        )

        # 内核分派：CUDA + head_dim=512 走 cutedsl（支持 full-cache 标志）；
        # 其余(indexer head_dim=128 / AMD / XPU)走 triton，签名不同故 extra_kwargs 为空。
        # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
        # does not, so the two callables have different signatures.
        compress_norm_rope_store_fn: Any
        if current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
                compress_norm_rope_store_cutedsl,
            )

            # head=512 on CUDA always uses cutedsl, for both the fp8_ds_mla
            # layout and the plain full-cache layout. The full-cache flags
            # are consumed only here.
            compress_norm_rope_store_fn = compress_norm_rope_store_cutedsl
            extra_kwargs: dict[str, Any] = dict(
                store_full_kv=store_full_kv,
                store_full_fp8=store_full_fp8,
                fp8_scale=fp8_scale,
            )
        else:
            # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {}

        # 调用融合内核：state_cache(含 kv+score) → 压缩聚合成单 head 向量 →
        # RMSNorm → RoPE(作用最后 rope_head_dim 维) → FP8 block-scaled 量化 →
        # 按 block_table/slot_mapping 写入主 KV cache。quant_block/token_stride/
        # scale_dim 控制量化布局（见 __init__ 注释）。
        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
