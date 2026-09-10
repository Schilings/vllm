# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    OffloadingSpec,
)

# 测试用的统一超参：10 个 block、block_size=16、4 个 KV head、head_size=64、fp16。
NUM_BLOCKS = 10
BLOCK_SIZE = 16
NUM_KV_HEADS = 4
HEAD_SIZE = 64
DTYPE = torch.float16
DEVICE_TYPE = current_platform.device_type

# 要在哪些注意力后端上跑测试（不同后端的 KV cache 重塑/布局不同，
# 需逐一验证规范化结果一致）。
ATTN_BACKENDS: list[str] = []
if current_platform.is_cuda():
    ATTN_BACKENDS = [
        "FLASH_ATTN",
        "FLEX_ATTENTION",
        "FLASHINFER",
        "TRITON_ATTN",
    ]
elif current_platform.is_rocm():
    ATTN_BACKENDS = ["TRITON_ATTN"]
elif current_platform.is_xpu():
    ATTN_BACKENDS = ["TRITON_ATTN", "FLASH_ATTN"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _allocate_and_reshape_kv_caches(
    kv_cache_config: KVCacheConfig,
    attn_groups: list[list],
    device: torch.device,
):
    """
    Use the real GPUModelRunner allocation and reshape methods to produce
    kv_caches, just like the model runner does during initialization.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # 某些后端（如 FlashAttention）在重塑 KV cache 时会查询布局，最终调到
    # get_current_vllm_config()。这里用 set_kv_cache_layout("NHD") 覆盖布局，
    # 避免为了造几个张量而搭建完整的 VllmConfig 上下文。
    set_kv_cache_layout("NHD")
    try:
        # 用 object.__new__ 绕过 GPUModelRunner.__init__（它要建完整模型/通信等），
        # 只手动塞入本测试需要的字段，然后调用真实的张量分配+重塑方法。
        # 这样 offloading 规范化拿到的张量形状与真实 model runner 初始化时一致。
        runner = object.__new__(GPUModelRunner)
        runner.device = device
        runner.runner_only_attn_layers = set()
        runner.attn_groups = attn_groups
        runner.kv_cache_config = kv_cache_config
        runner.cache_config = MagicMock(cache_dtype="auto")
        runner.shared_kv_cache_layers = {}
        runner.model_config = MagicMock()
        runner.model_config.hf_config.model_type = ""
        runner.compilation_config = MagicMock(
            static_forward_context=defaultdict(MagicMock)
        )
        runner.kv_caches = []

        kernel_block_sizes = [BLOCK_SIZE] * len(kv_cache_config.kv_cache_groups)
        # 直接复用真实路径：分配 GPU 张量并按注意力组重塑成各层视图。
        return runner.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
    finally:
        set_kv_cache_layout(None)


def _make_worker(kv_cache_config: KVCacheConfig):
    """
    Create an OffloadingConnectorWorker with mocked dependencies.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )

    # OffloadingSpec 与真正的传输 worker（负责 DMA 的部分）全部用 Mock 替代，
    # 这样测试只验证 register_kv_caches 的"规范化"逻辑，不触发真实 KV 搬运。
    # spec.get_worker 被替换为 Mock，稍后通过 call_args 取出规范化产物做断言。
    spec = MagicMock(spec=OffloadingSpec)
    spec.kv_cache_config = kv_cache_config
    spec.vllm_config = MagicMock()
    spec.get_worker.return_value = MagicMock()

    worker = OffloadingConnectorWorker(spec=spec)
    worker.worker = MagicMock()

    return worker, spec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_register_kv_caches(backend):
    """Test register_kv_caches with multiple groups covering all layer types.

    Creates one FullAttention group, one MLA group, one Mamba group, and
    one Mamba-padded group. Each group has GROUP_SIZE layers.

    KVCacheTensors are shared across all groups mirroring the real allocation
    in kv_cache_utils.py: tensor i is shared by layer i from every group.
    The padded-mamba group has a different page size so its layers get their
    own dedicated tensors.

    Uses the real GPUModelRunner.initialize_kv_cache_tensors to produce
    kv_caches, which automatically applies
    _update_hybrid_attention_mamba_layout for hybrid models.

    Verifies that the canonicalized CanonicalKVCaches has the correct
    block tensors, tensor_idx references, and page sizes across all groups.
    """
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerBackend,
    )
    from vllm.v1.worker.utils import AttentionGroup

    MLA_HEAD_SIZE = NUM_KV_HEADS * HEAD_SIZE * 2

    # padded mamba (missing HEAD_SIZE)
    CONV_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS, HEAD_SIZE)
    UNALIGNED_SSM_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS - 1, HEAD_SIZE)

    PAGE_SIZE_BYTES = 2 * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * get_dtype_size(DTYPE)
    unaligned_mamba_page_size = PAGE_SIZE_BYTES - HEAD_SIZE * get_dtype_size(DTYPE)

    # unpadded mamba (fills page exactly)
    ALIGNED_SSM_STATE_SHAPE = (BLOCK_SIZE * NUM_KV_HEADS, HEAD_SIZE)

    backend_cls = AttentionBackendEnum[backend].get_class()

    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    mla_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=MLA_HEAD_SIZE,
        dtype=DTYPE,
    )
    unaligned_mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(CONV_STATE_SHAPE, UNALIGNED_SSM_STATE_SHAPE),
        dtypes=(DTYPE, DTYPE),
        page_size_padded=PAGE_SIZE_BYTES,
    )
    aligned_mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(CONV_STATE_SHAPE, ALIGNED_SSM_STATE_SHAPE),
        dtypes=(DTYPE, DTYPE),
        page_size_padded=PAGE_SIZE_BYTES,
    )

    assert attn_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert mla_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert unaligned_mamba_spec.page_size_bytes == PAGE_SIZE_BYTES
    assert aligned_mamba_spec.page_size_bytes == PAGE_SIZE_BYTES

    GROUP_SIZE = 3

    # -- Build per-group layer info ----------------------------------------
    layer_idx = 0

    attn_layer_names = []
    for _ in range(GROUP_SIZE):
        attn_layer_names.append(f"model.layers.{layer_idx}.self_attn")
        layer_idx += 1

    mla_layer_names = []
    for _ in range(GROUP_SIZE):
        mla_layer_names.append(f"model.layers.{layer_idx}.self_attn")
        layer_idx += 1

    unaligned_mamba_layer_names = []
    for _ in range(GROUP_SIZE):
        unaligned_mamba_layer_names.append(f"model.layers.{layer_idx}.mamba_unpadded")
        layer_idx += 1

    aligned_mamba_layer_names = []
    for _ in range(GROUP_SIZE - 1):
        aligned_mamba_layer_names.append(f"model.layers.{layer_idx}.mamba_padded")
        layer_idx += 1

    layer_groups = [
        attn_layer_names,
        mla_layer_names,
        unaligned_mamba_layer_names,
        aligned_mamba_layer_names,
    ]

    # 模拟真实 kv_cache_utils 的分配：第 i 个 KVCacheTensor 被"每个组的第 i 层"
    # 共享（shared_by）。即多个 layer 指向同一块物理张量——这正是
    # worker.py 去重分支（225-279 行）要处理的场景。
    # 注意：aligned_mamba 组只有 GROUP_SIZE-1=2 层，所以它的第 2 层不参与共享，
    # 第 3 个 tensor（tensor index 2）只被 attn/mla/unaligned_mamba 共享。
    kv_cache_tensors: list[KVCacheTensor] = []
    for i in range(GROUP_SIZE):
        shared_by: list[str] = []
        for group_layer_names in layer_groups:
            if len(group_layer_names) > i:
                shared_by.append(group_layer_names[i])
        kv_cache_tensors.append(
            KVCacheTensor(
                size=PAGE_SIZE_BYTES * NUM_BLOCKS,
                shared_by=shared_by,
            )
        )

    kv_cache_groups = [
        KVCacheGroupSpec(layer_names=attn_layer_names, kv_cache_spec=attn_spec),
        KVCacheGroupSpec(layer_names=mla_layer_names, kv_cache_spec=mla_spec),
        KVCacheGroupSpec(
            layer_names=unaligned_mamba_layer_names, kv_cache_spec=unaligned_mamba_spec
        ),
        KVCacheGroupSpec(
            layer_names=aligned_mamba_layer_names, kv_cache_spec=aligned_mamba_spec
        ),
    ]

    attn_groups = [
        [
            AttentionGroup(
                backend=backend_cls,
                layer_names=attn_layer_names,
                kv_cache_spec=attn_spec,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,
                layer_names=mla_layer_names,
                kv_cache_spec=mla_spec,
                kv_cache_group_id=1,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,  # unused for mamba
                layer_names=unaligned_mamba_layer_names,
                kv_cache_spec=unaligned_mamba_spec,
                kv_cache_group_id=2,
            ),
            AttentionGroup(
                backend=DeepseekV32IndexerBackend,  # unused for mamba
                layer_names=aligned_mamba_layer_names,
                kv_cache_spec=aligned_mamba_spec,
                kv_cache_group_id=3,
            ),
        ]
    ]
    #
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )
    #
    kv_caches = _allocate_and_reshape_kv_caches(
        kv_cache_config,
        attn_groups,
        device=torch.device(f"{DEVICE_TYPE}:0"),
    )

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    # register_kv_caches 内部会把规范化结果传给 spec.get_worker（此处为 Mock），
    # 通过 call_args 取出产出的 CanonicalKVCaches 做断言。
    canonical = spec.get_worker.call_args[0][0]
    assert isinstance(canonical, CanonicalKVCaches)

    # -- Expected block tensors ----------------------------------------------
    # 去重后应有 3 个物理张量（因为 aligned_mamba 比其它组少一层，第 3 个 tensor
    # 只被 attn/mla/unaligned_mamba 共享，aligned_mamba 不参与）。
    # 所有 tensor 的 padded 页大小都是 PAGE_SIZE_BYTES。
    # Tensor 0: shared by attn[0], mla[0], mamba_unaligned[0], mamba_aligned[0]
    # Tensor 1: shared by attn[1], mla[1], mamba_unaligned[1], mamba_aligned[1]
    # Tensor 2: shared by attn[2], mla[2], mamba_unaligned[2]
    #           (mamba_aligned has only GROUP_SIZE-1 = 2 layers)
    expected_tensors = [
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
        (NUM_BLOCKS, PAGE_SIZE_BYTES),
    ]

    # -- Expected group data refs (order matches kv_cache_groups) -------------
    ref = CanonicalKVCacheRef
    expected_group_refs = [
        # attn group: layers attn[0..2] → tensors 0,1,2 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=2, page_size_bytes=PAGE_SIZE_BYTES),
        ],
        # mla group: layers mla[0..2] → tensors 0,1,2 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=2, page_size_bytes=PAGE_SIZE_BYTES),
        ],
        # unaligned mamba group: layers [0..2] → tensors 0,1,2 with unaligned page
        [
            ref(tensor_idx=0, page_size_bytes=unaligned_mamba_page_size),
            ref(tensor_idx=1, page_size_bytes=unaligned_mamba_page_size),
            ref(tensor_idx=2, page_size_bytes=unaligned_mamba_page_size),
        ],
        # aligned mamba group: layers [0..1] → tensors 0,1 with full page size
        [
            ref(tensor_idx=0, page_size_bytes=PAGE_SIZE_BYTES),
            ref(tensor_idx=1, page_size_bytes=PAGE_SIZE_BYTES),
        ],
    ]

    # Verify block tensors
    # 校验去重后的物理张量：数量对、形状为 (num_blocks, page_size)、dtype 为 int8
    # （offloading 按字节寻址的视图）、page_size_bytes 与预期一致。
    assert len(canonical.tensors) == len(expected_tensors)
    for block_tensor, (exp_num_blocks, exp_page_size) in zip(
        canonical.tensors, expected_tensors
    ):
        tensor = block_tensor.tensor
        assert tensor.dtype == torch.int8
        assert tensor.shape == (exp_num_blocks, exp_page_size)
        assert block_tensor.page_size_bytes == exp_page_size

    # Verify group data refs
    # 校验每个 KV group 的引用表：层数对，且每层指向正确的 tensor_idx 与
    # （未 padding 的）真实页大小。unaligned_mamba 用更小的真实页，其它组用满页。
    assert len(canonical.group_data_refs) == len(expected_group_refs)
    for actual_refs, exp_refs in zip(canonical.group_data_refs, expected_group_refs):
        assert len(actual_refs) == len(exp_refs)
        for actual, expected in zip(actual_refs, exp_refs):
            assert actual.tensor_idx == expected.tensor_idx
            assert actual.page_size_bytes == expected.page_size_bytes


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_register_kv_caches_uniform_type(backend):
    """Test register_kv_caches with UniformTypeKVCacheSpecs.

    Two attention layers use the same backend but different num_kv_heads,
    giving them different per-layer page sizes. Each has its own
    KVCacheTensor and are wrapped in a UniformTypeKVCacheSpecs group.
    Verifies that each layer gets the correct tensor_idx and
    page_size_bytes in its block data ref.
    """
    from vllm.v1.worker.utils import AttentionGroup

    backend_cls = AttentionBackendEnum[backend].get_class()

    layer_a = "model.layers.0.self_attn"
    layer_b = "model.layers.1.self_attn"
    # 两个注意力层用同一后端，但 num_kv_heads 不同（4 vs 8）→ 每页字节数不同。
    spec_a = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    spec_b = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS * 2,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    assert spec_a.page_size_bytes != spec_b.page_size_bytes

    # 同组但逐层 spec 不同 → 用 UniformTypeKVCacheSpecs 包起来（worker.py 里会走
    # per_layer_specs 分支取各自的 spec），验证每层的真实页大小能被正确保留。
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=BLOCK_SIZE,
        kv_cache_specs={layer_a: spec_a, layer_b: spec_b},
    )

    # 两层各自独占一个 KVCacheTensor（互不相同），因此规范化后应有 2 个物理张量。
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec_a.page_size_bytes * NUM_BLOCKS,
                shared_by=[layer_a],
            ),
            KVCacheTensor(
                size=spec_b.page_size_bytes * NUM_BLOCKS,
                shared_by=[layer_b],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_a, layer_b],
                kv_cache_spec=uniform_spec,
            )
        ],
    )

    attn_groups = [
        [
            AttentionGroup(
                backend=backend_cls,
                layer_names=[layer_a],
                kv_cache_spec=spec_a,
                kv_cache_group_id=0,
            ),
            AttentionGroup(
                backend=backend_cls,
                layer_names=[layer_b],
                kv_cache_spec=spec_b,
                kv_cache_group_id=0,
            ),
        ]
    ]

    kv_caches = _allocate_and_reshape_kv_caches(
        kv_cache_config,
        attn_groups,
        device=torch.device(f"{DEVICE_TYPE}:0"),
    )

    worker, spec = _make_worker(kv_cache_config)
    worker.register_kv_caches(kv_caches)

    canonical = spec.get_worker.call_args[0][0]
    assert isinstance(canonical, CanonicalKVCaches)

    for block_tensor in canonical.tensors:
        assert block_tensor.tensor.dtype == torch.int8

    # Single group with refs from both layers
    # ⚠️ 仅 1 个 KV group，但包含 2 层；规范化后应去重出 2 个物理张量，
    # 且每层的 page_size_bytes 与各自 spec 的真实页大小严格对应。
    assert len(canonical.group_data_refs) == 1
    group_refs = canonical.group_data_refs[0]
    assert len(group_refs) == 2

    assert len(canonical.tensors) == 2
    assert canonical.tensors[0].page_size_bytes == spec_a.page_size_bytes
    assert canonical.tensors[1].page_size_bytes == spec_b.page_size_bytes
    assert canonical.tensors[0].tensor.shape == (NUM_BLOCKS, spec_a.page_size_bytes)
    assert canonical.tensors[1].tensor.shape == (NUM_BLOCKS, spec_b.page_size_bytes)

    # 引用表顺序：layer_a → tensor 0（其页大小），layer_b → tensor 1（其更大页大小）。
    assert group_refs[0] == CanonicalKVCacheRef(
        tensor_idx=0, page_size_bytes=spec_a.page_size_bytes
    )
    assert group_refs[1] == CanonicalKVCacheRef(
        tensor_idx=1, page_size_bytes=spec_b.page_size_bytes
    )
