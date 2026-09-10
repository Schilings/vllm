# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for contiguous KV cache packing."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_config_packed,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def _make_mla_spec(page_size: int, block_size: int = 256) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        page_size_padded=page_size,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        # ⚠️ self.page_size_padded = real_page_size_bytes填充到alignment倍数
        alignment=576,
    )


def _make_full_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )


def _make_sw_spec() -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
        sliding_window=128,
    )


def _make_groups(n_c4, n_c128, n_swa):
    # 64×584 = 37376 → 37440
    PS_C4_MLA = 37440
    # 64×132 = 8448 → 8640
    PS_C4_IDX = 8640
    # 2×584 = 1168 → 1728
    PS_C128 = 1728
    # 64×584 = 37376 → 37440
    PS_SWA = 37440

    mla_specs = {}
    for i in range(n_c4):
        mla_specs[f"c4_mla.{i}"] = _make_mla_spec(PS_C4_MLA)
        mla_specs[f"c4_idx.{i}"] = _make_mla_spec(PS_C4_IDX)
    for i in range(n_c128):
        mla_specs[f"c128_mla.{i}"] = _make_mla_spec(PS_C128)

    # ⚠️ 都是 MLAAttentionSpec， 归为 UniformTypeKVCacheSpecs
    mla_group = KVCacheGroupSpec(
        layer_names=list(mla_specs.keys()),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=mla_specs),
    )

    swa_specs = {}
    for i in range(n_swa):
        swa_specs[f"swa.{i}"] = _make_mla_spec(PS_SWA)

    # ⚠️ 都是 MLAAttentionSpec， 归为 UniformTypeKVCacheSpecs
    swa_group = KVCacheGroupSpec(
        layer_names=list(swa_specs.keys()),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=256, kv_cache_specs=swa_specs),
    )

    return [mla_group, swa_group]


def _mock_vllm_config(kv_connector_extra_config: dict[str, str] | None = None):
    config = MagicMock()
    config.cache_config.num_gpu_blocks_override = None
    config.kv_transfer_config = None
    if kv_connector_extra_config is not None:
        config.kv_transfer_config = MagicMock()
        config.kv_transfer_config.kv_connector_extra_config = kv_connector_extra_config
    return config


def _run(n_c4=3, n_c128=2, n_swa=5, mem=100 * 1024 * 1024):
    groups = _make_groups(n_c4, n_c128, n_swa)
    """
        1️⃣假设有 2 个 group，layer 的 page_size 如下：
            Group 0:  L0(大=100)  L1(小=20)  L2(大=100)
                       ↑           ↑         ↑
            Group 1:  L3(大=100)  L4(小=20)  L5(大=100)  L6(小=20)
        2️⃣buckets = {
                    100: [ [L0, L3],     # slot 0：两个 group 的"第1个大页 layer"
                           [L2, L5] ],   # slot 1：两个 group 的"第2个大页 layer"
                    20: [ [L1, L4],     # slot 0：两个 group 的"第1个小页 layer"
                          [L6]     ],   # slot 1：只有 Group1 有第2个小页 layer
                  }
        3️⃣一个物理 block 的内存布局（block_stride = 各桶尺寸之和）：
         ┌─────────100x2────────┬─────────100x2────────┬────20───┬────20───┐
         │  buckets[100][0]     │  buckets[100][1]     │ [20][0] │ [20][1] │
         │  L0/L3 共享           │  L2/L5 共享           │ L1/L4   │  L6     │
         └──────────────────────┴──────────────────────┴─────────┴─────────┘
         offset=0

        [L0, L3] 为什么能挤进同一段内存而不冲突？
        因为它们来自不同 group，各自有独立的 block table（独立的 block-id 命名空间）。
        同一时刻，Group0 用 block-id=5、Group1 也用 block-id=5，但它们指向的是不同物理 block——block table 映射不同，永不撞车。
        反过来，同一个 group 内的两个大页 layer（L0、L2）不能共享，因为它们共用一张 block table，block-id=5 对两者是同一块，会互相覆盖
        所以它们被分到 slot 0 和 slot 1 两个不同的桶。

        ============================================================
        ⚠️ DeepSeek-V4 类似示例（这才是 packed 布局的主力场景）
        ============================================================
        区别: DeepSeek-V4 只有 2 个 group, 但每个 group 是
        UniformTypeKVCacheSpecs —— 同一个 group 内部各 layer 的 page_size 不同
        取 n_c4=2, n_c128=1, n_swa=3 (n_swa = n_c4 + n_c128,
        SWA 组层数与 MLA 组的 c4+c128 层数一一对应; page_size 单位: 字节):
            Group 0 (mla_group):
                c4_mla.0 (37440)  c4_idx.0 (8640)
                c4_mla.1 (37440)  c4_idx.1 (8640)
                c128_mla.0 (1728)
            Group 1 (swa_group):
                swa.0 (37440)     swa.1 (37440)     swa.2 (37440)

        最终 buckets:
            37440: [ [c4_mla.0, swa.0],    # slot0: 跨 group 共享
                     [c4_mla.1, swa.1],    # slot1: 跨 group 共享
                     [swa.2] ]             # slot2: 只有 Group1 (第3个大页)
             8640: [ [c4_idx.0],           # slot0: 只有 Group0
                     [c4_idx.1] ]          # slot1: 只有 Group0
             1728: [ [c128_mla.0] ]        # slot0: 只有 Group0

        ⚠️ 一个物理 block 的内存布局 ==> 放了所有layer同个block的kv cache
        (block_stride = 37440*3 + 8640*2 + 1728 = 131328):
         ┌────37440─────┬────37440─────┬───37440─────┬──8640───┬──8640───┬─1728─┐
         │ [37440][0]   │ [37440][1]   │[37440][2]   │[8640][0]│[8640][1]│[1728]│
         │c4_mla.0/swa.0│c4_mla.1/swa.1│ swa.2       │c4_idx.0 │c4_idx.1 │c128.0│
         └──────────────┴──────────────┴─────────────┴─────────┴─────────┴──────┘
         off=0          off=37440       off=74880    off=112320 off=120960 off=129600
        
         ⚠️ 这里是 c4_mla.0/swa.0 是 或 的关系，不是 和 的关系！！！！！！
    """
    return _get_kv_cache_config_packed(_mock_vllm_config(), groups, mem)


def _page_sizes_by_layer(
    groups: list[KVCacheGroupSpec],
) -> dict[str, int]:
    page_sizes = {}
    for group in groups:
        specs = group.kv_cache_spec.kv_cache_specs
        for layer_name in group.layer_names:
            page_sizes[layer_name] = specs[layer_name].page_size_bytes
    return page_sizes


class TestInterleavedPacking:
    def test_all_tensors_have_block_stride(self):
        _, tensors = _run()
        for t in tensors:
            assert t.block_stride > 0

    def test_all_tensors_share_same_size(self):
        _, tensors = _run()
        # ⚠️ size都是一样的！
        # ⚠️ layer（slot）通过 block_id * total_num_bytes_per_block + offset来访问自己的部分
        sizes = set(t.size for t in tensors)
        assert len(sizes) == 1
        assert sizes.pop() > 0

    def test_offsets_within_one_block(self):
        _, tensors = _run()
        for t in tensors:
            # ⚠️ layer（slot）通过 block_id * total_num_bytes_per_block + offset来访问自己的部分
            assert t.offset < t.block_stride

    def test_all_layers_accounted_for(self):
        n_c4, n_c128, n_swa = 5, 4, 7
        _, tensors = _run(n_c4=n_c4, n_c128=n_c128, n_swa=n_swa)
        all_names = set()
        for t in tensors:
            all_names.update(t.shared_by)
        # 只有 min(5,7) * [c4_mla.i, swa.i]
        expected = n_c4 * 2 + n_c128 + n_swa
        assert len(all_names) == expected

    def test_strided_views_are_independent(self):
        groups = _make_groups(n_c4=3, n_c128=2, n_swa=5)
        page_sizes = _page_sizes_by_layer(groups)
        # ⚠️ max(n_swa, n_c4) + n_c4 + n_c128 = 5 + 3 + 2 = 10 个 KVCacheTensor
        num_blocks, tensors = _get_kv_cache_config_packed(
            _mock_vllm_config(), groups, 100 * 1024 * 1024
        )
        # ⚠️ 完整的物理KV Cache
        backing = torch.zeros(tensors[0].size, dtype=torch.uint8)
        views = []

        # ⚠️ max(n_swa, n_c4) + n_c4 + n_c128 = 5 + 3 + 2 = 10 个 KVCacheTensor
        for t in tensors:
            page_size = page_sizes[t.shared_by[0]]
            v = torch.as_strided(
                backing,
                size=(num_blocks, page_size),
                stride=(t.block_stride, 1),
                storage_offset=t.offset,
            )
            views.append(v)

        for i, v in enumerate(views):
            v.fill_(i + 1)

        for i, v in enumerate(views):
            assert (v == i + 1).all(), f"View {i} was corrupted"

    def test_hma_attention_groups_keep_default_backing(self):
        full = _make_full_spec()
        sw = _make_sw_spec()
        page_size = full.page_size_bytes
        groups = [
            KVCacheGroupSpec(["full.0", "full.1"], full),
            KVCacheGroupSpec(["sw.0", "sw.2"], sw),
            KVCacheGroupSpec(["sw.1", "sw.3"], sw),
        ]

        # ⚠️ 走 3️⃣ general 混合。强制 所有group的page size完全一致。
        # group size = 3， 那么 创建 2 个KV Cache Tensor, 每个 ==> availale memory // 2
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(), groups, available_memory=page_size * 2 * 32
        )

        assert config.num_blocks == 32
        assert sum(t.size for t in config.kv_cache_tensors) == page_size * 2 * 32
        assert config.kv_cache_tensors == [
            KVCacheTensor(size=page_size * 32, shared_by=["full.0", "sw.0", "sw.1"]),
            KVCacheTensor(size=page_size * 32, shared_by=["full.1", "sw.2", "sw.3"]),
        ]

    def test_hma_attention_groups_use_packed_backing_with_enable_cross_layers(self):
        full = _make_full_spec()
        sw = _make_sw_spec()
        page_size = full.page_size_bytes
        groups = [
            KVCacheGroupSpec(["full.0", "full.1"], full),
            KVCacheGroupSpec(["sw.0", "sw.2"], sw),
            KVCacheGroupSpec(["sw.1", "sw.3"], sw),
        ]
        # ==> 产生 2 个 slot, slot 0 = [full.0, sw.0, sw.1 ], slot 1 = [full.1, sw.2, sw.3 ]

        config = get_kv_cache_config_from_groups(
            # ⚠️ 强制走 2️⃣ DSv4 or packed 。将所有layer的信息放在单block
            # ⚠️ 一个物理 block 的内存布局 ==> 放了所有layer同个block的kv cache
            # ⚠️⚠️⚠️ 这种设计就是： 同样的block，可以通用，不同group的kv可以拿去存储！！！！
            # ⚠️⚠️⚠️ 例如dsv4，两种kv group( Uniform(CSA+CIA+HCA), Uniform(SWA) ) 都可以直接拿这种 block 进行存储
            # ⚠️⚠️⚠️ 缺点就是：可能每个block都存在显存浪费！！！！
            _mock_vllm_config({"enable_cross_layers_blocks": "True"}),
            groups,
            available_memory=page_size * 2 * 32,
        )

        assert config.num_blocks == 32
        assert {t.size for t in config.kv_cache_tensors} == {page_size * 2 * 32}
        assert config.kv_cache_tensors == [
            KVCacheTensor(
                size=page_size * 2 * 32,
                shared_by=["full.0", "sw.0", "sw.1"],
                offset=0,
                block_stride=page_size * 2,
            ),
            KVCacheTensor(
                size=page_size * 2 * 32,
                shared_by=["full.1", "sw.2", "sw.3"],
                offset=page_size,
                block_stride=page_size * 2,
            ),
        ]

    def test_single_group_attention_keeps_unpacked_layout(self):
        spec = _make_full_spec()
        groups = [KVCacheGroupSpec(["full.0", "full.1"], spec)]

        # ⚠️ 走 3️⃣ general 混合。强制 所有group的page size完全一致。
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(), groups, available_memory=spec.page_size_bytes * 2 * 32
        )

        assert sum(t.size for t in config.kv_cache_tensors) == (
            spec.page_size_bytes * 2 * 32
        )
        assert [t.block_stride for t in config.kv_cache_tensors] == [0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
