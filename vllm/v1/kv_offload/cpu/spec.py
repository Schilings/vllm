# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            )
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)
        # ⚠️
        # 用户通过 kv_connector_extra_config["cpu_bytes_to_use"] 指定的、可用于
        # KV offload 的 CPU 内存总量（字节）。这是整个 CPU offload 池的容量上限。
        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # world_size = 所有并行 worker 数（TP*PP*...），用于把 per-worker 的 KV
        # 字节数换算成跨 worker 聚合后的总量。
        world_size = vllm_config.parallel_config.world_size
        # 先给容量相关字段一个默认值，下面只有在确实有 GPU block 时才会被覆盖。
        self.num_blocks = 0
        self.kv_bytes_per_offloaded_block = 0
        self.cpu_page_size_per_worker = 0
        assert kv_cache_config is not None
        if kv_cache_config.num_blocks > 0 and world_size > 0:
            # is_packed: KV 是否采用“打包”布局——即一个 block 内所有层的 KV 连续
            # 存放（block_stride != 0）。要求所有 tensor 要么都打包、要么都不打包。
            is_packed = any(t.block_stride for t in kv_cache_config.kv_cache_tensors)
            assert not is_packed or all(
                t.block_stride for t in kv_cache_config.kv_cache_tensors
            )
            # 单个 worker 上全部层的 GPU KV 总字节数：
            # 打包时所有层已经在 tensor[0] 一个张量里；否则把各层 tensor 尺寸相加。
            total_gpu_kv_bytes = (
                kv_cache_config.kv_cache_tensors[0].size
                if is_packed
                else sum(t.size for t in kv_cache_config.kv_cache_tensors)
            )
            # 跨所有 worker 聚合后，一个 GPU KV block 的字节数。
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * world_size
            # block_size_factor>1 时，一个“offloaded block”会打包多个 GPU 子 block，
            # 因此 offload 到 CPU 的 block 比 GPU 上的一个 block 更大。
            kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor

            # 一个 offloaded block 在 CPU 上按 worker 切分成若干“页”，
            # 每个 worker 负责传输其中一页：本 worker 的页字节数 = 总量 / world_size。
            self.cpu_page_size_per_worker = kv_bytes_per_offloaded_block // world_size

            # 按 BLOCK_SIZE_ALIGNMENT 向上对齐（本类为 1，即不对齐；子类可覆盖）。
            aligned_kv_bytes_per_offloaded_block = round_up(
                kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT
            )
            # CPU offload 池能容纳的 offloaded block 总数 = 总 CPU 字节 / 每块字节。
            self.num_blocks = (
                int(cpu_bytes_to_use) // aligned_kv_bytes_per_offloaded_block
            )

            # 对外暴露对齐后的值（可能含 padding）。每个 offloaded block 在内存中的
            # 布局形如：|--- W0-B0 ---|---- W1-B0 ---| ... |---- Wn-B0 ---| *** 尾部对齐 pad ***|
            self.kv_bytes_per_offloaded_block = aligned_kv_bytes_per_offloaded_block

        # scheduler 侧管理器与 worker 侧执行器都延迟创建（首次 get_manager/get_worker 时），
        # 这里先置空占位。
        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        # KV 块淘汰策略，默认 LRU（也可配其他），用于 CPU offload 池满时挑出被覆盖的块。
        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            # ⚠️
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        # ⚠️
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
        )

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
