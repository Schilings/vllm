# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from dataclasses import replace

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
)

logger = init_logger(__name__)


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker: OffloadingWorker | None = None

        # job_id -> req_id for in-flight loads.
        self._load_jobs: dict[int, ReqId] = {}
        self._unsubmitted_store_jobs: list[
            tuple[int, GPULoadStoreSpec, LoadStoreSpec]
        ] = []
        self._connector_worker_meta = OffloadingWorkerMetadata()

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self.worker = self.spec.get_worker(kv_caches)

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        """"""
        """5️⃣ Worker: 初始化时注册各层 KV cache 张量（仅一次）。
        转发给 ``OffloadingConnectorWorker.register_kv_caches``。
        仅当连接器未使用跨层 block（``prefer_cross_layer_blocks`` 为 False）时走这里。
        """
        # 把 attention backend 在 GPU 上分配好的 KV cache 规范化（canonicalize）成统一的形状 (num_blocks, page_size_bytes) 的 int8 视图，
        # 交给 worker做 GPU<->CPU 的 KV 传输。
        # 本函数不分配新显存，只是重新 view 已有存储。
        kv_cache_config = self.spec.kv_cache_config
        num_blocks = kv_cache_config.num_blocks

        # Packed layouts (e.g. DSv4) set block_stride > 0; their tensors use
        # stride(0) as the manager-block stride (equals total_num_bytes_per_block).
        # General (non-packed) layouts size the tensor at page_size_bytes per
        # manager block, so page_size_bytes is the correct offloading stride.
        # 判断每个 layer 是否使用了 packed 布局（如 DeepSeek-v4）。
        # Packed 布局会设置 block_stride > 0，其张量 stride(0) 等于一个
        # manager-block 的总字节数；普通布局则按 page_size_bytes 对齐。
        layer_is_packed: dict[str, bool] = {
            ln: bool(kv_tensor.block_stride)
            for kv_tensor in kv_cache_config.kv_cache_tensors
            for ln in kv_tensor.shared_by
        }

        # 三个 per-layer 的映射，循环结束后组装成 CanonicalKVCaches：
        # - tensors_per_block: layer_name -> 该层 (num_blocks, page_size_bytes) 的 int8 视图
        # - page_size_bytes: 含 padding 的页字节数（传输时拷贝的字节量）
        # - unpadded_page_size_bytes: 真实（未 padding）页字节数（逻辑上有效数据量）
        # layer_name -> (num_blocks, page_size_bytes) tensor
        tensors_per_block: dict[str, tuple[torch.Tensor, ...]] = {}
        # layer_name -> size of (un-padded) page in bytes
        unpadded_page_size_bytes: dict[str, int] = {}
        # layer_name -> size of page in bytes
        page_size_bytes: dict[str, int] = {}

        # 遍历所有 KV cache group 的每一层，构造规范化的 (num_blocks, page) 视图。
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_layer_names = kv_cache_group.layer_names
            group_kv_cache_spec = kv_cache_group.kv_cache_spec
            # 非均匀 group（如混合架构中同组各层 page_size/dtype 不同）才有逐层 spec 字典；
            # 均匀 group 所有层共用 group 级 spec，此时 per_layer_specs留空。
            # 下面用 .get(layer_name, group_kv_cache_spec) 兜底：查得到就取逐层 spec，查不到（含空字典情况）就回退到 group 级 spec。
            if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
                per_layer_specs = group_kv_cache_spec.kv_cache_specs
            else:
                per_layer_specs = {}
            for layer_name in group_layer_names:
                layer_kv_cache_spec = per_layer_specs.get(
                    layer_name, group_kv_cache_spec
                )
                if isinstance(layer_kv_cache_spec, AttentionSpec):
                    # ⚠️ 普通 attention 层：拿到该层在 GPU 上的 KV cache 张量（单个 Tensor）
                    layer_kv_cache = kv_caches[layer_name]
                    assert isinstance(layer_kv_cache, torch.Tensor)

                    # page：含 padding 的每 block 字节数。
                    page = layer_kv_cache_spec.page_size_bytes
                    elem_size = layer_kv_cache.element_size()
                    # 张量在底层 storage 中的字节偏移（处理多张量共享同一 storage 的情况）
                    byte_offset = layer_kv_cache.storage_offset() * elem_size
                    # 步长（block 之间隔多少字节）：
                    # - packed 布局下用 stride(0)，即一个 manager-block 的总字节数；
                    # - 普通布局下相邻 block 紧挨着，步长就直接等于 page。
                    block_stride_bytes = (
                        layer_kv_cache.stride(0) * elem_size
                        if layer_is_packed[layer_name]
                        else page
                    )
                    # 关键：不拷贝数据，仅用 .set_() 在已有 storage 上重新解释为
                    # (num_blocks, page) 的 int8 视图；步长保证跨 block 寻址正确。
                    #
                    # 思路：把一层 KV cache 底层那块原始显存，重新看成一个(num_blocks, page_size_bytes) 的 int8 矩阵
                    # ——每块一行、每元素一个字节，使后面的 DMA 拷贝引擎能用"字节指针+每块字节数"直接搬，而不用关心 KV 原本是 fp16/bf16 还是哪种attention 布局。
                    tensors_per_block[layer_name] = (
                        # 载体张量：[] 只是占位哑元，内容会被 set_ 覆盖；
                        # dtype=int8 才能按"字节"寻址（每元素=1 字节,故列数=每 block 字节数）；
                        # device 必须与原 KV cache 同设备。
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=layer_kv_cache.device,
                        ).set_(
                            # storage: 原 KV 张量底层字节级缓冲区(untyped_storage抹掉 dtype)，新张量直接挂上去 -> 零拷贝。
                            layer_kv_cache.untyped_storage(),
                            # storage_offset: 本层数据在该共享 storage 内的起始字节偏移(=storage_offset()*elem_size)，
                            # 多层共享同一 storage 时定位本层起点。
                            byte_offset,
                            # size: 新形状 (num_blocks, page)。num_blocks=block
                            # 总数(行)；page=每 block 字节数(列,int8 下即字节)。
                            (num_blocks, page),
                            # stride: 行/列步长(以 1 字节为单位)。
                            #  block_stride_bytes: 相邻 block 间字节间隔 (packed 布局下含 padding 更大,普通布局=page);
                            #  1: 同 block 内相邻字节列步长。
                            (block_stride_bytes, 1),
                        ),
                    )
                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = (
                        layer_kv_cache_spec.real_page_size_bytes
                    )

                elif isinstance(layer_kv_cache_spec, MambaSpec):
                    # Mamba 等 state-space 层：KV cache 是多个 state 张量的列表，
                    # 而非单个注意力张量。
                    state_tensors = kv_caches[layer_name]
                    assert isinstance(state_tensors, list)

                    # 从第一个 state 张量出发，把整块底层 storage 重建成
                    # (num_blocks, page_size) 的连续 int8 视图。
                    # re-construct the raw (num_blocks, page_size) tensor
                    # from the first state tensor
                    assert len(state_tensors) > 0
                    first_state_tensor = state_tensors[0]
                    assert first_state_tensor.storage_offset() == 0
                    tensor = (
                        torch.tensor(
                            [],
                            dtype=torch.int8,
                            device=first_state_tensor.device,
                        )
                        .set_(first_state_tensor.untyped_storage())
                        .view((num_blocks, layer_kv_cache_spec.page_size_bytes))
                    )
                    tensors_per_block[layer_name] = (tensor,)

                    page_size_bytes[layer_name] = layer_kv_cache_spec.page_size_bytes
                    unpadded_page_size_bytes[layer_name] = replace(
                        layer_kv_cache_spec, page_size_padded=None
                    ).page_size_bytes

                else:
                    raise NotImplementedError

        # 检测是否存在 packed KV cache：即某个 kv_cache_tensor 同时有
        # block_stride 且被多个 layer 共享（如 DSv4 把多层拼进一块连续存储）。
        packed_kv_cache_tensor = next(
            (
                t
                for t in kv_cache_config.kv_cache_tensors
                if t.block_stride and t.shared_by
            ),
            None,
        )
        if packed_kv_cache_tensor is not None:
            # 所有共享层落到同一个物理张量上，因此只需要 1 个 CanonicalKVCacheTensor。
            # 用 as_strided 把它展开成 (num_blocks, 总字节数) 的大视图，
            # 每个 KV cache group 都指向 tensor 0。
            (tensor,) = tensors_per_block[packed_kv_cache_tensor.shared_by[0]]
            block_stride = tensor.stride(0)
            packed_tensor = tensor.as_strided(
                (num_blocks, block_stride),
                (block_stride, 1),
                storage_offset=0,
            )
            self._init_worker(
                CanonicalKVCaches(
                    [CanonicalKVCacheTensor(packed_tensor, block_stride)],
                    [
                        [CanonicalKVCacheRef(0, block_stride)]
                        for _ in kv_cache_config.kv_cache_groups
                    ],
                )
            )
            return

        # 非 packed 的普通布局：多个 layer 可能各自拥有独立的物理张量，
        # 也可能多 layer 共享同一张量。需要逐个 kv_cache_tensor 处理，
        # 产出"去重后的唯一物理张量列表"(block_tensors) 以及
        # "每个 layer 指向哪些 tensor"的引用表(block_data_refs)。
        block_tensors: list[CanonicalKVCacheTensor] = []
        block_data_refs: dict[str, list[CanonicalKVCacheRef]] = defaultdict(list)
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # 只保留本循环已成功构造视图的 layer。packed 分配会为每一种
            # (tuple_idx, page_size) 槽位都生成 KVCacheTensor 条目，
            # 没有对应模型的槽位是空 shared_by 的保留内存，需跳过。
            # Filter to layers that were actually processed above.
            # Packed KV allocation emits KVCacheTensor entries for
            # every (tuple_idx, page_size) slot; slots where no group has a
            # layer at that index produce an empty shared_by (reserved memory
            # with no corresponding model layer).
            tensor_layer_names = [
                n for n in kv_cache_tensor.shared_by if n in tensors_per_block
            ]
            if not tensor_layer_names:
                continue

            # 校验：同一物理张量上的所有 layer 必须指向完全相同的
            # 张量、data_ptr 和 stride（否则无法安全地统一视图）。
            # verify all layers in the group reference the exact same tensors
            assert len({len(tensors_per_block[n]) for n in tensor_layer_names}) == 1
            assert (
                len({tensors_per_block[n][0].data_ptr() for n in tensor_layer_names})
                == 1
            )
            assert (
                len({tensors_per_block[n][0].stride() for n in tensor_layer_names}) == 1
            )

            # 用第一个 layer 代表整组，把它的 (num_blocks, page) 视图登记为
            # 一个 CanonicalKVCacheTensor，并记录它对应的物理索引 curr_tensor_idx。
            # pick the first layer to represent the group
            first_layer_name = tensor_layer_names[0]
            for tensor in tensors_per_block[first_layer_name]:
                block_tensors.append(
                    CanonicalKVCacheTensor(
                        tensor=tensor,
                        page_size_bytes=page_size_bytes[first_layer_name],
                    )
                )

                curr_tensor_idx = len(block_tensors) - 1
                # 该物理张量被哪些 layer 共享：每个 layer 用 CanonicalKVCacheRef
                # 指回 tensor_idx，并记录自己未 padding 的真实 page 字节数。
                for layer_name in tensor_layer_names:
                    block_data_refs[layer_name].append(
                        CanonicalKVCacheRef(
                            tensor_idx=curr_tensor_idx,
                            page_size_bytes=(unpadded_page_size_bytes[layer_name]),
                        )
                    )

        # 把 per-layer 的引用表聚合成 per-group：每个 KV cache group 由若干
        # layer 组成，把它们的 CanonicalKVCacheRef 依次拼起来即为该 group 的
        # group_data_refs。这样 worker 就能按 group 定位要传输的 tensor 与字节范围。
        group_data_refs: list[list[CanonicalKVCacheRef]] = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_refs: list[CanonicalKVCacheRef] = []
            for layer_name in kv_cache_group.layer_names:
                group_refs += block_data_refs[layer_name]
            group_data_refs.append(group_refs)

        # 组装最终的 CanonicalKVCaches 并交给 worker 初始化：
        # - tensors: 去重后的唯一物理张量列表
        # - group_data_refs: 每个 group 由哪些 tensor/层组成、每层的真实页大小
        canonical_kv_caches = CanonicalKVCaches(
            tensors=block_tensors,
            group_data_refs=group_data_refs,
        )

        self._init_worker(canonical_kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        # verify that num_blocks is at physical position 0 in the cross-layers
        # tensor layout.
        test_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1234, block_size=16, num_kv_heads=1, head_size=256
        )
        num_blocks_logical_dim = test_shape.index(1234) + 1
        physical_to_logical = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        num_blocks_physical_dim = physical_to_logical.index(num_blocks_logical_dim)
        assert num_blocks_physical_dim == 0

        kv_cache_groups = self.spec.kv_cache_config.kv_cache_groups
        assert len(kv_cache_groups) == 1
        kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        num_layers = len(kv_cache_groups[0].layer_names)
        page_size_bytes = kv_cache_spec.page_size_bytes * num_layers

        assert kv_cache.storage_offset() == 0
        storage = kv_cache.untyped_storage()
        assert len(storage) % page_size_bytes == 0
        num_blocks = len(storage) // page_size_bytes
        tensor = (
            torch.tensor(
                [],
                dtype=torch.int8,
                device=kv_cache.device,
            )
            .set_(storage)
            .view(num_blocks, page_size_bytes)
        )
        kv_cache_tensor = CanonicalKVCacheTensor(
            tensor=tensor, page_size_bytes=page_size_bytes
        )
        # in cross layers layout, there's currently only a single group
        kv_cache_data_ref = CanonicalKVCacheRef(
            tensor_idx=0, page_size_bytes=page_size_bytes
        )
        canonical_kv_caches = CanonicalKVCaches(
            tensors=[kv_cache_tensor], group_data_refs=[[kv_cache_data_ref]]
        )

        self._init_worker(canonical_kv_caches)

    def handle_preemptions(self, kv_connector_metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        #
        if kv_connector_metadata.jobs_to_flush:
            self.worker.wait(kv_connector_metadata.jobs_to_flush)

    def start_kv_transfers(self, metadata: OffloadingConnectorMetadata):
        assert self.worker is not None
        for job_id, src_spec, dst_spec in self._unsubmitted_store_jobs:
            #
            success = self.worker.submit_store(job_id, src_spec, dst_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for job_id, entry in metadata.load_jobs.items():
            self._load_jobs[job_id] = entry.req_id
            assert isinstance(entry.dst_spec, GPULoadStoreSpec)
            #
            success = self.worker.submit_load(job_id, entry.src_spec, entry.dst_spec)
            assert success

    def prepare_store_kv(self, metadata: OffloadingConnectorMetadata):
        for job_id, entry in metadata.store_jobs.items():
            # NOTE(orozery): defer the store to the beginning of the next
            # engine step, so that offloading starts AFTER transfers related
            # to token sampling, thereby avoiding delays to token generation.
            assert isinstance(entry.src_spec, GPULoadStoreSpec)
            #
            self._unsubmitted_store_jobs.append(
                (job_id, entry.src_spec, entry.dst_spec)
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Returns:
            tuple of (finished_sending, finished_recving). Stores never
            emit finished_sending — the scheduler tracks store completion
            via kv_connector_worker_meta.completed_jobs and fences any
            block reuse via jobs_to_flush. Loads still emit
            finished_recving so the base scheduler can resume requests
            blocked on remote KV (and free aborted-during-load reqs).
        """
        assert self.worker is not None
        finished_recving: set[str] = set()
        for transfer_result in self.worker.get_finished():
            # we currently do not support job failures
            job_id = transfer_result.job_id
            assert transfer_result.success
            is_load = job_id in self._load_jobs
            if (
                transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                if is_load:
                    stats = self._connector_worker_meta.transfer_stats.load
                else:
                    stats = self._connector_worker_meta.transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )
            #
            self._connector_worker_meta.mark_completed(job_id)
            req_id = self._load_jobs.pop(job_id, None)
            if req_id is not None:
                finished_recving.add(req_id)

        return set(), finished_recving

    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        """Return completed transfer job IDs since the last call."""
        if not self._connector_worker_meta.completed_jobs:
            return None
        meta = self._connector_worker_meta
        self._connector_worker_meta = OffloadingWorkerMetadata()
        return meta

    def shutdown(self) -> None:
        self._unsubmitted_store_jobs.clear()
        self._load_jobs.clear()
        self._connector_worker_meta = OffloadingWorkerMetadata()
        if self.worker is not None:
            self.worker.shutdown()
