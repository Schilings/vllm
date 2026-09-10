# vLLM AttentionGroup 设计、架构、调度、创建与使用全流程分析

> 配套源码:`vllm/v1/worker/gpu/attn_utils.py`(`init_attn_backend`)、
> `vllm/v1/worker/utils.py`(`AttentionGroup` / `prepare_kernel_block_sizes` /
> `add_kv_sharing_layers_to_kv_cache_groups`)、
> `vllm/v1/kv_cache_interface.py`(`KVCacheGroupSpec` / `KVCacheConfig`)、
> `vllm/v1/attention/backend.py`(`AttentionCGSupport`)。

---

## 0. 一句话结论

**AttentionGroup 是"运行时注意力计算单元"的分组**,它沿 KV Cache Group 再切一刀,
把**后端相同、KV 规格相同、每 rank 的 `num_heads_q` 相同**的层聚到一起,使它们
**共享同一个 `AttentionMetadataBuilder`、同一份 attention metadata、同一块后端
workspace buffer**。

- **KV Cache Group** 回答的是"**KV 存在哪、怎么存**"(共享同一张 block table / 同一块
  显存),由内存布局驱动。
- **AttentionGroup** 回答的是"**attention 怎么算、metadata 怎么造**"(共享同一个 kernel
  backend 实例与其 scratch buffer),由**计算正确性 + 性能**驱动。

一个 KV Cache Group **内部可以拆成多个 AttentionGroup**(组数 ≤ KV 组数)。

---

## 1. 为什么需要 AttentionGroup?(与 KV Cache Group 的本质区别)

### 1.1 KV Cache Group 是什么

`vllm/v1/kv_cache_interface.py:950`

```python
@dataclass
class KVCacheGroupSpec:
    """Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager."""
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False
```

- 关注**物理存储**:同一组的层共用一张 block table、同一份 KV 显存张量。
- 由 `kv_cache_config.kv_cache_groups` 描述(见 `KVCacheConfig` 的 docstring:
  单类型注意力模型只有 1 个组;混合注意力如 Full+SWA 会有多个组)。
- 它驱动的是 **KV cache manager / block pool / block table** 那一套。

### 1.2 为什么 KV 组还不够——需要 AttentionGroup

KV 组只保证"显存共享",但**注意力计算**还需要:

1. **不同的注意力 backend**:同一块 KV 显存,可能要被不同的 kernel 后端读写
   (例如普通 Full-attention 用 FlashAttn,而 fast-prefill 路径被包了一层
   `FastPrefill` 自定义 backend,见 `gpu_model_runner.py:6843`)。
   不同 backend 的 `get_builder_cls()` 不同,必须各自造 metadata。
2. **`num_heads_q` 必须组内一致**:builder 的 scratch buffer(如 `triton_attn` 的
   `softmax_segm_*`、`FlashInfer` 的 `num_qo_heads`)是**按 `num_heads_q` 定尺**且
   **假设组内均匀**的(`gpu_model_runner.py:6815-6819` 注释)。因此 spec-decode 的
   draft head(头数可能比 target 少)必须和 target 拆到**不同** AttentionGroup。
3. **共享 workspace buffer**:所有 attention backend 实例共享一个后端 workspace
   (`attn_utils.py:160-165`),拆组后由 `set_workspace_buffer` 统一注入,避免重复分配。

→ 所以 `AttentionGroup` 在 KV 组**内部**,按 `(backend, kv_cache_spec, num_heads_q)`
三元组再分组。这正是 `attn_utils.py:114-136` 的 `group_map` key:

```python
key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)
```

### 1.3 三者关系图

```
KVCacheConfig
 ├─ kv_cache_groups[0]  (Full-attention 层, 共享 block table A)
 │    └─ attn_groups[0] = [ AttnGroup(FullAttn backend, heads=32),
 │                          AttnGroup(FastPrefill backend, heads=32),  # 若启用 fast-prefill
 │                          AttnGroup(Eagle draft backend, heads=8) ]  # spec-decode draft head
 ├─ kv_cache_groups[1]  (SWA 层, 共享 block table B)
 │    └─ attn_groups[1] = [ AttnGroup(SlidingWindow backend, heads=32) ]
 └─ ...
```

> 注意嵌套:`attn_groups: list[list[AttentionGroup]]`
> - 外层 `i` 对应 `kv_cache_group_id`(`init_attn_backend:600` 的 `for i in range(num_kv_cache_groups)`)。
> - 内层是该 KV 组下拆出的多个 AttentionGroup。

---

## 2. 数据结构:AttentionGroup

`vllm/v1/worker/utils.py:222`

```python
@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]          # 该组使用的注意力后端类
    layer_names: list[str]                   # 属于该组的层名
    kv_cache_spec: KVCacheSpec               # 该组的 KV 规格
    kv_cache_group_id: int                   # 所属的 KV Cache Group 编号
    metadata_builders: list[AttentionMetadataBuilder]  # 组内 builder(ubatch 时多个)
```

关键方法:

- `create_metadata_builders(vllm_config, device, kernel_block_size, num_metadata_builders=1)`
  (`utils.py:235`):用 `backend.get_builder_cls()` 为每个 builder 构造
  `AttentionMetadataBuilder`,并把 `kv_cache_spec` 用 `copy_with_new_block_size(
  kernel_block_size)` 换成 kernel 实际使用的块大小。
- `get_metadata_builder(ubatch_id=0)`:取第 `ubatch_id` 个 builder(ubatching 时每组
  每 ubatch 一个,避免内部 persistent buffer 互相打架)。

---

## 3. 创建流程:`init_attn_backend`(`attn_utils.py:83`)

> 注:本仓库存在**两套**实现:
> - `vllm/v1/worker/gpu/attn_utils.py:83` 的 `init_attn_backend(...)` —— V2 路径,
>   被 `model_runner.py:448` 调用,**三阶段合并在一个函数里**,返回
>   `(attn_groups, attn_cg_support_info, kernel_block_sizes)`。
> - `vllm/v1/worker/gpu_model_runner.py:6799` 的 `initialize_attn_backend` +
>   `initialize_metadata_builders` —— V1 路径,拆分更细,且含 ubatching 支持。
> 两者逻辑等价,下面以 V2 的 `attn_utils.py` 版本为主线。

### Phase 0:KV-sharing 层并入目标组(前置)
`attn_utils.py:94-99`

```python
add_kv_sharing_layers_to_kv_cache_groups(
    get_shared_kv_cache_layers(vllm_config), kv_cache_config.kv_cache_groups
)
```

- `get_shared_kv_cache_layers`(`attn_utils.py:74`):从 `static_forward_context` 取出
  `{ layer_name -> target_layer_name }`(哪些层复用别的层的 KV)。
- `add_kv_sharing_layers_to_kv_cache_groups`(`utils.py:428`):把这些"借用 KV"的层
  **加回它们目标层所在的 KV Cache Group**,从而 Phase 1 能和目标层一起被探测到,
  共享同一组 AttentionGroup / metadata。

### Phase 1:在 KV 组内发现 AttentionGroup
`attn_utils.py:101-138`

```python
for kv_cache_group_id, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
    layer_names = kv_cache_group_spec.layer_names
    if active_layer_names is not None:
        layer_names = list(active_layer_names.intersection(layer_names))
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase, layer_names)
    group_map = {}
    group_order = []
    for layer_name in layer_names:
        attn_backend = attn_layers[layer_name].get_attn_backend()
        layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
        if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
            layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
        num_heads_q = getattr(attn_layers[layer_name], "num_heads", 0)
        key = (attn_backend.full_cls_name(), layer_kv_cache_spec, num_heads_q)
        if key not in group_map:
            group_map[key] = AttentionGroup(attn_backend, [layer_name],
                                            layer_kv_cache_spec, kv_cache_group_id)
            group_order.append(key)
        else:
            group_map[key].layer_names.append(layer_name)
    attn_groups.append([group_map[key] for key in group_order])
```

- `get_layers_from_vllm_config`(`config/vllm.py:2313`):从
  `compilation_config.static_forward_context` 按 `layer_type` 过滤出 attention 层。
- 分组 key = `(backend 全类名, kv_cache_spec, num_heads_q)`。
- 同一 key 的层**合并**到同一个 `AttentionGroup`,追加进 `layer_names`。
- 结果 `attn_groups[i]` 是该 KV 组下的一组 AttentionGroup。

### Phase 2:选 kernel block size
`attn_utils.py:140-142` → `prepare_kernel_block_sizes`(`utils.py:331`)

```python
kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, attn_groups)
```

- 对 Attention 类 spec:调用 `select_common_block_size`(`utils.py:262`)在"KV manager 的
  block size"和"组内所有 backend 支持的 kernel block size"之间取**公共因子**——
  既满足所有 backend,又整除 manager 块大小(虚拟分块拆分)。
- 对 Mamba/非注意力 spec:不分块,直接用原 block size。
- 对 `EncoderOnlyAttentionSpec`:跳过(无 KV 块)。

### Phase 3:造 builder + 定 CG 支持
`attn_utils.py:144-178`

```python
attn_backend_workspace = None
min_cg_support = AttentionCGSupport.ALWAYS
min_cg_attn_backend = None
for kv_cache_group_id, groups in enumerate(attn_groups):
    kernel_block_size = kernel_block_sizes[kv_cache_group_id] if ... else None
    for group in groups:
        group.create_metadata_builders(vllm_config, device, kernel_block_size, 1)
        builder = group.get_metadata_builder(0)
        # 所有 backend 共享同一块 workspace buffer
        if attn_backend_workspace is None:
            if hasattr(builder, "_get_workspace_buffer"):
                attn_backend_workspace = builder._get_workspace_buffer()
        else:
            if hasattr(builder, "set_workspace_buffer"):
                builder.set_workspace_buffer(attn_backend_workspace)
        # 取全局最弱的 CG 支持等级
        cg_support = builder.get_cudagraph_support(vllm_config, group.kv_cache_spec)
        if cg_support.value < min_cg_support.value:
            min_cg_support = cg_support
            min_cg_attn_backend = group.backend.__name__
```

- **workspace 共享**:第一个 builder 创建 workspace,后续 builder 复用(`set_workspace_buffer`),
  避免为每组重复分配后端 scratch。
- **CG 支持取最小值**:`AttentionCGSupport`(`backend.py:583`)等级:
  `ALWAYS(3) > UNIFORM_BATCH(2) > UNIFORM_SINGLE_TOKEN_DECODE(1) > NEVER(0)`。
  全局取**最弱**等级,反哺图模式选择(`attn_cg_support_info`)。
- 返回 `AttentionCGSupportInfo(min_cg_support, min_cg_attn_backend)`。

---

## 4. 调度(Scheduling)维度

### 4.1 kernel block size(虚拟分块)
- 决定 attention kernel 实际操作的块粒度,与 KV manager 的 block size 解耦。
- 由 `select_common_block_size` 在 backend 支持集 ∩ manager 块大小 间求解。
- 影响 `CommonAttentionMetadata` 中 block table 的 stride 解读。

### 4.2 CUDA Graph 支持等级(`AttentionCGSupport`)
- 每个 builder 声明自己的 CG 能力,全局取 min。
- `model_runner.py` 用 `attn_cg_support_info.min_cg_support` 决定 `CUDAGraphMode`:
  - `ALWAYS` → 可 FULL(混合 prefill/decode 同图)。
  - `UNIFORM_BATCH` → spec-decode(每行 query 长度一致)。
  - `UNIFORM_SINGLE_TOKEN_DECODE` → 仅纯 decode(每行 query_len==1)。
  - `NEVER` → 不走 CUDA Graph。

### 4.3 workspace buffer 复用
- 跨 AttentionGroup 共享一个后端 workspace(Phase 3),是显存与带宽的优化。

### 4.4 reorder_batch(可选)
- builder 可声明 `reorder_batch_threshold`(`backend.py:607`):把短 query 行前置,
  提升 kernel  occupancy。该阈值在 `initialize_metadata_builders` 之后由
  `calculate_reorder_batch_threshold` 计算(`gpu_model_runner.py:6927`)。

### 4.5 ubatching(仅 V1 路径)
- `initialize_metadata_builders` 在 `use_ubatching` 时给每组创建
  `num_ubatches` 个 builder(`gpu_model_runner.py:6921`),每个 ubatch 独立 builder,
  避免 CUDA Graph persistent buffer 冲突。

---

## 5. 使用流程:从 prepare_attn 到每层 forward

### 5.1 入口:`GPUModelRunner.execute_model`(`model_runner.py`)
```
model_runner.py:448   self.attn_groups, attn_cg_support, self.kernel_block_sizes = init_attn_backend(...)
model_runner.py:497   kv_caches = init_kv_cache(..., self.attn_groups, ..., self.kernel_block_sizes, ...)
model_runner.py:1228  block_tables, slot_mappings = self.prepare_attn(input_batch)
model_runner.py:1278  attn_metadata = self.model_state.prepare_attn(input_batch, cg_mode,
                                                                     block_tables, slot_mappings,
                                                                     self.attn_groups, self.kv_cache_config)
```

- `attn_groups` 在第 448 行**一次性创建并常驻**,后续每步复用。
- `prepare_attn`(`model_runner.py:1059`)先 `gather_block_tables` 拿到
  `num_kv_cache_groups × [num_reqs_padded, max_num_blocks]` 的块表。

### 5.2 `model_state.prepare_attn`(`default.py:133`)
```python
def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                 attn_groups, kv_cache_config, for_capture=False):
    if cudagraph_mode == CUDAGraphMode.FULL:
        num_reqs = input_batch.num_reqs_after_padding   # padded 形状
    else:
        num_reqs = input_batch.num_reqs                 # 未 padded
    ...
    attn_metadata = build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs, num_tokens=num_tokens,
        query_start_loc_gpu=input_batch.query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        max_query_len=max_query_len, seq_lens=input_batch.seq_lens,
        max_seq_len=max_seq_len, block_tables=block_tables, slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound, ...,
        for_cudagraph_capture=for_capture, causal=causal, ...)
    return attn_metadata
```
- 根据 `CUDAGraphMode` 决定**形状策略**:FULL 用 padded,其余用未 padded。
- 把 `attn_groups` 透传给 `build_attn_metadata`。

### 5.3 `build_attn_metadata`(`attn_utils.py:571`)—— attn_groups 的"落地"
(详见 `attn_metadata.md`,此处只点出与 group 相关的部分)
```python
for i in range(num_kv_cache_groups):                 # 外层 = KV 组
    block_table = block_tables[i]; slot_mapping = slot_mappings[i]
    common = CommonAttentionMetadata(...)            # 每组一份通用元数据
    for attn_group in attn_groups[i]:                # 内层 = 该 KV 组下的 AttentionGroup
        builder = attn_group.get_metadata_builder(0)
        metadata = builder.build_for_cudagraph_capture(common) if for_capture \
                   else builder.build(common_prefix_len=0, common_attn_metadata=common)
        for layer_name in attn_group.layer_names:
            attn_metadata[layer_name] = metadata      # 同组多层共享同一份 metadata
```
- **每个 AttentionGroup 用其 builder 造一份 `metadata` 对象**。
- **同组所有层共享同一份 metadata**(按 `layer_name` 写入返回的 dict),零拷贝引用。
- 不同 AttentionGroup 即使在同一 KV 组,也各有各的 metadata(因为 backend / head 数不同)。

### 5.4 注入 forward 上下文 & 每层取用
- `model_runner` 通过 `set_forward_context(attn_metadata)` 把 dict 注入。
- 每层 `Attention`(`vllm/model_executor/layers/attention/attention.py:756`)用
  `attn_metadata[layer_name]` 取**自己那份** metadata,喂给对应 backend kernel。

---

## 6. 设计动机总结(为什么这样设计)

| 维度 | KV Cache Group | AttentionGroup |
|---|---|---|
| 回答的问题 | KV 存哪 / 怎么共享显存 | attention 怎么算 / metadata 怎么造 |
| 驱动因素 | 内存布局、block table | backend 类、`num_heads_q`、scratch 尺寸 |
| 分组 key | 由 `kv_cache_spec` 自动聚 | `(backend, kv_cache_spec, num_heads_q)` |
| 数量关系 | 基准(粗) | 在 KV 组内再细分,组数 ≤ KV 组数 |
| 产物 | 共享 block table / KV 张量 | 共享 `AttentionMetadataBuilder` + metadata + workspace |

**核心收益**:
1. **正确性**:`num_heads_q` 不一致(如 spec-decode draft head)的层被隔离,
   builder scratch 不会因组内 head 数不均而出错。
2. **性能**:同组共享 builder / metadata / workspace,减少构造开销与显存占用;
   kernel block size 按 backend 能力虚拟分块,最大化 kernel 效率。
3. **解耦**:KV 存储(group)与注意力计算(group)正交演进 —— 新增一种后端或 head
   配置,只需在 AttentionGroup 维度处理,不触碰 KV manager。
4. **CUDA Graph 友好**:每组 builder 声明 CG 能力,全局取最弱等级反哺图模式;
   同组 metadata 共享,形状在 capture 与 replay 间保持一致。

---

## 7. 端到端时序图

```
                ┌─────────────────────────────────────────────────────┐
初始化期(只一次) │ init_attn_backend (attn_utils.py:83)               │
                │  Phase0: add_kv_sharing_layers_to_kv_cache_groups    │
                │  Phase1: per KV group → AttentionGroup(key=backend+  │
                │           spec+num_heads_q)                          │
                │  Phase2: prepare_kernel_block_sizes                  │
                │  Phase3: create_metadata_builders + 共享 workspace   │
                │           + 取 min AttentionCGSupport                │
                │ → (attn_groups, attn_cg_support_info, kernel_bs)    │
                └─────────────────────────────────────────────────────┘
                                       │ 常驻 self.attn_groups
                                       ▼
每步 execute_model:
  prepare_attn() ──▶ gather_block_tables (按 KV 组取块表)
        │
        ▼
  model_state.prepare_attn(cg_mode, block_tables, slot_mappings, attn_groups)
        │  选形状(padded / 未 padded)
        ▼
  build_attn_metadata(attn_groups, ...)
        │  for KV group i:
        │     common = CommonAttentionMetadata(...)
        │     for attn_group in attn_groups[i]:
        │        metadata = builder.build(common)
        │        for layer_name in group: attn_metadata[layer_name] = metadata
        ▼
  set_forward_context(attn_metadata)
        │
        ▼
  每层 Attention: attn_metadata[layer_name] → backend kernel
```

---

## 8. 扩展点:新增一种模型/后端支持

1. **新 backend**:实现 `AttentionBackend.get_builder_cls()`,在 builder 中声明
   `get_cudagraph_support()` 与 `get_supported_kernel_block_sizes()`,
   分组逻辑会自动按 `full_cls_name()` 把它拆成独立 AttentionGroup。
2. **不同 head 数的层**(如 draft head):只要 `num_heads_q` 不同,自动分入不同组,
   无需手动干预。
3. **跨层 KV 复用**:在模型定义里声明 `shared_kv_cache_layers`,Phase 0 会自动把复用层
   并入目标 KV 组,从而共享其 AttentionGroup / metadata。
4. **新 KV 规格**:实现 `KVCacheSpec` 子类,`prepare_kernel_block_sizes` 会按
   `AttentionSpec` / `MambaSpec` / `EncoderOnlyAttentionSpec` 分支处理。

---

## 9. 相关阅读

- `attn_metadata.md` —— attention metadata 的字段、构造、消费全解。
- `model_states.md` —— `prepare_attn` 在 ModelState 策略中的位置。
- `hybrid_attention_kv_cache.md` / `kv_cache.md` —— KV Cache Group 的来龙去脉。
- `gpu_model_runner.py:6799-6939` —— V1 路径 `initialize_attn_backend` / `initialize_metadata_builders`
  (含更详尽的 `AttentionGroupKey` 注释,可作为本报告的补充参考)。
