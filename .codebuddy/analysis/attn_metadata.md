# vLLM V1 Attention Metadata 设计、架构、调度、创建与使用全流程分析

> 分析对象:`vllm/v1/attention/backend.py`、`vllm/v1/worker/gpu/attn_utils.py`、`vllm/v1/worker/gpu/model_states/*`、`vllm/v1/worker/utils.py`、`vllm/model_executor/layers/attention/attention.py`
> 关联:`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/attention/backends/flash_attn.py`、`vllm/v1/kv_cache_interface.py`

---

## 1. 概览与定位

Attention Metadata(注意力元数据)是 vLLM V1 在每次 `execute_model` 中**为当前这批请求构造的一组"注意力执行指令"**,它告诉底层注意力 kernel:

- 这批请求在扁平 query 张量里**从哪到哪**(query_start_loc);
- 每个请求的**上下文长度 / KV 长度**(seq_lens);
- **KV 缓存的物理块表**(block_table)与**新 token 落盘位置**(slot_mapping);
- 是否因果、最大 query 长度、最大序列长度(用于 kernel launch 配置);
- 以及模型专属信息(交叉注意力的 encoder 序列长度、Mamba 的 `is_prefilling`、DCP 分片长度、R-SWA 前缀长度等)。

它是连接 **调度器(决定本步跑哪些 token)**、**KV 缓存管理(决定块表/slot)** 和 **注意力后端 kernel(真正算注意力)** 三者的关键枢纽,也是 CUDA Graph 能否安全重放的决定性因素之一。

设计核心思想:**"一份共享的通用元数据 + 后端/模型专属的扩展注入 + 每 KV 组独立构造"**。

---

## 2. 架构:四层核心抽象

```
                      ┌─────────────────────────────────────────────┐
   model_runner       │  build_attn_metadata(...)                   │  ← attn_utils.py
 (execute_model)      │   构造 CommonAttentionMetadata(共享)         │
        │             │   遍历 attn_groups × attn_group              │
        ▼             │   每个 builder.build() 产出 per-layer 元数据  │
 ┌──────────────┐     └─────────────────────────────────────────────┘
 │ ModelState   │                  │                      │
 │.prepare_attn│──────────────────┘                      │
 └──────────────┘                                         ▼
                                  ┌──────────────────────────────────────┐
                                  │ CommonAttentionMetadata (共享载体)     │
                                  │  query_start_loc / seq_lens /         │
                                  │  block_table / slot_mapping /         │
                                  │  max_query_len / max_seq_len / ...    │
                                  │  + model_specific 注入字段             │
                                  └──────────────────────────────────────┘
                                                  │ 被传入每个 builder.build()
                                                  ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ AttentionMetadataBuilder (每个 AttentionGroup 一个)                 │
   │   build() / build_for_cudagraph_capture() / build_for_drafting()   │
   │   产出 FlashAttentionMetadata / MLAAttnMetadata / ... (per-layer)  │
   └──────────────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
   attn_metadata: dict[layer_name -> AttentionMetadata]
        │  set_forward_context(attn_metadata) 注入全局 forward_context
        ▼
   Attention 层 forward 时: attn_metadata[layer_name]  取出自己那一份
```

### 2.1 `CommonAttentionMetadata`(`backend.py:394`)
**跨层、跨后端共享的通用元数据载体**(dataclass)。关键字段:

| 字段 | 形状/dtype | 含义 |
|---|---|---|
| `query_start_loc` / `query_start_loc_cpu` | `(num_reqs+1,)` | 每个请求在扁平 query 张量中的起址(CPU 版供 host 端逻辑用,零拷贝共享) |
| `seq_lens` | `(num_reqs,)` GPU | 每个请求的当前上下文长度(KV 长度) |
| `seq_lens_cpu_upper_bound` | `(num_reqs,)` CPU | 乐观上界(上一轮已分析;prefill 精确,async-spec 乐观) |
| `num_reqs` / `num_actual_tokens` | int | 请求数 / token 总数(可能含 padding) |
| `max_query_len` / `max_seq_len` | int | 批内最长 query / 最长上下文(决定 kernel 规模) |
| `block_table_tensor` / `slot_mapping` | tensor | 本 KV 组的块表 / slot 映射 |
| `causal` | bool/tensor | 是否因果 |
| `dcp_local_seq_lens` | tensor\|None | DCP 下本 rank 持有的 KV 长度 |
| `positions` | tensor\|None | token 位置(供稀疏注意力预计算) |
| `is_prefilling` | tensor\|None | 是否仍在 prefill 阶段 |
| `mm_req_doc_ranges` | dict\|None | 多模态 PrefixLM 双向注意力区间 |
| `rswa_prefix_lens` | tensor\|None | 参考滑动窗口注意力(R-SWA)前缀长度 |

它内部还有若干**惰性缓存方法**:`compute_num_computed_tokens()`(设备端算 `seq_lens - query_lens`)、`token_to_req_indices()`(构造 token→请求索引,带缓存)、`unpadded()`(构造去掉 padding 的视图用于 spec-decode)。注意 `seq_lens_cpu` / `num_computed_tokens_cpu` 已被标为 deprecated——设计上**刻意避免隐式 H→D 同步**,优先用 GPU 上的 `seq_lens`(`backend.py:483-511` 注释 "avoid implicit H<>D sync which breaks full async scheduling")。

### 2.2 `ModelSpecificAttnMetadata`(`model_states/interface.py:21`)
模型/后端专属元数据的**注入接口基类**(dataclass)。只定义两个钩子:
- `get_extra_common_attn_kwargs(kv_cache_group_id, num_reqs) -> dict`:把字段注入到 `CommonAttentionMetadata` 构造里(如 EncoderDecoder 的 `encoder_seq_lens`)。
- `get_extra_attn_kwargs(attn_metadata_builder, num_reqs) -> dict`:把字段注入到特定 builder 的 `build()` 里(如 Mamba 的 `num_accepted_tokens` / `num_decode_draft_tokens_cpu`)。

这样**后端差异通过多态收敛**,`build_attn_metadata` 主干不用 `if model == xxx`。

### 2.3 `AttentionMetadataBuilder`(`backend.py:600`)
**后端专属的元数据构造器**(每个 `AttentionGroup` 持有一个实例,常驻)。核心方法:
- `build(common_prefix_len, common_attn_metadata, fast_build=False)`:真正构造 per-layer 元数据,是中枢方法。
- `build_for_cudagraph_capture(common_attn_metadata)`:CUDA graph **捕获阶段**用,默认调 `build(common_prefix_len=0, ...)`,子类(如 FlashMLA)会覆盖以用最坏 `max_seq_len` 捕获。
- `build_for_drafting(...)`:投机解码 draft 模型用(`fast_build=True`,优先构造速度)。
- `update_block_table(...)`:多个 KV 组共享同一结构仅块表不同时的快速更新。
- `_cudagraph_support`(类变量)/ `get_cudagraph_support()`:声明该后端的 CUDA Graph 支持等级。
- `reorder_batch_threshold`:是否/如何对 batch 重排(把短 query 提到前面,见 §3.3)。

### 2.4 `AttentionGroup`(`worker/utils.py:222`)
把"共享同一**后端 + KV cache spec + Q 头数**"的若干层归为一组:

```python
@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    metadata_builders: list[AttentionMetadataBuilder]
```

- 同一组的层**共用一个 metadata builder 实例**,只需构造一次元数据,再按 `layer_names` 复制分发(见 §5)。
- 分组键(`attn_utils.py:129`):`(backend.full_cls_name(), kv_cache_spec, num_heads_q)`。不同 Q 头数的层(如 spec-decode 的 draft head 与 target head)会被分到不同组,得到独立 builder。
- `attn_groups` 是**两层列表**:`[kv_cache_group_id][AttentionGroup]`。每个 KV cache group 可拆成多个 attention group。

---

## 3. 调度(Scheduling)维度

"调度"在此有双重含义:(A) 本步 batch 的 CUDA Graph 模式如何决定元数据形状;(B) 注意力后端内部的执行调度策略。

### 3.1 CUDA Graph 模式 → 元数据 padding 形状
`model_state.prepare_attn`(如 `default.py:142`)根据 `cudagraph_mode` 选择 `num_reqs` / `num_tokens`:
- `FULL` 模式:用 `num_reqs_after_padding` / `num_tokens_after_padding`(对齐到捕获图形状)。
- 其他(PIECEWISE / eager):用未 padding 的 `num_reqs` / `num_tokens`。

`max_seq_len` 也随模式变化:`for_capture=True` 时用最坏情况 `self.max_model_len`(保证图对任意重放都有效),否则取 `seq_lens_cpu_upper_bound[:num_reqs].max()`(`default.py:153-157`)。这直接决定了注意力 kernel 的 buffer 规模,是 CUDA Graph 安全的关键。

### 3.2 `AttentionCGSupport` 等级(`backend.py:583`)
后端声明其 CUDA Graph 支持级别,取所有组中的**最小值**作为整体支持(`attn_utils.py:171`):

| 等级 | 含义 |
|---|---|
| `ALWAYS` (3) | 始终支持,含混合 prefill-decode(如 FA3) |
| `UNIFORM_BATCH` (2) | 仅当批内 query 长度一致(投机解码的 spec-as-decode) |
| `UNIFORM_SINGLE_TOKEN_DECODE` (1) | 仅 query_len==1 的纯 decode |
| `NEVER` (0) | 不支持 |

这个等级会反向影响 `dispatch_cg_and_sync_dp` 选哪条图(即 §1 你之前看的那步),形成"后端能力 → 图模式 → 元数据形状"的闭环。

### 3.3 Batch 重排(Reorder)与 AOT 调度
- `reorder_batch_threshold`(`backend.py:634`):部分后端(如 MLA)会把 query 长度 ≤ 阈值的请求重排到 batch 前部,使一个 kernel launch 能高效处理。阈值会根据投机 token 数、DCP 自动抬高。
- AOT 调度(FA3,`flash_attn.py:381-402`):在 builder `__init__` 里预分配 `scheduler_metadata` 和 `max_num_splits`,`build()` 中按 `aot_schedule` 在 host 端完成 kernel 调度计划;`fast_build=True`(spec-decode)时关闭以省开销。

### 3.4 多后端 / 多 KV 组的块大小协商
`prepare_kernel_block_sizes` + `select_common_block_size`(`worker/utils.py:262-372`)为**每个 KV 组**选一个被组内所有后端支持的 kernel block size(支持 virtual block splitting 的后端可拆)。MambaSpec 组不做拆分。这保证了同一组的 builder 用统一的块大小构造 block_table。

---

## 4. 创建流程(Creation)

### 4.1 初始化期:`init_attn_backend`(`attn_utils.py:83`)
在 `GPUModelRunner.__init__` 中(`model_runner.py:447`)一次性完成,**每个 step 不再重复**:

1. **Phase 1 发现分组**:遍历 `kv_cache_config.kv_cache_groups`,按 `(backend, kv_cache_spec, num_heads_q)` 把层归入 `AttentionGroup`,形成 `attn_groups[kv_cache_group_id][group]`。
2. **Phase 2 选 kernel block size**:`prepare_kernel_block_sizes`。
3. **Phase 3 建 builder 并定 CG 支持**:对每个 group 调 `create_metadata_builders(...)`(`utils.py:235`),内部 `backend.get_builder_cls()(kv_cache_spec, layer_names, vllm_config, device)` 实例化 builder。同时挑出全局 `min_cg_support`。

返回 `(attn_groups, attn_cg_support_info, kernel_block_sizes)`,runner 长期持有 `self.attn_groups`。

### 4.2 每个 step:`prepare_attn` → `build_attn_metadata`
以 `DefaultModelState.prepare_attn`(`default.py:132`)为例:
1. 按 `cudagraph_mode` 决定 padded / unpadded 的 `num_reqs/num_tokens`(§3.1)。
2. 取 `query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)`(零拷贝,上轮已分析)。
3. 算 `max_seq_len`。
4. 调 `build_attn_metadata(...)`(`attn_utils.py:571`)——**主控函数**。

### 4.3 `build_attn_metadata` 内部(`attn_utils.py:592-649`)
```python
seq_lens = seq_lens[:num_reqs]                       # 切片(视图,无拷贝)
dcp_local_seq_lens = dcp_local_seq_lens[:num_reqs]
seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound[:num_reqs]

attn_metadata = {}
num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
for i in range(num_kv_cache_groups):                 # 遍历每个 KV 组
    block_table = block_tables[i]                    # 该组的块表
    slot_mapping = slot_mappings[i]
    # 1) 模型专属字段注入 CommonAttentionMetadata
    extra = model_specific_attn_metadata.get_extra_common_attn_kwargs(i, num_reqs)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        seq_lens=seq_lens, block_table_tensor=block_table,
        slot_mapping=slot_mapping, ...,
        **extra)
    # 2) 该组下每个 attention group 用其 builder 构造
    for attn_group in attn_groups[i]:
        builder = attn_group.get_metadata_builder(0)
        if for_cudagraph_capture:
            metadata = builder.build_for_cudagraph_capture(common)
        else:
            extra2 = model_specific_attn_metadata.get_extra_attn_kwargs(builder, num_reqs)
            metadata = builder.build(0, common, **extra2)
        # 3) 同一 group 的所有层共享这份 metadata
        for layer_name in attn_group.layer_names:
            attn_metadata[layer_name] = metadata
return attn_metadata
```

要点:
- **一份 `CommonAttentionMetadata` 按 KV 组构造,传入各自 builder**;builder 产出 per-layer 元数据。
- **同组多层共享同一 metadata 对象**(按 `layer_names` 复制引用),省去重复构造。
- `for_cudagraph_capture` 决定走 `build_for_cudagraph_capture`(最坏形状捕获)还是 `build`(实际执行形状)。

### 4.4 具体 builder 做了什么(以 FlashAttention `build`,`flash_attn.py:428`)
从 `CommonAttentionMetadata` 取出 `query_start_loc / seq_lens / block_table / slot_mapping / causal` 等,结合 builder 自身持有的常量(头数、block_size、DCP/CP 配置、R-SWA 持久 buffer),构造 `FlashAttentionMetadata`——里面是真正喂给 FlashAttention kernel 的张量(如 `cu_seqlens_q/k`、`block_table`、`slot_mapping`、AOT scheduler 计划等)。FA3 还会首次 build 时惰性确定 `aot_sliding_window`。

---

## 5. 使用流程(Usage)

### 5.1 注入 forward_context
`execute_model` 在真正 forward 之前(`model_runner.py:1360`):
```python
with set_forward_context(
    attn_metadata,                       # 就是 §4.3 返回的 dict
    self.vllm_config,
    num_tokens=input_batch.num_tokens_after_padding,
):
    hidden_states = self.model(**model_inputs)
```
`attn_metadata` 被放进全局 `forward_context`。投机解码时它可能是 `list[dict]`(多步),取 `[0]` 为 base。

### 5.2 每层取自己的元数据
模型里每个 `Attention` 层 forward 时(`attention.py:756`):
```python
forward_context = get_forward_context()
attn_metadata_raw = forward_context.attn_metadata
if isinstance(attn_metadata_raw, dict):
    attn_metadata = attn_metadata_raw[layer_name]   # 按层名取
elif isinstance(attn_metadata_raw, list):
    attn_metadata = attn_metadata_raw[0][layer_name]
else:
    attn_metadata = attn_metadata_raw
```
然后 `attn_metadata` 被传给 `attn_layer.impl.forward(...)`,最终驱动具体 kernel。MLA、Mamba、GDN 等也通过 `forward_context.attn_metadata[layer_name]` / `attn_metadata[layer.prefix]` 取用(`mla_attention.py:570`、`mamba_mixer.py:263`、`*_gdn_*.py` 等),印证了"**一个 dict 按 layer_name 分发**"的统一消费模式。

### 5.3 生命周期收尾
forward + 采样后,metadata dict 随 `ExecuteModelState` 暂存(`model_runner.py:1400`)供 speculator 复用,随后在下一步被新构造的 metadata 替换——**metadata 是每步重建的临时对象**,而 `attn_groups` / builder 实例、`CommonAttentionMetadata` 的持久 buffer(如 R-SWA 的 `persistent_rswa_prefix_lens`、FA3 的 `scheduler_metadata`)是常驻复用的,以保证 CUDA Graph 安全。

---

## 6. 设计亮点总结

1. **共享 + 注入的解耦**:`CommonAttentionMetadata` 承载所有后端公共字段,`ModelSpecificAttnMetadata` 通过两个 `get_extra_*` 钩子做多态注入,主干 `build_attn_metadata` 零 `if model` 分支。
2. **分组复用**:`AttentionGroup` 把"同后端同 spec 同 Q 头数"的层合并,一个 builder 构造、按 `layer_names` 分发,避免逐层重复构造。
3. **CUDA Graph 友好**:`for_cudagraph_capture` 用最坏形状捕获;`max_seq_len` 取上界;持久 buffer 常驻;GPU/CPU 双版本字段规避 H→D 同步——一切为图安全重放服务。
4. **调度闭环**:后端 `AttentionCGSupport` 等级 → 图模式选择 → padding 形状 → 元数据规模,形成自洽链路;`reorder_batch` / AOT schedule 在 kernel 层进一步优化执行效率。
5. **零拷贝与懒计算**:`query_start_loc_cpu` 用 `torch.from_numpy` 共享 numpy buffer;`seq_lens_cpu_upper_bound` 放 CPU 避免 D2H;`compute_num_computed_tokens` / `token_to_req_indices` 带缓存惰性计算。

---

## 7. 端到端时序一览

```
GPUModelRunner.__init__
  └─ init_attn_backend()  →  self.attn_groups (常驻), builders 实例化
        │
execute_model(scheduler_output)
  ├─ dispatch_cg_and_sync_dp()             # 决定 cg_mode / padding (§3.1)
  ├─ prepare_inputs() / prepare_attn()     # 填 input_batch, 取 block_tables/slot
  ├─ model_state.preprocess_state()        # Mamba 等前置 (可选)
  ├─ model_state.prepare_attn()            # §4.2 选形状/算 max_seq_len
  │     └─ build_attn_metadata()           # §4.3 主控
  │           ├─ for each KV group i:
  │           │     CommonAttentionMetadata (+ model_specific 注入)
  │           │     for each AttentionGroup:
  │           │        builder.build() / build_for_cudagraph_capture()
  │           │        → attn_metadata[layer_name] = metadata
  │           └─ return dict[layer_name -> AttentionMetadata]
  ├─ get_mm_embeddings()                   # 多模态 (可选)
  ├─ set_forward_context(attn_metadata)    # §5.1 注入
  ├─ model.forward(**inputs)                # 每层 attention.py:756 取自己的 metadata
  └─ postprocess_state()                   # Mamba 等后置 (可选)
```

至此,attention metadata 从"调度结果 + KV 块表"被完整翻译为"每个注意力层可直接消费的 kernel 指令",完成其在 vLLM V1 中的核心使命。
