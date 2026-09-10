# vLLM V1 `model_states` 模块深度分析

> 分析对象:`vllm/v1/worker/gpu/model_states/`
> 关联代码:`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/worker/gpu/input_batch.py`、`vllm/v1/worker/gpu/attn_utils.py`、`vllm/v1/worker/gpu/states.py`

---

## 1. 定位:它是什么角色

`model_states` 是 **GPUModelRunner 与具体模型架构之间的"适配层 / 策略层"**。

GPUModelRunner 负责的是**与模型无关的通用批处理流程**:调度输入、填 input buffer、算 batch 顺序与索引、准备 block table / slot mapping、跑 forward、采样。但"模型到底需要什么样的 attention metadata?要不要跑多模态 encoder?位置编码(mRoPE)怎么算?混合并行(Mamba+Attention)的循环状态怎么迁移?"这些都是**模型架构相关**的,不应该污染通用 runner。

`ModelState` 就是把这些"模型特定的胶水逻辑"收拢起来的抽象基类。Runner 只通过 `self.model_state.xxx()` 调用一组统一接口,而不关心背后是纯 decoder-only(Default)、Whisper 这类编解码器(EncoderDecoder),还是 Mamba 混合模型(MambaHybrid)。

一句话角色定位:
- **对上(Runner)**:提供一组"每个 step 都要被回调"的钩子(`add_request` / `prepare_inputs` / `prepare_attn` / `postprocess_state` …)。
- **对下(模型)**:封装 attention metadata 构造、多模态嵌入流水线、位置编码状态、混合循环状态等专属逻辑,并把它们以统一签名返回给 Runner。

---

## 2. 设计动机(为什么需要这层抽象)

| 问题 | 若无 ModelState | 有 ModelState |
|---|---|---|
| 模型差异 | Runner 里一堆 `if isinstance(model, Whisper)` 分支 | 工厂 `init_model_state` 选择实现类 |
| attention metadata | 每个后端自己拼 metadata,重复造轮子 | 统一 `prepare_attn` → `build_attn_metadata` |
| 多模态 | Runner 直接调 encoder_runner / rope_state | DefaultModelState 内部编排 |
| 混合模型状态 | 循环状态迁移散落各处 | MambaHybridModelState 在 `preprocess/postprocess` 钩子里收口 |
| 新增模型 | 改 Runner 主干 | 新建一个 ModelState 子类或模型定义 `get_model_state_cls` |

核心收益:**把"模型架构特异性"从通用执行循环里剥离,让 GPUModelRunner 保持通用,模型差异通过多态收敛**。这与 vLLM 整体"用 interface/registry 解耦"的风格一致(类似 `AttentionImpl`、`ModelRunner` 接口本身)。

---

## 3. 类结构

```
interface.py
├── ModelSpecificAttnMetadata      # 数据类:模型专属 attention 元数据载体
└── ModelState(ABC)               # 抽象基类:定义所有钩子与属性

__init__.py
└── init_model_state(...)         # 工厂:按模型类型选择实现类

default.py       → DefaultModelState(ModelState)             # 纯 decoder-only + 可选多模态
encoder_decoder.py → EncoderDecoderModelState(ModelState)    # 交叉注意力编解码器
mamba_hybrid.py  → MambaHybridModelState(DefaultModelState)  # Mamba/线性注意力 + 注意力混合
mm_pruning.py    → MultiModalPruner                          # (被 Default 组合使用)多模态剪枝
```

### 工厂选择顺序(`__init__.py:init_model_state`)
1. 若模型自己定义了 `model.get_model_state_cls()` → 用模型自定义类(最高优先级,模型可完全接管)。
2. 否则,若模型里**存在 `CrossAttention` 模块** → `EncoderDecoderModelState`(Whisper / CohereASR / NemotronParse / FireRedLID …)。
3. 否则,若 `model_config.is_hybrid` → `MambaHybridModelState`。
4. 否则 → `DefaultModelState`(绝大部分文本生成模型)。

注意 `MambaHybridModelState` 继承自 `DefaultModelState`,说明它是在"默认行为 + 多模态 + 注意力"基础上**叠加**了 Mamba 循环状态管理,而不是另起炉灶。

---

## 4. 接口逐方法解析(`interface.py`)

### 4.1 生命周期钩子
- `add_request(req_index, new_req_data)`:新请求进入时回调(默认:用 RoPE 状态初始化 prefill 位置)。
- `remove_request(req_id)`:请求结束时回调(默认空;需在 `req_states.remove_request` **之前**调用,以便还能查到 slot index)。
- `apply_staged_writes()`:把"暂存"的更新刷入持久状态(默认:刷 RoPE 的 staged 位置更新)。

### 4.2 Step 内前后置钩子
- `preprocess_state(input_batch, block_tables, kv_cache_config, num_computed_tokens)`:**forward 之前**(block table 收集后)运行。默认空;Mamba 用它把循环状态跨 block 边界迁移("align" prefix caching)。
- `postprocess_state(idx_mapping, num_sampled, num_computed_tokens)`:**forward + 采样之后**运行。默认空;Mamba 用它记录 `num_accepted_tokens`、保存未对齐的循环状态。

### 4.3 多模态相关
- `get_mm_embeddings(scheduled_encoder_inputs, input_batch, req_states)`:**核心**。跑多模态 encoder、按 mm_hash 缓存、gather 嵌入、与 input_ids 合成 `inputs_embeds`。
- `gather_mm_embeddings(input_batch, draft_lookahead)`:从 encoder cache 收集已缓存的 MM 嵌入(默认实现委托 `encoder_runner`)。
- `dummy_inputs_embeds(num_tokens)`:为 dummy/profiling run 预分配 shape 正确的 `inputs_embeds`(内容无意义,只占位供 compiled model 捕获图)。
- `prepare_dummy_inputs(num_reqs, num_tokens)`:dummy run 的模型输入(位置/mrope 占位)。

### 4.4 真正喂给模型的输入(`prepare_inputs`)
- `prepare_inputs(input_batch, req_states) -> dict`:返回一个 dict,其键值会**覆盖/合并**进 `model_inputs`(`model_runner.py:1316` 的 `**self.model_state.prepare_inputs(...)`)。例:Default 返回 `{"positions": ...}`(mRoPE 重算后的位置);EncoderDecoder 返回 `{"encoder_outputs": ...}`。

### 4.5 Attention metadata 构造(`prepare_attn`)—— 最关键的方法
- `prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config, for_capture) -> dict`:构造 attention metadata,最终返回的是传给 `build_attn_metadata` 的参数字典(或直接是 metadata)。它是模型差异最集中的地方。

### 4.6 任务 / 采样协商
- `get_supported_generation_tasks() -> tuple[GenerationTask, ...]`:声明支持的任务(generate / transcription / realtime)。Runner 在初始化时据此决定 `SupportedTask`。
- `custom_sampler(sampler) -> (sampler, rejection_sampler) | None`:允许模型/state 替换默认采样器。
- `num_new_sampled_tokens_per_step: int = 1`:每个 decode step 新采样 token 数(不含投机解码接受的 draft)。Runner 用它算 `decode_query_len`。

---

## 5. 三个实现类详解

### 5.1 DefaultModelState(`default.py`)—— 主力实现
覆盖绝大多数 decoder-only 模型 + 可选多模态。

核心持有:
- `rope_state`:来自 `get_rope_state(...)`。负责 mRoPE 位置计算与重算。无 RoPE 文本模型时为 `None`(此时 `prepare_inputs` 直接返回 `{}`,用 1D positions)。
- `mm_pruner`:来自 `maybe_create_mm_pruner(...)`。仅当模型启用多模态剪枝(EVS,如 Qwen2.5-VL)时非 None。

关键方法行为:
- `prepare_attn`(`default.py:132`):
  - FULL CUDA graph 下用 `num_reqs_after_padding` / `num_tokens_after_padding`(padding 形状),否则用未 padding 形状。
  - `max_seq_len` 计算:`for_capture` 时用最坏情况 `max_model_len`(保证图重放安全),否则取 `seq_lens_cpu_upper_bound[:num_reqs].max()`(即上一轮你问的那块 CPU 上界,host 端算、避免 D2H sync)。
  - 若模型是 `is_mm_prefix_lm`,调用 `compute_mm_prefix_ranges` 算 `req_doc_ranges`(多模态前缀 LM 的文档区间)。
  - 最终调 `build_attn_metadata(...)`,把 `seq_lens / positions / dcp_local_seq_lens / rswa_prefix_lens` 等一并以 `input_batch` 提供的值喂入。
- `get_mm_embeddings`(`default.py:67`):encoder_runner 准备并**执行** MM encoder → `encoder_cache.encoder_outputs` 按 mm_hash 缓存 → gather 嵌入 → (若有 pruner)EVS 重算 mRoPE 位置并刷入 RoPE staged → 合成 `inputs_embeds`。注意**最后裁剪到 `num_tokens_after_padding`**,与 CUDA graph 的 padding 对齐。
- `prepare_inputs`(`default.py:108`):仅当有 `rope_state` 时,调 `rope_state.prepare_positions(...)` 并在 GPU 上取 `positions` 返回。普通 1D 位置模型直接 `return {}`(因为 positions 已在 `input_batch.positions` 由 model_runner 填好)。

### 5.2 EncoderDecoderModelState(`encoder_decoder.py`)
专用于**交叉注意力编解码器**(语音/图像类 ASR、解析等)。差异点:

- 构造即要求 `encoder_cache is not None`,并额外持有 `encoder_seq_lens_gpu([max_num_reqs])` 与 `encoder_outputs` 列表。
- **不使用 `inputs_embeds`**,而是把 encoder 输出作为 `encoder_outputs` forward kwarg 直接喂给解码器。交叉注意力的 K/V 在第一步写入 KV cache,之后 decode 步直接读 cache(`encoder_decoder.py:83-91` 注释)。
- `prepare_inputs` 返回 `{"encoder_outputs": ...}`(且立即清空,防泄漏)。
- `prepare_attn` 构造 `EncoderDecoderAttnMetadata`(子类 `ModelSpecificAttnMetadata`),其中 `_get_encoder_seq_lens` 按 `CrossAttentionSpec` 把每个请求的 encoder 序列长度(真实或 capture 时最坏 `max_encoder_len`)注入到对应的 kv_cache_group。`get_extra_common_attn_kwargs` 在 `build_attn_metadata` 内部被调用,把这些 `encoder_seq_lens` 透传给交叉注意力后端。
- 注:此类**未继承 Default**,所以多模态剪枝/RoPE 等逻辑它不含(编解码器模型不走 mRoPE 剪枝路径)。

### 5.3 MambaHybridModelState(`mamba_hybrid.py`)
继承 Default,叠加 **Mamba / 线性注意力混合模型的循环状态管理**。是 `ModelState` 钩子被用得最充分的一个子类。

核心持有:
- `num_accepted_tokens_gpu([max_num_reqs])`:每个请求本步接受的 token 数(投机解码用,默认 1 为中性值)。
- 当 `mamba_cache_mode == "align"` 时,额外持有:
  - `_mamba_state_idx_gpu`(当前循环状态所在 block)
  - `_mamba_src_col_gpu / _mamba_src_off_gpu`(align 预拷贝的源列/偏移)
  - `_mamba_ctx: MambaSpecDecodeGPUContext`(预拷贝的 fused GPU 上下文)

关键钩子:
- `preprocess_state`(`mamba_hybrid.py:166`):**forward 前**,在 GPU 上用 `preprocess_mamba_align_fused_kernel` 把每个请求的 recurrent state **跨 block 边界迁移**(V1 align 语义),并 `run_fused_precopy`。仅在 `align` 模式 + 真实 batch 下运行(虚拟 DP/profile 跳过)。kernel 对不跨边界的请求 fast-exit,开销极低。
- `prepare_attn`(`mamba_hybrid.py:215`):在 Default 基础上额外构造 `MambaHybridAttnMetadata`(携带 `is_prefilling`、`num_accepted_tokens`、`num_decode_draft_tokens_cpu`),经 `get_extra_common_attn_kwargs` / `get_extra_attn_kwargs` 注入给 Mamba2 / GDN 注意力后端。仅在非 capture 且开启投机解码时计算这些 spec 字段。
- `postprocess_state`(`mamba_hybrid.py:295`):把采样结果(接受 token 数)scatter 回 `num_accepted_tokens_gpu`(经 `_scatter_num_accepted_kernel`),并在 align 模式下 `run_fused_postprocess_align` 把未 block 对齐的状态保存回对齐位置。

### 5.4 MultiModalPruner(`mm_pruning.py`)—— 被 Default 组合使用的辅助类
不是 `ModelState` 子类,而是 Default 持有的一个**可选工具**,专门处理**多模态嵌入剪枝(EVS)** 模型(如 Qwen2.5-VL / Qwen3-VL / Nemotron-Nano-VL 的 Efficient Video Sampling)。

- 剪枝模型会在 media embedding 末尾**追加 mRoPE 位置通道**。Pruner 负责:在 target forward 时 `recompute()` 把这些通道拆出、重算正确的 mRoPE 位置并 stage 回 `RopeState`;在 draft forward 时 `strip()` 仅剥掉末尾通道(投机器复用 target 已重算的位置)。
- 通过 `maybe_create_mm_pruner` 创建,只有"启用多模态剪枝 + 模型支持 + 有 RopeState + 有 encoder_cache"才非 None。
- `recompute` 内部逐请求按 window 切分嵌入、调 `model.recompute_mrope_positions`,再 `rope_state.update_prefill_positions` 写回。逻辑刻意与 gather 路径重复(`_num_window_embeds`),保持主路径干净。

---

## 6. 在 GPUModelRunner 中的调用时序

结合 `model_runner.py` 的 `execute_model` 流程,`model_state` 的钩子按如下顺序被调用:

```
execute_model(scheduler_output, ...)
│
├─ dispatch_cg_and_sync_dp(...)                      # 1️⃣ DP 协商(上轮分析)
│
├─ prepare_inputs(scheduler_output, batch_desc)      # 2️⃣ 填 InputBatch
├─ prepare_attn(input_batch)                         # 3️⃣ 取 block_tables / slot_mappings
│
├─ model_state.preprocess_state(...)                 # 4️⃣ 前置钩子(Mamba align 预拷贝)
│
├─ model_state.prepare_attn(...)                     # 5️⃣ 构造 attention metadata(核心)
│        └─ build_attn_metadata(...) [default/encdec/mamba 各自注入 model_specific_attn_metadata]
│
├─ model_state.get_mm_embeddings(...) / dummy_inputs_embeds(...)  # 6️⃣ 多模态嵌入(仅首 PP rank)
│
├─ model_inputs = {
│       "input_ids/positions/inputs_embeds",
│       **model_state.prepare_inputs(input_batch, req_states)     # 7️⃣ 覆盖式注入(positions / encoder_outputs)
│   }
│
├─ model.forward(**model_inputs)                     # 8️⃣ 真正前向
├─ sampler / speculator                             # 9️⃣ 采样
│
└─ model_state.postprocess_state(...)                # 🔟 后置钩子(记录 num_accepted_tokens / align 保存)
```

此外,在**请求生命周期**中:
- `model_state.add_request(req_index, new_req_data)` —— 在 `req_states.append_block_ids` 之前(`model_runner.py:815`)。
- `model_state.apply_staged_writes()` —— 与新请求 staged 写入一起刷新(`model_runner.py:836`)。
- `model_state.remove_request(req_id)` —— 在 `req_states.remove_request` **之前**(`model_runner.py:749`),以便仍能查到 slot。

初始化期:
- `self.model_state = init_model_state(...)`(`model_runner.py:321`)。
- `self.model_state.get_supported_generation_tasks()` 决定 SupportedTask(`:272`)。
- `self.model_state.custom_sampler(self.sampler)` 可能替换默认采样器(`:341`)。
- `self.model_state.num_new_sampled_tokens_per_step` 用于 `decode_query_len`(`:325-328`)。
- CUDA graph 捕获时 `capture(self.model, self.model_state, ...)` 把 model_state 传进去(`:719`),dummy run 走 `prepare_dummy_inputs` / `dummy_inputs_embeds`。

---

## 7. 扩展点总结:如何新增一种模型支持

1. **最小侵入(模型自定义)**:在模型类里定义 `get_model_state_cls()`,直接返回自己的 `ModelState` 子类,工厂优先采用。
2. **通用扩展**:新建 `MyModelState(ModelState)`,在 `init_model_state` 工厂里按检测条件(如某模块类型、`model_config` 标志)插入分支。
3. **继承复用**:若只是 Default 之上加料(如 MambaHybrid),继承 `DefaultModelState` 并覆写 `prepare_attn` / `preprocess_state` / `postprocess_state`,复用的多模态 / RoPE / 基础 attn 逻辑。
4. **模型专属 attention 元数据**:继承 `ModelSpecificAttnMetadata`,实现 `get_extra_common_attn_kwargs`(注入到 `CommonAttentionMetadata`)与 `get_extra_attn_kwargs`(注入到特定后端 builder),再由 `prepare_attn` 通过 `model_specific_attn_metadata=` 参数传给 `build_attn_metadata`。

---

## 8. 小结

`model_states` 模块的本质是 **GPUModelRunner 的"模型多态适配层"**,承担四类职责:

1. **Attention metadata 构造的收口**:所有模型差异最终在 `prepare_attn` 里通过 `build_attn_metadata` + `ModelSpecificAttnMetadata` 收敛,Runner 无需感知后端差异。
2. **多模态流水线的编排**:encoder 执行、mm_hash 缓存、嵌入 gather、EVS 剪枝重算 mRoPE 位置,全部封装在 Default(及可选 Pruner)内。
3. **非注意力状态的管理**:RoPE/mRoPE 位置(Default)、交叉注意力 encoder 输出与序列长度(EncoderDecoder)、Mamba 循环状态的 block 对齐迁移与接受计数(MambaHybrid)。
4. **生命周期与协商钩子**:请求增删、staged 写入、任务声明、采样器替换、每步新增采样数。

它让 `GPUModelRunner` 这个"通用执行引擎"保持架构无关,而把"模型长什么样"这一易变维度,通过**工厂 + 策略模式 + 前后置钩子**优雅隔离。这也是为什么同一个 Runner 能同时支撑纯文本 LLM、Whisper 类 ASR、Qwen-VL 多模态、以及 Mamba 混合模型。
