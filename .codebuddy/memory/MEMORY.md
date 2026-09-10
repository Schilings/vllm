# 项目记忆

## Git 分支策略（2026-07-12）
- `main` 分支：纯净跟踪上游 vllm-project/vllm，不加任何修改
- `annotated` 分支：从 main 分出，用于添加源码阅读注释
- 上游更新时：`main` pull upstream → `annotated` rebase main

## 远程仓库
- origin: https://github.com/Schilings/vllm.git (fork)
- upstream: https://github.com/vllm-project/vllm.git (官方)

## 源码分析工作流偏好（2026-07-31 更新）
- 当前分支 `comments-on-v0.25.1` 目标是给 vLLM 做源码阅读分析。
- 用户偏好：**调研+源码分析+报告三步走**。
  - 报告：讲"为什么这样设计、全局关系、调度/生命周期"，建立全局认知。
  - **带注释的源码片段应放进 `.codebuddy/analysis/*.md` 文档里（Markdown 代码块 + 中文行间注释），而不是修改 `vllm/` 真实源码文件。**
- ⚠️ **重要纠正（2026-07-31）**：用户明确反对直接改 `vllm/` 源码加注释。源码文件必须保持干净。
  所有中文行间注释一律写在文档的代码块内。文档中代码块用 `行号:行号:文件路径` 标注，方便按图索骥回源码。
- 报告产出目录：`.codebuddy/analysis/`。
- **注释语言**：一律用**中文**（用户明确要求，曾纠正过英文注释）。正式 API docstring 可保留英文。

## 源码分析文档
- `.codebuddy/analysis/blockpool_hybrid_attention.md`：vLLM 单 BlockPool 服务多 Attention Group 的深度解剖文档（v0.25.1），覆盖分组全生命周期、物理 Tensor 布局、前缀缓存隔离等
- `.codebuddy/analysis/encoder_integration.md`：vLLM Encoder 集成机制深度解剖（v0.25.1），覆盖多模态 Encoder、Encoder Cache、Encoder 调度、Cross-Attention KV Cache、Encoder-Decoder 模型等
- `.codebuddy/analysis/scheduler_deep_dive.md`：vLLM V1 Scheduler 调度流程深度解剖（v0.25.1），覆盖 schedule() 全流程、两阶段调度、KV Cache 分配、Encoder 调度、抢占机制、投机解码集成等，含完整时序图
- `.codebuddy/analysis/model_runner_v1_v2.md`：vLLM Model Runner V1 vs V2 深度对比解剖（v0.25.1），覆盖 Persistent Batch 解耦、StagedWriteTensor、GPU-Native 输入准备、Triton 采样器、CUDA Graph 管理、V1/V2 完整时序图、性能对比
- `.codebuddy/analysis/vllm_async_scheduler.md`：vLLM 异步调度器深度解剖（v0.25.1），覆盖 placeholder 两种形态、prev_sampled_token_ids GPU 内接力、batch queue overlap、EOS 兜底等
- `.codebuddy/analysis/persistent_batch.md`：vLLM Persistent Batch（持久批）深度解剖（v0.25.1），覆盖 input_batch 单例证据（仅 __init__ 创建、7050 条件重建例外）、req_id_to_index 间接层、add/remove/condense/swap 座位管理、与异步调度 prev_sampled_token_ids 的连接、V1/V2 对比、设计假设与退化场景
- `.codebuddy/analysis/kv_cache_offloading.md`：vLLM KV Cache Offloading 深度解剖（v0.25.1），覆盖配置桥接、OffloadingConnector 双角色分派、Scheduler 侧 Manager/lookup、Worker 侧 DMA、Tiering 多层级、与 prefix cache/异步调度关系
- `.codebuddy/analysis/kv_connector.md`：vLLM KV Connector 解耦与集成机制深度解剖（v0.25.1，理解 offloading 的前置基础），覆盖 KVConnectorBase_V1 双角色钩子划分、KVTransferConfig、注册表+工厂可插拔、Scheduler 六回调点、Worker 全局单例 _KV_CONNECTOR_AGENT、KVConnectorModelRunnerMixin 生命周期、OffloadingConnector 复用框架、V0 vs V1 对比
- `.codebuddy/analysis/model_states.md`：vLLM ModelState 适配层深度解剖（v0.25.1），覆盖 interface/default/encoder_decoder/mamba_hybrid/mm_pruning 四类职责、工厂选择顺序、model_runner 调用时序、扩展点
- `.codebuddy/analysis/attn_metadata.md`：vLLM Attention Metadata 深度解剖（v0.25.1），覆盖 CommonAttentionMetadata/ModelSpecificAttnMetadata/AttentionMetadataBuilder/AttentionGroup 四层抽象、调度三维度、创建与使用流程、字段表与时序图
- `.codebuddy/analysis/attn_group.md`：vLLM AttentionGroup 设计深度解剖（v0.25.1），覆盖与 KV Cache Group 的本质区别、分组 key(backend+spec+num_heads_q)、init_attn_backend 三阶段、kernel block size 虚拟分块、AttentionCGSupport 取最弱、workspace 共享、端到端时序图。（注：曾误在源码 attn_utils.py/utils.py 加过中文注释，已要求改为文档内嵌）
- `.codebuddy/analysis/decode_context_parallel.md`：vLLM DCP（Decode Context Parallel）深度解剖（v0.25.1），覆盖 DCP 通信域(TP 组内子组)、KV 按 token 位置切分(round-robin/interleave)、prefix caching 集成(block_size*=dcp + hash 粒度放大保跨 rank 一致)、_forward_with_dcp(AllGather Q → 本地KV段注意力causal=False → dcp_combine LSE加权 → 新KV注意力causal=True → merge_attn_states)、ag_rs/a2a 两种通信策略、MLA 支持、混合注意力现状(HybridKVCacheCoordinator 仍 assert dcp==1)
- `.codebuddy/analysis/deepseek_v4_kv_cache_layout.md`：DeepSeek-V4 KV Cache 布局与全局管理深度解剖（v0.25.1），覆盖 AttentionLayerBase 约定属性、static_forward_context 双读取(收集spec+注入)、DSv4 **五种 cache**(主压缩 KV / SWA KV / 主 Compressor State / Indexer KV(k_cache) / **Indexer 内 Compressor State**)、packed 布局+共享 block table、并以 **ModelRunnerV2 为主线**重写分配/注入/执行期逻辑（init_kv_cache @ gpu/attn_utils.py:530、initialize_kv_cache @ gpu/model_runner.py:407、bind_kv_cache @ worker/utils.py:462、execute_model 两阶段、prepare_attn→ModelState.prepare_attn→attn_metadata）。V1 仅作对照，不展开。
  - **⚠️ 重要修正(2026-08-06)**：之前漏了 indexer 内部的 compressor。事实：`DeepseekV4Indexer`(`attention.py:855`)持有自己的 `DeepseekCompressor`(key=`self_attn.indexer.compressor`)，其内又有 `CompressorStateCache`(key=`self_attn.indexer.compressor.state_cache`)。所以 C4A 层有**两份** CompressorStateCache：主(head_dim=512 路径, state_dim=2×2×512=2048, page=32768) + indexer 内(head_dim=128 路径, state_dim=2×2×128=512, page=8192)；两者对 C4 的 `(block_size=4, sliding_window=8)` 相同，被 `group_and_unify_kv_cache_specs` 分到同一**容器组③**，但到 packed 的 **page_size 分桶**层因 32768≠8192 拆成**两个独立 slot**。C4A=5 key，C128A=3 key(无 indexer)，SWA-only=1 key(主 KV 返回 None)。文档与 kv_cache_utils.py docstring 均已同步修正。
- `.codebuddy/analysis/eplb_ep_dispatch_combine.md`：vLLM EPLB + MoE 专家并行(EP) Dispatch/Combine 全流程深度解剖（v0.25.1）。**关键结论：vLLM 已实现并接入 DeepSeek EPLB**（不是那篇微信文章讲的，文章只解析官方仓库）。覆盖：EPLB 算法(rebalance_experts_hierarchical 分层三步法 @ policy/default.py)、运行时状态触发重排(EplbState/EplbLayerState @ eplb_state.py)、**副本选择核心**(base_router.py:18 的 Triton kernel 用 token_idx 的 Knuth 哈希 % replica_count 选副本，逻辑→物理映射)、dispatch(all-to-all，目标rank=物理id//n_local @ all2all.py/naive_dp_ep.py)、Grouped GEMM、combine(加权求和+反向 all-to-all+late all-reduce)、EP vs TP 区别、12 条 FAQ(含多副本选路/确定性/异步重排一致性等场景)。
