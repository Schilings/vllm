# 项目记忆

## Git 分支策略（2026-07-12）
- `main` 分支：纯净跟踪上游 vllm-project/vllm，不加任何修改
- `annotated` 分支：从 main 分出，用于添加源码阅读注释
- 上游更新时：`main` pull upstream → `annotated` rebase main

## 远程仓库
- origin: https://github.com/Schilings/vllm.git (fork)
- upstream: https://github.com/vllm-project/vllm.git (官方)

## 源码分析文档
- `.codebuddy/analysis/blockpool_hybrid_attention.md`：vLLM 单 BlockPool 服务多 Attention Group 的深度解剖文档（v0.25.1），覆盖分组全生命周期、物理 Tensor 布局、前缀缓存隔离等
- `.codebuddy/analysis/encoder_integration.md`：vLLM Encoder 集成机制深度解剖（v0.25.1），覆盖多模态 Encoder、Encoder Cache、Encoder 调度、Cross-Attention KV Cache、Encoder-Decoder 模型等
- `.codebuddy/analysis/scheduler_deep_dive.md`：vLLM V1 Scheduler 调度流程深度解剖（v0.25.1），覆盖 schedule() 全流程、两阶段调度、KV Cache 分配、Encoder 调度、抢占机制、投机解码集成等，含完整时序图
- `.codebuddy/analysis/model_runner_v1_v2.md`：vLLM Model Runner V1 vs V2 深度对比解剖（v0.25.1），覆盖 Persistent Batch 解耦、StagedWriteTensor、GPU-Native 输入准备、Triton 采样器、CUDA Graph 管理、V1/V2 完整时序图、性能对比
- `.codebuddy/analysis/vllm_async_scheduler.md`：vLLM 异步调度器深度解剖（v0.25.1），覆盖 placeholder 两种形态、prev_sampled_token_ids GPU 内接力、batch queue overlap、EOS 兜底等
- `.codebuddy/analysis/persistent_batch.md`：vLLM Persistent Batch（持久批）深度解剖（v0.25.1），覆盖 input_batch 单例证据（仅 __init__ 创建、7050 条件重建例外）、req_id_to_index 间接层、add/remove/condense/swap 座位管理、与异步调度 prev_sampled_token_ids 的连接、V1/V2 对比、设计假设与退化场景
- `.codebuddy/analysis/kv_cache_offloading.md`：vLLM KV Cache Offloading 深度解剖（v0.25.1），覆盖配置桥接、OffloadingConnector 双角色分派、Scheduler 侧 Manager/lookup、Worker 侧 DMA、Tiering 多层级、与 prefix cache/异步调度关系
- `.codebuddy/analysis/kv_connector.md`：vLLM KV Connector 解耦与集成机制深度解剖（v0.25.1，理解 offloading 的前置基础），覆盖 KVConnectorBase_V1 双角色钩子划分、KVTransferConfig、注册表+工厂可插拔、Scheduler 六回调点、Worker 全局单例 _KV_CONNECTOR_AGENT、KVConnectorModelRunnerMixin 生命周期、OffloadingConnector 复用框架、V0 vs V1 对比
