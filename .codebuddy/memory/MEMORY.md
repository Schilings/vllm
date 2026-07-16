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
