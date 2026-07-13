# 项目记忆

## Git 分支策略（2026-07-12）
- `main` 分支：纯净跟踪上游 vllm-project/vllm，不加任何修改
- `annotated` 分支：从 main 分出，用于添加源码阅读注释
- 上游更新时：`main` pull upstream → `annotated` rebase main

## 远程仓库
- origin: https://github.com/Schilings/vllm.git (fork)
- upstream: https://github.com/vllm-project/vllm.git (官方)
