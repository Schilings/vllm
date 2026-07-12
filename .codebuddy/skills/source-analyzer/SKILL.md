---
name: source-analyzer
description: 深度阅读源码，生成图文并茂的 Markdown 分析报告。结合代码探索、调用链追踪、Mermaid 架构图/时序图/流程图/对比表，按"全景架构→逐层解剖→关键数据结构→完整时序"结构输出。当用户要求"分析XX原理/源码"、"解剖XX机制"、"写XX分析报告"、"XX是怎么实现的"时使用。
---

# Source Analyzer（源码原理分析报告生成器）

## Overview

深度分析源码模块，生成图文并茂的中文 Markdown 分析报告。核心特点：**代码探索 → Mermaid 架构图 → 逐层深度解剖 → 完整调用链时序图**。报告要能让"小白"看懂原理，让"老手"快速定位关键路径。

## When To Use

- 用户要求"分析/解剖/理解 XX 模块的实现原理"
- 用户想要了解某个子系统的完整调用链
- 用户需要一份可视化的架构分析文档

## 与 code-commenter 的分工

| | source-analyzer | code-commenter |
|---|---|---|
| 产物 | 独立 `.md` 分析报告 | 源码内注释 |
| 配图 | **Mermaid 图**（架构图/时序图/流程图） | ASCII box-drawing 图 |
| 存放 | `.codebuddy/analysis/` | 原源码文件中 |
| 触发 | "分析XX原理" "解剖XX机制" | "加注释" "注释这个类" |

## Workflow

按顺序执行 6 个阶段，**不要跳过探索**。

### Phase 1: 全景探索

用 `code-explorer` subagent 全面搜索相关代码：

1. 找到所有相关文件（目录结构、关键类、接口）
2. 追踪调用链：谁创建对象、谁调用核心方法、数据如何流转
3. 记录关键数据结构及其字段含义
4. 梳理模块间依赖关系（scheduler / cache / attention / worker / connector 等）

**输出**: 模块全景图 + 核心文件清单

### Phase 2: 架构总览

输出报告的 **"全景架构概览"** 章节，包含：

1. **Mermaid 架构图**（用 `graph TD` 或 `flowchart` 画模块分层和交互关系）
2. 一句话概括每个模块的职责
3. 关键数据流的高层描述

```mermaid
graph TD
    subgraph "Layer 1: 调度层"
        S[Scheduler] --> K[KVCacheManager]
        K --> C[HybridKVCacheCoordinator]
    end
    subgraph "Layer 2: 执行层"
        C --> R[GPUModelRunner]
        R --> B[BlockTable / slot_mapping]
    end
    subgraph "Layer 3: 计算层"
        B --> A[Attention Backend]
    end
```

### Phase 3: 逐层深度解剖

对每一层，输出：

1. **Mermaid 流程图/时序图** 展示该层内部逻辑
2. **关键代码段引用**（文件 + 行号，用 ` ```py:line:file ` 格式）
3. **设计决策说明**（为什么这样设计、有什么取舍）
4. **边界/陷阱标注**（?? 容易出错的地方）

示例时序图（Mermaid `sequenceDiagram`）:

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant K as KVCacheManager
    participant C as HybridKVCacheCoordinator
    participant F as FullAttentionManager
    participant W as SlidingWindowManager

    S->>K: ① get_computed_blocks(request)
    K->>C: find_longest_cache_hit()
    C->>F: 左扫前缀
    C->>W: 右扫连续窗口
    C-->>K: 交集 hit_length
    K-->>S: (KVCacheBlocks, num_hit)

    S->>K: ② allocate_slots(request, ...)
    K->>C: remove_skipped_blocks() → 窗口外淘汰
    K->>C: get_num_blocks_to_allocate() → 容量检查
    K->>C: allocate_new_computed_blocks() → 前缀块追加
    K->>C: allocate_new_blocks() → 新 block 分配
    K->>C: cache_blocks() → 写入前缀哈希
    K-->>S: new_blocks
```

### Phase 4: 关键数据结构速查表

用 Markdown 表格总结核心数据结构：

| 数据结构 | 关键字段 | 作用 |
|---------|---------|------|
| KVCacheBlocks | blocks: tuple[Sequence[KVCacheBlock], ...] | 跨 group 的 block 分配结果 |
| CommonAttentionMetadata | block_table_tensor, slot_mapping | 传给后端的 KV cache 寻址信息 |
| ... | ... | ... |

### Phase 5: 完整调用链时序图

输出报告的核心卖点——**一个覆盖全流程的 Mermaid 时序图**：

- 从 Scheduler 开始到 Attention 计算结束
- 标注每个步骤的编号 ①~?
- 标注每个步骤对应的文件和函数

### Phase 6: 对比总结（按需）

如果被分析的系统有多个可比的实现/分支：

1. **对比表格**（Mermaid 或 Markdown 表格）
2. 各自的适用场景和性能特征
3. 如何选择

如果没有可比实现，跳过此节。

---

## 报告结构模板

按以下固定结构输出到 `.codebuddy/analysis/<topic>.md`:

```
# <主题>深度解剖

## 目录（自动生成锚点链接）

## 1. 全景架构概览
   - Mermaid 架构图
   - 一句话说明每层职责

## 2. Layer 1: <名称> 深度解剖
   - Mermaid 流程图/时序图
   - 关键代码引用
   - 设计决策

## 3. Layer 2: <名称> 深度解剖
   ...

## N. 完整调用链时序图
   - 端到端 Mermaid sequenceDiagram

## N+1. 关键数据结构速查表

## N+2. 快速问题解答（FAQ）
```

---

## Mermaid 画图规范

参考 `references/mermaid-guide.md` 获取完整语法参考。要点：

| 图类型 | 用途 | 关键语法 |
|-------|------|---------|
| `graph TD` / `flowchart` | 模块架构、分层关系 | `subgraph`, `-->` |
| `sequenceDiagram` | 调用链时序 | `participant`, `->>`, `-->>` |
| `stateDiagram-v2` | 状态流转 | `[*]`, `-->` |
| `classDiagram` | 类关系 | `class`, `<|--` |

要求:
- 每个图有标题和编号
- 节点用中文标注
- 关键数据流标注方向（`→` `↓` `↑`）
- 不要过度复杂，一个图说清一件事

---

## 铁律（Hard Rules）

1. **调用链必须基于真实代码探索**，调用 `code-explorer` subagent 查清对端文件 + 函数 + 行号
2. **Mermaid 语法必须正确**，必要时对照 `references/mermaid-guide.md` 校验
3. **图不要过度复杂**——一个图说清一件事，复杂流程拆成多个子图
4. **语言用简体中文**
5. **输出位置**: `.codebuddy/analysis/<topic>.md`
