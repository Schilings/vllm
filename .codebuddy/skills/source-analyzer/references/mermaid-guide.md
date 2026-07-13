# Mermaid 语法快速参考

用于源码分析报告的各种 Mermaid 图表语法。

## 1. 架构图 / 流程图 (`flowchart` / `graph`)

最常用的图，用于展示模块分层和交互关系。

```mermaid
graph TD
    subgraph "Layer Name"
        A[Module A] --> B[Module B]
        B --> C[Module C]
    end

    A -->|data flow| D[External Module]

    style A fill:#e1f5fe
    style D fill:#fff3e0
```

关键语法：
- `graph TD` = 从上到下，`graph LR` = 从左到右
- `subgraph "Title" ... end` = 分组框
- `A -->|label| B` = 带标签的箭头
- `A --> B` = 实线箭头，`A -.-> B` = 虚线箭头

## 2. 时序图 (`sequenceDiagram`)

展示调用链的时间顺序。

```mermaid
sequenceDiagram
    participant A as Module A
    participant B as Module B

    A->>B: ① sync call
    B-->>A: async response
    Note over A,B: This is a note
    activate B
    B->>B: internal processing
    deactivate B
```

关键语法：
- `participant X as "Display Name"` = 定义参与者
- `->>` = 同步调用，`-->>` = 异步响应
- `activate`/`deactivate` = 激活/停用生命线
- `Note over A,B: text` = 跨参与者注释

## 3. 状态流转图 (`stateDiagram-v2`)

展示对象的状态迁移。

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Running: start
    Running --> Finished: complete
    Running --> Preempted: OOM
    Preempted --> Running: resume
    Finished --> [*]
```

## 4. 对比表格

Mermaid 不支持原生表格，用 Markdown 表格替代：

```markdown
| | Implementation A | Implementation B |
|---|---|---|
| Feature 1 | ✅ | ❌ |
| Performance | Fast | Slow |
| Memory | High | Low |
```

## 5. 特殊小技巧

### 节点样式（颜色）
```mermaid
graph TD
    A[Critical Path]:::critical
    B[Optional]:::optional
    classDef critical fill:#ffcdd2,stroke:#b71c1c
    classDef optional fill:#e8eaf6,stroke:#1a237e
```

### 引用换行
```mermaid
graph TD
    A["Module A<br/>(file.py:42)"]
```

### 避免过度复杂
- 超过 10 个节点 → 考虑拆成多个子图
- 箭头太多 → 检查是否真的需要全部展示
- 一个时序图不要超过 8 个 participant
