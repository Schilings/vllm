# DP Prefill-Balancing 深度剖析：vLLM 为什么要有这个操作

> 调研对象：vLLM v1 调度器里的 **DP prefill balancing**（数据并行下的 prefill 负载均衡）。
> 涉及文件：
> - `vllm/v1/core/sched/scheduler.py`（调度主逻辑、throttle 判定、安全阀）
> - `vllm/v1/core/sched/interface.py`（SchedulerInterface 上 `throttle_prefills` 形参）
> - `vllm/v1/engine/core.py`（`DPEngineCoreProc._should_throttle_prefills`、cadence 判定、wave 机制）
> - `vllm/config/scheduler.py:153`（`prefill_schedule_interval` 配置项）
> - `vllm/engine/arg_utils.py:611/1458/2167`（命令行暴露）
>
> 所有行号基于分支 `comments-on-v0.25.1` 实际源码核对。
> 阅读约定：源码块用 ```` ```python ```` 标注，块内第一行 `# 行号:行号:文件路径` 标出来源，方便回源码定位。

---

## 0. 一句话答案

> **DP prefill balancing 解决的问题是：在多卡数据并行（DP）部署下，各 DP rank 各自独立决定"何时做 prefill"，会因为请求到达的随机性导致"有的 rank 在猛算 prefill（重）、有的 rank 在轻量 decode（轻）"——两者必须**同步**才能一起进下一步（DP 要求各 rank 步调一致），于是快的 rank 干等慢的 rank，GPU 利用率暴跌。**
>
> vLLM 的解法：**让所有 DP rank 约定一个"步距"（cadence），只有 cadence 对齐的步（step_counter % N == 0）才统一接纳新 prefill，中间步只做 decode。** 这样各 rank 的每步 forward 计算量都趋向"以 decode 为主 + 周期性同步灌入 prefill"，步与步之间时间差被抹平，DP 的同步等待被降到最低。

---

## 1. 为什么 DP 必须步调一致（问题根源）

### 1.1 DP 的本质：同模型、多副本、各吃不同请求

数据并行（Data Parallel）下，`dp_size` 张卡各跑一份**完全相同权重**的模型副本，每张卡服务**不同**的请求子集。它们**不通信激活值**（不像 TP/PP），只在某些全局同步点（如 EP 的 all-reduce、或 wave 结束的 all-reduce）对齐。

### 1.2 prefill 与 decode 的计算量天差地别

| 阶段 | 每步 token 数 | 计算特征 | 耗时 |
| --- | --- | --- | --- |
| **prefill** | 整段 prompt（可能数千 token） | 大矩阵乘、compute-bound | **重**（毫秒~百毫秒） |
| **decode** | 1 token（或 spec 的 K+1） | memory-bandwidth-bound | **轻**（亚毫秒） |

关键事实：**prefill 一步的耗时可能是 decode 一步的几十到上百倍。**

### 1.3 独立调度的灾难：木桶效应

假设 `dp_size=2`，rank0 此刻在 decode（轻），rank1 此刻在 prefill（重）。DP 的全局同步点（如 wave 结束的 `_has_global_unfinished_reqs` all-reduce，`core.py:1991`）要求两者都到齐。结果：

```
rank0: decode ── 完成 ──► 等 ► 等 ► 等 ► 等 ►（空转，GPU 闲置）
rank1: prefill ──────────────────────── 完成 ►
                                          ↑ rank0 干等 rank1
```

**rank0 的 GPU 在 rank1 算 prefill 期间完全空转**——这就是"好端端的有了 DP prefill balancing"要消除的浪费。更糟的是：如果 rank0 在不断接收新请求、rank1 的请求都长，两者的"重步"永远错开，DP 整体吞吐被最慢 rank 绑架（木桶效应）。

---

## 2. 核心机制：cadence 对齐 + throttle

### 2.1 配置入口

```python
# vllm/config/scheduler.py:153
prefill_schedule_interval: int = Field(default=1, ge=1)
# 文档: "For data-parallel deployments, only admit new prefill requests
#        once every N engine steps, aligned across DP ranks, to better balance
#        per-step forward-pass times."
```

`interval=1`（默认）即**不开启**均衡（每步都接纳 prefill，原 vLLM 行为）。`interval>1` 才启用。

### 2.2 谁决定"这一步该不该压抑 prefill"—— cadence 判定

```python
# vllm/v1/engine/core.py:1916-1923   （仅 DPEngineCoreProc 重写，普通 EngineCore 永远返回 False）
def _should_throttle_prefills(self) -> bool:
    # Throttle new prefills to cadence-aligned steps for DP balancing.
    # step_counter is identical across DP ranks. On a fresh wave the
    # counter is 0, so prefills are admitted immediately after idle.
    return (
        self.prefill_schedule_interval > 1
        and self.step_counter % self.prefill_schedule_interval != 0
    )
```

**关键点**：`step_counter` 是各 DP rank 的**逻辑步计数器**，在每个 wave 开始时归零（`core.py:1981` `self.step_counter = 0`），且各 rank 推进节奏一致（都是每个 forward 步 +1，见 `:1987`）。所以 `step_counter % interval == 0` 在**所有 rank 上同时为真/假**——这就是"cadence 对齐"的数学保证：无需通信，各 rank 用同一个公式算出"现在是不是 prefill 窗口"。

> ⚠️ **为什么不用通信来对齐，而用计数器？** 因为 `step_counter` 本身就是在 wave 边界通过 all-reduce 同步的（`_has_global_unfinished_reqs` 每 32 步做一次 `sync_dp_state` all-reduce，`core.py:1988-1991`），且每个 rank 每步 +1，所以"步号"天然是跨 rank 一致的全局逻辑时钟。用计数器判断比每步 all-reduce 一个 bool 便宜得多。

### 2.3 调度器如何消费这个信号—— `defer_prefills`

`EngineCore.step` 把 `_should_throttle_prefills()` 的结果传给 `scheduler.schedule(throttle_prefills=...)`（`core.py:490` / `:547`）。

```python
# vllm/v1/core/sched/scheduler.py:444-461
# --- DP prefill balancing ---
# DP (Data Parallel) 多卡推理时，prefill 计算量远大于 decode。若各 rank 各自
# 决定何时做 prefill，快慢不均导致部分 rank 空闲等待。解决方案：所有 rank 只在
# "cadence-aligned step" 同时做 prefill (step_counter % interval == 0)，
# 中间步只做轻量 decode，保证 load balance。
#
# defer_prefills 由 3 个条件控制：
#   ① throttle_prefills: DP engine core 传入 (non-cadence-aligned step 为 True)
#   ② not self.prefill_capacity_bound: 安全阀
#   ③ any(not r.is_prefill_chunk for r in self.running): running 中至少有一个纯 decode 请求
defer_prefills = (
    throttle_prefills and not self.prefill_capacity_bound
) and any(not r.is_prefill_chunk for r in self.running)
```

**三个条件缺一不可**：

| 条件 | 含义 | 为什么需要 |
| --- | --- | --- |
| `throttle_prefills` | 这是 cadence 非对齐步（由 `interval` 决定） | 对齐步（%N==0）**放行** prefill，错开步**压抑** |
| `not prefill_capacity_bound` | 安全阀（见 §3） | 若上一轮 cadence 步都没把 waiting 清空，说明积压严重，必须破例放行，否则 TTFT 雪崩 |
| `any(not r.is_prefill_chunk ...)` | running 里至少有一个纯 decode 请求 | 否则压抑 prefill 会导致"本步啥活都没有"→ 空转（比不均衡更糟） |

### 2.4 压抑的两个作用点

**作用点 A：推迟 running 队列里正在 chunked-prefill 的请求**

```python
# vllm/v1/core/sched/scheduler.py:493-498
# -- 跳过条件 3: DP prefill 均衡 —— 非 cadence 对齐步延迟 prefill --
if defer_prefills and request.is_prefill_chunk:
    # DP prefill balancing: defer this in-progress prefill chunk to a
    # cadence-aligned step; decodes still run to fill this step.
    req_index += 1
    continue
```

正在 chunked-prefill 的请求本步**直接跳过**，但它**仍留在 running 队列**（没被抢占、没被移除），只是本步不分配新 token。等下一个 cadence 对齐步再来算。

**作用点 B：推迟 waiting 队列里新请求的接纳**

```python
# vllm/v1/core/sched/scheduler.py:848-851
elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
    # DP prefill balancing: defer this step's local prefill
    # compute to a cadence-aligned step.
    break
```

注意 `break` 而非 `continue`：一旦遇到第一个该被推迟的 prefill 新请求，就**停止整个 waiting 遍历**——因为 waiting 队列按某种顺序排，前面都是 decode/已计算请求，遇到首个长 prefill 就截断，本步不再接纳任何新 prefill。

### 2.5 一个完整 cadence 周期长这样

设 `interval=4`：

```
step:        0      1      2      3      4      5      6      7
cadence?:    ✅对齐  ❌      ❌      ❌     ✅对齐  ❌      ❌      ❌
prefill:     接纳    压抑    压抑    压抑    接纳    压抑    压抑    压抑
decode:      照常    照常    照常    照常    照常    照常    照常    照常
```

- **对齐步（0,4,...）**：所有 rank 同时灌入一批新 prefill，各 rank 这步都"偏重"，但因为都重，时间接近，DP 同步点等待小。
- **错开步（1,2,3,...）**：所有 rank 都只做 decode（轻），步与步时间都短且接近，DP 同步点等待小。

**结果**：每个 rank 的步时间分布趋同，木桶效应被抹平。

---

## 3. 安全阀：`prefill_capacity_bound`（防止雪崩）

### 3.1 为什么需要安全阀

设想：cadence 对齐步接纳了一批 prefill，但 waiting 队列**太长，一个对齐步塞不下**（token budget 上限 `max_num_batched_tokens`）。下一轮错开步本想压抑 prefill 让位 decode，但此时 waiting 里**全是积压的 prefill 请求**——如果真压抑了，这些请求要再等 `interval-1` 步才能被接纳，**TTFT（首 token 延迟）雪崩**。

### 3.2 实现：记录 cadence 步是否清空 waiting

```python
# vllm/v1/core/sched/scheduler.py:287-293
# DP prefill balancing: Flag to track whether the last cadence-aligned
# prefill batch fully drained the waiting queue. Prefill throttling
# is disabled in this case.
#
# 安全阀: 上一轮 cadence 步若未能清空 waiting (capacity_bound=True),
# 则后续 non-cadence 步也放行 prefill，防止积压导致 TTFT 雪崩。
self.prefill_capacity_bound = False
```

```python
# vllm/v1/core/sched/scheduler.py:1076-1082
# DP prefill balancing: on a step that admitted prefills (release),
# record whether it was capacity-bound.
#
# cadence-aligned step 结束时若 waiting 仍未清空 → capacity_bound=True,
# 下轮 non-cadence 步也放行 prefill (安全阀)
if not defer_prefills:
    self.prefill_capacity_bound = bool(self.waiting)
```

逻辑闭环：
- **cadence 对齐步**（`defer_prefills=False`）：本步接纳了 prefill。步末检查 `waiting` 是否还有剩。
  - 空了 → `capacity_bound=False` → 下轮错开步正常压抑（健康状态）。
  - 没空 → `capacity_bound=True` → **下轮错开步也放行 prefill**（破例，优先清积压）。
- 一旦某次 cadence 步把 `waiting` 清空了，`capacity_bound` 回到 `False`，安全阀关闭，恢复周期性压抑。

> ⭐ **这是个典型的"弹性限流"设计**：默认严格 cadence 节律保均衡；一旦检测到系统跟不上（waiting 积压），立刻切换到"能接多少接多少"的尽力模式，等 backlog 消化完再回到节律。避免了"为了均衡而把请求饿死"的副作用。

---

## 4. 它跟哪些 vLLM 机制是正交/耦合的

### 4.1 与 async scheduling 正交

`throttle_prefills` 是 `schedule()` 的独立形参（`interface.py:72`），async scheduling 用它自己的 `num_output_placeholders` 机制（见 `vllm_async_scheduler.md`）。两者可同时开：async 管"step N 结果未回就调度 N+1"，DP prefill balancing 管"本步要不要接纳新 prefill"。

### 4.2 与 chunked prefill 强耦合

压抑 prefill 时，正在 chunk 的 prefill 请求**留在 running 队列不抢占**（§2.4 作用点 A 是 `continue` 不是 preemption）。这依赖 chunked prefill 本来就能把长 prefill 切成多步，压抑只是"延后下一刀"，不会破坏正确性。

### 4.3 与 MoE / EP 的关系

`DPEngineCoreProc` 明确要求 `is_moe`（`core.py:1759` `assert ... is_moe`）。为什么？因为 **DP prefill balancing 的真正价值在 MoE 模型上最显著**：

- MoE 的 decode 也是 compute-bound（每个 token 要过所有 expert 的 router + 部分 expert），prefill/decode 差距虽在，但更关键的是 **EP（Expert Parallel）下的 all-to-all 通信要求各 rank 的 token 数尽量对齐**——否则有的 rank 发 100 个 token 的 all-to-all、有的发 10 个，快 rank 等慢 rank。
- 周期性、跨 rank 对齐地灌入 prefill，让各 rank 的"本步总 token 数"波动被平滑，EP all-to-all 更易饱和、等待更少。

> 注意：源码 `assert is_moe` 是**当前实现约束**，不代表该机制理论上只能用于 MoE；它是为 MoE 的 EP 同步痛点量身打磨的。

### 4.4 与 wave 机制的关系

`step_counter` 在每个 wave 开始归零（`core.py:1981`）。这意味着 **每个新 wave（一轮"所有请求都跑完/暂停"的边界）的第一个 step 必然是 cadence 对齐步**（`0 % N == 0`），于是新 wave 启动后**立即接纳 prefill**——避免"刚启动却因计数器非零而压抑首波请求"的尴尬。源码注释也明说：`core.py:1919 "On a fresh wave the counter is 0, so prefills are admitted immediately after idle."`

---

## 5. ⭐ 为什么"好端端的"会有这个操作——设计哲学总结

vLLM 默认（单卡 / TP / PP，无 DP）**根本不需要** prefill balancing——每个调度步各 rank 自己算自己的，没有跨 rank 同步点强制步调一致，不均衡只是"单步快慢不同"，不影响整体吞吐。

**DP 改变了游戏规则**：DP rank 之间虽然不通信激活值，但**存在全局同步点**（wave 结束的 all-reduce、EP 的 all-to-all）。一旦有同步点，"最慢 rank 决定整体速度"。而 prefill 是天然的不均衡源（请求长度方差极大）。

所以 DP prefill balancing 是一个**专为"多副本 + 同步点"场景补的均衡层**：

1. **用计数器模拟跨 rank 协商**：`step_counter % interval` 无需通信就让所有 rank 在同一时刻打开/关闭 prefill 窗口。
2. **把"重活"打包、把"轻活"铺平**：重 prefill 集中在 cadence 步，轻 decode 铺在中间步，各 rank 步时间分布趋同。
3. **带安全阀的弹性节律**：积压时自动破例放行，防止均衡反噬 TTFT。
4. **与既有机制正交复用**：chunked prefill 提供"可中断的 prefill"语义，wave 提供计数器归零锚点，EP 提供同步痛点。

> **一句话**：它不是"多余的设计"，而是 **DP 部署下"同步点 + prefill 长尾"这对矛盾的必然解**。没有它，DP 的加速比会严重偏离线性（被最慢 rank 的 prefill 拖死）。

---

## 6. 速查表

| 概念 / 字段 / 函数 | 位置 | 作用 |
| --- | --- | --- |
| `prefill_schedule_interval` | `config/scheduler.py:153` | 配置项：每 N 步接纳一次 prefill（N=1 关闭） |
| 命令行暴露 | `engine/arg_utils.py:611/1458/2167` | `--prefill-schedule-interval` |
| `_should_throttle_prefills` (基类) | `engine/core.py:474` | 普通 EngineCore：永远返回 False（不启用） |
| `_should_throttle_prefills` (DP) | `engine/core.py:1916` | DP 版：`interval>1 and step_counter%interval!=0` |
| `step_counter` 归零 | `engine/core.py:1981` | 每 wave 开始重置，保证新 wave 首步即 cadence 对齐 |
| `step_counter` 推进 | `engine/core.py:1987` | 每步 +1（每 32 步才真正 all-reduce 同步一次） |
| `defer_prefills` | `core/sched/scheduler.py:459` | 三条件 AND：`throttle and not capacity_bound and 有decode请求` |
| 作用点 A（running chunk） | `core/sched/scheduler.py:494` | `is_prefill_chunk` 请求本步跳过（留 running，不抢占） |
| 作用点 B（waiting 新接纳） | `core/sched/scheduler.py:848` | 遇到首个待压抑 prefill 即 `break` 停止遍历 |
| `prefill_capacity_bound` 定义 | `core/sched/scheduler.py:293` | 安全阀标志 |
| `prefill_capacity_bound` 更新 | `core/sched/scheduler.py:1082` | cadence 步末 `= bool(waiting)` |
| `DPEngineCoreProc` 约束 | `engine/core.py:1759` | `assert is_moe`（当前实现要求 MoE 模型） |
| `schedule(throttle_prefills=)` | `core/sched/interface.py:72` | 接口形参文档说明 DP prefill balancing 语义 |
| 测试 | `tests/v1/core/test_scheduler.py:378` / `:383` | `test_throttle_defers_*` 验证压抑行为 |

---

## 7. 快速问答

**Q1：关掉 `prefill_schedule_interval`（保持 1）会怎样？**
A：退化为 vLLM 默认行为——每个 DP rank 各自独立、每步都接纳 prefill。在 DP 部署下，rank 间 prefill 步错开，同步点等待变多，GPU 利用率下降。单卡/TP/PP 部署不受影响（本来就没同步点痛点）。

**Q2：为什么用 `step_counter % interval` 而不是通信协商？**
A：`step_counter` 本身由 wave 边界的 all-reduce 间接同步，且每 rank 每步 +1，是跨 rank 一致的"逻辑时钟"。用它判断窗口无需每步额外通信，几乎零成本。详见 §2.2。

**Q3：`defer_prefills` 第三个条件"running 里至少有一个纯 decode"为什么必要？**
A：若 running 全是 prefill chunk，压抑它们后本步**没有任何请求可算**→ 空转（dummy batch 或干等），比不均衡更浪费。有 decode 请求在，压抑 prefill 才有意义（用 decode 填满本步）。见 `scheduler.py:454-461` 注释。

**Q4：安全阀会不会让均衡"失效"？**
A：不会长期失效。安全阀只在 `waiting` 积压时临时开启，一旦某 cadence 步清空 `waiting`，`capacity_bound` 立刻回 False，恢复节律。它是"积压时优先清队列"的弹性策略，详见 §3.2。

**Q5：它和 P/D 分离（disaggregated prefill）是一回事吗？**
A：不是。P/D 分离是把 prefill 和 decode 放到**不同实例/不同硬件**上，彻底解耦两者资源；DP prefill balancing 是在**同一个 DP 组内**让各 rank **步调对齐**地做 prefill。两者可叠加：P/D 分离解决"prefill 占 decode 资源"，DP balancing 解决"DP rank 间 prefill 不同步"。
