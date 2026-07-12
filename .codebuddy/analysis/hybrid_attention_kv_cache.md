# vLLM 混合注意力（Full + Sliding Window）KV 缓存机制深度解剖

## 目录
1. [全景架构概览](#1-全景架构概览)
2. [KV Cache 分组：混合模型的内存布局](#2-kv-cache-分组混合模型的内存布局)
3. [调度层：从 Schedule 到 Block Allocation 的完整调用链](#3-调度层从-schedule-到-block-allocation-的完整调用链)
4. [前缀缓存：Full 左扫 + SWA 右扫的交集算法](#4-前缀缓存full-左扫--swa-右扫的交集算法)
5. [KV 分配：Block 的分配与 Sliding Window 的环形复用](#5-kv-分配block-的分配与-sliding-window-的环形复用)
6. [KV 回收：Block 的释放与 Slide-out 淘汰](#6-kv-回收block-的释放与-slide-out-淘汰)
7. [从 Block ID 到 Attention 计算：Block Table 的使用](#7-从-block-id-到-attention-计算block-table-的使用)
8. [混合注意力运行时：Per-Layer Mask 动态切换](#8-混合注意力运行时per-layer-mask-动态切换)
9. [完整调用链时序图](#9-完整调用链时序图)
10. [关键数据结构速查表](#10-关键数据结构速查表)

---

## 1. 全景架构概览

vLLM 的混合注意力 KV 缓存系统是三层架构：

```
┌──────────────────────────────────────────────────────────────────┐
│                        Layer 1: 调度层                           │
│  Scheduler ──→ KVCacheManager（分配/前缀/释放的统一门面）         │
│                ├── HybridKVCacheCoordinator（协调多种注意力类型） │
│                │   ├── FullAttentionManager（全注意力 block 管理）│
│                │   └── SlidingWindowManager（SWA block 管理）    │
│                └── BlockPool（物理 block 池 + 前缀缓存哈希表）    │
└──────────────────────────────┬───────────────────────────────────┘
                               │ SchedulerOutput (block_ids)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Layer 2: 模型执行层                           │
│  GPUModelRunner ──→ InputBatch（block_table + slot_mapping）     │
│                     CommonAttentionMetadata                      │
│                     AttentionMetadataBuilder.build()             │
│                     FlexAttentionMetadata（per-layer）            │
└──────────────────────────────┬───────────────────────────────────┘
                               │ block_table, slot_mapping,
                               │ sliding_window per layer
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Layer 3: Attention 计算层                      │
│  FlexAttentionImpl.forward()                                     │
│    ├── KV Cache Update: reshape_and_cache_flash()                │
│    │   ├── slot_mapping → GPU 物理地址 → 写入 K/V                │
│    ├── Per-Layer Switch: sliding_window 差异检测                 │
│    │   ├── sliding_window 变化 → 重建 mask_mod → 重建 block_mask│
│    ├── Mask 组合: get_mask_mod()                                 │
│    │   ├── base: causal_mask_mod (q_idx >= kv_idx)               │
│    │   ├── + sliding_window_mask_mod (|q - kv| < window)         │
│    │   ├── + prefix_lm_mask_mod (多模态 PrefixLM)                │
│    │   └── + rswa_mask_mod (prefix 全局 + 生成 token 窗口)       │
│    └── flex_attention_compiled(query, KV_cache, mask_mod, …)     │
│        └── BlockMask 裁剪无用 KV blocks → 高效 kernel            │
└──────────────────────────────────────────────────────────────────┘
```

**核心概念**：vLLM 将混合注意力模型（如 Gemma3：10 full + 52 SWA）的层按注意力类型分组，每组共享一个 KV Cache Group。Full attention 层分配覆盖全部 token 的 block，而 Sliding Window 层只分配最近的 `sliding_window_size` 个 token 的 block，通过环形缓冲区复用物理 block。

---

## 2. KV Cache 分组：混合模型的内存布局

### 2.1 触发条件

当模型包含不同类型的注意力层时（如 Full + SlidingWindow），`_get_kv_cache_groups_uniform_page_size()` 被调用。

**关键文件**：`vllm/v1/core/kv_cache_utils.py:1138`

### 2.2 分组算法（以 Gemma3-27b 为例：10 Full + 52 SWA）

```
输入: kv_cache_spec = {
    "full.0": FullAttentionSpec,  "full.1": FullAttentionSpec,  ...  (10 layers)
    "sw.0":  SlidingWindowSpec,   "sw.1":  SlidingWindowSpec,   ...  (52 layers)
}

Step 1: 按类型分组
    same_type_layers = {
        FullAttentionSpec: [full.0, full.1, ..., full.9],     (10 layers)
        SlidingWindowSpec:  [sw.0,  sw.1,  ..., sw.51]       (52 layers)
    }

Step 2: 计算 group_size
    min_num_layers = min(10, 52) = 10
    max_num_layers = max(10, 52) = 52
    52 < 10 * 1.5? NO → group_size = 10

Step 3: 交错分配（确保 Pipeline Parallelism 正确）
    num_groups = max(ceil(10/10), ceil(52/10)) = 6
    最终 6 个 KV Cache Groups:
    
    ┌──────────────────────────────────────────────┐
    │ Group 0: full.0 - full.9    (10 layers)      │  → FullAttentionSpec
    │ Group 1: sw.0 - sw.9        (10 layers)      │  → SlidingWindowSpec
    │ Group 2: sw.10 - sw.19      (10 layers)      │  → SlidingWindowSpec
    │ Group 3: sw.20 - sw.29      (10 layers)      │  → SlidingWindowSpec
    │ Group 4: sw.30 - sw.39      (10 layers)      │  → SlidingWindowSpec
    │ Group 5: sw.40 - sw.49      (10 layers)      │  → SlidingWindowSpec
    │ Group 6: sw.50 - sw.51 + 8 padding           │  → SlidingWindowSpec
    └──────────────────────────────────────────────┘
```

### 2.3 两种 Spec 的行为差异

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│       属性         │   FullAttentionSpec  │  SlidingWindowSpec   │
├────────────────────┼──────────────────────┼──────────────────────┤
│ block_size         │ 16                   │ 16                   │
│ num_kv_heads       │ 8                    │ 8                    │
│ sliding_window     │ None                 │ 4096 (例)            │
│ 分配 block 数      │ num_tokens / 16      │ sliding_window / 16  │
│ 需要覆盖的 token   │ 全部历史              │ 最近 sw_size 个      │
│ 前缀缓存扫描方向   │ 从左到右              │ 从右到左             │
│ 物理 block 复用    │ 不适用               │ 环形缓冲区            │
│ 淘汰策略           │ 前缀缓存 refcount     │ 窗口外自动淘汰       │
└────────────────────┴──────────────────────┴──────────────────────┘
```

### 2.4 可选降级：禁用 Hybrid Manager

如果启动 `--disable-hybrid-kv-cache-manager`，所有 `SlidingWindowSpec` 会被统一为 `FullAttentionSpec`：
- SWA 层也会分配覆盖全部 token 的 block（浪费内存）
- 但计算时仍使用 sliding window mask（不浪费计算）
- 适用于显存充足但不想要混合管理复杂度的场景

---

## 3. 调度层：从 Schedule 到 Block Allocation 的完整调用链

**关键文件**：
- `vllm/v1/core/sched/scheduler.py` - 调度器
- `vllm/v1/core/kv_cache_manager.py` - KV 缓存管理门面
- `vllm/v1/core/kv_cache_coordinator.py` - 混合协调器

### 3.1 每个调度步的完整流程

```
Scheduler.schedule()                          [scheduler.py]
│
├── 1. 对每个 Token (do..while 循环)
│   │
│   ├── 1a. KVCacheManager.get_computed_blocks(request)
│   │   │                                    [kv_cache_manager.py:207]
│   │   │
│   │   ├── HybridKVCacheCoordinator.find_longest_cache_hit()
│   │   │   │                                [kv_cache_coordinator.py:674]
│   │   │   │
│   │   │   ├── ★ 迭代收敛算法 ★
│   │   │   │   │
│   │   │   │   ├── Step 1: FullAttentionManager.find_longest_cache_hit()
│   │   │   │   │   └── 从左到右扫描，每次增长一个 block 检查前缀哈希表
│   │   │   │   │     返回: (hit_blocks, hit_length_full)
│   │   │   │   │
│   │   │   │   ├── Step 2: SlidingWindowManager.find_longest_cache_hit()
│   │   │   │   │   └── 从右到左扫描，需要连续 sliding_window_contiguous_blocks
│   │   │   │   │     返回: (hit_blocks, hit_length_sw) 
│   │   │   │   │     ※ sw 的 hit_length <= full 的 hit_length（可能更少）
│   │   │   │   │
│   │   │   │   └── Step 3: 如果 sw 缩短了 → 重新检查 full（trim）
│   │   │   │       最终: hit_length = min(hit_length_full, hit_length_sw)
│   │   │   │
│   │   │   └── 返回: (KVCacheBlocks, num_hit_tokens)
│   │   │
│   │   └── 返回: (computed_blocks, num_new_computed_tokens)
│   │
│   ├── 1b. [可选] KV Connector 外部缓存查询
│   │   └── connector.get_num_new_matched_tokens()
│   │       → num_external_computed_tokens（PD 场景）
│   │
│   ├── 1c. KVCacheManager.allocate_slots()  [kv_cache_manager.py:269]
│   │   │
│   │   ├── Stage 1: 释放不必要的 blocks
│   │   │   └── coordinator.remove_skipped_blocks()
│   │   │       └── SWA 层：释放滑动窗口外的 block
│   │   │
│   │   ├── Stage 2: 容量检查
│   │   │   └── coordinator.get_num_blocks_to_allocate()
│   │   │       ├── Full: (num_tokens / block_size) 向上取整
│   │   │       └── SWA: (min(num_tokens, sliding_window) / block_size)
│   │   │   如果 required_blocks > free_blocks → 返回 None（无法调度）
│   │   │
│   │   ├── Stage 3: 分配 prefix blocks
│   │   │   └── coordinator.allocate_new_computed_blocks()
│   │   │       └── 将前缀命中 blocks 追加到 request 的 block 列表
│   │   │
│   │   ├── Stage 4: 分配 new/lookahead blocks
│   │   │   └── coordinator.allocate_new_blocks()
│   │   │       ├── Full: block_pool.allocate_new_block() for each block
│   │   │       └── SWA: 同样的分配，但受 sliding window 上限约束
│   │   │       如果 full_sequence_must_fit → 还需验证完整序列能放下
│   │   │
│   │   └── Stage 5: 缓存 blocks
│   │       └── coordinator.cache_blocks(request, num_tokens_to_cache)
│   │           ├── 计算 block_hash (基于 token_ids)
│   │           └── block_pool.cache_full_blocks() → 存入前缀哈希表
│   │
│   └── 1d. 构建 SchedulerOutput
│       └── new_block_ids = new_blocks.get_block_ids()
│           → tuple[list[int], ...]  # per KV cache group
│
├── 2. 其他调度操作
│   ├── finished_req_ids → KVCacheManager.free()
│   │   └── coordinator.free(request_id)
│   │       ├── 释放 block_pool 中的 blocks（ref_count -= 1）
│   │       └── 前缀缓存引用计数管理
│   │
│   └── get_num_common_prefix_blocks()
│       └── 为 Cascade Attention 找公共前缀
│
└── 返回: SchedulerOutput → GPUModelRunner
```

### 3.2 allocate_slots 的 Block 布局

```
──────────────────────────────────────────────────────────────────
| <  comp  > | <  new_comp > | < ext_comp > | <  new  > | < lkhd >|
──────────────────────────────────────────────────────────────────
                                             | < to be computed > |
──────────────────────────────────────────────────────────────────
                             |         < to be allocated >        |
──────────────────────────────────────────────────────────────────

comp      = request.num_computed_tokens  （已计算 token 数）
new_comp  = num_new_computed_tokens       （本次前缀命中新增）
ext_comp  = num_external_computed_tokens  （外部缓存命中，如 P/D）
new       = num_new_tokens                （本次要计算的 token）
lkhd      = num_lookahead_tokens          （推测解码 lookahead）
```

---

## 4. 前缀缓存：Full 左扫 + SWA 右扫的交集算法

### 4.1 核心算法

`HybridKVCacheCoordinator.find_longest_cache_hit()` 是混合前缀缓存的核心。

```
算法: 混合注意力前缀缓存交集查找

输入: block_hashes (所有 token 的 block hash), max_cache_hit_length
输出: (hit_blocks_per_group, hit_length)

while True:
    curr_hit_length = hit_length
    
    for each attention_group (Full or SWA):
    
        if FullAttention:
            # Full 是"向下封闭"的: 如果之前查过了，只需 trim 到新长度
            if cached_blocks is not None:
                curr_hit_length = min(curr_hit_length, prev_hit_length)
                continue
            
            # 从左到右扫描: 每次前进一个 block，查哈希表
            # [✓] [✓] [✓] [✗] → hit 3 blocks
            for i in range(0, max_num_blocks):
                if block_pool.get_cached_block(block_hashes[i]):
                    computed_blocks[i] = cached_block
                else:
                    break  # 第一个 miss 就停止
        
        if SlidingWindow:
            # 从右到左扫描: 需要找到连续 sliding_window_contiguous_blocks
            # [⊗] [⊗] [✓] [✓] [⊗] [✓] → 需要连续 2 个 = [✓] [✓]
            for i in range(max_num_blocks-1, -1, -1):
                if cached := block_pool.get_cached_block(block_hashes[i]):
                    contiguous_count++
                    if contiguous_count >= required:
                        match_found = True
                        break
                else:
                    contiguous_count = 0  # 中断，重置计数
        
        # SWA 可能返回比 Full 更少的 hit_length
        curr_hit_length = min(curr_hit_length, _new_hit_length)
    
    # 收敛检查
    if curr_hit_length >= hit_length:
        break
    hit_length = curr_hit_length
    
    # 简单混合（1 Full + 1 other）一次迭代就够了
    if is_simple_hybrid:
        break
```

### 4.2 为什么 Full 左扫、SWA 右扫？

```
Full Attention 需要 PREFIX 命中（从第一个 token 开始连续命中）:
  token: [0] [1] [2] [3] [4] [5] [6] [7] ...
  cache: [✓] [✓] [✓] [✗] ...      ← 左扫，第一个 miss 停止
  结果:   3 个 block 命中（token 0-47）

Sliding Window 需要窗口内命中（只需最近 N 个 token）:
  token:  [...   ...  ...  ...] [5] [6] [7] 
  cache:       [⊗] [⊗] [✓] [✓]   ← 右扫，找连续窗口块
  结果:  最近 2 个 block 命中（token 80-111）
  
交集: min(3, 2) = 2 个 block = 32 token 可以复用
```

### 4.3 为什么需要取交集？

因为 **所有 KV Cache Group 必须一致地步调前进**。模型的一次 forward pass 中，所有层共享相同的请求调度状态。如果 Full 层认为 5 个 block 是前缀可复用的，但 SWA 层只认 2 个 block，那么 SWA 层的前 3 个 block 必须重新计算。为了统一，调度器取交集：只有 2 个 block 的前缀是可以安全跳过的。

---

## 5. KV 分配：Block 的分配与 Sliding Window 的环形复用

### 5.1 Full Attention 的 Block 分配

```
序列长度 = 100 tokens, block_size = 16
→ 需要 ceil(100/16) = 7 个 blocks

block_table (逻辑到物理映射):
  log_idx:  0    1    2    3    4    5    6
  phy_id:   42   17   89   3   56   21   99

每个 block 存储 16 个 token 的 K/V:
  block 42: K/V for tokens  [0:16]
  block 17: K/V for tokens  [16:32]
  block 89: K/V for tokens  [32:48]
  ...
```

### 5.2 Sliding Window 的 Block 分配（环形复用）

```
sliding_window = 4096 tokens, block_size = 16
→ 最多需要 ceil(4096/16) = 256 个 blocks

序列长度从 0 增长到 10000 tokens:
  ┌──────────────────────────────────────────────────┐
  │ Token     0-4095   → block [0:256] 分配          │
  │ Token   4096-4111  → 复用 block[0]（旧 token 0-15 被覆盖）
  │ Token   4112-4127  → 复用 block[1]（旧 token 16-31 被覆盖）
  │ ...                                              │
  │ Token  9984-9999   → 复用 block[243]             │
  └──────────────────────────────────────────────────┘

物理 block 的环形复用:
  物理 block[0]:
    Logical     Physical   Content
    idx=0       block[0]   tokens 0-15    (时刻 T0)
    idx=256     block[0]   tokens 4096-4111 (时刻 T1, 覆盖!)
    idx=512     block[0]   tokens 8192-8207 (时刻 T2, 覆盖!)

  slot_mapping 负责将 token 分配到正确的物理 slot:
    token 0    → block[0], offset=0
    token 4096 → block[0], offset=0 (同一个物理位置，新内容)
```

**关键实现**：`SingleTypeKVCacheManager.allocate_new_blocks()` 中的逻辑保证了 SWA 层只在滑动窗口范围内分配 block。当新 token 超出窗口时，最旧的 block 被标记为可复用，新的 token 写入复用后的物理位置。

### 5.3 Physical-to-Logical 映射（处理环形复用）

```
特殊情况: 物理 block 40 被多个逻辑位置引用
  block_table:     [40, 41, 42, 40]  # block 40 出现在 idx=0 和 idx=3
  含义: idx=3 是 block 40 当前的有效逻辑位置
  
  physical_to_logical 映射:
    physical[40] = max(0, 3) = 3  ← 取最大逻辑索引（最新）
    physical[41] = 1
    physical[42] = 2
```

**关键代码**：`flex_attention.py:162` 的 `physical_to_logical_mapping()`，使用 `scatter_reduce_` 的 `reduce="amax"`。

---

## 6. KV 回收：Block 的释放与 Slide-out 淘汰

### 6.1 三种回收路径

```
┌─────────────────────────────────────────────────────────────────┐
│  路径 1: Request 完成 → free()                                  │
│                                                                 │
│  Scheduler → KVCacheManager.free(request)                       │
│      └── coordinator.free(request_id)                           │
│          ├── block_pool.free_block() × N                         │
│          │   └── ref_count -= 1                                 │
│          │   └── if ref_count == 0 → 归还给 free pool           │
│          └── prefix cache hash table → 移除相关条目              │
│              └── 注意: 如果 ref_count > 0，block 不会被释放       │
│                  （其他请求还在通过前缀缓存引用此 block）          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  路径 2: Sliding Window 自动淘汰 → remove_skipped_blocks()      │
│                                                                 │
│  Scheduler → KVCacheManager.remove_skipped_blocks()             │
│      └── coordinator.remove_skipped_blocks()                    │
│          ├── Full 层: 不操作（所有 block 都需要保留）            │
│          └── SWA 层: 释放滑动窗口外的 block                       │
│              ├── block[0] ← 被新的 token 覆盖（环形复用）        │
│              └── block_pool.free_block(old_block)                │
│                                                                 │
│  注意: 这是"安全"释放，只释放已处理且不在当前窗口内的 block       │
│  边界: processed_computed_tokens = total - inflight              │
│        (考虑到可能的 speculative token 回滚)                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  路径 3: 前缀缓存淘汰 → evict_blocks()                           │
│                                                                 │
│  KV Connector → KVCacheManager.evict_blocks(block_ids)           │
│      └── block_pool.evict_blocks(block_ids)                      │
│          └── 从 hash table 移除，但不释放 block                   │
│              （如果 block 还在被使用，只是不再作为缓存靶向）      │
│                                                                 │
│  触发条件: 外部 KV 传输发现某些 blocks 已失效                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 前缀缓存的引用计数管理

```
           ┌──────────────────────────────────┐
           │       BlockPool (物理 blocks)     │
           │                                  │
           │  block[42]: ref_count=3          │
           │    ├── Request A (full/suffix)    │
           │    ├── Request B (full/prefix)    │
           │    └── Request C (swa/window)     │
           │                                  │
           │  block[99]: ref_count=1          │
           │    └── Request A (full/prefix)    │
           └──────────────────────────────────┘

           哈希表: hash → [block_ids for each group]
           
           当 Request A 完成:
           - block[42]: ref_count 3→2, 不释放
           - block[99]: ref_count 1→0, 释放到 free pool
           
           当所有引用都释放后:
           - block 归还 free pool，哈希表条目删除
```

---

## 7. 从 Block ID 到 Attention 计算：Block Table 的使用

### 7.1 数据流转

```
SchedulerOutput
  ├── scheduled_new_reqs[].block_ids    → tuple[list[int], ...]
  │                                      per KV cache group
  └── scheduled_cached_reqs.new_block_ids → dict[str, tuple[list[int], ...]]
                                         只传增量（diff）

         ↓ GPUModelRunner._update_states()
         
InputBatch.CachedRequestState
  ├── block_ids: tuple[list[int], ...]  # 每个请求的完整 block 表
  └── num_computed_tokens: int          # 已计算的 token 数

         ↓ GPUModelRunner._prepare_inputs()
         
CommonAttentionMetadata
  ├── block_table_tensor                # [num_reqs, max_blocks_per_seq]
  │                                     # 2D tensor，值为物理 block_id
  ├── slot_mapping                      # [num_tokens]
  │                                     # 每个 token → 物理 slot 位置
  ├── query_start_loc                   # [num_reqs+1] 累积偏移
  └── seq_lens                          # [num_reqs]

         ↓ AttentionMetadataBuilder.build()
         
FlexAttentionMetadata
  ├── block_table                       # 同 block_table_tensor
  ├── slot_mapping                      # 同 slot_mapping
  ├── sliding_window: int | None        # per-layer 的滑动窗口
  └── physical_to_logical               # 物理→逻辑 block 映射
```

### 7.2 Block Table 的构建

`GPUModelRunner._prepare_inputs()` 中：

```python
# 伪代码简化
for req_id, state in input_batch.req_states.items():
    # 每个请求的 block_table 是从 request state 中拿到的
    block_ids_per_group = state.block_ids  
    # e.g., ([42,17,89,3,56], [21,99])  
    #        ↑ group0 (full)   ↑ group1 (swa)
    
    # 填充到 block_table_tensor 中
    for group_idx, block_ids in enumerate(block_ids_per_group):
        layer_indices = kv_cache_config.get_layer_indices(group_idx)
        for layer_idx in layer_indices:
            block_table_tensor[layer_idx, req_idx, :len(block_ids)] = block_ids

# block_table_tensor shape: [num_layers, num_reqs, max_blocks]
```

### 7.3 Slot Mapping 的构建

```
slot_mapping 的核心作用: 告诉 GPU kernel 每个 token 的 K/V 应该写入哪个物理位置

构建逻辑:
  for each request:
      for each new_token in request.new_tokens:
          token_pos = num_computed + idx
          block_idx = token_pos // block_size      ← 逻辑 block 索引
          block_offset = token_pos % block_size    ← block 内偏移
          physical_block = block_table[block_idx]  ← 查表得到物理 block
          slot = physical_block * block_size + block_offset
          slot_mapping[token_idx] = slot
```

```
示例: block_size=16, num_computed=40
  new_token_idx=0 (全局 token 40):
    block_idx = 40 // 16 = 2
    offset = 40 % 16 = 8
    physical_block = block_table[2] = 89
    slot = 89 * 16 + 8 = 1432
    
  new_token_idx=1 (全局 token 41):
    block_idx = 41 // 16 = 2
    offset = 41 % 16 = 9  
    physical_block = block_table[2] = 89
    slot = 89 * 16 + 9 = 1433
```

### 7.4 Attention 计算中使用 Block Table

在 `FlexAttentionImpl.forward()` 中，block_table 用于两个关键操作：

**操作 1: KV Cache 更新（写入新 token 的 K/V）**
```python
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    # kv_cache shape: [num_blocks, num_kv_heads, block_size * 2 * head_size]
    # 即 [num_blocks, H_kv, block_size, 2 * head_size] → 转置
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    #                          → shape: [num_blocks, block_size, H_kv, head_size]
    
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping, ...
    )
    # slot_mapping[i] → GPU 物理索引 → 直接写入对应的 cache 位置
```

**操作 2: Attention 计算（读取缓存的 K/V）**
```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
    # 将 KV cache 展平为 [total_slots, num_kv_heads, head_size]
    key_cache = key_cache.view(-1, num_kv_heads, head_size)
    
    # 通过 paged_mask_mod 将物理 KV 索引转换为逻辑索引
    def paged_mask_mod(b, h, q_idx, physical_kv_idx):
        # physical_kv_idx → physical_kv_block → 查 physical_to_logical → logical_kv_idx
        logical_block_idx = physical_to_logical[req, physical_kv_idx // block_size]
        logical_kv_idx = logical_block_idx * block_size + offset
        # 检查有效性 + causal mask + sliding window
        ...
    
    # FlexAttention kernel 通过 block_mask 知道哪些 KV block 需要加载
    out = flex_attention_compiled(
        query, key_cache, value_cache,
        mask_mod, block_mask, scale
    )
```

---

## 8. 混合注意力运行时：Per-Layer Mask 动态切换

### 8.1 FlexAttentionImpl 的 forward 核心逻辑

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
    # ============ Step 1: 检测 per-layer 差异 ============
    
    needs_rebuild_block_mask = False
    
    # 1a: sliding_window 是否与 metadata 中预设的不同？
    if attn_metadata.sliding_window != self.sliding_window:
        attn_metadata.sliding_window = self.sliding_window
        # Full CUDA graph 模式下，metadata 预取了第一层的 sliding_window
        # 其他层若不同，需要切换
        attn_metadata.mask_mod = attn_metadata.get_mask_mod()
        needs_rebuild_block_mask = True
    
    # 1b: Layer 是否携带自定义 mask_mod？
    # 例: Mistral 系列的特定层可能有不同的 mask
    layer_mask_mod = getattr(layer, "logical_mask_mod", None)
    if layer_mask_mod is not None and attn_metadata.logical_mask_mod is not layer_mask_mod:
        attn_metadata.logical_mask_mod = layer_mask_mod
        attn_metadata.mask_mod = attn_metadata.get_mask_mod()
        needs_rebuild_block_mask = True
    
    # 1c: 自定义 block sparsity hint？
    layer_hint = getattr(layer, "block_sparsity_hint", None)
    if layer_hint is not None and attn_metadata.block_sparsity_hint is not layer_hint:
        attn_metadata.block_sparsity_hint = layer_hint
        needs_rebuild_block_mask = True
    
    # ============ Step 2: 重建 block_mask（如果需要） ============
    if needs_rebuild_block_mask or attn_metadata.block_mask is None:
        if attn_metadata.direct_build:
            attn_metadata.block_mask = attn_metadata._build_block_mask_direct()
        else:
            attn_metadata.block_mask = attn_metadata.build_block_mask()
    
    # ============ Step 3: 执行 Attention ============
    # kv_cache → 展平 → 作为 key_tensor, value_tensor
    kv_cache = kv_cache.transpose(1, 2)
    key_cache, value_cache = kv_cache.split(head_size, dim=-1)
    key_cache = key_cache.view(-1, num_kv_heads, head_size)    # 展平
    value_cache = value_cache.view(-1, num_kv_heads, head_size)
    
    out = flex_attention_compiled(
        query, key_cache, value_cache,
        mask_mod,         # 包含 causal + sliding_window + prefix_lm
        block_mask,       # 预裁剪的 KV block 索引
        scale
    )
```

### 8.2 Mask 组合机制

```
get_mask_mod() 的 mask 管道:

    Stage 1: 基础 mask
    ┌─────────────────────────────────────────┐
    │ paged attention: causal_mask_mod        │  q_idx >= kv_idx
    │ encoder only:    bidirectional_mask_mod │  总是 True
    └──────────────┬──────────────────────────┘
                   │
    Stage 2: AND 组合（全部满足才可见）
    ┌──────────────▼──────────────────────────┐
    │ + sliding_window_mask_mod               │  |q_idx - kv_idx| < window
    │   (if sliding_window is not None)       │
    ├─────────────────────────────────────────┤
    │ + rswa_mask_mod                         │  kv in_prefix OR kv in_window
    │   (if rswa enabled)                     │
    └──────────────┬──────────────────────────┘
                   │
    Stage 3: OR 组合（满足任一项即可见）
    ┌──────────────▼──────────────────────────┐
    │ + prefix_lm_mask_mod                    │  多模态 VLPrefixLM
    │   (if mm_prefix_range defined)          │  双向可见区域
    └─────────────────────────────────────────┘
```

### 8.3 Block Mask 的裁剪优化

```python
def _build_block_mask_direct(self):
    # 1. 获取每个请求使用的物理 pages
    used_pages = self.block_table[self.doc_ids, :num_blocks]
    
    # 2. 基础裁剪: causal mask 去除未来 blocks
    if self.causal:
        future_blocks = block_starts[None, :] > logical_q_idx[:, None]
        used_pages.masked_fill_(future_blocks, 0)
    
    # 3. Sliding Window 裁剪: 窗口外的 blocks 不需要加载
    if self.sliding_window:
        min_kv_idx = logical_q_idx - (self.sliding_window - 1)
        min_block_idx = min_kv_idx // self.block_size
        sliding_mask = logical_block_ids >= min_block_idx[:, None]
        used_pages.masked_fill_(~sliding_mask, 0)
    # 效果: 大幅减少 kernel 需要处理的 KV blocks
    
    # 4. 使用 BlockMask.from_kv_blocks() 构建
    return BlockMask.from_kv_blocks(used_pages, ...)
```

**优化效果示例**（4096 sliding window, block_size=16, 10000 token 序列）：
- 不裁剪: kernel 遍历 ceil(10000/16) = 625 个 KV blocks
- SWA 裁剪: kernel 遍历 ceil(4096/16) = 256 个 KV blocks
- **节省 59% 的 kernel 遍历量**

---

## 9. 完整调用链时序图

```
时间 ─────────────────────────────────────────────────────────────────→

┌─ 调度阶段 ────────────────────────────────────────────┐
│                                                       │
│  Scheduler.schedule()                                 │
│  │                                                    │
│  ├── for each request:                                │
│  │   │                                                │
│  │   ├─① get_computed_blocks(request)                 │
│  │   │ └→ HybridKVCacheCoordinator                    │
│  │   │   ┌─ FullAttentionManager  (左扫前缀)          │
│  │   │   └─ SlidingWindowManager  (右扫连续窗口)      │
│  │   │     → 取交集 hit_length                         │
│  │   │   → 返回 (KVCacheBlocks, num_hit_tokens)       │
│  │   │                                                │
│  │   ├─② [Optional] KV Connector                     │
│  │   │   └→ external match tokens                    │
│  │   │                                                │
│  │   ├─③ allocate_slots(request, ...)                 │
│  │   │   ├─ remove_skipped_blocks()   # SWA: 窗口外淘汰│
│  │   │   ├─ get_num_blocks_to_allocate()               │
│  │   │   ├─ allocate_new_computed_blocks()            │
│  │   │   ├─ allocate_new_blocks()      # 从 pool 分配 │
│  │   │   └─ cache_blocks()             # 写入哈希表   │
│  │   │                                                │
│  │   └─④ 记录 block_ids → SchedulerOutput             │
│  │                                                    │
│  ├── free(finished_requests)                          │
│  └── get_num_common_prefix_blocks()  # Cascade Attn    │
│                                                       │
└── SchedulerOutput ─────────────────────────────────────┘
         │
         │  block_ids, slot_mapping, ...
         ▼
┌─ 模型执行阶段 ────────────────────────────────────────┐
│                                                       │
│  GPUModelRunner                                       │
│  │                                                    │
│  ├─⑤ _update_states(SchedulerOutput)                  │
│  │   └→ InputBatch.CachedRequestState.block_ids 更新 │
│  │                                                    │
│  ├─⑥ _prepare_inputs()                                │
│  │   ├─ 构建 block_table_tensor                       │
│  │   │   [num_layers, num_reqs, max_blocks]           │
│  │   ├─ 构建 slot_mapping                             │
│  │   └─ 构建 query_start_loc, seq_lens               │
│  │                                                    │
│  ├─⑦ _build_attention_metadata()                      │
│  │   └→ CommonAttentionMetadata                       │
│  │                                                    │
│  └─⑧ execute_model()                                  │
│      │                                                │
│      └── for each layer:                              │
│          ├─ Attention.forward()                       │
│          │                                            │
│          │  ┌─ AttentionMetadataBuilder.build()       │
│          │  │  └→ FlexAttentionMetadata               │
│          │  │    ├── block_table ← block_table_tensor  │
│          │  │    ├── slot_mapping                      │
│          │  │    ├── sliding_window ← kv_cache_spec   │
│          │  │    └── physical_to_logical ← 反查表     │
│          │  │                                          │
│          │  └─ FlexAttentionImpl.forward()             │
│          │     ├─⑨ KV Cache Update                     │
│          │     │  └→ reshape_and_cache_flash()         │
│          │     │     slot_mapping → 写入 K/V           │
│          │     │                                        │
│          │     ├─⑩ Per-Layer Switch Detection          │
│          │     │  sliding_window changed? mask changed?│
│          │     │  └→ 重建 block_mask (if needed)      │
│          │     │                                        │
│          │     └─⑪ flex_attention_compiled()           │
│          │        ├── mask_mod = causal & sliding_window│
│          │        ├── block_mask = pre-pruned blocks   │
│          │        └── output → next layer              │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 关键编号快速导航

| 编号 | 步骤 | 关键文件 | 关键函数 |
|------|------|---------|---------|
| ① | 前缀缓存查找 | `kv_cache_coordinator.py` | `find_longest_cache_hit()` |
| ② | 外部 KV 匹配 | `kv_connector*.py` | `get_num_new_matched_tokens()` |
| ③ | Block 分配 | `kv_cache_manager.py` | `allocate_slots()` |
| ④ | 调度输出 | `scheduler.py` | `schedule()` → `SchedulerOutput` |
| ⑤ | 状态更新 | `gpu_model_runner.py` | `_update_states()` |
| ⑥ | 输入准备 | `gpu_model_runner.py` | `_prepare_inputs()` |
| ⑦ | Metadata 构建 | `gpu_model_runner.py` | `_build_attention_metadata()` |
| ⑧ | 模型执行 | `gpu_model_runner.py` | `execute_model()` |
| ⑨ | KV 写入 | `flex_attention.py` | `do_kv_cache_update()` |
| ⑩ | 类型检测 | `flex_attention.py` | `forward()` 开头 |
| ⑪ | 注意力计算 | `flex_attention.py` | `forward()` 末尾 |

---

## 10. 关键数据结构速查表

### KVCacheBlocks
```
blocks: tuple[Sequence[KVCacheBlock], ...]
  - 外层 tuple 索引 = KV cache group index
  - 内层 sequence = 该 group 的 block 列表
  - get_block_ids() → tuple[list[int], ...]
```

### CommonAttentionMetadata
```
block_table_tensor  Tensor [num_layers, num_reqs, max_blocks]
slot_mapping        Tensor [num_tokens]   # 每个 token → 物理 slot
query_start_loc     Tensor [num_reqs + 1] # 累计偏移
seq_lens            Tensor [num_reqs]
rswa_prefix_lens    Tensor | None
```

### FlexAttentionMetadata
```
block_table         Tensor  # 当前层的 block_table（可能共享或独立）
slot_mapping        Tensor
sliding_window      int | None  # 当前层的滑动窗口
mask_mod            callable     # 组合后的 mask 函数
block_mask          BlockMask    # 预裁剪的 block mask
logical_mask_mod    callable     # 基础逻辑 mask（causal/bidirectional）
physical_to_logical Tensor       # 物理→逻辑 block 映射
```

### Request
```
request_id          str
num_tokens          int         # 总 token 数
num_computed_tokens int         # 已计算 token 数
block_hashes        list[BlockHash]  # 用于前缀缓存查找
```

### SchedulerOutput
```
scheduled_new_reqs  list[NewRequestData]
scheduled_cached_reqs CachedRequestData
  ├── new_block_ids  list[tuple[list[int], ...]]  # per request, per group
  ├── num_computed_tokens
  └── num_output_tokens
finished_req_ids    set[str]
num_common_prefix_blocks list[int]
```

---

## 附录：快速问题解答

**Q: 为什么 Full 和 SWA 的前缀缓存要取交集？**
A: 因为一次 forward pass 中所有层同步执行。如果 Full 层认为可以复用 5 个 block 而 SWA 层只能复用 2 个，那就取 2 个，其余 3 个重新计算。

**Q: SWA 的前缀缓存为什么是"右扫找连续块"？**
A: 因为 SWA 只需要最近 `window_size` 内的 token。即使前面的 block 不命中，只要窗口内的 block 命中就可以复用。连续性要求是因为窗口是连续的 token 范围。

**Q: per-layer sliding_window 怎么切换不会影响性能？**
A: 在 CUDA Graph 模式下，第一层构建 metadata 时预取了 sliding_window。后续层如果相同就复用，不同时才触发重建。`torch.compile` 的动态性保证了切换开销可控。

**Q: Physical-to-Logical 映射为什么要用 `amax`？**
A: 因为 SWA 环形复用导致同一个物理 block 在 block_table 中出现多次（不同逻辑位置）。`amax` 确保取最新的逻辑索引，旧位置的映射被自动覆盖。
