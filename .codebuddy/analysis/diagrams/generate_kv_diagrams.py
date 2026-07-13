"""Generate PNG diagrams for KV Cache Manager analysis reports."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))


def box(ax, x, y, w, h, label, color, fontsize=8, bold=False, edge='#333'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, label='', dashed=False, color='#555', lw=1.3):
    ls = '--' if dashed else '-'
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dy = 0.15 if y1 == y2 else 0
        ax.text(mx, my + dy, label, ha='center', va='bottom' if dy >= 0 else 'top',
                fontsize=6.5, color=color, style='italic')


def note_box(ax, x, y, w, text, color_bg='#fffbe6', color_edge='#d79b00'):
    rect = FancyBboxPatch((x, y-0.25), w, 0.5, boxstyle="round,pad=0.08",
                          facecolor=color_bg, edgecolor=color_edge, linewidth=1)
    ax.add_patch(rect)
    ax.text(x + w/2, y, text, ha='center', va='center', fontsize=7,
            color='#8a6100', style='italic')


# ============================================================
# Report 1: KV Cache Manager backbone diagrams
# ============================================================

def draw_kv_arch_overview():
    """图1: KV Cache Manager 三层架构."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

    # Layer 1: Scheduler
    ax.text(0.5, 6.5, 'Layer 1: Scheduler（调度层）', fontsize=10, fontweight='bold', color='#d79b00')
    box(ax, 0.5, 4.0, 3.2, 2.2, '', '#fff2cc')
    box(ax, 1.0, 5.2, 2.2, 0.6, 'Scheduler.schedule()', '#ffe6cc', bold=True)
    box(ax, 1.0, 4.3, 2.2, 0.6, 'get_computed_blocks()', '#fff2cc')
    box(ax, 1.0, 3.5, 2.2, 0.6, 'allocate_slots()', '#fff2cc')

    # Layer 2: KVCacheManager
    ax.text(4.5, 6.5, 'Layer 2: KVCacheManager（门面层）', fontsize=10, fontweight='bold', color='#6c8ebf')
    box(ax, 4.5, 4.0, 3.2, 2.2, '', '#dae8fc')
    box(ax, 5.0, 5.2, 2.2, 0.6, 'KVCacheManager', '#a8d0f0', bold=True)
    box(ax, 5.0, 4.3, 2.2, 0.6, 'coordinator', '#dae8fc')
    box(ax, 5.0, 3.5, 2.2, 0.6, 'block_pool', '#dae8fc')

    # Layer 3: Coordinator + Pool
    ax.text(8.5, 6.5, 'Layer 3: Coordinator + BlockPool（执行层）', fontsize=10, fontweight='bold', color='#82b366')
    box(ax, 8.5, 4.0, 5.0, 2.2, '', '#d5e8d4')
    box(ax, 9.0, 5.2, 2.0, 0.6, 'HybridCoordinator', '#b8e0b8', bold=True)
    box(ax, 11.3, 5.2, 2.0, 0.6, 'BlockPool', '#b8e0b8')
    box(ax, 9.0, 4.3, 1.5, 0.5, 'FullAttnMgr', '#d5e8d4')
    box(ax, 10.7, 4.3, 1.5, 0.5, 'SWAMgr', '#d5e8d4')
    box(ax, 9.0, 3.7, 1.5, 0.5, 'ChunkedMgr', '#d5e8d4')
    box(ax, 10.7, 3.7, 1.5, 0.5, 'MambaMgr', '#d5e8d4')
    box(ax, 12.3, 4.3, 1.0, 0.9, 'LRU\nEvict', '#e8f5e8')

    # Arrows between layers
    for x1, x2 in [(3.7, 4.5), (7.7, 8.5)]:
        ax.annotate('', xy=(x2, 5.2), xytext=(x1, 5.2),
                   arrowprops=dict(arrowstyle='->', color='#888', lw=2))
    ax.text(4.1, 5.6, 'delegates', ha='center', fontsize=7, color='#888')

    plt.tight_layout()
    path = os.path.join(OUT, 'kv_arch_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def draw_req_lifecycle():
    """图2: 请求的一生 —— 与 KVCacheManager 的交互."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis('off')

    participants = ['API\nServer', 'Engine\nCore', 'Scheduler', 'KVCache\nManager', 'Coordinator', 'BlockPool', 'GPU\nWorker']
    x_pos = [1.5, 3.5, 5.5, 7.5, 9.8, 12.5, 15.0]

    # Lifelines
    for i, (name, x) in enumerate(zip(participants, x_pos)):
        ax.plot([x, x], [0.3, 11.5], color='#ddd', lw=2, zorder=1)
        ax.text(x, 11.5, name, ha='center', va='bottom', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f5f5', edgecolor='#ccc'))

    def seq_arrow(y, i1, i2, label, dashed=False, color='#333'):
        x1, x2 = x_pos[i1], x_pos[i2]
        ls = '--' if dashed else '-'
        if x1 < x2:
            ax.annotate('', xy=(x2-0.05, y), xytext=(x1+0.05, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.3, ls=ls))
        else:
            ax.annotate('', xy=(x1-0.05, y), xytext=(x2+0.05, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.3, ls=ls))
        mx = (x1 + x2) / 2
        dy = 0.2 if x1 < x2 else -0.25
        ax.text(mx, y + dy, label, ha='center', va='bottom' if dy >= 0 else 'top',
                fontsize=6.5, color=color, style='italic' if dashed else 'normal')

    y = 10.8
    seq_arrow(y, 0, 1, 'add_request()')
    seq_arrow(10.2, 1, 2, 'add_request()')
    note_box(ax, 5.5, 9.6, 3.0, 'Scheduler → waiting 队列')

    y = 9.0
    note_box(ax, 5.5, y, 4.5, '1.1 get_computed_blocks() — 前缀缓存匹配')
    seq_arrow(8.5, 2, 3, 'get_computed_blocks(req)')
    seq_arrow(8.0, 3, 4, 'find_longest_cache_hit(block_hashes)')
    seq_arrow(7.5, 4, 5, 'lookup hash → ref_cnt++')
    seq_arrow(7.0, 5, 4, 'computed_blocks', dashed=True)
    seq_arrow(6.7, 4, 3, 'KVCacheBlocks, hit_length', dashed=True)

    y = 6.2
    note_box(ax, 5.5, y, 5.5, '1.2 allocate_slots() — 三阶段分配')
    seq_arrow(5.7, 2, 3, 'allocate_slots(req, num_new_tokens)')
    note_box(ax, 7.5, 5.2, 3.2, '① remove_skipped_blocks()', '#ffe6e6', '#cc6666')
    seq_arrow(4.8, 3, 4, 'remove_skipped_blocks()')
    seq_arrow(4.5, 4, 5, 'free sliding window外blocks')
    note_box(ax, 7.5, 4.2, 3.2, '② allocate_new_computed_blocks()', '#e6ffe6', '#66cc66')
    seq_arrow(3.7, 4, 5, 'allocate cached blocks')
    note_box(ax, 7.5, 3.2, 3.2, '③ allocate_new_blocks()', '#e6e6ff', '#6666cc')
    seq_arrow(2.7, 4, 5, 'allocate new blocks')
    seq_arrow(2.4, 5, 4, 'new_blocks', dashed=True)
    seq_arrow(2.2, 4, 3, 'new_blocks', dashed=True)
    seq_arrow(1.9, 3, 2, 'new KVCacheBlocks', dashed=True)
    note_box(ax, 7.5, 1.7, 4.0, 'cache_blocks() — 写入prefix cache', '#fffbe6', '#d79b00')

    y = 1.2
    seq_arrow(y, 1, 6, 'execute_model()', dashed=False, color='#82b366')
    note_box(ax, 10.0, 0.7, 3.0, 'GPU Forward Pass', '#e6ffe6', '#66cc66')

    y = 0.3
    note_box(ax, 5.5, y-0.2, 5.0, '请求结束 → free() → blocks回收至BlockPool → ref_cnt--')

    plt.tight_layout()
    path = os.path.join(OUT, 'req_lifecycle.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def draw_allocate_blocks_layout():
    """图3: allocate_slots 的 block 布局."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4); ax.axis('off')

    # Block layout diagram
    blocks = [
        ('comp', '#dae8fc', '已缓存的\ncomputed'),
        ('comp', '#dae8fc', '已缓存的\ncomputed'),
        ('new_comp', '#b8e0b8', '新前缀\ncache hit'),
        ('new_comp', '#b8e0b8', '新前缀\ncache hit'),
        ('ext_comp', '#fff2cc', '外部\ncomputed'),
        ('new', '#ffe6cc', '未缓存\n待计算'),
        ('new', '#ffe6cc', '未缓存\n待计算'),
        ('lookahead', '#f8cecc', 'lookahead\n(spec decode)'),
    ]

    x = 0.5
    for btype, color, label in blocks:
        box(ax, x, 2.0, 1.4, 1.2, label, color)
        x += 1.6

    # Labels below
    ax.annotate('', xy=(0.5, 1.5), xytext=(13.5, 1.5),
               arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
    ax.text(0.5+6.5, 1.2, '|← cached by vLLM →|← not cached →|← to be computed →|← lookahead →|',
            ha='center', fontsize=8, color='#555')

    # Annotations above
    ax.annotate('', xy=(0.5, 3.5), xytext=(13.5, 3.5),
               arrowprops=dict(arrowstyle='<->', color='#d79b00', lw=1.5, ls='--'))
    ax.text(7.0, 3.75, '|← 释放冗余 (remove_skipped_blocks) →|← 分配已计算 (allocate_new_computed_blocks) →|← 分配新块 (allocate_new_blocks) →|',
            ha='center', fontsize=7.5, color='#d79b00', style='italic')

    plt.tight_layout()
    path = os.path.join(OUT, 'block_layout.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def draw_kv_scheduler_interaction():
    """图4: KV Cache Manager 与 Scheduler 的紧密协作."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

    # Left: Scheduler
    box(ax, 0.5, 4.0, 3.0, 2.5, '', '#fff2cc')
    ax.text(2.0, 6.3, 'Scheduler', ha='center', fontsize=10, fontweight='bold', color='#a67c00')
    box(ax, 0.8, 5.2, 2.4, 0.5, 'schedule()', '#ffe6cc')
    box(ax, 0.8, 4.5, 2.4, 0.5, 'update_from_output()', '#ffe6cc')
    box(ax, 0.8, 3.8, 2.4, 0.5, 'finish_requests()', '#ffe6cc')

    # Middle: KVCacheManager
    box(ax, 4.2, 4.0, 3.2, 2.5, '', '#dae8fc')
    ax.text(5.8, 6.3, 'KVCacheManager', ha='center', fontsize=10, fontweight='bold', color='#4a6fa5')
    box(ax, 4.5, 5.2, 2.6, 0.5, 'get_computed_blocks()', '#a8d0f0')
    box(ax, 4.5, 4.5, 2.6, 0.5, 'allocate_slots()', '#a8d0f0')
    box(ax, 4.5, 3.8, 2.6, 0.5, 'cache_blocks()', '#a8d0f0')

    # Right: Coordinator + Pool
    box(ax, 8.2, 4.0, 5.3, 2.5, '', '#d5e8d4')
    ax.text(10.8, 6.3, 'HybridCoordinator + BlockPool', ha='center', fontsize=9, fontweight='bold', color='#3a6b3a')
    box(ax, 8.5, 5.2, 2.3, 0.5, 'find_longest_cache_hit()', '#b8e0b8')
    box(ax, 11.0, 5.2, 2.3, 0.5, 'allocate_new_blocks()', '#b8e0b8')
    box(ax, 8.5, 4.5, 2.3, 0.5, 'remove_skipped_blocks()', '#b8e0b8')
    box(ax, 11.0, 4.5, 2.3, 0.5, 'cache_blocks()', '#b8e0b8')
    box(ax, 8.5, 3.8, 2.3, 0.5, 'get_blocks()', '#b8e0b8')
    box(ax, 11.0, 3.8, 2.3, 0.5, 'free()', '#e8c8c8')

    # Arrows
    for y, x1, x2, label in [
        (5.45, 3.5, 4.2, 'call'), (4.8, 3.5, 4.2, 'call'),
        (4.15, 3.5, 4.2, 'call'),
        (5.45, 7.4, 8.2, 'delegate'),
        (4.8, 7.4, 8.2, 'delegate'),
        (4.15, 7.4, 8.2, 'delegate'),
    ]:
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
        ax.text((x1+x2)/2, y+0.15, label, ha='center', fontsize=7, color='#888', style='italic')

    # Bottom: request flow
    note_box(ax, 1.2, 1.5, 11.6, '每步调度流程: add_request → get_computed_blocks → allocate_slots → execute → cache_blocks / free',
             '#fffbe6', '#d79b00')

    plt.tight_layout()
    path = os.path.join(OUT, 'kv_scheduler_interaction.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


# ============================================================
# Report 2: Speculative Decoding diagrams
# ============================================================

def draw_spec_decode_overview():
    """图5: 推测解码集成架构."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')

    # Proposer side
    box(ax, 0.5, 5.0, 3.5, 2.5, '', '#fff2cc')
    ax.text(2.25, 7.3, 'Proposer（提议者）', fontsize=10, fontweight='bold', color='#a67c00')
    box(ax, 0.8, 6.2, 2.9, 0.7, 'EAGLE / MTP / DFlash / Medusa', '#ffe6cc', bold=True)
    box(ax, 0.8, 5.3, 1.4, 0.6, 'N-gram\n匹配', '#fff2cc')
    box(ax, 2.3, 5.3, 1.4, 0.6, 'Draft\nModel', '#fff2cc')

    # Target Model
    box(ax, 5.0, 5.0, 4.0, 2.5, '', '#d5e8d4')
    ax.text(7.0, 7.3, 'Target Model（验证者）', fontsize=10, fontweight='bold', color='#3a6b3a')
    box(ax, 5.3, 6.2, 3.4, 0.7, 'GPU Forward Pass', '#b8e0b8', bold=True)
    box(ax, 5.3, 5.3, 3.4, 0.6, '拒绝采样 → 验证 draft tokens', '#d5e8d4')

    # KV Cache
    box(ax, 10.0, 5.0, 3.5, 2.5, '', '#dae8fc')
    ax.text(11.75, 7.3, 'KV Cache 集成', fontsize=10, fontweight='bold', color='#4a6fa5')
    box(ax, 10.3, 6.2, 2.9, 0.7, 'allocate_slots()\n+ lookahead tokens', '#a8d0f0')
    box(ax, 10.3, 5.3, 2.9, 0.6, 'cache_blocks() 仅缓存\n已验证的 tokens', '#dae8fc')

    # Arrows
    arrow(ax, 4.0, 6.5, 5.0, 6.5, 'draft tokens', color='#a67c00')
    arrow(ax, 9.0, 6.2, 10.0, 6.2, 'lookahead slots', color='#4a6fa5', dashed=True)
    arrow(ax, 10.0, 5.6, 9.0, 5.6, 'verified tokens', color='#4a6fa5', dashed=True)
    arrow(ax, 7.0, 4.8, 7.0, 3.2, 'rejected → 修正 KV', color='#cc6666', lw=1.5, dashed=True)

    # Bottom: flow
    note_box(ax, 2.0, 2.5, 10.0, '流程: Proposer生成draft tokens → Scheduler为它们分配lookahead slots → GPU验证 → reject修正 → cache已验证部分',
             '#ffe6e6', '#cc6666')

    # Scheduler connection
    box(ax, 2.8, 1.0, 8.4, 1.0, '', '#f5f5f5')
    ax.text(7.0, 1.5, 'Scheduler 调度: num_spec_tokens + num_output_placeholders（异步占位） + draft token占位符[-1,-1,...]',
            ha='center', fontsize=8, color='#555', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUT, 'spec_decode_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def draw_spec_cache_flow():
    """图6: 推测解码的缓存管理流程."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')

    participants = ['Proposer', 'Scheduler', 'KVCache\nManager', 'Coordinator', 'BlockPool', 'GPU']
    x_pos = [1.5, 4.0, 6.8, 9.5, 12.5, 15.0]

    for i, (name, x) in enumerate(zip(participants, x_pos)):
        ax.plot([x, x], [0.3, 9.5], color='#ddd', lw=2, zorder=1)
        ax.text(x, 9.5, name, ha='center', va='bottom', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f5f5', edgecolor='#ccc'))

    def sa(y, i1, i2, label, dashed=False, color='#333'):
        x1, x2 = x_pos[i1], x_pos[i2]
        ls = '--' if dashed else '-'
        if x1 < x2:
            ax.annotate('', xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.3, ls=ls))
        else:
            ax.annotate('', xy=(x1 - 0.05, y), xytext=(x2 + 0.05, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.3, ls=ls))
        mx = (x1 + x2) / 2
        dy = 0.2 if x1 < x2 else -0.22
        ax.text(mx, y + dy, label, ha='center', va='bottom' if dy >= 0 else 'top',
                fontsize=6.5, color=color)

    y = 8.8
    sa(y, 0, 1, 'draft_token_ids: [t1,t2,t3]')

    y = 8.2
    note_box(ax, 4.0, y, 5.0, 'Scheduler: num_new_tokens += num_spec_tokens')

    y = 7.5
    sa(y, 1, 2, 'allocate_slots(num_lookahead=N)')
    y = 7.0
    note_box(ax, 6.8, y, 4.0, '为 draft tokens 分配 lookahead slots', '#fffbe6', '#d79b00')
    y = 6.5
    sa(y, 2, 3, 'allocate_new_blocks(含lookahead)')
    y = 6.0
    sa(y, 3, 4, 'get_free_blocks()')
    y = 5.5
    sa(y, 4, 3, 'new_blocks', dashed=True)
    y = 5.0
    sa(y, 3, 2, 'KVCacheBlocks(含lookahead)', dashed=True)

    y = 4.3
    note_box(ax, 4.0, y, 8.5, 'Scheduler: spec_token_ids = [-1,-1,-1]（占位，worker侧填充）')

    y = 3.7
    sa(y, 1, 5, 'execute_model(draft_tokens)', color='#82b366')

    y = 3.1
    note_box(ax, 6.0, y, 3.5, 'GPU验证: accept/reject')

    y = 2.3
    note_box(ax, 4.0, y, 9.5, 'update_from_output: num_accepted=2, num_rejected=1 → 修正 num_computed_tokens, output_placeholders')

    y = 1.5
    sa(y, 1, 2, 'cache_blocks(已验证tokens)', dashed=True)
    y = 1.0
    note_box(ax, 6.8, y, 4.5, 'cache_blocks: 仅缓存 num_tokens 以内的已验证 token', '#ffe6e6', '#cc6666')
    y = 0.5
    note_box(ax, 4.0, y, 8.0, '关键: num_tokens_to_cache = min(computed + new, request.num_tokens) — 排除draft tokens')

    plt.tight_layout()
    path = os.path.join(OUT, 'spec_cache_flow.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def draw_eagle_integration():
    """图7: EAGLE 的 KV Cache 集成."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')

    # Target Model
    box(ax, 0.5, 2.0, 3.5, 3.5, '', '#d5e8d4')
    ax.text(2.25, 5.3, 'Target Model', fontsize=10, fontweight='bold', color='#3a6b3a')
    box(ax, 0.8, 4.3, 2.9, 0.7, 'Layer 1..N-1 → hidden states', '#b8e0b8')
    box(ax, 0.8, 3.4, 2.9, 0.6, 'Last Layer → logits', '#b8e0b8')
    box(ax, 0.8, 2.5, 2.9, 0.6, 'KV Cache: 完整填充', '#e6ffe6')

    # EAGLE Proposer
    box(ax, 5.0, 2.0, 3.5, 3.5, '', '#fff2cc')
    ax.text(6.75, 5.3, 'EAGLE Proposer', fontsize=10, fontweight='bold', color='#a67c00')
    box(ax, 5.3, 4.3, 2.9, 0.7, '输入: hidden states + tokens', '#ffe6cc')
    box(ax, 5.3, 3.4, 2.9, 0.6, '输出: draft tokens', '#ffe6cc')
    box(ax, 5.3, 2.5, 2.9, 0.6, 'KV Cache: lookahead slots', '#fffbe6')

    # KV Cache Manager
    box(ax, 9.5, 2.0, 4.0, 3.5, '', '#dae8fc')
    ax.text(11.5, 5.3, 'KVCacheManager (EAGLE适配)', fontsize=9, fontweight='bold', color='#4a6fa5')
    box(ax, 9.8, 4.3, 3.4, 0.7, 'use_eagle=True\n→ coordinator eagle_group_ids', '#a8d0f0')
    box(ax, 9.8, 3.4, 3.4, 0.6, 'num_lookahead_tokens\n= num_spec_tokens', '#a8d0f0')
    box(ax, 9.8, 2.5, 3.4, 0.6, 'cache_blocks(): 不缓存\ndraft/lookahead blocks', '#dae8fc')

    # Arrows
    arrow(ax, 4.0, 4.6, 5.0, 4.6, 'hidden_states', color='#a67c00', lw=1.8)
    arrow(ax, 8.5, 3.1, 9.5, 3.1, 'draft tokens', color='#a67c00', lw=1.5)
    arrow(ax, 11.5, 2.0, 11.5, 1.0, '不缓存 draft blocks', color='#cc6666', lw=1.3, dashed=True)

    # Bottom note
    note_box(ax, 2.0, 0.7, 10.0, 'EAGLE特有: 需要从Target Model传hidden states给Proposer → pass_hidden_states_to_model=True → KV Cache完整保留')

    plt.tight_layout()
    path = os.path.join(OUT, 'eagle_integration.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    draw_kv_arch_overview()
    draw_req_lifecycle()
    draw_allocate_blocks_layout()
    draw_kv_scheduler_interaction()
    draw_spec_decode_overview()
    draw_spec_cache_flow()
    draw_eagle_integration()
    print('\nAll 7 diagrams generated!')
