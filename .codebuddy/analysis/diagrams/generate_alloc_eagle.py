"""PNG diagrams for allocate_slots deep dive + EAGLE3 report."""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.dirname(os.path.abspath(__file__))


def box(ax, x, y, w, h, label, color, fs=8, bold=False, ec='#333', lw=1.5):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                       facecolor=color, edgecolor=ec, linewidth=lw)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal')


def arrow(ax, x1, y1, x2, y2, c='#555', lw=1.3, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, ls=ls))


def note(ax, x, y, w, text, cbg='#fffbe6', cec='#d79b00'):
    FancyBboxPatch((x, y - 0.28), w, 0.56, boxstyle="round,pad=0.06",
                   facecolor=cbg, edgecolor=cec, linewidth=1, zorder=10)
    ax.add_patch(FancyBboxPatch((x, y - 0.28), w, 0.56, boxstyle="round,pad=0.06",
                                facecolor=cbg, edgecolor=cec, linewidth=1, zorder=10))
    ax.text(x + w / 2, y, text, ha='center', va='center', fontsize=6.5, color='#8a6100')


# ===== Report 1: allocate_slots diagrams =====

def draw_alloc_flow():
    """allocate_slots 完整决策树."""
    fig, ax = plt.subplots(figsize=(14, 12)); ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis('off')

    # Entry
    box(ax, 4.5, 11.0, 5.0, 0.7, 'Scheduler → allocate_slots(req, num_new_tokens)', '#ffe6cc', bold=True, fs=9)
    arrow(ax, 7.0, 10.9, 7.0, 10.3)

    # Step 0: full_sequence_must_fit
    box(ax, 0.3, 9.3, 4.5, 0.8, '① full_sequence_must_fit?\n计算整条请求所需block', '#dae8fc', fs=7)
    box(ax, 5.5, 9.3, 3.0, 0.8, '不够?\n→ return None', '#f8cecc', fs=7)
    arrow(ax, 4.8, 9.7, 5.5, 9.7)
    arrow(ax, 7.0, 9.3, 7.0, 8.7)

    # Step 1: remove skipped blocks
    box(ax, 2.0, 7.7, 10.0, 0.8, '② remove_skipped_blocks()\nFree sliding window外blocks (Full=noop, SWA=淘汰, Mamba=清上步state)', '#d5e8d4', fs=7)
    arrow(ax, 7.0, 7.7, 7.0, 7.1)

    # Step 2: get_num_blocks_to_allocate
    box(ax, 0.3, 6.1, 5.5, 0.8, '③ get_num_blocks_to_allocate()\nFast-path: running req → 直接差值\nNew req: num_new = req - max(skip,local)', '#dae8fc', fs=7)
    arrow(ax, 7.0, 6.1, 7.0, 5.5)

    # Step 3: capacity check
    box(ax, 0.3, 4.5, 13.4, 0.8, '④ 容量检查: available = free - reserved_block - watermark (waiting/preempted)\n                    required > available? → return None (Scheduler 抢占)', '#f8cecc', fs=7)
    arrow(ax, 7.0, 4.5, 7.0, 3.9)

    # Step 4: allocate computed + new
    box(ax, 0.3, 2.9, 5.5, 0.8, '⑤ allocate_new_computed_blocks()\n追加 prefix-cache-hit blocks\ntouch + ref_cnt++', '#b8e0b8', fs=7)
    box(ax, 6.5, 2.9, 5.5, 0.8, '⑥ allocate_new_blocks()\n从 BlockPool 取 free blocks\n追加到 req 的 block_table', '#b8e0b8', fs=7)
    arrow(ax, 5.8, 3.3, 6.5, 3.3)
    arrow(ax, 7.0, 2.9, 7.0, 2.3)

    # Step 5: cache
    box(ax, 2.0, 1.3, 10.0, 0.8, '⑦ cache_blocks() → 写入 prefix cache hash表\nnum_tokens_to_cache = min(computed + new, request.num_tokens)\n⚠️ 上限 request.num_tokens → 自动排除 draft tokens', '#fff2cc', fs=7)
    arrow(ax, 7.0, 1.3, 7.0, 0.7)
    box(ax, 4.5, 0.0, 5.0, 0.6, 'return KVCacheBlocks(new_blocks)', '#ffe6cc', bold=True, fs=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'alloc_flow.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved alloc_flow.png')


def draw_block_stages():
    """三阶段 block 操作详解."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ['阶段1: remove_skipped_blocks()', '阶段2: allocate_computed + 容量预估', '阶段3: allocate_new + cache']
    colors = ['#e8f0fe', '#e6ffe6', '#fffbe6']
    data = [
        [('SWA窗口外的\nblocks', '#f8cecc'), ('Full Attn\nnoop', '#e8e8e8'), ('Mamba状态\n块释放', '#f0e6ff')],
        [('Fast-path:\n已track请求\n直接差值', '#b8e0b8'), ('新请求:\nnum_new =\nreq-max(skip,local)', '#b8e0b8'), ('evictable\nblocks\n计数', '#fff2cc')],
        [('追加prefix\ncache-hit\nblocks', '#b8e0b8'), ('从BlockPool\n取free\nblocks', '#b8e0b8'), ('cache_blocks\n仅缓存\n已验证token', '#fffbe6')],
    ]
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 6); ax.set_ylim(0, 5.5); ax.axis('off')
        ax.set_title(titles[i], fontsize=10, fontweight='bold', pad=8)
        bg = FancyBboxPatch((0.2, 1.0), 5.6, 4.0, boxstyle="round,pad=0.2",
                            facecolor=colors[i], edgecolor='#ccc', linewidth=1)
        ax.add_patch(bg)
        for j, (label, c) in enumerate(data[i]):
            box(ax, 0.5 + j * 1.85, 2.0, 1.6, 2.0, label, c, fs=7)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'block_stages.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved block_stages.png')


def draw_watermark_model():
    """Watermark / full_sequence_must_fit / reserved 保护机制."""
    fig, ax = plt.subplots(figsize=(12, 5)); ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis('off')

    # Pool bar
    pool_w, pool_h = 10, 1.2
    FancyBboxPatch((1, 3.2), pool_w, pool_h, boxstyle="round,pad=0.1",
                   facecolor='#e8e8e8', edgecolor='#999', linewidth=1.5)
    ax.add_patch(FancyBboxPatch((1, 3.2), pool_w, pool_h, boxstyle="round,pad=0.1",
                                 facecolor='#e8e8e8', edgecolor='#999', linewidth=1.5))

    # Used
    FancyBboxPatch((1, 3.25), 5.5, 1.1, boxstyle="round,pad=0.08",
                   facecolor='#f8cecc', edgecolor='#cc6666', linewidth=1)
    ax.add_patch(FancyBboxPatch((1, 3.25), 5.5, 1.1, boxstyle="round,pad=0.08",
                                 facecolor='#f8cecc', edgecolor='#cc6666', linewidth=1))
    ax.text(3.75, 3.8, 'used blocks', ha='center', fontsize=7, color='#a33')

    # Reserved
    FancyBboxPatch((6.5, 3.25), 1.5, 1.1, boxstyle="round,pad=0.08",
                   facecolor='#fff2cc', edgecolor='#d79b00', linewidth=1)
    ax.add_patch(FancyBboxPatch((6.5, 3.25), 1.5, 1.1, boxstyle="round,pad=0.08",
                                 facecolor='#fff2cc', edgecolor='#d79b00', linewidth=1))
    ax.text(7.25, 3.8, 'reserved/\nwatermark', ha='center', fontsize=6, color='#a67c00')

    # Free
    FancyBboxPatch((8.0, 3.25), 3, 1.1, boxstyle="round,pad=0.08",
                   facecolor='#d5e8d4', edgecolor='#6aa84f', linewidth=1)
    ax.add_patch(FancyBboxPatch((8.0, 3.25), 3, 1.1, boxstyle="round,pad=0.08",
                                 facecolor='#d5e8d4', edgecolor='#6aa84f', linewidth=1))
    ax.text(9.5, 3.8, 'available', ha='center', fontsize=7, color='#3a6b3a')

    ax.text(6, 2.5, 'available = free - reserved_blocks - (watermark for waiting/preempted)', ha='center', fontsize=8, color='#555')
    ax.text(6, 1.8, 'required  = get_num_blocks_to_allocate()', ha='center', fontsize=8, color='#555')
    ax.text(6, 1.1, 'required > available ? → return None → Scheduler 抢占!', ha='center', fontsize=9, fontweight='bold', color='#cc0000')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'watermark_model.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved watermark_model.png')


# ===== Report 2: EAGLE3 diagrams =====

def draw_eagle3_arch():
    """EAGLE3 模型架构."""
    fig, ax = plt.subplots(figsize=(14, 8)); ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')

    # Target Model (Verifier) left
    box(ax, 0.3, 2.5, 3.0, 5.0, '', '#d5e8d4')
    ax.text(1.8, 7.3, 'Target Model (Verifier)', fontsize=9, fontweight='bold', color='#3a6b3a')
    box(ax, 0.6, 5.5, 2.4, 1.0, 'Embedding', '#b8e0b8')
    box(ax, 0.6, 4.2, 2.4, 1.0, 'Layer 1..29', '#b8e0b8')
    box(ax, 0.6, 2.8, 2.4, 1.0, 'Layer 30\n→ hidden states', '#e6ffe6', bold=True)

    # EAGLE3 Draft Model right
    box(ax, 4.5, 2.5, 4.5, 5.0, '', '#fff2cc')
    ax.text(6.75, 7.3, 'EAGLE3 Draft Model', fontsize=9, fontweight='bold', color='#a67c00')
    box(ax, 4.8, 5.8, 3.9, 0.8, 'fc: 融合层 (3×hidden → hidden)', '#ffe6cc', fs=7)
    box(ax, 4.8, 4.7, 3.9, 0.8, 'First Layer: Eagle3DecoderLayer\nQKV = fc(concat(embeds, hidden))', '#ffe6cc', fs=7)
    box(ax, 4.8, 3.6, 3.9, 0.8, 'Layer 2..N: Standard Decoder', '#fff2cc', fs=7)
    nose(ax, 4.8, 2.8, 3.9, 0.6, 'norm → lm_head', '#fff2cc', fs=7)

    # Scheduler / KV Cache right
    box(ax, 10.0, 2.5, 3.5, 5.0, '', '#dae8fc')
    ax.text(11.75, 7.3, 'vLLM Integration', fontsize=9, fontweight='bold', color='#4a6fa5')
    box(ax, 10.3, 5.8, 2.9, 0.8, 'pass_hidden_states\n_to_model=True', '#a8d0f0', fs=7)
    box(ax, 10.3, 4.7, 2.9, 0.8, 'num_lookahead\n_tokens', '#a8d0f0', fs=7)
    nose(ax, 10.3, 3.6, 2.9, 0.8, 'use_eagle=True\n→ eagle_group_ids', '#dae8fc', fs=7)
    nose(ax, 10.3, 2.8, 2.9, 0.6, 'cache排除\ndraft tokens', '#dae8fc', fs=7)

    # Arrows
    arrow(ax, 3.3, 3.3, 4.5, 6.1, c='#a67c00', lw=2.5)  # hidden states
    ax.text(3.9, 4.8, 'hidden\nstates', ha='center', fontsize=6, color='#a67c00', fontweight='bold')
    arrow(ax, 9.0, 6.2, 10.0, 6.2, c='#a67c00', lw=1.5, ls='--')  # draft to scheduler
    arrow(ax, 11.75, 5.7, 11.75, 4.8, c='#4a6fa5', lw=1.5, ls='--')

    # Bottom note
    ax.text(7, 1.5, 'EAGLE3 关键: 从 Target 的3层取 hidden states → 融合 → 与 embedding 拼接输入首层 → 自回归生成 draft tokens → Target 批量验证',
            ha='center', fontsize=8, color='#555', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#ccc'))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'eagle3_arch.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved eagle3_arch.png')


def draw_eagle3_flow():
    """EAGLE3 完整推理流程."""
    fig, ax = plt.subplots(figsize=(16, 10)); ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis('off')

    parties = ['Target\nModel', 'Hidden\nStates', 'Fusion\nLayer', 'First\nLayer', 'Std\nLayers', 'lm_head', 'GPU\nVerify']
    xp = [1.5, 3.3, 5.2, 7.0, 9.0, 11.0, 13.5]
    for n, x in zip(parties, xp):
        ax.plot([x, x], [0.3, 9.5], color='#ddd', lw=2)
        ax.text(x, 9.5, n, ha='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f5f5f5', edgecolor='#ccc'))

    def sa(y, i1, i2, label, c='#333', ls='-'):
        x1, x2 = xp[i1], xp[i2]
        if x1 < x2:
            ax.annotate('', xy=(x2 - 0.05, y), xytext=(x1 + 0.05, y),
                       arrowprops=dict(arrowstyle='->', color=c, lw=1.3, ls=ls))
        else:
            ax.annotate('', xy=(x1 - 0.05, y), xytext=(x2 + 0.05, y),
                       arrowprops=dict(arrowstyle='->', color=c, lw=1.3, ls=ls))
        ax.text((x1+x2)/2, y+0.15, label, ha='center', fontsize=6.5, color=c)

    def nb(x, y, w, text):
        ax.add_patch(FancyBboxPatch((x, y-0.28), w, 0.56, boxstyle="round,pad=0.06",
                                     facecolor='#fffbe6', edgecolor='#d79b00', linewidth=1))
        ax.text(x+w/2, y, text, ha='center', fontsize=6.5, color='#8a6100')

    y = 8.6
    sa(y, 0, 1, 'forward()', c='#82b366')
    nb(4.0, y-0.4, 3.0, '取3层 hidden states')
    y = 7.8
    sa(y, 1, 2, '3×hidden concat', c='#a67c00')
    y = 7.0
    sa(y, 2, 3, 'fc → hidden_size')
    y = 6.2
    nb(5.0, y-0.2, 4.2, 'concat(embeds, fused_hidden) → 2×hidden → QKV')
    sa(y, 3, 4, 'output') if True else None
    sa(5.6, 3, 4, 'std decoder')
    nb(8.0, 5.4, 2.5, '自回归多步')
    y = 4.8
    sa(y, 4, 5, 'norm → logits')
    nb(10.0, y-0.2, 2.5, '采样 token')
    y = 4.0
    sa(y, 5, 0, 'draft tokens', c='#a67c00', ls='--')
    y = 3.2
    nb(2.0, y-0.2, 10.0, 'Target Model 批量验证 draft tokens → 拒绝采样')
    y = 2.5
    sa(y, 0, 6, 'forward(verified + draft)')
    y = 1.8
    nb(7.0, y-0.2, 5.0, 'accept/reject → 修正 KV Cache')
    y = 1.0
    nb(2.0, y-0.2, 10.0, 'cache_blocks: 仅缓存 accept 的 tokens (num_tokens cap 排除 draft)')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'eagle3_flow.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved eagle3_flow.png')


def draw_eagle_vs_others():
    """EAGLE3 与其他推测解码方案对比."""
    fig, ax = plt.subplots(figsize=(14, 5.5)); ax.set_xlim(0, 14); ax.set_ylim(0, 5.5); ax.axis('off')

    methods = [
        ('N-gram\nMatching', '搜索历史序列\n匹配n-gram', '无', '★☆☆', '#e8e8e8'),
        ('Draft Model', '独立小模型\n并行生成', '无', '★★☆', '#dae8fc'),
        ('Medusa', '基础模型+\n多个预测头', '无', '★★☆', '#dae8fc'),
        ('EAGLE3', 'target hidden\nstates 外推', 'use_eagle+\nlookahead', '★★★', '#fff2cc'),
    ]
    for i, (name, desc, kv_req, perf, c) in enumerate(methods):
        x = 1.0 + i * 3.2
        box(ax, x, 3.0, 2.4, 1.5, '', c)
        ax.text(x + 1.2, 4.3, name, ha='center', fontsize=9, fontweight='bold', color='#333')
        ax.text(x + 1.2, 3.8, desc, ha='center', fontsize=7, color='#666')
        ax.text(x + 1.2, 3.3, f'KV Cache: {kv_req}', ha='center', fontsize=6.5, color='#999')
        ax.text(x + 1.2, 3.0, f'加速: {perf}', ha='center', fontsize=7, color='#d79b00')

    ax.text(7, 1.5, 'EAGLE3 最大特点: 需要 Target Model 的 hidden states → pass_hidden_states_to_model=True → 对 KV Cache 和模型运行器有额外要求',
            ha='center', fontsize=8, color='#555', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#ccc'))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'eagle_vs_others.png'), dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print('Saved eagle_vs_others.png')


def nose(ax, x, y, w, h, label, c, fs=7):
    box(ax, x, y, w, h, label, c, fs=fs)


if __name__ == '__main__':
    draw_alloc_flow()
    draw_block_stages()
    draw_watermark_model()
    draw_eagle3_arch()
    draw_eagle3_flow()
    draw_eagle_vs_others()
    print('\nAll 6 diagrams generated!')
