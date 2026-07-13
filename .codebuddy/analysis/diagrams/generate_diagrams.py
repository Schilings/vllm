"""Generate PNG diagrams for vLLM Async Scheduler analysis report."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

# Use Chinese font on Windows
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Color palette ──
C_API    = '#dae8fc'
C_ENGINE = '#d5e8d4'
C_SCHED  = '#fff2cc'
C_WORKER = '#e1d5e7'
C_ASYNC  = '#ffe6cc'
C_GPU    = '#dae8fc'
C_EDGE   = '#555555'
C_HIGHLIGHT = '#d79b00'


def draw_architecture():
    """System architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def box(ax, x, y, w, h, label, color, bold=False):
        """Draw a rounded rectangle with label."""
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                              facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', 
                fontsize=9, fontweight=weight, transform=ax.transData)

    def arrow(ax, x1, y1, x2, y2, label='', dashed=False, color=C_EDGE):
        ls = '--' if dashed else '-'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls=ls))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, ha='center', va='bottom', 
                    fontsize=7, color=color, style='italic')

    # Swinlanes (outer)
    box(ax, 0.3, 5.0, 3.8, 2.5, '', C_API)
    ax.text(0.5, 7.3, 'API Server 进程', fontsize=9, fontweight='bold', color='#4a6fa5')
    box(ax, 4.5, 3.0, 5.0, 4.5, '', C_ENGINE)
    ax.text(4.7, 7.3, 'EngineCore 进程', fontsize=9, fontweight='bold', color='#4a8c3c')
    box(ax, 10.0, 5.0, 3.5, 2.5, '', C_WORKER)
    ax.text(10.2, 7.3, 'Worker 进程(es)', fontsize=9, fontweight='bold', color='#7a4f8c')

    # Components
    box(ax, 1.0, 5.8, 2.4, 0.6, 'AsyncLLM', C_API)
    box(ax, 1.0, 5.0, 2.4, 0.6, 'ZMQ Client', '#fff2cc')
    box(ax, 5.0, 6.5, 2.0, 0.6, 'EngineCore', '#f8cecc', bold=True)
    box(ax, 5.0, 3.5, 4.2, 2.8, '', C_SCHED)  # AsyncScheduler container
    ax.text(5.15, 6.1, 'AsyncScheduler', fontsize=8, fontweight='bold', color='#a67c00')
    box(ax, 5.3, 4.5, 1.8, 0.6, 'AsyncScheduler', C_ASYNC, bold=True)
    box(ax, 7.4, 4.5, 1.5, 0.5, 'RequestQueue', '#f5f5f5')
    box(ax, 5.3, 3.8, 1.8, 0.5, 'KVCacheManager', '#f5f5f5')
    box(ax, 7.4, 3.8, 1.5, 0.5, 'SchedulerOutput', '#e1d5e7')
    box(ax, 10.5, 5.8, 2.4, 0.6, 'ModelExecutor', '#f8cecc')
    box(ax, 10.5, 5.0, 2.4, 0.6, 'GPUModelRunner', '#f8cecc')

    # Arrows
    arrow(ax, 3.4, 6.1, 5.0, 6.8, 'ZMQ Request', color='#b85450')
    arrow(ax, 7.0, 6.5, 6.2, 5.4, 'schedule()', color=C_HIGHLIGHT)
    arrow(ax, 8.9, 4.1, 10.5, 6.1, 'SchedulerOutput', dashed=True)
    arrow(ax, 10.5, 5.6, 12.9, 5.6, '', dashed=True)
    arrow(ax, 10.5, 5.3, 10.5, 3.0, 'execute_model()')
    # Highlight: async scheduling hint
    ax.text(5.0, 2.4, '不等GPU就发下一次 schedule()', fontsize=8, color=C_HIGHLIGHT, 
            style='italic', bbox=dict(boxstyle='round', facecolor='#fffbe6', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(OUT, 'architecture.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def draw_call_chain():
    """Complete call chain sequence diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')

    participants = ['API\nServer', 'Engine\nCore', 'Async\nScheduler', 'KV Cache\nManager', 'Model\nExecutor', 'GPU']
    x_pos = [1.5, 3.5, 5.5, 7.5, 10.0, 12.5]

    # Draw lifelines
    for i, (name, x) in enumerate(zip(participants, x_pos)):
        ax.plot([x, x], [0.5, 15.5], color='#ccc', lw=2, zorder=1)
        box_color = '#f0f0f0'
        ax.text(x, 15.5, name, ha='center', va='bottom', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=box_color, edgecolor='#999'))

    def arrow_pair(y, x1, x2, label, dashed=False, left_to_right=True, step=''):
        """Draw arrow from x1 to x2 at height y."""
        ls = '--' if dashed else '-'
        color = '#333' if not dashed else '#888'
        lw = 1.5
        if left_to_right:
            ax.annotate('', xy=(x2-0.05, y), xytext=(x1+0.05, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls))
        else:
            ax.annotate('', xy=(x1+0.05, y), xytext=(x2-0.05, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=ls))
        prefix = f'{step} ' if step else ''
        mx = (x1 + x2) / 2
        offset = 0.25 if left_to_right else -0.25
        ax.text(mx, y + offset, f'{prefix}{label}', ha='center', va='center' if offset>0 else 'top',
                fontsize=7, color=color)

    def note(y, text, x_st=0.8, x_end=13.2):
        """Draw a note box spanning across."""
        rect = FancyBboxPatch((x_st, y-0.3), x_end-x_st, 0.6, 
                             boxstyle="round,pad=0.1", facecolor='#fffbe6', 
                             edgecolor='#d79b00', linewidth=1, zorder=3)
        ax.add_patch(rect)
        ax.text((x_st+x_end)/2, y, text, ha='center', va='center', fontsize=7.5,
                color='#8a6100', style='italic')

    y = 14.5
    arrow_pair(y, x_pos[0], x_pos[1], 'add_request(req)', step='')

    y -= 1.0
    arrow_pair(y, x_pos[1], x_pos[2], 'add_request()', step='')

    # ── schedule() block ──
    note(13.0, '① schedule() — vllm/v1/core/sched/scheduler.py:432')
    y = 12.5
    note(y, '遍历 running 请求, 计算 num_new_tokens')
    y = 11.8
    arrow_pair(y, x_pos[2], x_pos[3], '② allocate_slots()', step='')
    y = 11.2
    note(y, 'remove_skipped_blocks() → allocate_new_blocks() → 失败则抢占')
    y = 10.5
    arrow_pair(y, x_pos[3], x_pos[2], 'new_blocks / None', dashed=True, left_to_right=False, step='')
    y = 9.8
    note(y, '遍历 waiting 请求 → 满足条件 → 移入 running')
    y = 9.1
    note(9.1, '③ _update_after_schedule() — output_placeholders += N (假设!)', x_st=3.5, x_end=7.5)

    # ── Execution phase ──
    y = 8.3
    arrow_pair(y, x_pos[2], x_pos[1], 'SchedulerOutput', dashed=True, left_to_right=False, step='')
    y = 7.6
    arrow_pair(y, x_pos[1], x_pos[4], 'execute_model()', step='')
    y = 6.9
    arrow_pair(y, x_pos[4], x_pos[5], '④ Forward Pass', step='')
    y = 6.2
    arrow_pair(y, x_pos[5], x_pos[4], 'logits, tokens', dashed=True, left_to_right=False, step='')
    y = 5.5
    arrow_pair(y, x_pos[4], x_pos[1], 'ModelRunnerOutput', dashed=True, left_to_right=False, step='')

    # ── update_from_output() block ──
    note(4.7, '⑤ update_from_output() — vllm/v1/core/sched/scheduler.py:1550')
    y = 4.0
    arrow_pair(y, x_pos[1], x_pos[2], 'update_from_output()', step='')
    y = 3.3
    note(y, '⑥ _update_request_with_output() — output_placeholders -= len(tokens) (修正!)', x_st=3.5, x_end=7.5)
    y = 2.6
    arrow_pair(y, x_pos[2], x_pos[3], '⑦ cache_blocks()', step='')
    y = 1.9
    note(y, '⑧ check_stop() → EOS / stop string / max_tokens')
    y = 1.2
    arrow_pair(y, x_pos[2], x_pos[1], 'EngineCoreOutputs', dashed=True, left_to_right=False, step='')
    y = 0.5
    arrow_pair(y, x_pos[1], x_pos[0], 'ZMQ 返回生成结果', dashed=True, left_to_right=False, step='')

    plt.tight_layout()
    path = os.path.join(OUT, 'call_chain.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def draw_sync_vs_async():
    """Sync vs Async scheduling comparison diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Dividers
    ax.axvline(x=7.0, ymin=0.03, ymax=0.97, color='#aaa', lw=1, ls='--')
    ax.text(3.5, 9.5, '同步调度', ha='center', fontsize=13, fontweight='bold', color='#4a6fa5')
    ax.text(10.5, 9.5, '异步调度', ha='center', fontsize=13, fontweight='bold', color='#a67c00')

    participants_sync = ['Sync\nScheduler', 'Worker\nGPU']
    participants_async = ['Async\nScheduler', 'Worker\nGPU']

    def draw_lane(ax, x_offset, name1, name2, color1, color2):
        x1, x2 = x_offset + 1.5, x_offset + 4.5
        ax.plot([x1, x1], [1.0, 9.0], color='#ccc', lw=2)
        ax.plot([x2, x2], [1.0, 9.0], color='#ccc', lw=2)
        ax.text(x1, 9.2, name1, ha='center', fontsize=8, fontweight='bold')
        ax.text(x2, 9.2, name2, ha='center', fontsize=8, fontweight='bold')

    draw_lane(ax, 0, 'Sync\nScheduler', 'Worker\nGPU', C_API, C_GPU)
    draw_lane(ax, 7, 'Async\nScheduler', 'Worker\nGPU', C_HIGHLIGHT, C_GPU)

    # Sync side: step 0
    xl, xr = 1.5, 4.5
    y = 8.0
    ax.annotate('', xy=(xr, y), xytext=(xl, y),
               arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(3.0, y+0.25, 'S0.schedule()', ha='center', fontsize=7.5)
    
    y = 7.2
    ax.annotate('', xy=(xl, y), xytext=(xr, y),
               arrowprops=dict(arrowstyle='->', color='#888', lw=1.5, ls='--'))
    ax.text(3.0, y-0.35, 'S0 执行完成', ha='center', fontsize=7.5, color='#666')
    
    y = 6.5
    note_sync = FancyBboxPatch((xl-0.3, y-0.3), 3.6, 0.6, boxstyle="round,pad=0.1",
                               facecolor='#e8f0fe', edgecolor='#4a6fa5', linewidth=1)
    ax.add_patch(note_sync)
    ax.text(3.0, y, 'update状态', ha='center', fontsize=7.5)

    # Sync side: step 1
    y = 5.5
    ax.annotate('', xy=(xr, y), xytext=(xl, y),
               arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
    ax.text(3.0, y+0.25, 'S1.schedule() 基于准确状态', ha='center', fontsize=7.5)
    y = 4.7
    ax.annotate('', xy=(xl, y), xytext=(xr, y),
               arrowprops=dict(arrowstyle='->', color='#888', lw=1.5, ls='--'))
    ax.text(3.0, y-0.35, 'S1 返回', ha='center', fontsize=7.5, color='#666')
    y = 4.0
    note_sync2 = FancyBboxPatch((xl-0.3, y-0.3), 3.6, 0.6, boxstyle="round,pad=0.1",
                                facecolor='#e8f0fe', edgecolor='#4a6fa5', linewidth=1)
    ax.add_patch(note_sync2)
    ax.text(3.0, y, 'update状态', ha='center', fontsize=7.5)

    # Async side: S0 schedule
    xl2, xr2 = 8.5, 11.5
    y = 8.0
    ax.annotate('', xy=(xr2, y), xytext=(xl2, y),
               arrowprops=dict(arrowstyle='->', color='#a67c00', lw=1.5))
    ax.text(10.0, y+0.25, 'S0.schedule() 下发执行', ha='center', fontsize=7.5)
    
    y = 7.3
    note_as = FancyBboxPatch((xl2-0.3, y-0.3), 3.6, 0.6, boxstyle="round,pad=0.1",
                             facecolor='#fffbe6', edgecolor='#a67c00', linewidth=1)
    ax.add_patch(note_as)
    ax.text(10.0, y, 'output_placeholders += N (假设!)', ha='center', fontsize=7.5, color='#a67c00')

    # Async side: S1 schedule (without waiting)
    y = 6.2
    ax.annotate('', xy=(xr2, y), xytext=(xl2, y),
               arrowprops=dict(arrowstyle='->', color='#a67c00', lw=1.5))
    ax.text(10.0, y+0.25, 'S1.schedule() 不等S0,基于假设!', ha='center', fontsize=7.5, color='#a67c00')
    y = 5.5
    note_as2 = FancyBboxPatch((xl2-0.3, y-0.3), 3.6, 0.6, boxstyle="round,pad=0.1",
                              facecolor='#fffbe6', edgecolor='#a67c00', linewidth=1)
    ax.add_patch(note_as2)
    ax.text(10.0, y, '继续累加 placeholder', ha='center', fontsize=7.5, color='#a67c00')

    # S0 result comes back
    y = 4.8
    ax.annotate('', xy=(xl2, y), xytext=(xr2, y),
               arrowprops=dict(arrowstyle='->', color='#888', lw=1.5, ls='--'))
    ax.text(10.0, y-0.35, 'S0 结果返回', ha='center', fontsize=7.5, color='#666')
    y = 4.1
    note_as3 = FancyBboxPatch((xl2-0.5, y-0.3), 4.0, 0.6, boxstyle="round,pad=0.1",
                              facecolor='#f0e6ff', edgecolor='#7a4f8c', linewidth=1)
    ax.add_patch(note_as3)
    ax.text(10.0, y, 'output_placeholders -= len(tokens) (修正!)', ha='center', fontsize=7.5, color='#6a3f8c')

    # S2 schedule
    y = 3.4
    ax.annotate('', xy=(xr2, y), xytext=(xl2, y),
               arrowprops=dict(arrowstyle='->', color='#a67c00', lw=1.5))
    ax.text(10.0, y+0.25, 'S2.schedule()', ha='center', fontsize=7.5)

    # S1 result returns
    y = 2.7
    ax.annotate('', xy=(xl2, y), xytext=(xr2, y),
               arrowprops=dict(arrowstyle='->', color='#888', lw=1.5, ls='--'))
    ax.text(10.0, y-0.35, 'S1 结果返回, 修正 placeholder', ha='center', fontsize=7.5, color='#666')

    # Key insight
    y = 1.8
    note_key = FancyBboxPatch((xl2-0.5, y-0.4), 4.0, 0.8, boxstyle="round,pad=0.2",
                              facecolor='#ffedd5', edgecolor='#d97706', linewidth=2)
    ax.add_patch(note_key)
    ax.text(10.0, y, 'Scheduler 始终领先\nWorker 1~2 步!', ha='center', fontsize=8, 
            fontweight='bold', color='#92400e')

    plt.tight_layout()
    path = os.path.join(OUT, 'sync_vs_async.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    draw_architecture()
    draw_call_chain()
    draw_sync_vs_async()
    print('All diagrams generated!')
