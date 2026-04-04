"""
クラスタリング詳細可視化（改善版）
Figure 1: 初期盤面 + 全候補着弾点（重なり注記付き）
Figure 2: クラスタ別の着弾点マップ（クラスタごとに分割表示）
Figure 3: クラスタ内類似性の検証（同一クラスタのメンバー結果盤面を並べて表示）
Figure 4: 歩 vs DC 最良手の結果盤面比較
"""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter

matplotlib.rcParams['font.family'] = 'MS Gothic'

JSON_PATH = "../build/Release/experiments/pool_clustering_results/detail_opp2_my1_r20.json"

with open(os.path.join(os.path.dirname(__file__), JSON_PATH), 'r') as f:
    data = json.load(f)

initial = data['initial_stones']
candidates = data['candidates']
pool_best = data['pool_best_idx']
dc_best = data['dc_best_idx']
k = data['k']
state_name = data['state_name']

n_clusters = max(c['cluster_id'] for c in candidates) + 1
CLUSTER_COLORS = list(mcolors.TABLEAU_COLORS.values())[:10] + list(mcolors.CSS4_COLORS.values())[::20]
while len(CLUSTER_COLORS) < n_clusters:
    CLUSTER_COLORS += CLUSTER_COLORS

TYPE_MARKERS = {'Draw': 'o', 'Hit': 'X', 'Freeze': 'D', 'Guard': 's', 'Pass': 'P', 'Other': '*'}

HCX, HCY, HR = 0.0, 38.405, 1.829


def draw_house(ax, y_range=3.5):
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(HCY - y_range, HCY + 2.2)
    ax.set_aspect('equal')
    ax.set_facecolor('#E8F4F8')
    for r, c, a in [(1.829, '#A8C8E8', 0.4), (1.22, 'white', 0.6), (0.61, '#E8A8A8', 0.35), (0.15, 'white', 0.9)]:
        ax.add_patch(Circle((HCX, HCY), r, color=c, alpha=a, zorder=0))
    ax.axhline(y=HCY, color='gray', lw=0.5, ls='--', zorder=0)
    ax.axvline(x=HCX, color='gray', lw=0.5, ls='--', zorder=0)


def draw_initial_stones(ax, stones, alpha=1.0, label=True):
    for st in stones:
        color = '#FFD700' if st['team'] == 0 else '#C41E3A'
        ec = '#B8860B' if st['team'] == 0 else '#8B0000'
        circle = Circle((st['x'], st['y']), 0.145, color=color, ec=ec, lw=1.5, alpha=alpha, zorder=10)
        ax.add_patch(circle)
        if label:
            t = 'M' if st['team'] == 0 else 'O'
            ax.text(st['x'], st['y'], t, ha='center', va='center', fontsize=7, fontweight='bold', zorder=11)


def draw_result_stones(ax, result_stones, initial_stones, alpha=0.9):
    init_set = set((s['team'], s['index']) for s in initial_stones)
    for st in result_stones:
        is_new = (st['team'], st['index']) not in init_set
        color = '#FFD700' if st['team'] == 0 else '#C41E3A'
        ec_color = '#00AA00' if is_new else ('#B8860B' if st['team'] == 0 else '#8B0000')
        lw = 2.5 if is_new else 1.2
        circle = Circle((st['x'], st['y']), 0.145, color=color, ec=ec_color, lw=lw, alpha=alpha, zorder=10)
        ax.add_patch(circle)
        if is_new:
            ax.text(st['x'], st['y'], 'NEW', ha='center', va='center', fontsize=5, fontweight='bold',
                    color='#006600', zorder=11)


# ========== Figure 1: 着弾点マップ（重なり注記付き）==========
fig1, ax1 = plt.subplots(figsize=(10, 10))
draw_house(ax1, y_range=4.5)
draw_initial_stones(ax1, initial)

# 重なりグループを検出（0.15m以内を同一グループとする）
landing_points = []
for c in candidates:
    if c['new_stone_found']:
        landing_points.append((c['new_stone_x'], c['new_stone_y'], c))

# 重なりグループ化
groups = []
used = set()
for i, (x1, y1, c1) in enumerate(landing_points):
    if i in used:
        continue
    group = [c1]
    used.add(i)
    for j, (x2, y2, c2) in enumerate(landing_points):
        if j in used:
            continue
        if abs(x1 - x2) < 0.2 and abs(y1 - y2) < 0.2:
            group.append(c2)
            used.add(j)
    groups.append(group)

# 各グループをプロット
for group in groups:
    cx = np.mean([c['new_stone_x'] for c in group])
    cy = np.mean([c['new_stone_y'] for c in group])

    if len(group) == 1:
        c = group[0]
        color = CLUSTER_COLORS[c['cluster_id'] % len(CLUSTER_COLORS)]
        marker = TYPE_MARKERS.get(c['type'], 'o')
        size = 150 if c['is_medoid'] else 60
        ec = 'black' if c['is_medoid'] else 'gray'
        lw = 2.0 if c['is_medoid'] else 0.5
        ax1.scatter(c['new_stone_x'], c['new_stone_y'], c=color, marker=marker,
                    s=size, edgecolors=ec, linewidths=lw, zorder=5, alpha=0.8)
    else:
        # 重なりグループ: 円で囲んで注記
        for idx_in_group, c in enumerate(group):
            angle = 2 * np.pi * idx_in_group / len(group)
            jitter_r = 0.08 * min(len(group), 6)
            jx = cx + jitter_r * np.cos(angle)
            jy = cy + jitter_r * np.sin(angle)
            color = CLUSTER_COLORS[c['cluster_id'] % len(CLUSTER_COLORS)]
            marker = TYPE_MARKERS.get(c['type'], 'o')
            size = 120 if c['is_medoid'] else 50
            ec = 'black' if c['is_medoid'] else 'gray'
            lw = 2.0 if c['is_medoid'] else 0.5
            ax1.scatter(jx, jy, c=color, marker=marker,
                        s=size, edgecolors=ec, linewidths=lw, zorder=5, alpha=0.8)

        # グループを囲む円 + 個数注記
        radius = 0.12 * len(group) + 0.1
        circle = Circle((cx, cy), radius, fill=False, ec='#555555', lw=1.0, ls=':', zorder=4, alpha=0.6)
        ax1.add_patch(circle)
        # タイプ内訳を表示
        type_counts = Counter(c['type'] for c in group)
        type_str = '/'.join(f"{t[0]}{n}" for t, n in sorted(type_counts.items()))
        ax1.annotate(f"{len(group)}shots\n({type_str})", (cx, cy - radius - 0.05),
                     fontsize=6, ha='center', va='top', color='#333333',
                     bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='gray', alpha=0.8))

# 歩/DCの最良手ハイライト
pool_c = candidates[pool_best]
dc_c = candidates[dc_best]
if pool_c['new_stone_found']:
    ax1.scatter(pool_c['new_stone_x'], pool_c['new_stone_y'], c='none', marker='o',
                s=500, edgecolors='blue', linewidths=3, zorder=15)
    ax1.annotate(f"Pool best\n{pool_c['label']}\nscore={pool_c['score']:.0f}",
                 (pool_c['new_stone_x'], pool_c['new_stone_y']),
                 textcoords="offset points", xytext=(-60, 25), fontsize=8, color='blue',
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                 bbox=dict(boxstyle='round', fc='#E0E8FF', ec='blue', alpha=0.9))
if dc_c['new_stone_found'] and dc_best != pool_best:
    ax1.scatter(dc_c['new_stone_x'], dc_c['new_stone_y'], c='none', marker='D',
                s=400, edgecolors='red', linewidths=3, zorder=15)
    ax1.annotate(f"DC best\n{dc_c['label']}\nscore={dc_c['score']:.0f}",
                 (dc_c['new_stone_x'], dc_c['new_stone_y']),
                 textcoords="offset points", xytext=(40, -25), fontsize=8, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 bbox=dict(boxstyle='round', fc='#FFE0E0', ec='red', alpha=0.9))

# 凡例
from matplotlib.lines import Line2D
legend_elems = []
for t, m in TYPE_MARKERS.items():
    cnt = sum(1 for c in candidates if c['type'] == t)
    if cnt > 0:
        legend_elems.append(Line2D([0], [0], marker=m, color='w', markerfacecolor='gray',
                                   markersize=8, label=f'{t} ({cnt})'))
legend_elems.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                           markeredgecolor='black', markeredgewidth=2, markersize=10, label='Medoid'))
legend_elems.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
                           markeredgecolor='#B8860B', markersize=10, label='My stone (T0)'))
legend_elems.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#C41E3A',
                           markeredgecolor='#8B0000', markersize=10, label='Opp stone (T1)'))
ax1.legend(handles=legend_elems, loc='upper left', fontsize=8, framealpha=0.9)

ax1.set_title(f'{state_name}: {len(candidates)}候補 → {k}クラスタ (保持率20%)\n'
              f'M=自石, O=相手石, 色=クラスタ, 形=タイプ, 点線円=重なりグループ', fontsize=11)
ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
fig1.tight_layout()
fig1.savefig('detail_landing_points.png', dpi=150)
print("Saved: detail_landing_points.png")


# ========== Figure 2: クラスタ別の着弾点マップ ==========
cols = 4
rows = (n_clusters + cols - 1) // cols
fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
axes_flat = np.array(axes2).flatten()

for ci in range(len(axes_flat)):
    ax = axes_flat[ci]
    if ci < n_clusters:
        draw_house(ax)
        draw_initial_stones(ax, initial, alpha=0.3, label=False)

        members = [c for c in candidates if c['cluster_id'] == ci]
        medoid = [c for c in members if c['is_medoid']]

        # メンバーの着弾点
        for c in members:
            if not c['new_stone_found']:
                continue
            marker = TYPE_MARKERS.get(c['type'], 'o')
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            if c['is_medoid']:
                ax.scatter(c['new_stone_x'], c['new_stone_y'], c=color, marker=marker,
                          s=200, edgecolors='black', linewidths=2, zorder=6)
                # メドイドの結果盤面のストーンも表示
                draw_result_stones(ax, c['result_stones'], initial, alpha=0.6)
            else:
                ax.scatter(c['new_stone_x'], c['new_stone_y'], c=color, marker=marker,
                          s=60, edgecolors='gray', linewidths=0.5, zorder=5, alpha=0.7)

        # タイプ内訳
        type_counts = Counter(c['type'] for c in members)
        type_str = ', '.join(f"{t}:{n}" for t, n in sorted(type_counts.items()))

        # Pool bestがこのクラスタ内にいるか
        pool_in = any(c['index'] == pool_best for c in members)
        dc_in = any(c['index'] == dc_best for c in members)
        marks = []
        if pool_in:
            marks.append('Pool best')
        if dc_in:
            marks.append('DC best')
        mark_str = ' | '.join(marks) if marks else ''

        title_color = 'blue' if pool_in else ('red' if dc_in else 'black')
        ax.set_title(f"Cluster {ci} ({len(members)}members)\n{type_str}"
                     + (f"\n{mark_str}" if mark_str else ""),
                     fontsize=8, color=title_color, fontweight='bold' if marks else 'normal')
    else:
        ax.set_visible(False)

fig2.suptitle(f'{state_name}: クラスタ別の着弾点と代表手(Medoid)の結果盤面 (K={k})', fontsize=13, y=1.01)
fig2.tight_layout()
fig2.savefig('detail_cluster_map.png', dpi=150, bbox_inches='tight')
print("Saved: detail_cluster_map.png")


# ========== Figure 3: クラスタ内類似性の検証 ==========
# 大きいクラスタを3つ選び、各クラスタ内の結果盤面を3〜4つ並べる
cluster_sizes = Counter(c['cluster_id'] for c in candidates)
top_clusters = [cid for cid, _ in cluster_sizes.most_common(3)]

fig3, axes3 = plt.subplots(3, 4, figsize=(16, 13))

for row, cid in enumerate(top_clusters):
    members = [c for c in candidates if c['cluster_id'] == cid]
    # メドイドを先頭にして最大4つ表示
    medoids_first = sorted(members, key=lambda c: -c['is_medoid'])[:4]

    type_counts = Counter(c['type'] for c in members)
    type_str = ', '.join(f"{t}:{n}" for t, n in sorted(type_counts.items()))

    for col in range(4):
        ax = axes3[row][col]
        if col < len(medoids_first):
            c = medoids_first[col]
            draw_house(ax)
            draw_result_stones(ax, c['result_stones'], initial)

            border_color = 'black'
            if c['index'] == pool_best:
                border_color = 'blue'
            elif c['index'] == dc_best:
                border_color = 'red'

            role = ""
            if c['is_medoid']:
                role = " [MEDOID]"
            if c['index'] == pool_best:
                role += " [Pool best]"
            if c['index'] == dc_best:
                role += " [DC best]"

            ax.set_title(f"{c['label']}\nscore={c['score']:.0f}{role}", fontsize=8,
                        color=border_color, fontweight='bold' if role else 'normal')
            # 枠色でクラスタを表現
            for spine in ax.spines.values():
                spine.set_edgecolor(CLUSTER_COLORS[cid % len(CLUSTER_COLORS)])
                spine.set_linewidth(3)
        else:
            ax.set_visible(False)

    # 行ラベル
    axes3[row][0].set_ylabel(f"Cluster {cid}\n({len(members)} members)\n{type_str}",
                              fontsize=9, fontweight='bold', rotation=0, labelpad=80, va='center')

fig3.suptitle(f'{state_name}: クラスタ内の結果盤面比較\n'
              f'同一クラスタ内のショットは類似した盤面を生み出すか？ (緑枠=新石)', fontsize=13)
fig3.tight_layout()
fig3.savefig('detail_cluster_similarity.png', dpi=150, bbox_inches='tight')
print("Saved: detail_cluster_similarity.png")


# ========== Figure 4: 歩 vs DC 最良手の結果盤面比較 ==========
fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5.5))

# 初期盤面
draw_house(axes4[0])
draw_initial_stones(axes4[0], initial)
axes4[0].set_title(f"初期盤面\n{state_name}\nshot={data.get('initial_stones', [{}])[0].get('index', '?')}",
                   fontsize=11)

# 歩の最良手
draw_house(axes4[1])
draw_result_stones(axes4[1], pool_c['result_stones'], initial)
axes4[1].set_title(f"歩(Pool) の最良手\n{pool_c['label']}\nType={pool_c['type']}, Score={pool_c['score']:.0f}",
                   fontsize=11, color='blue')
for spine in axes4[1].spines.values():
    spine.set_edgecolor('blue')
    spine.set_linewidth(2)

# DCの最良手
draw_house(axes4[2])
draw_result_stones(axes4[2], dc_c['result_stones'], initial)
axes4[2].set_title(f"DeltaClustered の最良手\n{dc_c['label']}\nType={dc_c['type']}, Score={dc_c['score']:.0f}",
                   fontsize=11, color='red')
for spine in axes4[2].spines.values():
    spine.set_edgecolor('red')
    spine.set_linewidth(2)

fig4.suptitle(f'初期盤面 → 最良手の結果盤面比較 (保持率20%)', fontsize=13)
fig4.tight_layout()
fig4.savefig('detail_best_comparison.png', dpi=150)
print("Saved: detail_best_comparison.png")

print("\nAll done.")
print(f"  Candidates: {len(candidates)}, Clusters: {n_clusters}")
print(f"  Pool best: [{pool_best}] {pool_c['label']} (score={pool_c['score']:.0f})")
print(f"  DC best:   [{dc_best}] {dc_c['label']} (score={dc_c['score']:.0f})")
print(f"  Same shot: {pool_best == dc_best}")
