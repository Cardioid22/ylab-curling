"""
クラスタリング効果の可視化
3つのグラフを生成:
1. 保持率 vs Exact Match / Same Type / Same Cluster（DeltaClustered）
2. 保持率 vs Cluster Purity
3. 保持率ごとの盤面内訳（EXACT / Same Cluster / Miss の積み上げ）
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'MS Gothic'

csv_path = "../build/Release/experiments/pool_clustering_results/three_method_comparison.csv"
df = pd.read_csv(csv_path)

# DeltaClustered のみ抽出
dc = df[df['method'] == 'DeltaClustered'].copy()
dc['ratio_pct'] = (dc['ratio'] * 100).round().astype(int)

# --- 保持率別の集計 ---
agg = dc.groupby('ratio_pct').agg(
    n=('same_shot', 'count'),
    exact_match=('same_shot', 'sum'),
    same_type=('same_type', 'sum'),
    same_cluster=('pool_best_in_same_cluster', 'sum'),
    avg_score_diff=('score_diff', lambda x: x.abs().mean()),
    avg_purity=('weighted_purity', 'mean'),
).reset_index()

agg['exact_pct'] = agg['exact_match'] / agg['n'] * 100
agg['same_type_pct'] = agg['same_type'] / agg['n'] * 100
agg['same_cluster_pct'] = agg['same_cluster'] / agg['n'] * 100
agg['miss_pct'] = 100 - agg['same_cluster_pct']

ratios = agg['ratio_pct'].values

# ========== Figure 1: 保持率 vs 一致率 ==========
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(ratios, agg['exact_pct'], 'o-', color='#2196F3', linewidth=2, markersize=8, label='Exact Match')
ax1.plot(ratios, agg['same_type_pct'], 's-', color='#FF9800', linewidth=2, markersize=8, label='Same Type')
ax1.plot(ratios, agg['same_cluster_pct'], '^-', color='#4CAF50', linewidth=2, markersize=8, label='Same Cluster')

ax1.set_xlabel('保持率 (%)', fontsize=13)
ax1.set_ylabel('一致率 (%)', fontsize=13)
ax1.set_title('DeltaClustered: 保持率と歩の最良手との一致率 (N=200)', fontsize=14)
ax1.legend(fontsize=11, loc='lower right')
ax1.set_xticks(ratios)
ax1.set_ylim(0, 105)
ax1.grid(True, alpha=0.3)

# 数値ラベル
for i, r in enumerate(ratios):
    ax1.annotate(f"{agg['exact_pct'].iloc[i]:.0f}%", (r, agg['exact_pct'].iloc[i]),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#2196F3')
    ax1.annotate(f"{agg['same_cluster_pct'].iloc[i]:.0f}%", (r, agg['same_cluster_pct'].iloc[i]),
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#4CAF50')

fig1.tight_layout()
fig1.savefig('clustering_agreement_rates.png', dpi=150)
print("Saved: clustering_agreement_rates.png")

# ========== Figure 2: 保持率 vs Purity ==========
fig2, ax2 = plt.subplots(figsize=(8, 5))
bars = ax2.bar(ratios, agg['avg_purity'] * 100, width=6, color='#9C27B0', alpha=0.8, edgecolor='white')
ax2.set_xlabel('保持率 (%)', fontsize=13)
ax2.set_ylabel('平均クラスタ純度 (%)', fontsize=13)
ax2.set_title('クラスタ純度: タイプ分類との一致度 (N=200)', fontsize=14)
ax2.set_xticks(ratios)
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, agg['avg_purity']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

# 解釈ライン
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(ratios[-1] + 2, 101, '1.0 = タイプ分類と同等', fontsize=9, color='red', va='bottom')

fig2.tight_layout()
fig2.savefig('cluster_purity.png', dpi=150)
print("Saved: cluster_purity.png")

# ========== Figure 3: 積み上げ棒グラフ（内訳） ==========
fig3, ax3 = plt.subplots(figsize=(8, 5))

exact_vals = agg['exact_pct'].values
same_cluster_only = agg['same_cluster_pct'].values - agg['exact_pct'].values
miss_vals = agg['miss_pct'].values

bar_width = 6
ax3.bar(ratios, exact_vals, width=bar_width, color='#2196F3', label='Exact Match（完全一致）')
ax3.bar(ratios, same_cluster_only, width=bar_width, bottom=exact_vals,
        color='#4CAF50', label='Same Cluster（同一クラスタ内）')
ax3.bar(ratios, miss_vals, width=bar_width, bottom=exact_vals + same_cluster_only,
        color='#F44336', alpha=0.7, label='Miss（クラスタ外）')

ax3.set_xlabel('保持率 (%)', fontsize=13)
ax3.set_ylabel('盤面の割合 (%)', fontsize=13)
ax3.set_title('200盤面の内訳: 歩の最良手との関係', fontsize=14)
ax3.legend(fontsize=10, loc='upper right')
ax3.set_xticks(ratios)
ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3, axis='y')

# 数値ラベル
for i, r in enumerate(ratios):
    if miss_vals[i] > 2:
        ax3.text(r, 100 - miss_vals[i]/2, f'{miss_vals[i]:.0f}%',
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax3.text(r, exact_vals[i]/2, f'{exact_vals[i]:.0f}%',
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')

fig3.tight_layout()
fig3.savefig('clustering_breakdown.png', dpi=150)
print("Saved: clustering_breakdown.png")

plt.show()
print("\nDone.")
