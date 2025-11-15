#!/usr/bin/env python3
"""
シンプルな実験結果分析スクリプト

Usage:
    python analyze_simple_results.py <results.csv>

Example:
    python analyze_simple_results.py experiment_results.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results(csv_file):
    # データ読み込み
    df = pd.read_csv(csv_file)

    print("=" * 60)
    print("実験結果分析")
    print("=" * 60)
    print(f"\nデータファイル: {csv_file}")
    print(f"総テスト数: {len(df)}")

    # 両方見つかった場合のみ抽出
    valid = df[(df['Clustered_Found'] == 1) & (df['AllGrid_Found'] == 1)]

    print(f"両方が理想の手を見つけた: {len(valid)}/{len(df)}")

    if len(valid) == 0:
        print("\n警告: 有効な比較データがありません")
        return

    # 統計情報
    print("\n" + "-" * 60)
    print("探索数の比較")
    print("-" * 60)

    print(f"\nクラスタリング版:")
    print(f"  平均探索数: {valid['Clustered_Iterations'].mean():.1f}")
    print(f"  最小: {valid['Clustered_Iterations'].min()}")
    print(f"  最大: {valid['Clustered_Iterations'].max()}")

    print(f"\n全グリッド版:")
    print(f"  平均探索数: {valid['AllGrid_Iterations'].mean():.1f}")
    print(f"  最小: {valid['AllGrid_Iterations'].min()}")
    print(f"  最大: {valid['AllGrid_Iterations'].max()}")

    # 効率比
    avg_ratio = valid['Iteration_Ratio'].mean()
    print("\n" + "-" * 60)
    print("効率性の分析")
    print("-" * 60)

    print(f"\n平均効率比 (クラスタリング/全グリッド): {avg_ratio:.3f}")

    if avg_ratio < 1.0:
        reduction = (1.0 - avg_ratio) * 100
        print(f"\n✓ クラスタリング版は平均 {reduction:.1f}% 少ない探索数で理想の手に到達")
        print("  → クラスタリングは効果的です！")
    elif avg_ratio > 1.0:
        increase = (avg_ratio - 1.0) * 100
        print(f"\n✗ クラスタリング版は平均 {increase:.1f}% 多い探索数が必要")
        print("  → クラスタリングは逆効果です")
    else:
        print("\n≈ 両方法で同等の探索数")

    # 詳細表示
    print("\n" + "-" * 60)
    print("テストケース別の結果")
    print("-" * 60)

    for idx, row in valid.iterrows():
        print(f"\nState {idx+1} (End={row['End']}, Shot={row['Shot']}):")
        print(f"  クラスタリング: {row['Clustered_Iterations']} 回")
        print(f"  全グリッド: {row['AllGrid_Iterations']} 回")
        print(f"  効率比: {row['Iteration_Ratio']:.3f}")

        if row['Iteration_Ratio'] < 1.0:
            reduction = (1.0 - row['Iteration_Ratio']) * 100
            print(f"  → {reduction:.1f}% 削減")

    # グラフ作成
    create_graphs(valid)

    print("\n" + "=" * 60)
    print("分析完了")
    print("=" * 60)

def create_graphs(df):
    # 2つのグラフを作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # グラフ1: 探索数の比較（棒グラフ）
    ax1 = axes[0]
    x = range(len(df))
    width = 0.35

    ax1.bar([i - width/2 for i in x], df['Clustered_Iterations'],
            width, label='クラスタリング版', color='#3498db', alpha=0.8)
    ax1.bar([i + width/2 for i in x], df['AllGrid_Iterations'],
            width, label='全グリッド版', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('テストケース', fontsize=12)
    ax1.set_ylabel('探索数（反復回数）', fontsize=12)
    ax1.set_title('探索数の比較', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'State {i+1}' for i in x])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # グラフ2: 効率比
    ax2 = axes[1]
    colors = ['#2ecc71' if r < 1.0 else '#e74c3c' for r in df['Iteration_Ratio']]

    ax2.bar(x, df['Iteration_Ratio'], color=colors, alpha=0.8)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='同等（比=1.0）')

    ax2.set_xlabel('テストケース', fontsize=12)
    ax2.set_ylabel('効率比（クラスタリング/全グリッド）', fontsize=12)
    ax2.set_title('効率比の分布', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'State {i+1}' for i in x])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = 'experiment_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nグラフを保存しました: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_simple_results.py <results.csv>")
        print("\n例:")
        print("  python analyze_simple_results.py experiment_results.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    try:
        analyze_results(csv_file)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
