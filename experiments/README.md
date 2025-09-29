# クラスタリング MCTS 効率性実験

このディレクトリには、クラスタリング MCTS の効率性を検証するための実験プログラムが含まれています。

## ファイル構成

### 実験実行関連

- `experiment_runner.h/cpp` - メイン実験実行クラス
- `efficiency_experiment.h/cpp` - 効率性比較実験の実装
- `statistical_analysis.h/cpp` - 統計分析機能

### MCTS 拡張

- `mcts_with_tracking.h/cpp` - 正解手発見を追跡する MCTS クラス
- `ground_truth_finder.h/cpp` - 正解手決定システム

## 実験の実行方法

### 1. ビルド

```bash
mkdir build
cd build
cmake ..
make
```

### 2. 実験実行

```bash
# 通常のゲームモード
./ylab_client localhost 10000

# 実験モード
./ylab_client --experiment
```

## 実験の流れ

1. **正解手決定**: 長時間 MCTS で各盤面の最適解を決定
2. **比較実験**: クラスタリング版と全グリッド版で正解発見までの探索数を比較
3. **統計分析**: 効率比の平均、標準偏差、有意性検定
4. **結果出力**: CSV 形式で詳細データを出力

## 出力ファイル

実験結果は `efficiency_experiment_YYYYMMDD_HHMMSS/` フォルダに保存されます：

- `statistical_summary.csv` - 統計サマリー
- `detailed_results.csv` - 全実験の詳細結果
- `efficiency_histogram.csv` - 効率比の分布データ
- `learning_curves.csv` - 学習曲線データ
- `success_rate_analysis.csv` - 成功率分析

## 実験設定

`ExperimentConfig`構造体で以下のパラメータを調整可能：

- `max_iterations`: 最大探索数（デフォルト: 5000）
- `max_time`: 最大時間（デフォルト: 180 秒）
- `trials_per_state`: 状態毎の試行回数（デフォルト: 10）
- `ground_truth_iterations`: 正解手決定用の探索数（デフォルト: 50000）

## 期待される結果

- **効率比 < 1.0**: クラスタリング版がより少ない探索数で正解発見
- **p < 0.05**: 統計的有意な差
- **高い成功率**: 両手法で安定した正解発見

この実験により、クラスタリングアプローチの定量的な有効性を証明できます。
