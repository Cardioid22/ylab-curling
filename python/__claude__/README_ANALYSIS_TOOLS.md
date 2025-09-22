# MCTS Curling AI Analysis Tools

このディレクトリには、ylab-curlingプロジェクトで生成されたMCTSデータを詳細に分析するための高度な分析ツールが含まれています。既存の`controller.py`を大幅に拡張し、より深い洞察を得ることができます。

## 📊 分析ツール一覧

### 1. `comprehensive_mcts_analyzer.py`
**総合MCTS性能比較分析**

- **機能**:
  - クラスター化MCTS vs 全グリッドMCTSの詳細性能比較
  - 統計的有意性検定による性能差の評価
  - 計算効率vs精度のトレードオフ分析
  - 複数グリッドサイズでの性能スケーリング分析

- **出力**:
  - `mcts_analysis_output/` ディレクトリ
  - 性能比較グラフ (`*_performance_comparison.png`)
  - 戦略一致度分析 (`*_strategy_agreement.png`)
  - 総合分析レポート (`comprehensive_analysis_report.txt`)

- **使用方法**:
```bash
python comprehensive_mcts_analyzer.py
```

### 2. `strategy_evolution_tracker.py`
**戦略進化パターン分析**

- **機能**:
  - ゲーム進行に伴う戦略選択の変化パターン分析
  - 重要ターン（戦略転換点）の自動検出
  - 戦略遷移行列の生成
  - ゲームフェーズ別戦略特徴の抽出

- **出力**:
  - `strategy_evolution_output/` ディレクトリ
  - 戦略進化グラフ (`*_strategy_evolution.png`)
  - クリティカルターン分析 (`*_critical_turns.png`)
  - 戦略遷移ヒートマップ (`*_strategy_transitions.png`)
  - 進化分析レポート (`strategy_evolution_report.txt`)

- **使用方法**:
```bash
python strategy_evolution_tracker.py
```

### 3. `spatial_exploration_analyzer.py`
**空間探索パターン分析**

- **機能**:
  - カーリングアイス上での空間探索密度分析
  - 戦略的ホットスポット（頻繁選択領域）の特定
  - ゾーン別戦略傾向分析（ハウス、ガードゾーン等）
  - 探索効率メトリクスの算出

- **出力**:
  - `spatial_exploration_output/` ディレクトリ
  - 空間密度ヒートマップ (`*_spatial_exploration.png`)
  - ホットスポット分析 (`*_hotspots.png`)
  - 探索効率比較 (`*_efficiency.png`)
  - 空間分析レポート (`spatial_exploration_report.txt`)

- **使用方法**:
```bash
python spatial_exploration_analyzer.py
```

### 4. `clustering_optimization_analyzer.py`
**クラスタリング最適化分析**

- **機能**:
  - シルエット係数、Calinski-Harabasz係数等による品質評価
  - 最適クラスター数の自動決定（エルボー法、シルエット法）
  - クラスター安定性の時系列分析
  - 異なるクラスタリングアルゴリズムの比較

- **出力**:
  - `clustering_optimization_output/` ディレクトリ
  - 品質進化グラフ (`*_quality_evolution.png`)
  - 最適クラスター分析 (`*_optimal_clusters.png`)
  - アルゴリズム比較 (`*_algorithm_comparison.png`)
  - 安定性分析 (`*_stability_analysis.png`)
  - 最適化レポート (`clustering_optimization_report.txt`)

- **使用方法**:
```bash
python clustering_optimization_analyzer.py
```

### 5. `predictive_accuracy_analyzer.py`
**予測精度と実戦効果分析**

- **機能**:
  - MCTS予測スコアと実際の性能の相関分析
  - 予測一致率と信頼性の評価
  - 時間経過による予測精度の変化分析
  - アンサンブル予測手法の効果検証

- **出力**:
  - `predictive_accuracy_output/` ディレクトリ
  - 予測一貫性分析 (`*_prediction_consistency.png`)
  - アルゴリズム一致度 (`*_algorithm_agreement.png`)
  - 時系列精度変化 (`*_temporal_accuracy.png`)
  - エラーパターン分析 (`*_error_patterns.png`)
  - 精度分析レポート (`predictive_accuracy_report.txt`)

- **使用方法**:
```bash
python predictive_accuracy_analyzer.py
```

### 6. `interactive_performance_dashboard.py`
**インタラクティブパフォーマンスダッシュボード**

- **機能**:
  - Webベースのインタラクティブ分析UI
  - リアルタイムパラメータ調整
  - 複数実験の同時比較表示
  - 動的フィルタリング機能

- **依存関係**:
```bash
pip install streamlit plotly pandas numpy seaborn
```

- **使用方法**:
```bash
streamlit run interactive_performance_dashboard.py
```

## 🔧 セットアップと使用方法

### 前提条件

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn kneed
```

### データ構造要件

分析ツールは以下のディレクトリ構造を想定しています：

```
remote_log/
├── Grid_10x10/
│   ├── Iter_43/
│   │   ├── root_children_score_clustered_*.csv
│   │   ├── root_children_score_allgrid_*.csv
│   │   └── best_shot_comparison_*.csv
│   └── MCTS_Output_ClusteringId_*/
│       └── cluster_ids_*.csv
├── Grid_20x20/
└── Iter_*/
```

### 基本的な使用手順

1. **個別分析の実行**:
```bash
# 各分析ツールを個別に実行
python comprehensive_mcts_analyzer.py
python strategy_evolution_tracker.py
python spatial_exploration_analyzer.py
python clustering_optimization_analyzer.py
python predictive_accuracy_analyzer.py
```

2. **統合ダッシュボードの起動**:
```bash
streamlit run interactive_performance_dashboard.py
```

3. **結果の確認**:
   - 各ツールが生成する`*_output/`ディレクトリを確認
   - `.png`ファイルで可視化結果を確認
   - `.txt`レポートファイルで数値分析結果を確認

## 📈 分析の特徴

### データ統合処理
- remote_logの全実験データを自動検出・統合
- 異なるグリッドサイズ、反復数での横断分析
- 欠損データの自動補完と異常値検出

### 高度な統計分析
- Mann-Whitney U検定による性能比較
- 信頼区間付き効果サイズ計算
- 相関分析と回帰分析

### 機械学習活用
- クラスタリング品質の自動評価
- 最適パラメータの自動探索
- 異常パターンの自動検出

### 可視化の強化
- インタラクティブなPlotlyグラフ
- カーリングアイス上での3D空間分析
- アニメーション付き時系列分析

## 🎯 従来のcontroller.pyとの違い

| 従来 | 新ツール群 |
|------|------------|
| 基本的なクラスター一致性分析 | 6つの専門分析ツール |
| 静的なmatplotlibグラフ | インタラクティブなPlotlyダッシュボード |
| 単一実験の分析 | 複数実験の横断比較 |
| 手動パラメータ設定 | 自動最適化と推奨 |
| 限定的な統計分析 | 包括的な統計・機械学習分析 |

## 📚 出力ファイルの説明

### CSV出力
- `*_performance_comparison.csv`: 性能比較データ
- `*_strategy_transitions.csv`: 戦略遷移データ
- `*_spatial_analysis.csv`: 空間分析データ
- `*_clustering_quality.csv`: クラスタリング品質データ

### グラフ出力
- `.png`: 高解像度の分析グラフ
- インタラクティブHTML（ダッシュボード使用時）

### レポート出力
- `.txt`: 数値分析結果の詳細レポート
- 統計的有意性、推奨パラメータ、改善提案を含む

## 🚀 今後の拡張可能性

1. **リアルタイム分析**: ゲーム進行中のライブ分析
2. **AI対戦分析**: 異なるAI間の対戦結果分析
3. **パラメータ最適化**: ベイズ最適化による自動調整
4. **予測モデル**: 次手予測の機械学習モデル
5. **3D可視化**: WebGLを使用した3次元分析

これらのツールにより、MCTSカーリングAIの性能を多角的に分析し、アルゴリズムの改善点を特定することができます。