# GPW2026 アブスト用 データ索引 (2026-07-27時点)

A9(ClusterValue)を主軸にしたアブストで使えるデータの所在一覧。
**すべて修正木 (エンド跨ぎ汚染バグ e9f0dde 修正後) のデータのみ引用可**。
旧run50/run50v2以前の対戦結果・regretはバグ木の計測なので使用禁止。

## ① A9の仕組み・性能（劣化なしの証明）

| 内容 | パス | 状態 |
|---|---|---|
| 6アームregret生データ (P=200, 5seed) | `reinvest_experiment/run50v2/{A1,A2,A5,A7,A8,A9}/seed_{42..46}/reinvest_results.csv` | 済 |
| 6アーム集計・検定 | `reinvest_experiment/run50v2/regret/reinvest_summary.csv`, `regret_stats.txt` | 済 (全アーム0.50-0.51で横並び) |
| A9 vs A2 直接対決 | `reinvest_experiment/run50v2/regret/` (`--pair A9,A2` で再集計可) | Δq=-0.002, ほぼ完全同値 |
| A9のクラスタ割当生データ | `reinvest_experiment/run50v2/A9/seed_{42..46}/cluster_table.csv` | 済 (e_score, cluster_value, land_x/y列あり) |
| 審判 (q_ref, K=200) | `reinvest_experiment/scorescreen/run50/referee/score_move_qtable.csv` | 済 (regret系全部で共通利用) |
| クラスタ価値の妥当性 (Spearman ρ) | (未保存、都度計算) seed42で ρ=+0.435 | **要: 5シード分計算してCSV化** |

## ② エリア価値マップ

| 内容 | パス | 状態 |
|---|---|---|
| 可視化バリエーション比較 (A-E案) | `reinvest_experiment/run50v2/A9/variants/variant_*.png` | 済 (B案=凸包に決定) |
| B案スクリプト | `scripts/plot_area_variants.py` (draw_hull) | 済、本番用に整理が必要 |
| 旧版 (散布図/ヒートマップ, 参考) | `scripts/plot_cluster_value_map.py`, `scripts/plot_area_heatmap.py` | 参考用 |
| 局面データ (盤面座標) | `test_positions50/batch_0001.csv` | 済 |

## ③ 深さ探索とクラスタの関係

| 内容 | パス | 状態 |
|---|---|---|
| A1 (d3, 修正木, 5seed) | `reinvest_experiment/run50v2/A1/seed_{42..46}/reinvest_results.csv` | 済 |
| A1d1 (d1, 修正木) seed42 | `reinvest_experiment/run50_A1d1fix/seed_42/reinvest_results.csv` | 済 |
| A1d1 (d1, 修正木) seed43-46 | `reinvest_experiment/run50v2/A1d1/seed_{43..46}/` | **実行中 (bear)** |
| d3敗因解剖 (旧, バグ木, 参考) | `reinvest_experiment/scorescreen/run50/d3_failure/` | 参考用のみ (バグ木) |
| d3敗因解剖 (修正木, seed42のみ) | `reinvest_experiment/d3fix_autopsy/`, `d3fix_autopsy_depth/` | 済、**5seed化待ち** |
| クラスタ乗り換え分析 (seed42のみ, n=35) | (未保存、都度計算スクリプト化) | **要スクリプト化+5seed化** |
| Δ曲線 (修正木, 3seed, 52局面) | `reinvest_experiment/treedepth_fix/analysis/tree_depth_curve.{csv,png}` | 済 |
| 深さ別局面セット (52局面, r=1-4) | `test_positions_depth/batch_0001.csv` | 済 |
| depth用審判 | `reinvest_experiment/depth/referee/score_move_qtable.csv` | 済 |

## ④ 使わない/参考程度に留めるデータ

| 内容 | パス | 理由 |
|---|---|---|
| 旧6アームregret表 (A7=0.554等) | `reinvest_experiment/scorescreen/run50/regret/` | バグ木の計測、無効 |
| 全自己対戦 (A7vs A1/A2/歩, 旧65.5%等) | `reinvest_experiment/selfplay2end/` | バグ木の計測、無効 |
| 修正木でのA7vsA1再戦 (42.6%, n.s.) | `reinvest_experiment/selfplay2end_v2/` | 正しいが「性能で語らない」方針のため今回は不使用 |
| max-max/エンド跨ぎバグの発見経緯 | メモリ `project_maxmax_ucb_bug.md`, `project_endcross_bug.md` | 今回は成果に含めない方針。触れるなら「検証基盤の頑健化」として一文で |
| A10 (ClusterValueDeep) | `reinvest_experiment/run50_A10fix/` 等 | 修正木での再測定なし。今回のアブストでは割愛推奨 |

## ⑤ コード・分析スクリプト参照

| 何 | パス |
|---|---|
| A9のMCTS実装 (selectClusterValue) | `experiments/reinvest_experiment.cpp` |
| クラスタリング共通処理 (distDelta) | `experiments/mcts_shared.cpp` |
| regret集計 | `scripts/aggregate_reinvest.py`, `scripts/regret_stats.py` |
| d3敗因解剖 (一致率・Δq・種別遷移) | `scripts/analyze_d3_failure.py` |
| エリア可視化 | `scripts/plot_area_variants.py` |
| Δ曲線 | `scripts/analyze_tree_depth.py`, `scripts/plot_treedepth_compare.py` |

## ⑥ 研究の背景（過去原稿）

| 何 | パス |
|---|---|
| GI58原稿 (Proposedの初出, IPSJスタイル一式) | `gi58/draft.tex`, `gi58/ipsj.cls`, `gi58/figures/` |
| プロジェクト現況ドキュメント (全実験の時系列まとめ) | `docs/PROJECT_STATUS.md` |

---
**このファイルは進行中のメモ。データが増えたら追記し、"済"↔"要"を更新すること。**
