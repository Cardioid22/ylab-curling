# 研究の現状まとめ（引き継ぎ用）

## 1. 研究目的

カーリングAIにおいて、候補手生成後にクラスタリングで冗長な候補を除去し、MCロールアウトで評価する候補数を削減する手法の有効性を検証する。

## 2. 大渡さんのプログラム（歩 / Ayumu）

### 概要
- 論文: 大渡勝己, 田中哲朗. "モンテカルロ木探索によるデジタルカーリングAI" GPW 2016
- ソースコード: https://github.com/u-tokyo-gps-tanaka-lab/gpw2016
  - パス: `projects/curling/c/ayumu/`

### パイプライン
```
1. genRootVMove() で候補手を生成（20〜60手）
   - 13種のショットタイプ: Draw, Hit, Freeze, Guard, Peel, ComeAround, PostGuard, Pass等
   - 盤面の石の配置に応じて動的に候補を生成（戦略的知識ベース）
2. 各候補手をMCロールアウトで評価（じりつくん方式 = 深さ1のフラットMC）
   - UCB1で有望な手に適応的にロールアウトを配分
   - ロールアウト中は学習済みポリシー関数(gPolicy)で手を選択
3. 平均報酬最大の候補を最良手として選出
```

### 公開されている学習済みモデル
リポジトリ: `data/curling/params/` に以下のファイルがある

| ファイル | 内容 |
|---|---|
| `policy_param.dat` | ロールアウト用ポリシー関数のパラメータ（42,072パラメータ） |
| `estimator_param.dat` | 静的盤面評価関数のパラメータ（34,816パラメータ） |
| `tscore_to_escore.dat` | 現在スコア→エンドスコア変換テーブル |
| `to2score_to_escore.dat` | 相手セカンドスコア→エンドスコア変換テーブル |
| `guard_prob.dat` | ガード確率テーブル |
| `param_double.dat` | ダブルテイクアウト事前計算テーブル |
| `param_fronttake.dat` | フロントテイクアウト事前計算テーブル |

### gPolicy（ロールアウト用ポリシー関数）の詳細
- **構造**: 線形ソフトマックスモデル（ニューラルネットではない）
- **パラメータ数**: 42,072
- **温度**: 0.8
- **入力特徴量**（5グループ）:
  1. POL_1STONE: 石の絶対位置（16ゾーン離散化 × 手タイプ × ターン番号6段階）
  2. POL_2STONES: 石ペアの相対位置（16カテゴリ離散化）
  3. POL_2SMOVE: 2石ムーブの内部関係
  4. POL_END_SCORE_TO_TYPE: 現在スコア × 手タイプ
  5. POL_2ND_SCORE_TO_TYPE: 相手セカンドスコア × 手タイプ
- **出力**: 各候補手の選択確率（ソフトマックス）
- **学習方法**: 教師あり学習（方策勾配）
  - データ: 230万ショット（自己対戦記録）
  - 最適化: SGD + L1/L2正則化（FOBOS）
  - エポック: 50、ミニバッチ: 256
- **ゾーン離散化（getPositionIndex）**: 石の位置を16ゾーンに分類
  - ハウス内（内側/中間/外側リング × 左右）、ガードゾーン、遠方等
- **相対位置ラベル（labelRelPos）**: 2石間の関係を16カテゴリに分類
  - アプローチ角度、スピン方向、前後関係に基づく

### gEstimator（静的盤面評価関数）の詳細
- **構造**: 線形ソフトマックスモデル
- **パラメータ数**: 34,816
- **主要な有効特徴**: No.1石の位置 + ガード状態のみ（他の特徴は無効化済み）
- **出力**: エンドスコアの確率分布
- **学習データ**: 270万ショット

### 学習データの生成方法
- 自己対戦: 3,200試合（8種のAI組み合わせ × 400試合）
- 各試合 ≈ 160ショット → 約51万ショット
- 追加自己対戦を含め230万ショット
- 保存形式: `shotlog_pol.bin`（バイナリ）
- 対戦組み合わせ: mf(MCTS full), mpf(MCTS policy full), p(pure random), pp(pure policy)
- 記録アーカイブ: `used_record.zip` がリポジトリに含まれる

### ソースコード主要ファイル
| ファイル | 内容 |
|---|---|
| `ayumu/mc/mcPlayer.hpp` | MCTSトップレベルコントローラ |
| `ayumu/mc/root.hpp` | ルートノード、UCB1選択 |
| `ayumu/mc/leaf.hpp` | ロールアウト（doUCT, doSimulation） |
| `ayumu/move/generator.hpp` | 候補手生成（genRootVMove, genChosenVMove） |
| `ayumu/policy/policy.hpp` | ポリシー特徴量計算 |
| `ayumu/eval/estimator.hpp` | 評価関数特徴量計算 |
| `ayumu/pglearner.cc` | ポリシー学習メイン |
| `ayumu/evlearner.cc` | 評価関数学習メイン |
| `ayumu/logic/logic.hpp` | getPositionIndex, labelRelPos等の離散化 |
| `ayumu/ayumu.hpp` | gPolicy, gEstimatorのグローバル宣言 |
| `jiritsu/search.hpp` | 実際に使われているMC探索（じりつくん方式） |
| `common/util/softmaxPolicy.hpp` | SoftmaxPolicy/SoftmaxPolicyLearnerクラス |

### 歩が使っていないが実装されている機能
- マルチスレッド探索（コメントアウト済み）
- 置換表（空場用 + 通常用）
- 速度誤差のガウス積分（20×20サブグリッド）

### 移植時の注意点
- 歩は独自物理シミュレータを使用。ylab-curlingはDigitalCurling3を使用。座標系が異なる可能性あり
- 特徴量の16ゾーン離散化やlabelRelPosの実装が必要
- 13種ショットタイプとylab-curlingのShotGeneratorとの対応付けが必要

---

## 3. 提案手法の現状

### パイプライン
```
1. ShotGenerator で候補手を生成（40〜100手）
   - Draw: 4×4グリッド × 2スピン = 32手（固定）
   - PreGuard: 3位置 × 2スピン = 6手（固定）
   - Hit: 相手石ごと × 重さ(TOUCH/WEAK/MIDDLE/STRONG) × 2スピン（可変）
   - Freeze: ハウス内相手石 × 2スピン（可変）
   - PostGuard: 相手石ごと × 2スピン（可変）
   - Pass: 1手（固定）
2. 全候補をシミュレーション → 結果盤面を取得
3. Delta距離関数v2 で結果盤面間の距離テーブルを構築（O(N²)）
4. 階層的クラスタリング（平均連結法）で K個のクラスタに分類
5. 各クラスタのメドイド（代表手）を選出 → K候補に絞る
6. K候補をMCロールアウトで評価 → 最良手選出
```

### Delta距離関数v2
2つのショットの結果盤面を、入力盤面からの「変化量」として比較する距離関数。

| 要素 | 重み | 何を見ているか |
|---|---|---|
| 新石の位置差 | NEW_STONE_WEIGHT = 4.0 | 新しい石がどれだけ離れた場所に止まったか |
| 既存石の移動差 | MOVED_STONE_WEIGHT = 2.0 | 既存石がどれだけ違う動きをしたか |
| 石の有無 | PENALTY_EXISTENCE = 30.0 | 一方で石が残り他方で消えた |
| ゾーン差 | PENALTY_ZONE = 12.0 | ハウス内/ガードゾーン/遠方のゾーンが違う |
| インタラクション差 | PENALTY_INTERACTION = 15.0 | 接触型(Hit) vs 非接触型(Draw) |
| 盤面スコア差 | SCORE_WEIGHT = 8.0 | 結果のスコアが異なる |
| 近接度差 | PROXIMITY_WEIGHT = 5.0 | Freeze(密着) vs Draw(離れて配置) |
| No.1石チーム差 | 10.0 | ティー最近石のチームが異なる |

注意: 距離関数内にevaluateBoard()が組み込まれている（SCORE_WEIGHT=8.0）

### 実験結果（確定済み、200盤面、静的評価）

#### 最良手一致率
| 保持率 | Exact Match | Same Type | Avg|ScoreDiff| |
|---|---|---|---|
| 10% | 31% | 68% | 0.06 |
| 20% | 49% | 81% | 0.01 |
| 30% | 64% | 86% | 0.00 |
| 50% | 88% | 100% | 0.00 |
| 70% | 100% | 100% | 0.00 |

#### クラスタ純度 & 同一クラスタ分析
| 保持率 | Avg Purity | Same Cluster | Exact Match |
|---|---|---|---|
| 10% | 0.71 | 86% | 31% |
| 20% | 0.78 | 97% | 49% |
| 30% | 0.83 | 100% | 64% |
| 50% | 0.96 | 100% | 88% |
| 70% | 1.00 | 100% | 100% |

- 保持率30%でScoreDiff=0.00、Same Cluster=100%
- Purity 0.71〜0.83 → クラスタはタイプ分類とは異なるグルーピング
- 評価関数は整数値（カーリング得点ルール）のため、連続値での差は未検証

### 未解決の課題
1. **MCロールアウトのポリシー関数がない**
   - ランダムグリッドポリシーでは候補手の影響がノイズに埋もれる（mean≈0問題）
   - ShotGeneratorベースのポリシーはgenerateCandidates()のコストが高すぎる
   - ルールベースポリシー or 大渡さんのモデル移植 or 自前学習が必要
2. **距離テーブルのO(N²)コスト**（候補数増加時の課題）
3. **評価関数の粒度**（整数スコアでは候補間の差が出にくい）

### 実装済みファイル
| ファイル | 内容 |
|---|---|
| `src/shot_generator.h/cpp` | ShotGenerator（歩相当の候補手生成） |
| `src/mcts.h/cpp` | MCTS（DeltaClustered統合済み） |
| `src/simulator.h/cpp` | SimulatorWrapper（物理シミュレーション） |
| `src/clustering.h/cpp` | 旧クラスタリング |
| `src/clustering-v2.h/cpp` | ClusteringV2 |
| `experiments/pool_clustering_experiment.h/cpp` | クラスタリング品質検証実験（200盤面） |
| `experiments/depth1_mcts_experiment.h/cpp` | 深さ1フラットMC実験（ポリシー問題で中断） |
| `analysis/visualize_clustering_detail.py` | クラスタリング可視化 |
| `analysis/plot_clustering_effectiveness.py` | 集計グラフ |

### 実行方法
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release

# クラスタリング品質検証（200盤面、数分で完了）
./ylab_client --pool-clustering

# 深さ1MC実験（ポリシー問題のため現状使えない）
./ylab_client --depth1-mcts
```

---

## 4. 次にやるべきこと（優先順）

1. **MCロールアウト用ポリシー関数の準備**
   - 選択肢A: 大渡さんのgPolicyを移植（1〜2週間、特徴量実装が主な工作）
   - 選択肢B: 自前で学習（2〜3週間、自己対戦データ生成が必要）
   - 選択肢C: ルールベースポリシー（2〜3日、ShotGenerator+静的評価で確率的選択）
2. **深さ1MC実験の完遂**（ポリシー準備後）
3. **距離関数の重み微調整**
4. **論文執筆**（IPSJ SIG-GI 58, TCGA2026, 台北, 5/15-16, 論文締切4/20）
