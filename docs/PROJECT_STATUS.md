# プロジェクト現況 (最終更新: 2026-07-16)

AIアシスタント (Codex / Claude / Gemini) 向けの現況同期ドキュメント。
**過去の設計文書ではなく「今」の状態を書く。実験が進んだら必ずここを更新すること。**

## 1. 研究の主線 (何を証明しようとしているか)

デジタルカーリングのMCTSにおいて、**候補手の削減（枝刈り）は「盤面類似クラスタリング」ではなく
「得点期待値 E[score] による価値スクリーン」で行うべき**、が中心的主張。
ただし盤面類似クラスタは**価値選択と組み合わせれば意思決定品質のコストほぼゼロで**、
説明可能性（エリア価値マップ）と網羅性を足せる（=Proposedの名誉回復）。

- 出口: 大会優勝 + 学会発表（GPW2026、締切7月中旬）
- 3本柱の論拠: ①regret（審判基準の手の質） ②予算スイープ ③自己対戦勝率（バイアスフリー）

## 2. アーム用語集

| アーム | 名前 (--method) | root選択 | depth>0 | 一言 |
|---|---|---|---|---|
| A1 | AllGrid | 全候補 | 全候補 | 基準 (枝刈りなし) |
| A2 | Proposed | distDeltaクラスタ+medoid | 同左 | 旧提案。価値盲目→regretはrandom並み |
| A5 | RandomK | 決定的乱択 K=ceil(N×0.2) | 同左 | クラスタ効果の単離用 |
| A7 | ScoreScreen | R_pre推定→ε帯→リスク多様性でK_cap | distDelta medoid | **現最強** |
| A8 | ScoreTopK | R_pre推定→E[score]上位K_capのみ | 同上 | A7のablation (P=50ではA7と同一!) |
| A9 | ClusterValue | distDeltaクラスタ+クラスタ平均E[score]上位K_cap×最良メンバー | 同上 | Proposed改。A7と統計的同等 |
| A10 | ClusterValueDeep | A9を全深さに適用 (相手番は符号反転・K=cv_k_opp) | (全深さ価値選択) | 予算1.72倍でA9と同等=まだ割に合わない |
| 歩 | Ayumu (selfplayのみ) | gPolicyで即決 (探索なし) | — | **本物の歩ではない** (UCT+gEstimator無し)。表記は「gPolicy単体」 |

- K_cap = playouts / v_target (P=200, v=50 → 4)。**A7/A8/A9/A10 は P=50 だと K_cap=1 で縮退・同一化する。評価は P≥100 で。**
- R_pre: MCTS前の各候補の安価なE[score]推定 (既定3回ロールアウト、A10は深さ逓減 3→2→1)

## 3. 評価基盤

- **審判 (referee)**: `score_move_experiment`。各候補をK=200回外乱ありリサンプル+ロールアウト → q_ref_mean = 単一エンド純得点期待値。**木を使わないのでmax-maxバグと無関係（全期間有効）**
- **regret** = q*(最良候補のq_ref) − q_ref(アームの選択手)。`aggregate_reinvest.py` + `regret_stats.py` (Friedman/Wilcoxon+Holm)
- **等予算**: actual_total_sims (物理シミュ回数カウンタ)。標準アーム予算 P=200 R=10 ≈ **150k sims**
- **自己対戦**: `--selfplay --method-a X --method-b Y --games N --ends 2` (ハンマー入替、エンド別純得点記録)。集計 `aggregate_selfplay_2end.py`
- **局面セット**: `test_positions50`(regret用50局面) / `test_positions_depth`(52局面、残り手数r=1-4×13) / `test_positions_categorized8`
- **マシン**: bear(最速・128論理コア) / lion / tiger / jaguar。Linuxビルド産物は `./build/ylab_client`

## 4. 主要結果 (2026-06中旬〜07-14)

### 4.1 regret 6アーム表 (P=200 R=10, 50局面×5seed, 等予算) ※max-max木時代の計測
| A1 | A2 | A5 | A7 | A8 | A9 |
|---|---|---|---|---|---|
| 0.781 | 0.769 | 0.738 | **0.554** | 0.571 | 0.636 |

- 主分離軸=「価値を見るか」(A7/A8/A9 ≪ A1/A2/A5)。Friedman p=0.056
- **A7≈A8** (p=0.67) → A7の強さの本体はスクリーン、リスク多様性の寄与は小
- **A9≈A7/A8** (p=0.56/0.33、A9中央値0.388は全アーム最良) → クラスタ表現の保持はほぼ無コスト
- **A9 vs A2**: −0.133 (勝敗27-20, d=0.43) → **A2の敗因はクラスタ表現でなく価値盲目のmedoid選択**

### 4.2 自己対戦 (2エンド, P=50 R=3, 等予算, lion/tiger/jaguar分割)
| 対戦 | n | A7勝率 | p | 平均純得点 |
|---|---|---|---|---|
| A7 vs A1 | 72 | 65.5% | 0.03 | +0.83 |
| A7 vs A2 | 50 | 88.9% | 1.9e-6 | +1.26 |
| A7 vs 歩(gPolicy単体) | 50 | 94.9% | 2.8e-9 | +2.72 |
| A7 vs A8 (P=200) | 50 | 62.5% | 0.215(ns) | +0.30 (引分36%) |

- 多エンド構造: end0でリード→得点側は次エンドのハンマーを渡す→end1で一部戻る (A7vsA1: end0 +1.11 / end1 −0.28)
- ハンマー(後攻)効果: P=200では +1.28点 (p=0.006) で有意

### 4.3 木の深さの価値 (treedepth → negamax再測定完了 2026-07-15)
Δ = q* − q_ref(depth-k MCTSの手)。**max-max版とnegamax版でΔ曲線はほぼ同一**
(r=2: 2.02→2.23 / r=3: 1.52→1.55 / r=4: 0.24→0.18, CI大きく重複)。
**しかし選択手そのものは激変**: max-max vs negamax の選択手一致率 r=2で0%, r=3で5%, r=4で54%。
= バグ修正は終盤の手をほぼ全部変えたが、どちらの手も「審判からの距離」は同じ。

**重要な帰結: Δ(q_ref基準)では深さの価値を判定できない。** Δの正体は
「ノイズフリー・ミニマックス計画(木)」と「外乱込み・方策継続評価(審判)」の**モデル不一致**であり、
協力的相手でも敵対的相手でも同程度に審判とズレる。深さの真の価値の決着には
**negamax版 depth-1 vs depth-3 自己対戦** (要 --depth-a/-b アーム別深さフラグ, 小改修) が必要。
副次示唆: 木の展開が無外乱(simulateNoRand)なので「精密だが外乱に脆い計画」を立てる可能性
→ 外乱込み展開 (A9 Phase 2の分布サンプル) が木の質の本命フロンティアかもしれない。
データ: `reinvest_experiment/treedepth_fix/` + `analysis/` (Δ曲線, before/after比較図, 大小Δ盤面)。

### 4.3b 深さ対決 (A7 d3 vs d1 自己対戦, negamax, 2026-07-16) — 深さの初の直接証拠
P=200(K_cap=4)・2エンド・50ゲーム・4マシン: **d3 17勝 / d1 24勝 / 分9**。
d3勝率 41.5% CI[0.28,0.57] (p=0.35)、平均純得点 −0.28 [−0.82,+0.30]。4マシン全てで d1≥d3。
**結論: 現行の木では深さ3の読みは同一プレイアウト数の深さ1(平坦MC)に勝る証拠なし(点推定は負)。**
treedepth_fix(深さは終盤の手をほぼ全部変える)と合わせると「深読みが変えた手は良くなっていない」
= ボトルネックは深さでなく木の質(無外乱展開)。→ A9 Phase 2 の動機を最終確認。
ハンマー効果 +1.36点 (Welch p=0.011) を再現。データ: `reinvest_experiment/A7d3vsA7d1_*/`。

### 4.3c 審判のε感度・収束監査 (2026-07-15/16)
- 収束: q_ref値のSE≈0.10点(K=200)で**値は収束**。ただし1位-2位差(中央値0.073)がSE以下
  → **約7割の局面でargmaxの同一性は不安定**。regret/Δ等「値」を使う指標は有効、
  「審判最良手と一致」系の指標は割引いて読む。
- ε感度 (ε∈{0,0.1}をK=200で追試 vs 既存0.3): 候補ランキングのSpearman ρ中央値 **0.76〜0.82**
  = **序列はεに頑健**。top1一致28-36%はε間でもノイズ域(上記の不安定性と整合)、q*シフト+0.05〜0.07のみ。
  → regret系の結論はεの選択に対して頑健。データ: `reinvest_experiment/referee_eps/`。

### 4.4 max-max UCBバグ (発見2026-07-13 → 修正 b84b22b)
- `selectBestChildUCB` が全ノードでroot視点meanを最大化=**協力的相手モデル**だった (rolloutは正常に敵対的)
- 修正: negamax符号反転 (相手番ノードは相手最良=root視点最小を選ぶ)。reinvest版・depth_n版とも
- **実害の因果的証明**: A10 同一条件(seed42, 1.72倍予算)で regret 0.776(max-max) → **0.655(negamax)** (25勝14敗)
- **b84b22b以前の木系結果 (4.1-4.3の全て) はmax-max木での計測**。相対比較は公平(全アーム同一コード)だが絶対値は要注意

## 5. 実行中 / 直近タスク

- **[準備完了] noisy-tree深さ対決**: `--noisy-tree` (開ループ・リサンプリング木, commit 481d7be) を実装済み。
  木の構造は決定的着地のまま、価値評価だけ毎訪問エッジを外乱ありで再シミュレーション
  (R_pre も候補手を打ち直し=審判と同規約)。d3-noisy vs d1-noisy の対決で
  「外乱を考慮した木なら深さは価値を持つか」(ユーザー仮説) を検証する。
  注: --noisy-tree は selfplay の両アーム共通 (アーム別 noisy フラグは未実装)。
- **[次候補・優先順]** (深さ対決①とε監査は完了 → 4.3b/4.3c)
  ① **発表図表・原稿の更新** (GPW締切接近): regret図にA8/A9追加、深さ対決・エリアヒートマップの整形。
    ストーリー: 「候補削減は価値で」(A7/A8) + 「クラスタは価値選択と組めば無コストで説明可能」(A9)
    + 「深さより木の質」(4.3b) の3部構成が完結した。
  ② A9 Phase 2: 分布クラスタリング (候補ごと外乱あり3-5サンプル) — 4.3bで「木の質が本命」と確定し最有力の次打ち手
  ③ 6アームregretのnegamax再走 (発表表を修正版の木で確定させたい場合)

## 6. 既知の注意点・ハマりどころ

- **「歩」表記**: selfplayのAyumuはgPolicy単体。論文・図では「gPolicy単体(探索なし)」と表記 (本物の歩はUCT+gEstimator)
- **P=50でA7/A8/A9/A10は縮退** (K_cap=1で同一手法になる)
- `data/policy_param.dat` (gPolicy重み161KB) は .gitignore 対象。サーバーへは手動scp
- **実行中バイナリの上書き**: Linuxで `cmake --build` は ETXTBSY で失敗 → `rm -f build/ylab_client` してからビルド (実行中プロセスは旧inodeで安全に継続)
- **結果ファイルのgit衝突**: サーバーで生成→ローカルで回収コミット→サーバーpull時に "untracked would be overwritten" → サーバー側で `mv` 退避してから pull
- 2エンド対戦は ef187f4 のクラッシュ修正が前提 (`game is already over`)
- CSVは**全局面/全ゲーム完了後に一括書き出し** (途中killで全損)。treedepth系の [done N/M] が長時間止まるのは末尾の重い局面 (混戦盤面は1局面4-8h) で正常

## 7. 主要パス

| 何 | どこ |
|---|---|
| アーム実装 | `experiments/reinvest_experiment.{h,cpp}` (MctsMode enum は `depth_n_mcts_experiment.h`) |
| 自己対戦 | `experiments/selfplay_experiment.{h,cpp}` |
| 審判 | `experiments/score_move_experiment.{h,cpp}` |
| 共通rollout/UCB | `experiments/mcts_shared.{h,cpp}` |
| regret結果 (6アーム+A10) | `reinvest_experiment/scorescreen/run50/{A1,A2,A5,A7,A8,A9,A10,A10fix}/seed_*/` + `regret/` |
| 審判qテーブル | `reinvest_experiment/scorescreen/run50/referee/score_move_qtable.csv` (depth用: `reinvest_experiment/depth/referee/`) |
| 2エンド自己対戦 | `reinvest_experiment/selfplay2end/` |
| treedepth (max-max版) | `reinvest_experiment/treedepth/` + `analysis/` |
| treedepth (negamax版, 回収予定) | `reinvest_experiment/treedepth_fix/` |
| 分析スクリプト | `scripts/aggregate_reinvest.py, regret_stats.py, aggregate_selfplay_2end.py, analyze_tree_depth.py, plot_treedepth_compare.py, plot_cluster_value_map.py, plot_board.py` |

## 8. 代表的な実行コマンド

```bash
# ビルド (Windows: build/Release/ylab_client.exe, Linux: build/ylab_client)
cmake --build build --config Release --target ylab_client

# regretアーム (標準予算)
./build/ylab_client --reinvest-arm --method ScoreScreen --depth 3 --playouts 200 \
  --rollouts-per-visit 10 --r-pre 3 --v-target 50 --retention 0.20 --seed 42 \
  --threads 50 --load-positions test_positions50 --output-dir <OUT>

# 2エンド自己対戦
./build/ylab_client --selfplay --method-a ScoreScreen --method-b AllGrid --games 24 \
  --threads 24 --seed 6001 --ends 2 --depth 3 --playouts 50 --rollouts-per-visit 3 \
  --r-pre 3 --v-target 50 --retention 0.20 --output-dir <OUT>

# regret集計+検定
python scripts/aggregate_reinvest.py --reinvest-dir reinvest_experiment/scorescreen/run50 \
  --referee-dir reinvest_experiment/scorescreen/run50/referee --pair A7,A9 --out <OUT>
python scripts/regret_stats.py --joined <OUT>/reinvest_joined.csv --out <OUT>
```
