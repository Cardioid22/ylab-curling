# プロジェクト現況 (最終更新: 2026-07-25)

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

### 4.1 regret 6アーム表 — **【2026-07-25 全面改訂】旧表の分離はエンド跨ぎバグの人工物だった**

**修正版 (run50v2, エンド跨ぎ修正済みの木, P=200 R=10, 50局面×5seed, commit ebb5f80):**
| A1 | A2 | A5 | A7 | A8 | A9 |
|---|---|---|---|---|---|
| 0.514 | 0.506 | 0.505 | 0.502 | 0.511 | 0.508 |

**6アーム完全横並び (CIも同一)。P=200では削減方式による差は存在しない。**
フェーズ分解による原因確定: 序盤・中盤のregretは元々全アーム同一 (~0.33 / ~0.55, 修正前後で不変)。
旧表の構造は全て終盤行にあり、価値盲目アームはバグに直撃され (A1終盤 1.18→0.61)、
スクリーンアームは e_pre の end_done ガードのおかげで**バグに免疫だっただけ** (A7終盤 0.69→0.58)。
= **「スクリーンが良い意思決定をしていた」のではなく「バグのワクチンを持っていた」**。

<details><summary>旧表 (バグ有り木, 無効。記録用)</summary>
A1 0.781 / A2 0.769 / A5 0.738 / A7 0.554 / A8 0.571 / A9 0.636。
「価値スクリーン群≪価値盲目群」「A9のA2改善」等の主張はすべてこの表に依存しており、現在は無効。
</details>

**無効化された主張**: 「候補削減は価値で行うべき」(P=200 regret根拠)、A9のA2改善マージン、A7優位。
**残存する成果**: バグ発見と定量的な因果検証そのもの・審判基盤とε頑健性・盤面/エリア可視化ツール群。
**未確定 (要・修正版での再測定)**: 予算スイープ (低予算での分離は理論上あり得る)、全自己対戦
(旧対決は両者バグ木でプレー。A7の勝利はスクリーンの免疫による可能性)、ハンマー効果 (対称なので頑健見込み)。

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

**再戦 (noisy-tree, 2026-07-17)**: --noisy-tree (開ループ・リサンプリング木) を両アームに適用した
d3-noisy vs d1-noisy (同条件50ゲーム): **d3 18勝/d1 22勝/分10、勝率45.0% CI[0.31,0.60] p=0.64**。
= **外乱を考慮した木でも深さ3は価値を生まない**。深さの問いは
「max-max / negamax / noisy の3方式すべてで深さ3≈深さ1 (点推定は負)」という**三重に頑健な負の結果**で決着。
残る説明: 賢いε-greedyロールアウト方策が続きの価値をほぼ取り切っている (+深部訪問の薄さ)。
注意: この対決は「noisy vs 決定的」の強さ比較ではない (d1-noisy vs d1-det は未測定・任意の追試)。
ハンマー効果3回目の再現 +1.48点 (p=0.010)。データ: `reinvest_experiment/A7d3n_vs_A7d1n_*/`。

### 4.3b-2 深さの追試2本 (2026-07-23) — 深さの問い完全決着 (5方式)
- **A1-d1 regret (スクリーンなし全候補・深さ1平坦MC, 5seed)**: regret **0.671** vs A1-d3 0.781
  (局面×seed対決 128勝89敗33分, Δq=+0.110)。しかも実シミュ**105k = A1-d3の32%減**。
  → **スクリーン無しの世界でも深さ1が深さ3に勝つ** =「深さはスクリーンの代替」仮説を棄却。
  序列: 価値スクリーン群(0.55-0.64) < 平坦MC無スクリーン(0.67) < 深さ3全変種(0.74-0.78)。
- **フェーズ適応対決 (前半d1×P100+終盤d3×P500 vs 定数d1×P200, 同一総プレイアウト, 50ゲーム)**:
  適応側 **14勝24敗12分, 勝率36.8% CI[0.23,0.53], 平均純得点 −0.64 [−1.24,−0.06] (CIが0を除外)**。
  → **終盤への深さ集中投資も定数浅配分に負け**。方策が終盤ショットをほぼ解いており、
  序盤のスクリーン品質(P100でK=2に縮小)を削る代償の方が大きい。
- 深さの結論は5経路 (定数対決/noisy対決/適応対決/treedepth手総入替/無スクリーンregret) で一致:
  **実用予算+現行方策では深さ3は一度も価値を示さなかった**。

### 4.3b-3 d3敗因解剖 (2026-07-23, `scripts/analyze_d3_failure.py`, 出力 `run50/d3_failure/`)
A1(d3) vs A1d1(d1) の同一局面×同一seed実走行を審判と突き合わせた解剖:
- **一致率は全域で低い (6〜18%)** が、序盤の分岐は Δq=−0.006 で**価値的に等価** (同価値手の高原)。
- **損害は終盤に集中**: r=2-4 で分かれた d3 の手は審判EVで **−0.45点/回**。
- d3の終盤逸脱は博打ではない: SD低 (1.10 vs 1.35)・大勝率低 (23% vs 31%)・大敗率高 (28% vs 21%)
  = 「上振れの小さい低EVの受動手」。
- **種別遷移の主犯: Hit→PostGuard ×14, Hit→Draw ×8** — EV正解の石排除をガード/ドローに置換。
  機構: r≤3で木が終端到達→評価が「方策継続EV」から「無外乱・離散候補ミニマックス」に切替わり、
  外乱下では漏れるブロック線に**幻の安全性**を見る。
- 対戦の負け方と整合: 大敗13%より小差負け33%が主体、end0ハンマー時の得点 d3 +0.40 vs d1 +1.02
  (**ハンマー活用が0.6点劣る**)。

### 4.3b-4 【重大】エンド跨ぎ汚染バグ発見・修正 (2026-07-23, commit e9f0dde) — **深さの結論を再審理へ**
d3敗因解剖の追跡中に決定的木の重大バグを発見:
- **バグ1 (エンド跨ぎ汚染)**: 木の枝がエンド最終ショットを跨ぐと、次エンドの盤面を展開して
  **次エンドのロールアウト得点**を価値にしていた (現エンドの得点が価値に入らない)。しかも
  得点→ハンマー喪失の規則により「今得点しない消極手」を**系統的に選好する誤った目的関数**。
  → 敗因解剖の Hit→PostGuard/Draw 置換、Δ曲線の r=2-3 集中と r=4 消失、negamax不変性を全て説明。
- **バグ2 (ガード飢餓)**: noisy木の跨ぎガード (481d7be) が子の訪問統計を更新せず、UCB優先度∞の
  同一子だけが選ばれ続け、相手最終ショットが「固定の1候補」扱いになっていた (noisy対決に影響)。
- 修正: 跨ぎ子=終端の葉として現エンド得点を報酬に (reinvest版・depth_n版とも)。
- **影響**: 決定的木の終盤系結果 (Δ曲線2本・対決#1・敗因解剖) はバグ1下、noisy対決はバグ2下の計測。
  **「深さは価値なし」の終盤成分は再審理**。中盤以遠 (r≥5) は跨ぎが起きないため影響なし。
  再検証プローブ: `run50_A1d3fix/seed_42` (regret) → 改善すれば深さ対決の再々戦へ。

### 4.3b-5 修正検証プローブ (2026-07-25, commit abc5554) — バグ1が主犯と確定 / **regret表は要再走**
- run50再解剖 (d3fix vs d1fix, seed42): 終盤Δq **−0.451→−0.086 (≈0)**、Hit→PostGuard **×14→0**、
  リスク異常も消滅。エンド跨ぎ汚染がrun50終盤病理のほぼ全てを説明。
- **regret激変 (seed42)**: A1-d3 **0.852→0.508**、A1-d1 0.744→0.477 (r=1汚染もあった)。
  → **公表用6アームregret表 (A1=0.781等) はバグ下の計測。A1の真の実力は大幅に上で、
  価値スクリーン群のマージンは縮む見込み。表の再走が発表前に必須。**
- depthセット解剖 (終盤39局面): 残差あり — d3fixはまだ **−0.340/分岐** でd1fixに劣る。
  ただし病理の型が変化: 受動置換は消え、今度は攻撃的シフト (Draw→Hit×7, SD増・大敗率増)。
  残る容疑: 深部訪問の薄さ / 相手応手の離散化 / 方策が終盤をほぼ解いている。
- 次: ①**6アームregretの全再走 (最優先, 表が現状不正確)** ②深さ対決 再々戦 (d3fix vs d1fix)。

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

- (実行中のサーバージョブなし。全マシン空き)
- **[次候補・優先順] — run50v2の横並び判明を受けた再建計画**
  ① **低予算regretの修正版再走 (P=50, 6アーム)**: 「価値スクリーンは低予算ほど有利」という
    元々の理論 (分散削減) が正しければ、分離はP=50で再出現するはず。ここが提案手法の生死を分ける。
  ② **自己対戦の修正版再走 (A7 vs A1, 2エンド)**: 旧65.5%勝利が「スクリーンのバグ免疫」由来か
    真の強さかを決着。①と並行可。
  ③ 結果を受けて発表ナラティブの再設計 (ユーザーと合議):
    案a: 分離が低予算で復活 → 「低予算領域における価値スクリーンの優位」に主張をスコープ
    案b: 復活せず → 「等予算ではあらゆる削減方式が同等」+ 2つの木バグの発見・診断方法論を主軸に
  ④ 保留: 深さ対決再々戦 / A9 Phase 2 / 学習化 (①②の結果次第で優先度再評価)

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
