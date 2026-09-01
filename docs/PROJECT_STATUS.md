# プロジェクト現況 (最終更新: 2026-09-01)

AIアシスタント (Codex / Claude / Gemini) 向けの現況同期ドキュメント。
**過去の設計文書ではなく「今」の状態を書く。実験が進んだら必ずここを更新すること。**

## 1. 研究の主線 (何を証明しようとしているか)

**【2026-07-27 転換】性能優位では語らない。** 修正木では全削減方式の regret が横並び (4.1) なので、
主張は「盤面類似クラスタ + 価値選択 (ClusterValue/A9) は **性能を損なわずに** 候補手を意味のある
エリアへ分類し (エリア価値マップ・クラスタ価値の妥当性 ρ)、探索深さによる手の変化をクラスタ単位で
説明できる」という **分析的価値** に置く。GPW2026 拡張アブストはこの線で 7/31 提出済み
(`gpw2026/`, 提出 PDF `仲_202611_GPW.pdf`)。

- 出口: GPW2026 本論文 (採否 9/4 → **最終原稿 10/23** → 発表 11/20-22 箱根) + 大会
- アブストで約束した本論文の内容 (= 4.5 の run200 キャンペーン):
  ① 局面 50→200 での統計的評価 ② 得点期待値と実行リスクを統合したクラスタ価値 ③ どんな局面で候補削減が有効か

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

### 4.1b 再建実験の結果 (2026-07-27, commit 8aa2a1a, 95aac72) — 【再建失敗】主張の生存証拠なし
- **P=50 修正版4アームregret** (A1/A2/A5/A7, 5seed): A1 0.511 / A2 0.529 / A5 0.592 / A7 0.529。
  Friedman p=0.88 (完全に有意差なし)。全6ペアp_holm=1、必要n=222〜13700+ (真の効果があっても極小)。
  「低予算ほど価値スクリーンが有利」という分散削減の理論的期待は**再建に失敗**。
- **修正版 A7 vs A1 自己対戦** (P=50, 2エンド, 72ゲーム): A7勝率 **42.6%** CI[0.30,0.56] p=0.34。
  旧65.5%(p=0.03)から反転し非有意に。マシン別0.36-0.54で方向も不一致=真のnullとの整合。
  → **旧勝利はスクリーンのバグ免疫による幻だったと確定**。ハンマー効果は+1.47点(p=0.002)で再現。
- **結論: P=50/P=200のregret・自己対戦いずれにも、候補削減方式間の性能差を示す証拠が残っていない。**
  提案手法(A7等)が全探索(A1)に優る主張は、現時点で統計的裏付けを失った。

**未確定**: 予算スイープP=100/500/1000の修正版再走 (未実施。理論上P=50/200両方で不出現なら
中間予算でも出ない可能性が高いが、正式には未検証)。深さ対決・treedepthの修正版再走 (別トラック)。

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

### 4.5 run200 キャンペーンの準備 (2026-08-25) — 実装完了・サーバー投入待ち
GPW 本論文 (`scripts/README_run200.md` が runbook):
- **局面 200**: `test_positions200/` (先頭50 = `test_positions50` と同一順序 → run50v2 の結果と審判を
  `scripts/run200_reuse_first50.sh` で `*_idx0.csv` として流用。+150 は `pick_more_positions.py --seed 23` の層化サンプル)。
- **リスク統合クラスタ価値**: `--risk-lambda λ` (ClusterValue 系のみ有効)。クラスタ価値 = μ_c − λσ_c、
  σ_c = メンバー全ロールアウト標本のプールSD (= √mean(sd_i² + (e_i−μ_c)²))。代表手も e_i − λ·sd_i 最大。λ=0 は A9 と完全一致。
  `cluster_table.csv` に `cluster_sd, cluster_value_risk, risk_lambda` 列、`reinvest_results.csv` に `risk_lambda` 列を追加。
  アーム `A9R025/A9R05/A9R10` を `run_reinvest.sh` に追加 (6語目以降が追加フラグ)。`--start-index/--max-positions` も通した。
  スモーク (λ=0/0.5, 2局面, P=40) で配管確認済み。
- **リスク調整 regret**: `aggregate_reinvest.py --risk-lambda λ` → `regret_adj` 列 (q_adj = q_ref_mean − λ·q_ref_sd)。
  `regret_stats.py --metric regret_adj` で検定。run50v2 では λ=0.5 でも 7 アーム横並び (Friedman p=0.72)。
- **局面特徴**: `scripts/position_features.py` (盤面 38 列 + 審判由来の多峰性/リスク + クラスタ構造 η²/巨大クラスタ率/妥当性ρ/被覆 screen_loss
  + アーム別 regret 差) → `scripts/analyze_when_clustering_helps.py` (Spearman 相関・層別 Kruskal・図)。
  列の意味は出力先の `feature_dictionary.md`。
- **50 局面の予備結果** (`reinvest_experiment/run50v2/features/analysis/`, n=50 なので確定ではない):
  - `d_A9_A1` (A9−A1 の regret 差) と最も相関するのは `best_is_rep` (審判最良手がスクリーンに残ったか, ρ=−0.31 p=0.03) と
    `best_type_margin` (ρ=−0.26): **戦術種別間の差が大きい局面ほど A9 が A1 に勝つ**。
  - `d_A9_A2`: `md_screen_loss` ρ=−0.50 (medoid が最良手を落とす局面ほど A9 が A2 に勝つ)。
  - `screen_loss` は `n_near_best_050` (q* 近傍候補数) と負相関 = **正解が一意な局面ほどスクリーンで落とす**。
  - `rho_gain` (クラスタ平均の分散削減) は `n_house`/`n_within_0.61` と負相関 (混雑ハウスで消える; crowded_house Kruskal p=0.005)。
  - `eta2_q` は `largest_frac` と負相関、巨大クラスタ退化 (largest_frac≥0.5, 8/50 局面) で ≈0.05 (p<0.001)。ハンマー保持側で低い (0.43 vs 0.66, p=0.007)。

### 4.6 run200 キャンペーン結果 (2026-08-27 回収・集計完了) — 【200局面で初めて有意差】
データ: `gpw_experiment/` (A1/A2/A5/A9 = 先頭50流用 + 新規150, A9P5/A9R05 = 200局面, 全5seed, 審判 K=200 全200局面)。
集計: `gpw_experiment/regret/` (regret_stats.txt), `regret_adj/`, `features/` (position_features.csv 200×162列, analysis_all / analysis_pilot50 / analysis_new150)。

**regret (P=200 R=10, 200局面×5seed, 修正木)**: A1 **0.472** / A9P5 **0.480** / A9R05 0.504 / A9 0.506 / A5 0.540 / A2 **0.578**。
Friedman **p=0.002**。Holm後有意: **A2 vs A9P5 (p=0.022, 110勝75敗)**、**A9 vs A9P5 (p=0.006)**。A1 vs A9P5 は p=0.70 (同等)。
- 主張可能: 「価値選択 (ClusterValue, R_pre=5) は medoid (ClusterMedoid) を有意に改善し、全探索と同等」。
  注記: A9P5/A9R05 は R_pre=5 のぶん sims が **+8%** (A9 は A2 と同予算だが A2 との差は p=0.22 で非有意)。
- λ=0.5 (A9R05): A9P5 より 0.024 悪い (raw p=0.015, Holm n.s.)。**リスク調整 regret でも改善せず** (raw p=0.048 で悪化)。
  → リスク統合は性能に寄与しない (理論どおり)。説明レバーとしてのみ扱う。オフライン掃引: 採用上位K の一致率 0.87、
  入れ替わりは審判SDが大きい局面 (ρ=−0.34)・q_range が小さい局面 (+0.29)・多峰局面 (−0.18) に集中。真値側で最良手が変わるのは 23%。
- クラスタ価値の妥当性 (R_pre=5): 候補 ρ 0.27 → クラスタ ρ 0.37 (中央値)、局面の 74% でクラスタ側が上、Wilcoxon p=4e-15。
- **巨大クラスタ退化 = 最終ショット (r=1)**: largest_frac≥0.5 の32局面は r=1 の32局面と完全一致。そこでは価値アームは A1 と 100% 同じ手を選び
  (regret 同一)、medoid/ランダムは 34%/18% しか一致せず regret +0.40 (p=0.006)。= 終局盤面で距離関数が潰れ、クラスタ表現が無意味になる。
- **「どこで効くか」 (パイロット50→新規150 で再現したものだけ)**:
  - 価値選択が価値盲目削減に勝つ度合いは、候補の価値の広がり q_range (ρ=−0.30)・安価推定の精度 rho_cand (−0.25) が大きいほど大きく、
    q*近傍の候補数 n_near_best_050 (+0.26) が多いほど小さい。= **選択が重要で推定が効く局面で価値選択が効き、何でも同じ局面では効かない**。
  - A9P5 が A2 に勝つ最強の予測子は `md_screen_loss` (medoid が最良手を落とした損失, ρ=−0.41) と `best_cluster_size` (−0.29)。
  - screen_loss は最良手が一意な局面 (n_near_best_025 −0.36, q_gap12 +0.30) で大きい: 両方式とも最良手の被覆率は 20-24% と低い。
  - **パイロット50で見えた「混雑局面では価値盲目削減が勝つ」(ρ=+0.42) は新規150で再現せず (−0.06) → 棄却**。仮説生成/検証の分割が機能した。
  - 疎な序盤 (ハウス内≤2, r≥9, n=12): 全アーム regret 0.2-0.5 で差なし = クラスタリング不要だが害もない (ユーザー仮説を支持)。
- 具体例候補 (新規150, seed 間一貫): 最終ショットで medoid 壊滅 g7871 e7 s15 (A2 3.66 vs A9P5 0.70); A9P5 が A1 に勝つ混戦終盤 g5335 e1 s14 (A1 1.27 vs 0.00), g7859, g8298;
  クラスタが害 g2099 e0 s14, g1616 e0 s14 (A1 0.00 vs A9P5 ~1.0, screen_loss 0 → 損失は木の中)。
- 分析ツール: `analyze_when_clustering_helps.py` に problem_type 4分類 (degenerate/needle/multi_area/flat)・選択規則×局面群の交互作用・
  `--only-positions/--exclude-positions` (パイロット/検証分割) を追加。

### 4.7 run500 第1波 (2026-08-30 回収) — 【外乱込み木 A9P5N が regret 0.31 で他を圧倒。PW は効かず】
詳細 `gpw_experiment/regret500/WAVE1_SUMMARY.md`。局面 500 (= run200 の 200 + 層化 300 `test_positions500`)、5 seed、審判 K=200 全 500 局面。
- **regret**: A1 0.418 ≈ A9P5 0.423 ≈ A11b 0.425 ≈ A11a 0.427 ≫ **A9P5N 0.309** (Friedman p=3e-18; A9P5N は全対に Holm p<1e-10, d≈0.4)。
  先頭 200 局面 (9 アーム) でも A9P5N 0.324 ≪ A1 0.472 ≈ A9P5 0.480 < A9 0.506 < A5 0.540 < A2 0.578。
- **A9P5N の利得は木の評価から**: スクリーン指標 (ρ, 被覆, screen_loss) は A9P5 とほぼ同じ、tree_loss が 0.19→0.10。
  = 決定的木の「精密だが脆い計画」を外乱込み評価が罰する (7月の d3 解剖と同じ機構)。利得は混雑・終盤に集中
  (crowded_late −0.34, freeze_late −0.23, takeout_late −0.15; 序盤・疎は ≈0)。**A1N (AllGrid+noisy) の対照が未走 = 現時点では
  「外乱込み評価」の効果であってクラスタリング固有とは言えない**。
- **PW (A11)**: 被覆 0.20→0.29、screen_loss 0.23→0.15 だが tree_loss 0.19→0.27 で相殺、regret 不変。sims +5-6%。
  P=200 を 12 子に割ると 1 子 15 訪問で評価しきれない (幅/深さトレードオフ)。
- 空ハウス序盤 (n=40): 全アーム 0.30-0.34、A9P5 が A1 より 0.04 良い (p=0.015)。候補の 73% が q* 近傍 = flat。ユーザー仮説を支持。
- 出力: `regret500/stats_500main` (regret), `stats_500main_adj`, `stats_200all`; `features500_{A9P5,A9P5N,A11a,A11b}/`。

## 5. 実行中 / 直近タスク

- (実行中のサーバージョブなし。run500 全完了・回収済み 2026-09-01)
- **【run500 最終結果】** 500局面×5seed×9アーム (`gpw_experiment/regret500/stats_500full9`, WAVE1_SUMMARY.md 第2波節):
  **A9P5N 0.309 < A1N 0.334 (Holm p=3e-4) ≪ A1 0.418 ≈ A9P5 0.423 ≈ A11a/b < A9 0.442 < A2 0.488 ≈ A5 0.491**。
  ①外乱込み評価は全手法に効く主レバー (−0.08〜−0.11)。②その条件下でクラスタ価値選択が全探索に有意に勝つ
  (分散削減仮説の初実証)。③利得は混雑・終盤、**r=1 では逆転** (退化層はスクリーンをバイパスすべき)。
  ④A9P5>A2 (p=1e-3)、A9>A5 (p=0.017)、A2≈A5 = medoid はランダム同等。
- **【次】** 論文の図表化 (regret 表, noisy の利得, A9P5N−A1N の層別, r=1 退化, エリア価値マップ具体例)。
  小改修候補: r≤1 でスクリーンをバイパス。
- 第2波回収後: `aggregate_reinvest.py --reinvest-dir gpw_experiment` → A1N vs A9P5N で「外乱込み評価の利得がクラスタ固有か」を決着。
- 論文の図表化 (具体例・どこで効くか・r=1 退化・PW の訪問分布) は並行して着手可。
- **【次】GPW 本論文の図表化**: (a) 具体例 3 局面 (盤面+エリア価値マップ+候補 q のクラスタ色分け) を新規150から作図
  (`scripts/plot_area_variants.py`), (b) 「どこで効くか」の図 (q_range / n_near_best vs 価値−盲目差, problem_type 別), (c) r=1 退化の図,
  (d) λ 掃引の図 (overlap vs 審判SD)。(e) クラスタ「カード」出力 (`cluster_cards.py`, 未作成)。
- 任意の追加実験: A9 の K を候補数に比例させる派生 (`--v-target 25` 等; 松崎先生の「クラスタ数」質問への回答)、空ハウス序盤20局面セット。
- **【完了】run200 キャンペーン (2026-08-25 19:10 JST 投入 → 08-27 回収)**
  | マシン | ジョブ | 進捗の見方 |
  |---|---|---|
  | bear | 審判 K=200 新規150局面 (`referee/referee_idx50.log`, 64thr) + **A9R05** 200局面×5seed (12thr/seed) | `grep -ac "^\[done" run200/A9R05/seed_*/run.log` |
  | jaguar | **A9P5** 200局面×5seed (12thr/seed)。**A1,A5 (150局面) は未投入** → 空きコア60で追加投入すること | 同上 |
  | lion | **A2** 新規150局面×5seed (18thr/seed) | 同上 (150 で完了) |
  | tiger | **A9** 新規150局面×5seed (18thr/seed) | 同上 |
  - 全アーム P=200 R=10。A9R05/A9P5 は `--r-pre 5` (launch_*.log で確認済み)。ログの eta は序盤の軽い局面基準で過小 (実際は 12-20h)。
  - **サーバー運用の事実 (2026-08-25 判明)**: 4台の `/home` は同一 NFS → リポジトリ/バイナリ/出力は共有。pull・ビルドは1回でよい。
    ビルド用 docker イメージ `ylab-project` は **lion のみ** (bear は docker 権限なし):
    `docker run --rm -v "$(pwd):/app" ylab-project bash -c 'cd /app/build && cmake --build . --config Release --target ylab_client -j 48'`
    (`cmake` はホストに無い。`rm -f build/ylab_client` を先に)。スクリプトは実行ビットが無いので `bash scripts/...` で起動。
- 回収後 (ローカル): `rsync` で run200 を取得 → `bash scripts/run200_reuse_first50.sh` → `aggregate_reinvest.py --risk-lambda 0.5`
  → `regret_stats.py` (regret / regret_adj) → `position_features.py` → `analyze_when_clustering_helps.py` (`scripts/README_run200.md` §4)
- 本論文の執筆 (10/23): 4.5 の 3 点 + アブスト図の更新 (エリア価値マップ, クラスタ乗り換え率を 200 局面で)。
- 保留 (GPW 後): Ward/complete linkage で巨大クラスタ退化の救済 / K (v_target) 感度 / A9 vs A1 修正木自己対戦 / A9 Phase 2 / 学習化。

## 6. 既知の注意点・ハマりどころ

- **「歩」表記**: selfplayのAyumuはgPolicy単体。論文・図では「gPolicy単体(探索なし)」と表記 (本物の歩はUCT+gEstimator)
- **P=50でA7/A8/A9/A10は縮退** (K_cap=1で同一手法になる)
- `data/policy_param.dat` (gPolicy重み161KB) は .gitignore 対象。サーバーへは手動scp
- **実行中バイナリの上書き**: Linuxで `cmake --build` は ETXTBSY で失敗 → `rm -f build/ylab_client` してからビルド (実行中プロセスは旧inodeで安全に継続)
- **結果ファイルのgit衝突**: サーバーで生成→ローカルで回収コミット→サーバーpull時に "untracked would be overwritten" → サーバー側で `mv` 退避してから pull
- 2エンド対戦は ef187f4 のクラッシュ修正が前提 (`game is already over`)
- `--risk-lambda` は ClusterValue/ClusterValueDeep のみ有効 (他モードでは無視、ログに注記が出る)
- `position_features.py --arms` は **基準→提案の順** に並べる (差分列は d_後_前 = 後のregret − 前のregret。正なら後が悪い)
- `test_positions200` の先頭50行は `test_positions50` と同一。局面数=行数なら `sampleTestPositions` はシャッフルしない
- 提出版 `gpw2026/main.tex` (7/31) は working copy にあり **HEAD (7/28) より新しい・未コミット** (A1/A2/A9 コードネーム入り版が提出版)
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
| run200 (GPW本論文) | `test_positions200/`, `scripts/README_run200.md`, `scripts/run200_reuse_first50.sh`, `scripts/position_features.py`, `scripts/analyze_when_clustering_helps.py`; 予備結果 `reinvest_experiment/run50v2/features/` |
| GPW2026 アブスト | `gpw2026/` (main.tex, 提出PDF, DATA_INDEX.md, data/, figures/) |

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
