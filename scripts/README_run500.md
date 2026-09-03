# run500 キャンペーン — GPW2026 本論文 第2弾 (層化300局面 × 実現性込みスクリーン × progressive widening)

run200 の結果 (`docs/PROJECT_STATUS.md` §4.6) を受けた追加実験。締切: 採否 9/4 → **最終原稿 10/23**。

## 問い
1. **検出力**: 同予算の A9(R_pre=3) vs A2、A1 vs A2/A5 (d≈0.14-0.19, 必要 n≈220-400) を決着させる。
   A1 vs A9P5 (d=0.03) は 500 でも決まらない = 「同等」のまま。
2. **層の補強**: run200 で薄かった層 (needle=正解一意 5局面、空ハウス序盤 2局面) を意図的に足す。
3. **被覆の改善 (A11 ClusterPW)**: 審判最良手が代表 4 手に残る率が 20% しかない (screen_loss が regret の 56%)。
   root の子をクラスタ価値順に progressive widening で徐々に開く: k(N) = max(k0, ceil(C·N^α))。
4. **実現性込みスクリーン (A9P5N)**: 現行の e_i は無外乱着地からの継続値 (=狙いどおり決まった前提)、
   審判 q は「試みる価値」(初手を外乱ありで振り直す) で不整合。`--noisy-tree` で R_pre も候補手を外乱ありで打ち直す。

## 局面セット `test_positions500`
先頭 200 = `test_positions200` と同一順序 (run200 の結果と審判を流用)。+300 は `scripts/pick_positions_stratified.py --seed 31` の層化:
| 層 | 定義 | 数 |
|---|---|---|
| empty_early | 石 0-1, shot≤3 (「空のハウス」) | 40 |
| sparse_early | 石 2-4, shot≤5 | 30 |
| takeout_mid / takeout_late | 相手 No.1 を除去する形, shot 6-11 / 12-14 (needle の代理) | 40 / 40 |
| freeze_mid / freeze_late | 相手 No.1 に密着できる形 | 30 / 30 |
| draw_mid / draw_late | No.1 が遠く空きが多い | 20 / 20 |
| crowded_mid / crowded_late | 石 6 個以上 | 25 / 25 |
最終ショット (shot 15) は除外 (退化層; run200 に 32 局面あり十分)。1 game 1 局面、既存 200 と game_id 重複なし。

## アーム (P=200 R=10, 5 seed)
| アーム | 内容 | 走らせる局面 |
|---|---|---|
| A1 / A2 / A9 / (A5) | 既存 (先頭200は流用) | 新規 300 (`--start-index 200 --max-positions 300`) |
| A9P5 | ClusterValue R_pre=5 (run200 の最良) | 新規 300 |
| **A9P5N** | A9P5 + `--noisy-tree` (実現性込みスクリーン) | 全 500 |
| **A11a** | ClusterPW C=1 α=0.5 k0=2 → k(200)=15 (積極) | 全 500 |
| **A11b** | ClusterPW C=2 α=0.3 k0=2 → k(200)=10 (保守) | 全 500 |
| A11c / A11aN | k(200)=5 (最小) / PW+noisy | 任意 |
PW の開き方 (k(N), N=root 訪問数): A11a 2,4,8,10,15 (N=4,50,100,200) / A11b 2,4,7,8,10 / A11c 2,2,4,4,5。比較対象 A9P5 は 4 固定。

## 手順
```bash
# 0. lion で pull + docker ビルド (共有FS なので1回) — scripts/README_run200.md §0 と同じ
# 1. 先頭200の流用 (ローカル or 集計マシン)
# (実際の運用: run500 の回収先も gpw_experiment/ に統合した。run200 と run500 のファイル名は衝突しない
#  (*_idx0/_idx50/無印 = run200, *_idx200 = 新規300, 500局面アームは無印))
# 2. 審判 (新規300局面, K=200): 2台に分割 (各150局面, ~14h)
cd ~/ylab-curling && mkdir -p reinvest_experiment/run500/referee
nohup ./build/ylab_client --score-move --score-rollouts 200 --states 500 --threads 64 --seed 42 \
  --load-positions test_positions500 --start-index 200 --max-positions 150 \
  --output-dir reinvest_experiment/run500/referee > reinvest_experiment/run500/referee/referee_idx200.log 2>&1 &   # bear
nohup ./build/ylab_client --score-move --score-rollouts 200 --states 500 --threads 64 --seed 42 \
  --load-positions test_positions500 --start-index 350 --max-positions 150 \
  --output-dir reinvest_experiment/run500/referee > reinvest_experiment/run500/referee/referee_idx350.log 2>&1 &   # jaguar
# 3. アーム (例。マシン割当は空き次第)
mkdir -p reinvest_experiment/run500
nohup bash scripts/run_reinvest.sh --arms "A1,A2" --positions-dir test_positions500 --n-states 500 \
  --start-index 200 --max-positions 300 --num-seeds 5 --max-parallel 5 --threads-per-seed 18 \
  --parent-dir reinvest_experiment/run500 > reinvest_experiment/run500/launch_X.log 2>&1 &
nohup bash scripts/run_reinvest.sh --arms "A11a" --positions-dir test_positions500 --n-states 500 \
  --num-seeds 5 --max-parallel 5 --threads-per-seed 12 \
  --parent-dir reinvest_experiment/run500 > reinvest_experiment/run500/launch_Y.log 2>&1 &
# 4. 回収・集計 (ローカル)
rsync -av lion:~/ylab-curling/reinvest_experiment/run500/ gpw_experiment/
python scripts/aggregate_reinvest.py --reinvest-dir gpw_experiment --referee-dir gpw_experiment/referee --pair A11a,A9P5 --risk-lambda 0.5 --out gpw_experiment/regret
python scripts/regret_stats.py --joined gpw_experiment/regret/reinvest_joined.csv --out gpw_experiment/regret
python scripts/position_features.py --positions-dir test_positions500 --referee-csv gpw_experiment/referee \
  --cluster-dir gpw_experiment/A11a --medoid-dir gpw_experiment/A2 --joined gpw_experiment/regret/reinvest_joined.csv \
  --arms A1,A2,A9,A9P5,A9P5N,A11b,A11a --risk-lambda 0.5 --out gpw_experiment/features
python scripts/analyze_when_clustering_helps.py --features gpw_experiment/features/position_features.csv \
  --out gpw_experiment/features/analysis_new300 --exclude-positions test_positions200 \
  --targets d_A11a_A9P5,d_A9P5N_A9P5,d_A9P5_A2,d_A9_A2,screen_loss,rho_gain,eta2_q --value-arms A9P5,A11a --blind-arms A2
```
`cluster_table.csv` の `rep_rank` (開いた順) と `rep_visits` (root からの訪問数) で、PW がどのクラスタをいつ開き
どれだけ調べたかが局面ごとに追える。`reinvest_results.csv` の `num_children` = 最終的に開いた子数。

## 第1波の結果 (2026-08-30) → `gpw_experiment/regret500/WAVE1_SUMMARY.md`
A9P5N (noisy-tree) 0.309 ≪ A1 0.418 ≈ A9P5 0.423 ≈ A11a/b 0.425-0.427 (Holm p<1e-10)。利得は木の評価 (tree_loss 0.19→0.10)。
PW は被覆 0.20→0.29 だが tree_loss が同じだけ悪化し regret 不変。**第2波に A1N (対照) を必ず入れる**。

## 第3弾 (2026-09-01 実装): A12 ClusterTS = 階層ベイズ縮約 + Thompson sampling
先生の「ベイジアンアプローチ」の具体化。root のみ、depth>0 は A9 と同じ。
1. 縮約: 候補 i の真値に 事前 N(μ_c, τ_c²) (μ_c=クラスタ平均, τ_c²=クラスタ内分散)、観測 e_i ~ N(·, sd_i²/R_pre)
   → 事後 m̃_i (少標本で上振れた候補ほどクラスタ平均へ引き戻される = winner's curse 対策)。
2. 代表 = クラスタ内 m̃ 最大。**全クラスタ (~15-30) を子に持ち**、事前 N(m̃_rep, prior_scale·ṽ_rep) を付与。
3. 訪問配分 = Thompson sampling (事後 = 事前 ⊕ 子の訪問統計, 観測分散 ts_obs_var)。未訪問の低事前クラスタは
   自然に切られ、有望クラスタに厚く配る = **ベイズ版 progressive widening** (A11 の決め打ち式 k(N) の置き換え)。
4. 最終着手 = 事後平均最大の子 (最多訪問ではない)。
フラグ: `--method ClusterTS --ts-obs-var 2.0 --ts-prior-scale 2.0`。アーム: **A12N** (noisy, 主; vs A9P5N/A1N) / A12 (決定的; vs A9P5)。
検証: A12N が A9P5N (0.309) を下回るか。被覆 (best_is_rep) と screen_loss は position_features.py がそのまま出す。
**注意: C++ 変更ありのため各サーバーで git pull + lion の docker リビルドが必要。**

## 第4弾 (2026-09-03 設計): 査読対応 — 外乱込みアブレーション + 予算スイープ

### A. アブレーション (500局面, P=200, 全て --noisy-tree)  «最優先»
A9P5N (0.309) と A1N (0.334) の差を「子数 K × 選択規則 × クラスタ構造」に分解する。

| ペア | 単離できるもの | 予想 (事前登録) |
|---|---|---|
| **A8N vs A9P5N** (両方 K=4, 価値) | **クラスタ構造の寄与** (本命。決定的世界では A8≈A9 だった) | 差なし〜小 = 最大リスク |
| A5N4 vs A8N (両方 K=4) | 価値スクリーンの寄与 | 価値が大きく勝つ |
| A1N vs A5N4 / A5N | 削減そのものの寄与 | 小 (K=4 ランダムは被覆で損) |
| A2N vs A5N (両方 K≈18) | クラスタ構造の寄与 (価値なし側) | 差なし (決定的世界の A2≈A5 と同様) |
| A2N vs A9P5N | 既存手法+noisy vs 提案 (査読の「既存手法との比較」) | 提案が勝つ (ただし K 交絡あり→本文で A8N 経由の解釈を明記) |

新アーム: A8N (ScoreTopK+noisy, K=4) / A5N (RandomK+noisy, K≈18) / A5N4 (RandomK+noisy, `--k-abs 4`) / A2N (既定義)。
`--k-abs` は新設 (root の子数を絶対指定; Proposed/RandomK)。計 4 アーム × 500 × 5 = 10,000 単位 ≈ 4台で 2 日。

### B. 予算スイープ (200局面 = test_positions500 の先頭200, A1N vs A9P5N)
- **P ∈ {100, 400, 800}** (P=200 は既存)。K_cap = P/v_target の**仕様のまま** (K = 2/8/16 と予算に応じて自動調整
  されるのは手法の一部)。審判は予算非依存なので再利用。
- 見るもの: d(P) = regret_A9P5N − regret_A1N の曲線。低予算で負 (クラスタ有利)、高予算で 0 に近づく/交差するか。
- **P=10000 はやらない**: コスト ~50× (1局面 ~33h, 全体で数週間) で締切内に不可能。かつ不要 —
  ①主張は時間制約下 (低予算) の優位で、大会レギュレーションに対応するのは P=100-400 帯。
  ②高予算では両者とも審判の argmax に収束し差が審判分解能 (SE≈0.10) 以下に沈む。
  ③知りたい情報は「交差点がどこか」で、P=800 までに傾向が出なければ P=1600 を 100 局面だけ追加すれば足りる。
- 計 200 × 5 × 2 アーム × (0.5+2+4)/1 ≈ 13,000 単位 ≈ 4台で 2 日。A の後に投入。

### C. 小改修 (Aと同時)
- r≤1 スクリーンバイパス (未実装): 最終ショットの退化対策。実装後 32 局面だけ再走。

## 計算量の目安 (run200 = 5,000 局面·seed 単位で 4 台 1.5 日)
- 既存4アーム × 300 × 5 = 6,000、新3アーム × 500 × 5 = 7,500、審判 300 局面 ≈ 3,200 相当 → **run200 の約 3.3 倍 ≈ 4-5 日** (4台)。
  任意アーム (A5/A11c/A11aN) を足すと +2,500 ずつ。
