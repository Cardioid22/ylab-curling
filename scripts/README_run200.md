# run200 キャンペーン — GPW2026 本論文用 (200局面 × リスク統合クラスタ価値 × 局面特徴)

アブストで約束した 3 点を埋めるための実験。締切: 採否 9/4 → **最終原稿 10/23**。

1. **局面拡大** 50 → 200 (`test_positions200`: 先頭50 = `test_positions50` と同一、+150 は石数×進行度で層化サンプル、seed 23)
2. **リスク統合クラスタ価値** `ClusterValue --risk-lambda λ`: クラスタ価値 = μ_c − λ·σ_c
   (σ_c = メンバー全ロールアウト標本のプールSD)。リスク系は sd_i 安定化のため **R_pre=5**:
   `A9P5`(λ=0, 対照) / `A9R05`(λ=0.5, 主) / `A9R10`(任意)。`A9`(R_pre=3, λ=0) はアブストのまま。
   **実測**: R_pre 3→5 で actual_total_sims は **+8%** (A9 59.3k → A9P5 64.4k, 200局面)。A9P5/A9R05 は A1/A2/A9 より 8% 多く計算している点を本文で明記 (等予算判定 ±10% 内)。
   **λ のオフライン掃引**: A9P5 の cluster_table には cluster_sd が入るので、λ を変えたとき採用クラスタが
   どれだけ変わるか (`topk_overlap_l{λ}`) や μ−λσ の妥当性 ρ は **λ アームを回さなくても** position_features.py で計算できる。
   λ アームが必要なのは「木を通した最終的な手の質 (regret)」への影響だけ。
3. **どんな局面で有効か**: `scripts/position_features.py` → `scripts/analyze_when_clustering_helps.py`

全て **修正木 (エンド跨ぎ修正 e9f0dde 以降)** で計測する。旧 run50 のデータは使わない。

---

## 0. 各サーバーで 1 回 (ビルド)
```bash
git pull
rm -f build/ylab_client && cmake --build build --config Release --target ylab_client
./build/ylab_client --reinvest-arm --method ClusterValue --risk-lambda 0.5 --depth 3 --playouts 20 \
   --rollouts-per-visit 2 --states 200 --start-index 2 --max-positions 1 --threads 1 --seed 42 \
   --load-positions test_positions200 --output-dir /tmp/smoke_l05   # 配管確認 (1局面, 数十秒〜数分)
grep risk_lambda /tmp/smoke_l05/cluster_table_idx2.csv | head -2   # 末尾3列 cluster_sd,cluster_value_risk,risk_lambda
```

## 1. 先頭 50 局面の既存結果を流用 (ローカル or 集計するマシンで 1 回)
```bash
./scripts/run200_reuse_first50.sh     # run50v2 → run200/<ARM>/seed_S/*_idx0.csv, 審判 → run200/referee/*_idx0.csv
```

## 2. 審判 (bear, 新規 150 局面のみ, K=200) — 約 10h (推定)
```bash
mkdir -p reinvest_experiment/run200/referee
nohup ./build/ylab_client --score-move --score-rollouts 200 --states 200 --threads 120 --seed 42 \
    --load-positions test_positions200 --start-index 50 --max-positions 150 \
    --output-dir reinvest_experiment/run200/referee \
    < /dev/null > reinvest_experiment/run200/referee/referee_idx50.log 2>&1 &
# 出力: reinvest_experiment/run200/referee/score_move_qtable_idx50.csv
```
審判は予算非依存なので、アームより先に走らせても後でもよい (集計時に join するだけ)。

## 3. アーム (P=200, R=10, 5 seed; 等予算 ≈ 55k sims/局面)
既存アーム (A1/A2/A5/A7/A9) は **残り 150 局面だけ** (`--start-index 50 --max-positions 150`)。
新アーム (A9R05/A9R10) は **200 局面全部**。1 局面あたり 1 スレッド, 平均 ~40 分・混戦局面は 4-8h。

| マシン | アーム | コマンド | 目安 |
|---|---|---|---|
| bear (128) | A9P5, A9R05 (200局面) | 下記 (a) | 10 job × ~5h / 2 並列 ≈ 25h (審判と同時なら threads を 40 に)。余裕があれば A9R10 |
| jaguar (128) | A1, A9 (150局面) | 下記 (b) | 10 job × ~4h / 2 並列 ≈ 20h |
| lion (96) | A2 (150局面) | 下記 (c) | 5 job × ~6h / 2 並列 ≈ 15h |
| tiger (96) | A5 (150局面; ランダム削減の対照) | 下記 (d) | 同上 |

```bash
# (a) bear
nohup ./scripts/run_reinvest.sh --arms "A9P5,A9R05" --positions-dir test_positions200 --n-states 200 \
    --num-seeds 5 --max-parallel 2 --threads-per-seed 60 \
    --parent-dir reinvest_experiment/run200 < /dev/null > reinvest_experiment/run200/launch_bear.log 2>&1 &
# (b) jaguar
nohup ./scripts/run_reinvest.sh --arms "A1,A9" --positions-dir test_positions200 --n-states 200 \
    --start-index 50 --max-positions 150 --num-seeds 5 --max-parallel 2 --threads-per-seed 60 \
    --parent-dir reinvest_experiment/run200 < /dev/null > reinvest_experiment/run200/launch_jaguar.log 2>&1 &
# (c) lion
nohup ./scripts/run_reinvest.sh --arms "A2" --positions-dir test_positions200 --n-states 200 \
    --start-index 50 --max-positions 150 --num-seeds 5 --max-parallel 2 --threads-per-seed 48 \
    --parent-dir reinvest_experiment/run200 < /dev/null > reinvest_experiment/run200/launch_lion.log 2>&1 &
# (d) tiger
nohup ./scripts/run_reinvest.sh --arms "A5" --positions-dir test_positions200 --n-states 200 \
    --start-index 50 --max-positions 150 --num-seeds 5 --max-parallel 2 --threads-per-seed 48 \
    --parent-dir reinvest_experiment/run200 < /dev/null > reinvest_experiment/run200/launch_tiger.log 2>&1 &
```
監視: `tail -f reinvest_experiment/run200/*/seed_*/run.log` (`[done N/M]` 行)。
CSV は **全局面完了後に一括書き出し** (途中 kill で全損) なので、途中で止めない。
余裕があれば A9R10 / A7 (ScoreScreen) も同じ形で追加。

## 4. 回収と集計 (ローカル)
サーバー側の出力 `reinvest_experiment/run200/` (共有FS なのでどのサーバーからでも可) をローカルの **`gpw_experiment/`** に集める:
```bash
rsync -av lion:~/ylab-curling/reinvest_experiment/run200/ gpw_experiment/
bash scripts/run200_reuse_first50.sh --dst gpw_experiment     # 先頭50局面 (run50v2) を *_idx0.csv として流用
```
そのあと:
```bash
# regret (通常 + リスク調整 λ=0.5)
python scripts/aggregate_reinvest.py --reinvest-dir gpw_experiment \
    --referee-dir gpw_experiment/referee --pair A9R05,A9 --risk-lambda 0.5 \
    --out gpw_experiment/regret
python scripts/regret_stats.py --joined gpw_experiment/regret/reinvest_joined.csv --out gpw_experiment/regret
python scripts/regret_stats.py --joined gpw_experiment/regret/reinvest_joined.csv --metric regret_adj --out gpw_experiment/regret_adj

# 局面特徴 (盤面 + 審判 + クラスタ構造 + 成果) → どんな局面で効くか
python scripts/position_features.py --positions-dir test_positions200 \
    --referee-csv gpw_experiment/referee \
    --cluster-dir gpw_experiment/A9 --medoid-dir gpw_experiment/A2 \
    --joined gpw_experiment/regret/reinvest_joined.csv \
    --arms A1,A5,A2,A9,A9P5,A9R05 --risk-lambda 0.5 --out gpw_experiment/features
python scripts/analyze_when_clustering_helps.py \
    --features gpw_experiment/features/position_features.csv \
    --out gpw_experiment/features/analysis \
    --targets d_A9_A1,d_A2_A1,d_A9_A2,d_A9R05_A9P5,dadj_A9R05_A9P5,screen_loss,rho_gain,eta2_q
# クラスタ価値の妥当性 (ρ) は position_features.csv の rho_cand / rho_cluster / rho_cluster_risk_l0.5 列に局面ごと入っている
```
`--cluster-dir` を `run200/A9R05` にすると、リスク統合版のクラスタ表 (cluster_sd, cluster_value_risk 列つき) から同じ特徴が出る。

## 5. 読み方 (論文の節に対応)
- **性能を損なわないか**: regret_stats (Friedman/Wilcoxon+Holm)。A9R05 が A9P5/A1 と同水準なら「リスク統合しても劣化なし」。
  λ の比較は **A9R05 vs A9P5** (R_pre を揃えた対)。リスク調整 regret (`regret_adj`) で A9R05 が良くなるのは設計上当然の側面あり → 本文で明記。
  通常 regret で A9R05 ≥ A9P5 なら「推定ノイズへの頑健化 (winner's curse 抑制)」の証拠になる。
- **リスクの位置づけ (方針)**: 単一エンド E[score] の枠では λ>0 は平均 regret を改善できない (定義上)。本論文ではリスクを
  **性能レバーではなく説明レバー** として扱う: 各エリアに (μ_c, σ_c) = 「見込み ±ブレ」を付け、「安全なエリア/博打のエリア」を
  可視化し、λ で採用エリアが入れ替わるのはどんな局面か (q* 近傍に複数エリアがある多峰局面か) を示す。
- **クラスタ価値の妥当性**: `rho_cluster` vs `rho_cand` (アブストの 0.49 vs 0.30 を 200 局面で更新)、`rho_cluster_risk_l0.5` vs `rho_cand_risk_l0.5`。
- **どんな局面で有効か**: `analyze_when_clustering_helps.py` の相関表と層別表。50 局面の予備結果 (run50v2/features/analysis) では
  - `screen_loss`(代表手に最良手が残らない損失) は「q* 近傍の候補が少ない = 正解が一意な局面」ほど大きい
  - `rho_gain`(クラスタ平均の分散削減効果) はハウス混雑 (n_house, n_within_0.61) が増えるほど消える
  - `eta2_q`(クラスタが価値的に意味ある区分か) は巨大クラスタ退化 (largest_frac≥0.5) でほぼ 0 になる
  という構造が見えている。200 局面で確定させる。
