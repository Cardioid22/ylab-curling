# 計算再投資実験 — 実行手順 (GPW2026)

理論背景は [`experiments/REINVESTMENT_EXPERIMENT_GUIDE.md`](../experiments/REINVESTMENT_EXPERIMENT_GUIDE.md)。
ここは「ビルド済みバイナリで実験を回す運用手順」をまとめる。

**問い:** クラスタリングで候補を絞って浮いた計算予算を「探索の深さ(3→5)」と
「葉あたりロールアウト数」のどちらに再投資すると、等予算で手の質(リグレット)が上がるか。

## アームと予算
全アームを**同一の総物理シミュ予算 B**で走らせる。B = `run_single_simulation`(ロールアウト/審判)
+ `simulateNoRand`(ノード展開) の総コール数。各局面の `actual_total_sims` 列に実測が出る。

| アーム | method | depth | 配分 | 目的 |
|---|---|---|---|---|
| A1 | AllGrid  | 3 | 基準 (P,R) | 削減なし基準 |
| A2 | Proposed | 3 | A1と同じ | クラスタリング効果の単離 |
| A3 | Proposed | 5 | 深さ再投資 (P減) | 深さ再投資 |
| A4 | Proposed | 3 | ロールアウト再投資 (R増・P減) | 評価精度再投資 |
| A5 | RandomK  | 3 | A2と同じ | クラスタリング vs 単なる削減 |
| A6 | AllGrid  | 5 | (任意) | 深さ5が予算内で破綻する実証 |

**比較の読み:** A2 vs A1=クラスタリングの寄与 / A3 vs A2=深さ再投資 / A4 vs A2=ロールアウト再投資 /
**A3 vs A4=本実験の主問い** / A5=賢い削減(クラスタリング)が効いているか。

---

## 0. ビルド (各マシンで1回)
```bash
cmake --build build --config Release --target ylab_client
```

## 1. スモーク (配管確認 — 本番前に必ず)
1局面・小予算で各アームを回し、深さ切替・RandomK・simカウンタ・出力スキーマ・審判 join を確認:
```bash
# 速い局面 (shot_num=14, 残り2手) で全アーム + 審判を回し、join を検証
B=build/Release/ylab_client.exe
for m in AllGrid Proposed RandomK; do
  "$B" --reinvest-arm --method $m --depth 3 --playouts 20 --rollouts-per-visit 3 \
       --states 10 --start-index 7 --max-positions 1 --threads 1 --seed 42 \
       --load-positions test_positions_isobudget10 --output-dir experiments/smoke/$m
done
"$B" --reinvest-arm --method Proposed --depth 5 --playouts 12 --rollouts-per-visit 3 \
     --states 10 --start-index 7 --max-positions 1 --threads 1 --seed 42 \
     --load-positions test_positions_isobudget10 --output-dir experiments/smoke/Proposed_d5
"$B" --score-move --score-rollouts 30 --states 10 --start-index 7 --max-positions 1 \
     --threads 4 --seed 42 --load-positions test_positions_isobudget10 \
     --output-dir experiments/smoke/referee
```
確認: depth5 の `actual_total_sims` > depth3、RandomK を同 seed で2回回すと同じ `candidate_idx`、
審判 CSV と `(game_id,end,shot_num,candidate_idx)` で突き合わせ可能。

## 2. 予算 B の校正 (最重要)
予算は実シミュ回数で揃える(深さ/ロールアウト数では揃わない)。
1. `scripts/run_reinvest.sh` 冒頭の `P_BASE/R_BASE/P_DEEP/P_RINV/R_RINV/RETENTION` を編集。
2. 各アームを少 seed (例: `--num-seeds 1`) で回し、`aggregate_reinvest.py` の
   **Budget check** を見る。全アームの平均 `actual_total_sims` が B±10% に収まるよう P/R を調整。
   - A3(深さ5)は展開が増えるので `P_DEEP` を下げて B を維持。
   - A4 は `R_RINV` を上げ `P_RINV` を下げて B を維持。
3. **低予算厳守**(高予算は1局面で数日。ガイド §6)。A1 が各候補を平均数十 visit する程度に。

## 3. 審判 (1回だけ・高K)
全アーム共通の物差し。校正後、一度だけ高 K で実行:
```bash
./scripts/run_referee.sh --k 1000 --threads 16 \
    --positions-dir test_positions_isobudget10 --n-states 10 \
    --output-dir experiments/reinvest_referee
```

## 4. 本番 (マシンにアームを割り当てて分散)
**分散はアーム単位**。各マシンで担当アームだけ起動する。同一 `--parent-dir` を共有
(NFS など) するか、後で1か所に集める。
```bash
# 強いマシン (例: bear) — 重い深さ5を担当
./scripts/run_reinvest.sh --arms "A3" --num-seeds 8 --parent-dir experiments/reinvest/run1

# 他マシン — 軽いアームを分担
./scripts/run_reinvest.sh --arms "A1,A2"   --num-seeds 8 --parent-dir experiments/reinvest/run1
./scripts/run_reinvest.sh --arms "A4,A5"   --num-seeds 8 --parent-dir experiments/reinvest/run1
```
- `--num-seeds K`: 各アーム K seed (multi-seed; 低予算=高分散なので 5〜10 推奨)。
- `--threads-per-seed T`: 1プロセスのスレッド数 (既定=局面数=10)。同時 (arm,seed) は `--max-parallel`。
- 出力: `<parent>/<ARM>/seed_<S>/reinvest_results.csv`。
- 共有FSが無い場合は各マシンの `<parent>/<ARM>` を後で1つの parent 下にコピーすればよい。

## 5. 集計
```bash
python3 scripts/aggregate_reinvest.py \
    --reinvest-dir experiments/reinvest/run1 \
    --referee-dir  experiments/reinvest_referee \
    --pair A3,A4 \
    --out experiments/reinvest/run1/summary
```
出力: アームごと平均リグレット・平均 `actual_total_sims`(等予算検証)・**A3 vs A4 の勝率**。

---

## 再現性とシードが制御するもの (重要)
`--seed S` と `state_seed = S ^ (global_idx * 0x9E3779B97F4A7C15)` が制御するのは:
- **ロールアウト方策** (ε-greedy グリッドの選択 RNG)。
- **RandomK の候補サブセット選択** (`state_seed ^ 盤面ハッシュ` で決定的。同 seed なら必ず同じ K 個)。
- 候補生成 `generatePool` はノイズなし (`simulateNoRand`) で常に決定的。

一方、**ロールアウトの実行ノイズ (PlayerNormalDist の着地ばらつき) は seed で固定していない**
(既存 depth_n 実験と同じ挙動)。これは意図的で、実行不確実性そのものが本実験の測定対象だから。
したがって:
- 同一 (arm, seed) を再実行しても、最終選択手は実行ノイズで変わりうる (`actual_total_sims` はほぼ一定)。
- **だから multi-seed (5〜10 seed) で平均する**。1 seed の単発結果に意味を持たせない。
- A3 vs A4 の head-to-head は seed を複製インデックスとみなした対応比較 (同一ノイズ実現の共有まではしない)。

## 注意 (公平性のため必ず守る — ガイド §5)
- ロールアウト方策・葉評価は全アーム共通 (ε-greedy 4×4グリッド, ε=0.3)。再投資は深さ or ロールアウト数のみ。
- 総シミュ予算 B は全アーム同一。A3/A4 は同じ B の再配分。
- 同じ局面・同じ seed 規約 (`state_seed = base_seed ^ (global_idx * 0x9E3779B97F4A7C15)`)。
- 候補生成 `generatePool` は全アーム共通(候補 index の join キーが一致する)。
