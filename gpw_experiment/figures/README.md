# GPW2026 本論文・ミーティング用の図 (2026-09-21 生成)

生成スクリプト: `scripts/make_gpw_figures.py` (データを更新したら再実行するだけで全図が再生成される)。
入力: `gpw_experiment/regret500/joined_500full11.csv` (11アーム×500局面), `joined_noisy7.csv` (外乱込み7アーム×500局面),
`sweep_P{100,400,800}/reinvest_joined.csv` (予算スイープ, 先頭200局面)。regret はすべて審判 (score_move, K=200) 基準・5 seed 平均。

| ファイル | 内容 | 対応する査読指摘 |
|---|---|---|
| `fig_sweep_dP.png` | d(P) = regret(A9P5N)−regret(A1N) の 4点曲線 (P=100/200/400/800, ex-r1 と全局面)。低予算で最大 −0.094、単調減少で P800 で消滅 | R2⑤ 削減の効果 (予算依存性) |
| `fig_sweep_r1_split.png` | d(P) を通常局面 (n=168) と最終ショット r=1 (n=32) に分けた棒。高予算の見かけの逆転は r=1 退化層のみが原因 | R2⑤ の限界説明 / R1④ 失敗例の層 |
| `fig_regret500_arms.png` | 全15アーム × 500局面の平均 regret 序列。外乱込み評価 (青) が主レバー、その中で価値K=4 (A8N/A9P5N) が上位 | R1③ 正解・既存手法との比較 |
| `fig_ablation_decomp.png` | 外乱込み7アームのアブレーション分解: 価値スクリーン −0.257 (主因) / クラスタ構造 +0.015 (寄与なし) / 削減>全探索 −0.025 | R2⑤ 効果の分解 / R1① 差分の実証 |
| `fig_noisy_lever.png` | 決定的→外乱込み評価の対比較 (5手法対)。全手法で −0.08〜−0.12 = 手法によらない主レバー | 手法説明 (§2) の補強 |

数値の出典: `gpw_experiment/regret500/WAVE1_SUMMARY.md` (第2〜5弾) と `sweep_dP.csv`。
