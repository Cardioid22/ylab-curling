# GPW2026 Extended Abstract 作業ディレクトリ

## Overleafでの使い方
1. このディレクトリ (`gpw2026/`) の中身を丸ごとzipにするか、Overleafで新規プロジェクトを作って
   `draft.tex`, `ipsj.cls`, `ipsjtech.sty`, `figures/` をアップロード。
2. Overleafのコンパイラを **pLaTeX** に設定（メニュー "Menu" → "Compiler"）。
   IPSJスタイル(`ipsj.cls`)は和文pLaTeX前提。
3. `draft.tex` をメインファイルに指定してコンパイル。
4. ページ数・フォントサイズ(10pt以上)を確認し、はみ出す場合は本文を削るか図を縮小。

## 現状の未確定事項 (締切前に確認・更新すること)
- `draft.tex` 冒頭の `\newcommand` 群は **seed42のみの速報値**。
  bearで実行中の `A1d1` 残り4シード (43-46) が揃ったら:
  ```
  python scripts/analyze_cluster_switch.py \
    --d3-dir reinvest_experiment/run50v2/A1 \
    --d1-dir reinvest_experiment/run50v2/A1d1 \
    --d1-extra reinvest_experiment/run50_A1d1fix/seed_42 \
    --cluster-dir reinvest_experiment/run50v2/A9 \
    --out reinvest_experiment/run50v2/cluster_switch
  ```
  を再実行し、`\switchEndgame` 等と `\agreeEndgame` 等の値を更新。
  図も `figures/cluster_switch_draft_seed42.png` 等を新しい出力で差し替え
  (ファイル名の `_seed42` は外して良い)。
- 著者所属・メールアドレスはGI58から流用。変更があれば `draft.tex` の `\author`/`\affiliate` を修正。
- 参考文献 `naka_gi58` の巻号ページはGI58の正式な発行情報が確定次第、埋める。
- 図1 (`area_map_B_hull_draft.png`) は3局面を横に並べた版。2段組の1カラム幅に収めると
  小さくなりすぎる場合は、1局面だけの単独版を作り直すことを検討
  (`scripts/plot_area_variants.py --positions <1局面>` で生成可能)。

## データの出典
`DATA_INDEX.md` を参照。どの数値・図がどの実験データ由来かの対応表。

## 全体像 (何を主張する原稿か)
1. Proposed(A2)の代表手選出は価値盲目 → ClusterValue(A9)で得点期待値ベースに変更
2. 性能は劣化しない (A9 ≈ A2 ≈ A1、regret横並び、Friedman n.s.)
3. **有望な結果**: クラスタ価値の妥当性が5シード一貫して確認 (ρ_cluster > ρ_candidate)
   → エリア価値マップという分析基盤の提供
4. **有望な結果**: 深さ探索の効果をクラスタ単位で分析すると、終盤ほど「戦術グループの
   乗り換え」が起きることを発見
5. 性能面の優劣ではなく、価値を考慮したクラスタリングの分析的価値を主軸にした構成
