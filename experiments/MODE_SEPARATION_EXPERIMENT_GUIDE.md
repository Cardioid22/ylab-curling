# モード分離実験ガイド (GPW2026)

## 問い

AllGrid は同一局面でもシードを変えると選ぶ手がばらつく。これは "正解" が一意でない
（カーリングとして自然 — ドロー / テイクアウトなど戦略的に別物の好手が複数ある）ことを
意味する。本当に重要なのは、**Proposed のクラスタリングが「AllGrid が正解と判断しうる、
戦略的に異なる複数の手」を別々のクラスタに分離できているか**である。

失敗モード = 戦略的に異なる複数の正解が **1 つのクラスタに集約 (collapse)** されること。
代表点 (medoid) が 1 つしか出ないため、Proposed は複数の正解のうち 1 つしか検討できなくなる。

これは既存の `same_cluster`（Cluster Agreement）指標の**裏返し**にあたる:
- `same_cluster` は AllGrid の選択が Proposed の最良クラスタに入るかを測り、**collapse を報酬扱い**
  する（K=1 で常に 100%）。
- 本実験の Mode Recall は **collapse を罰する**。両者で「K をいくつにするか」のトレードオフの
  両端を成す。

## 定義（確定）

| 概念 | 定義 |
|---|---|
| 正解集合 A | AllGrid (A1) を R シード回し、選ばれた手 `candidate_idx` の集合（頻度つき） |
| モード | A 内の候補を `shot_type` でグループ化（Draw/Hit/Peel/Freeze/ComeAround…） |
| 多峰局面 | `m = AllGrid が選んだ異なる shot_type 数 ≥ 2`。**本実験の主対象** |

> Proposed の展開は `simulateNoRand`（決定的）なので、候補プールとクラスタリングは
> **局面ごとに seed 非依存** = 審判が採点したプールと同一。よって `cluster_table` は
> 任意の 1 seed を使えばよい。一方 AllGrid の選択は rollout の rng + 物理ノイズで
> **seed 依存**（= 観察された分散そのもの）。

## 指標

- **Mode Recall（主指標）**: AllGrid が選んだ各 shot_type が Proposed の K 代表点に
  現れる割合。`(代表点に現れたモード数) / (AllGrid が選んだ全モード数)`。
  2 モードが 1 クラスタに collapse すると代表点は片方しか出せず Recall < 1。
- **Separation Rate（診断）**: AllGrid が選んだ「異なる shot_type の手のペア」が
  別クラスタにある割合。
- **Collapse 件数**: 異なる shot_type の選択手が同一クラスタに同居した件数（+ 詳細）。
- **AllGrid 多峰性**: distinct idx 数 / distinct type 数 / top1 頻度 / type エントロピー。
- **RandomK 対照 (A5)**: 同じ K でランダム削減した場合の Mode Recall（seed 平均）。
  クラスタリングが「単なる削減」より正解モードをよく分離できるかの ablation。
- **（任意）審判による質検証**: A の各モードが `q_ref` で本当に好手か
  （最良 q_ref から ε 以内）。AllGrid の分散がノイズでなく真の多峰性であることの裏付け。

## 実行手順

### 1. ビルド（cluster_table 出力を含む）
```bash
cmake --build build --config Release --target ylab_client
```
`reinvest_experiment` が Proposed/RandomK 実行時に
`<output-dir>/cluster_table.csv`（`candidate_idx, cluster_id, is_representative, shot_type, label`）
を追加出力する。

### 2. アーム実行（A1=正解集合, A2=クラスタ割当, A5=対照）
```bash
./scripts/run_reinvest.sh \
    --arms "A1,A2,A5" \
    --num-seeds 20 \
    --positions-dir test_positions_categorized8 \
    --n-states 8 \
    --parent-dir experiments/mode_separation/run1
```
- A1 は R=20 シードで AllGrid の選択分布を作る（多峰性を捉えるため R は大きめ推奨）。
- A2 は決定的なので 1 seed で十分だが、20 seed 回しても結果は同一（無害）。
- A5 は seed 依存なので 20 seed の平均をとる。

### 3.（任意）審判で質検証
```bash
./scripts/run_referee.sh   # score_move_qtable.csv を出力
```

### 4. 集計
```bash
python3 scripts/aggregate_mode_separation.py \
    --reinvest-dir experiments/mode_separation/run1 \
    --referee-dir  experiments/reinvest_referee \
    --out          experiments/mode_separation/run1/analysis
```

出力:
- `mode_separation_per_position.csv` — 局面ごとの全指標
- `mode_separation_summary.txt` — 全体 + 多峰部分集合のサマリ、collapse 詳細
- `mode_recall_proposed_vs_randomk.png` — 多峰局面での Mode Recall 比較

## 読み方

1. **まず多峰局面が何個あるか**を確認（summary の「多峰局面 (m>=2)」）。
   - 0 件なら既存 8 局面では AllGrid が単峰に寄っている → R を増やす or 多峰局面を新規選定。
2. 多峰局面で **Proposed Mode Recall** が高い（→ 複数正解を検討できている）か、
   **RandomK** より明確に高い（→ クラスタリングの寄与）かを見る。
3. **collapse 詳細**でどの shot_type ペアが集約されたかを特定 → 距離関数 `dist()` /
   クラスタ数 K の改善対象がわかる。

## K 掃引（発展）

`run_reinvest.sh` の `RETENTION`（K = ceil(N × retention)）を振り、Mode Recall vs K を描く。
多峰局面で高 Recall を保つ最小 K がコスト効率の最適点。
