# 計算再投資実験 — 実装ガイド (GPW2026向け)

別の Claude Code セッションがこの仕様を参照して実験プログラムを作成するためのガイド。
このガイドだけで実装に着手できるよう、目的・設計・コード接点・出力形式・制約・スコープを記す。

---

## 1. 目的と仮説

クラスタリングで候補を絞ると、同じ計算予算をより少ない候補に集中できる。
**問い: 浮いた分を「探索の深さ(3→5)」と「葉あたりロールアウト数」のどちらに再投資すると、等予算で手の質(リグレット)が上がるか。**

- 深さ再投資: クラスタリングは分岐数を削るので、深さを増やすコスト(分岐の累乗爆発)を抑えられる。AllGrid は 63⁵ で深さ5不可能、Proposed は 12⁵ で可能。**深さ再投資こそクラスタリング固有の価値**。
- ロールアウト再投資: 葉が減るので葉評価を厚くでき、分散が下がる。
- ユーザー仮説: 実行ノイズ(外乱)のため深く読む価値は逓減する → 深さ5は効かず、ロールアウト再投資が勝つ可能性。**この仮説をデータで検証する**のが本実験。結果がどちらでも論文化可能。

詳細な戦略背景はプロジェクト記憶 `project_gpw2026_reinvestment_experiment` / `project_tournament_goal_and_clustering_role` を参照。

---

## 2. 実験設計

### アーム (すべて同一の総予算 B で実行)

| 記号 | mode | depth | 配分 | 目的 |
|---|---|---|---|---|
| A1 | AllGrid | 3 | 基準配分 (P, R) | 削減なし基準 |
| A2 | Proposed | 3 | A1と同じ (P, R) | クラスタリング効果の単離(等予算・等配分) |
| A3 | Proposed | 5 | 深さに再配分 (Pを減らしBを維持) | 深さ再投資 |
| A4 | Proposed | 3 | ロールアウトに再配分 (Rを増やしPを減らしBを維持) | 評価精度再投資 |
| (任意) A5 | RandomK | 3 | A2と同じ | クラスタリング vs 単なる削減の単離 |
| (任意) A6 | AllGrid | 5 | B維持 | 深さ5が予算内で破綻することの実証 |

**比較の読み:**
- A2 vs A1 → クラスタリングの寄与(等予算・等配分)
- A3 vs A2 → 深さ再投資の効果
- A4 vs A2 → ロールアウト再投資の効果
- **A3 vs A4 → 本実験の主問い(深さ vs ロールアウト、どちらに投資すべきか)**
- A5 → Proposed が A5 に勝てば「賢い削減=クラスタリングが効いている」(v2主張の肝)

### 等予算の定義 (最重要・ここを外すと実験が無意味)

**予算 B = 総物理シミュレーション呼び出し回数**(`SimulatorWrapper::run_single_simulation` の総コール数)。
理由: 物理シミュがコストの支配項。深さ5は展開(generatePool)が増えて1プレイアウトあたりのコストが上がるため、playout数やrollout数では公平に揃わない。**必ず実シミュ回数で揃える。**

実装: スレッドセーフなカウンタ(`std::atomic<long long>`)で `run_single_simulation` の呼び出しを数え、各アームの実測値を出力に含める。各アームの (P, R) を調整して実測シミュ数を目標 B ± 数% に合わせる。最低限、実測シミュ数と wall-clock を出力に記録し、後段で正規化・検証できるようにする。

- **すべてのアームは同じ B**。A3/A4 は「同じ B を別配分する」のであって予算を増やすのではない。
- 局面ごとに候補数が違う(33〜145)ので、B は局面ごとに「A1 が候補を各 m 回visitできる量」を基準に決めるのが自然(例: A1 が各候補を平均 ~数十visitできる低予算帯)。**低予算帯に留める**(v2主張は時間制約下の優位、かつ深さ5を現実的コストに収める)。

### 評価指標 (審判 Q_ref)

各アームが選んだ root 手を、**実行不確実性込みの審判** Q_ref で採点する。
- 審判 = `experiments/score_move_experiment.{h,cpp}`(実装済み)。`--score-move`、初手リサンプルがデフォルト(`--frozen-first-shot` で旧挙動)。
- 審判は全候補を採点した Q テーブルを出すので、**(game_id, end, shot_num, candidate_idx) でアームの選択手と join** できる。
- `regret = Q_best(局面) − Q_ref(選んだ手)`。アームごとに平均リグレット、A3 vs A4 は (局面×seed) ごとの head-to-head 勝率。
- 審判は高 K(例: 500〜1000)で1回だけ走らせ、全アーム共通の物差しにする。

### 統計 (低予算=高分散なので必須)

- 局面セット: `test_positions_isobudget10`(10局面、既存・コミット済み)。
- multi-seed: 各アーム 5〜10 seed。`scripts/run_multiseed_depth3.sh` の並列方式(seedごとに別プロセス)を流用/参考。
- 1局面あたり (アーム × seed) の選択手を集計。

---

## 3. コード接点(既存資産と追加点)

### 既存(再利用)
- `experiments/depth_n_mcts_experiment.{h,cpp}`:
  - `DepthNMctsConfig` に `depth`(汎用), `proposed_playouts`, `allgrid_playouts`, `proposed_rollouts_per_visit`, `allgrid_rollouts_per_visit`, `retention_rate` がある。
  - `enum MctsMode { Proposed, AllGrid }`。`expandNode` がクラスタリング(Proposed)/全候補(AllGrid)を分岐。`buildTree`/`runPlayout` は `config_.depth` に汎用。
  - **depth は config で可変だが、main.cpp が `cfg.depth = 3` でハードコードしている**(`main.cpp` の `depth3_mcts_mode` ブロック)。
- `experiments/score_move_experiment.{h,cpp}`: 審判(実行不確実性込み)。
- `experiments/mcts_shared.{h,cpp}`: `rolloutFromState`(ε-greedy 4×4グリッド), `makeDistanceTableDelta`, `runClustering`, `calculateMedoids`, 局面ローダ。
- `src/shot_generator.{h,cpp}`: `generatePool`(全アーム共通の候補生成。歩相当だが ComeAround/Peel 等は未生成・速度グリッド誤差積分なし。差分は監査済み、本実験では全アーム共通なので影響なし)。
- `scripts/run_multiseed_depth3.sh`: K seed 並列ランナー。

### 追加が必要なもち
1. **`--depth N` フラグ**(main.cpp): depth を 3/5 で切替可能に。現状ハードコードを引数化。
2. **`MctsMode::RandomK`(任意, A5用)**: `expandNode` で、クラスタリングの代わりに `K = ceil(N * retention_rate)` 個の候補を**ランダムに**選んで子に展開。seed 依存の決定的乱択にすること(再現性)。
3. **総シミュレーションカウンタ**: `SimulatorWrapper` に `std::atomic<long long> sim_count` を追加し `run_single_simulation` 先頭で `++`。各局面処理の前後で差分を取り、出力に `actual_total_sims` として記録。
4. **単一アーム実行への対応**: 現行 `runOneState` は Proposed と AllGrid を両方走らせる作り。再投資実験では (mode, depth, P, R) を独立に変えた**単一アームを1回の実行で回す**方が素直。`DepthNMctsConfig` に `mode`(Proposed/AllGrid/RandomK)と単一の `playouts`/`rollouts_per_visit` を持たせ、1アーム=1プロセスで回す薄いモード(例 `--reinvest-arm`)を追加するのが推奨。あるいは既存構造を活かし、アームをスクリプト側でループしてもよい。**出力スキーマ(下記)さえ守れば実装方式は任せる。**

---

## 4. 出力スキーマ(これを厳守。後段 Python の join がこれに依存)

### 各アームの選択結果 CSV (1行=1局面×1seed×1アーム)
```
game_id,end,shot_num,method,depth,playouts,rollouts_per_visit,seed,
candidate_idx,label,actual_total_sims,time_sec
```
- `method`: "AllGrid" / "Proposed" / "RandomK"
- `candidate_idx`: **`generatePool` の候補順序の index**(審判 Q テーブルと突き合わせるキー)。最重要。
- `actual_total_sims`: その局面で消費した実シミュ回数(等予算検証用)。

### 審判 Q テーブル CSV (既存スキーマ、変更不要)
```
game_id,end,shot_num,candidate_idx,label,shot_type,q_ref_mean,q_ref_sd,n_rollouts,resampled
```

### 集計(Python, 後段)
- アーム選択CSV と Q テーブルを `(game_id,end,shot_num,candidate_idx)` で left join → 各選択手の `q_ref_mean`。
- 局面ごと `q_best = max(q_ref_mean)`。`regret = q_best − 選択手の q_ref_mean`。
- アームごと平均リグレット、`actual_total_sims` が揃っているか検証、A3 vs A4 の head-to-head。

---

## 5. 方法論上の制約(公平性のため必ず守る)

1. **ロールアウト方策は全アーム共通**(ε-greedy 4×4グリッド, ε=0.3)。再投資はあくまで「深さ」か「ロールアウト数」のみ。方策自体は変えない(方策改善=評価関数学習=別トラック v3)。
2. **評価関数(葉評価)も全アーム共通**。
3. **総シミュ予算 B は全アームで同一**。A3/A4 は同じ B の再配分。
4. **同じ局面・同じ seed 規約**。`state_seed = base_seed ^ (global_idx * 0x9E3779B97F4A7C15)` の方式を踏襲(depth_n と整合)。
5. 候補生成 `generatePool` は全アーム共通(歩相当の現行版をそのまま使用)。

---

## 6. スコープと注意(1ヶ月締切)

- **必須**: A1, A2, A3, A4 + 審判 + 集計。これで主問い(深さ vs ロールアウト)に答えられる。
- **できれば**: A5(RandomK)= クラスタリングの寄与単離。論文の肝なので可能なら入れる。
- **任意**: A6(AllGrid深さ5の破綻実証)。
- **低予算厳守**: 高予算(AllGrid@10000は1局面5〜7日)は今回やらない。深さ5は低予算でのみ現実的。
- まず**スモーク**(1局面・小B・2seed)で配管(depthフラグ・RandomK・simカウンタ・出力スキーマ・審判join)を確認してから本番。
- depth5 のコストは「展開ノード増 → generatePool 呼び出し増」で効くので、スモークで実シミュ数を測り B を見積もること。

---

## 7. 完了チェックリスト
- [ ] `--depth N` で深さ3/5を切替実行できる
- [ ] `MctsMode::RandomK` が決定的乱択で動く(任意)
- [ ] `run_single_simulation` の総コール数を計測・出力できる
- [ ] 各アーム出力が §4 スキーマに一致し、candidate_idx が generatePool 順と一致
- [ ] 審判 Q テーブルと join してリグレットが算出できる
- [ ] 全アームの actual_total_sims が目標 B にほぼ揃っている
- [ ] multi-seed で 10局面 × 5-10seed × 4アームが回る
