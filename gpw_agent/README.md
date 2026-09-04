# gpw_agent — GPW2026 デジタルカーリング大会用エージェント

研究用コード (`src/`, `experiments/`) とは独立した、大会で勝つためのエージェント。
相手プログラム (Jiritsukun-Jr 等) はスパーリング相手として実行するだけで、コードは参照しない。

## 設計 (Phase 1 → Phase 2)

- **物理**: DigitalCurling3 FCV1 をスレッドごとに 1 インスタンス (`src/sim.*`)。逆運動学 (`VelocitySolver`) は
  ドリフト量をキャッシュし、起動時に主要速度をプレウォーム。
- **候補手** (`src/candidates.*`): draw / guard / come-around / freeze / hit(±offset) / peel / raise / through の語彙。
  5 ロックルール中は相手 FGZ 石へのヒットを生成しない。
- **評価** (`src/eval.*`): 盤面 → ハンマー側のエンド結果分布 p(k), k=-4..4 → 勝率表 WP(点差, 残エンド, ハンマー) で勝率へ。
  - Phase 1: 手作り (現在カウント × 残投数減衰 + 石の質/ガード保護)。
  - Phase 2: `--model model.txt` で自己対戦から学習した DeepSets モデル (`src/nn.*`) に置換。
- **探索** (`src/search.*`): 決定論プレスクリーン → 外乱込み逐次半減 (successive halving)。
  エンド最後の 4 投は相手の応手 (小さな語彙, 決定論) の min を取る深さ 2。締切超過は必ず防ぐ。
- **時間管理** (`src/timeman.h`): 残り時間 × 安全率 / 残り自分投数 × 投数重み (終盤ほど厚く)。
- **エージェント** (`src/agent.*`): 例外時フォールバック (ティーへのドロー)、自己対戦用の ε 探索。

## ビルド

```bash
# Windows (VS2022, Boost ヘッダは C:/boost_1_86_0)
cd gpw_agent && mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DBOOST_ROOT=C:/boost_1_86_0 -DBoost_INCLUDE_DIR=C:/boost_1_86_0
cmake --build . --config Release
# Linux (研究室サーバー, lion の docker ylab-project 内)
bash scripts/build_linux.sh
```

## 使い方

```bash
# 大会クライアント
./build/Release/gpw_agent.exe localhost 10000 --threads 8 --name gpw_agent --log shots.log [--model model.txt]
# 自己対戦 / A-B テスト (fixed budget 秒/投)
./build/Release/gpw_agent.exe --selfplay --games 20 --ends 10 --budget-a 1.0 --budget-b 1.0 --threads 8 \
    [--model-a m1.txt --model-b m2.txt] [--explore 0.15 --seed 3] --out data/sp.jsonl
# 物理ベンチ
./build/Release/gpw_agent.exe --bench
# ローカルで Jiritsukun-Jr と対戦 (server + jiritsu + gpw_agent を起動)
bash scripts/match_local_jiritsu.sh <logdir> [threads]
```

## 学習ループ (Phase 2)

1. サーバーでデータ生成: `bash scripts/gen_selfplay.sh data/gen1 16 40 6 0.4 0.15` (16 proc × 40 局, 10 エンド)
2. 学習: `python python/train_value.py "data/gen1/*.jsonl" --out model_v1.txt --eval-out eval_v1.txt`
3. 検証: `python python/check_model.py model_v1.txt data/gen1/sp_x_0.jsonl` と
   `./build/gpw_agent --check-model data/gen1/sp_x_0.jsonl --model model_v1.txt` の出力が一致すること
4. 強さ: `--selfplay --model-a model_v1.txt` vs 旧版 (or `--eval-b`) を 100 局以上
5. 勝ったモデルで次世代のデータを生成 (expert iteration)

JSONL レコードは投球前局面 + 選択手 + `end_result_hammer` (そのエンドのハンマー側の結果) + `result` (勝敗) を持つ。
