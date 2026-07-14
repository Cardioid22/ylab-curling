# AGENTS.md — ylab-curling

デジタルカーリングAIの研究リポジトリ (MCTS + 候補削減戦略の比較研究、C++17 + Python分析)。

## 最初に読むもの

1. **`docs/PROJECT_STATUS.md`** — 研究の現況・アーム用語集・主要結果・実行中タスク・ハマりどころの単一ソース。
   **作業前に必ず読むこと。実験状況を進めたら必ず更新すること。**
2. `CLAUDE.md` — ビルド手順・コード構成の詳細 (Claude Code向けだが内容は共通)

## ビルド・実行の要点

```bash
# ビルド (初回: mkdir build && cd build && cmake ..)
cmake --build build --config Release --target ylab_client
# バイナリ: Windows=build/Release/ylab_client.exe, Linuxサーバー=build/ylab_client
```

- 研究実験のエントリポイントは `main.cpp` のCLIフラグ (`--reinvest-arm`, `--selfplay`, `--score-move` 等)。
  サーバー(lion/tiger/jaguar/bear)での長時間実行が前提のものが多い。
- 分析は `scripts/*.py` (Python 3 + numpy/scipy/matplotlib)。

## 重要な約束事

- **実験結果 (regret表・勝率・Δ曲線) を引用するときは `docs/PROJECT_STATUS.md` の数値と注記
  (max-max木時代か negamax修正後か) を確認する**。b84b22b 以前の木系結果は旧仕様の木。
- アームの評価は P≥100 で行う (P=50 では A7/A8/A9/A10 が縮退して同一化する)。
- 図の色は Okabe-Ito CVD-safe パレット (A1=#D55E00, A2=#0072B2, A5=#CC79A7, A7=#009E73)。
- コミットメッセージは英語 (feat:/fix:/data:/chore:)、本文に実験の数値要約を含める慣習。
- 「歩」(Ayumu) と書かれた自己対戦相手は gPolicy 単体 (本物の歩ではない)。対外表記は「gPolicy単体」。
- Linuxで実行中バイナリがあるときは `rm -f build/ylab_client` してからビルド (ETXTBSY回避)。
