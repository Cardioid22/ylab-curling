# GPW2026 本論文 (最終原稿, 締切 2026-10-23) の Overleaf プロジェクト

## 構成
| ファイル | 内容 |
|---|---|
| `main.tex` | 原稿本体 (学会配布の `jsample-gpw.tex` がベース) |
| `references.bib` | 参考文献 (`ipsjunsrt.bst` = 引用順) |
| `ipsj.cls`, `ipsjunsrt.bst` | `gpw2026/ipsj_v4-1/ipsj_v4-1/UTF8/` からコピー (v4.1, UTF-8 版) |
| `latexmkrc` | platex + pbibtex + dvipdfmx でビルドする設定 |
| `figures/` | 図を置く場所 |

## Overleaf へのアップロード
1. `gpw2026/paper_overleaf.zip` を New Project → Upload Project でアップロード
   (zip が古ければ、このディレクトリの中身を zip し直す)。
2. Menu → Compiler は **LaTeX** にする (実際のコマンドは `latexmkrc` が platex に差し替える)。
   Main document は `main.tex`。

## メモ
- 学会配布の `jsample-gpw.tex` は EUC-JP 前提だが、Overleaf は UTF-8 なので UTF-8 版の `ipsj.cls` を使っている。
- 本文で `\cite` を使い始めたら `main.tex` 末尾の `\nocite{*}` を消す
  (引用 0 件だと文献リストが空になってエラーになるので置いてある)。
- PNG を入れる場合は `\includegraphics[width=...]` で幅を指定する (pLaTeX+dvipdfmx は画像の DPI を読まない)。
  サイズがおかしければ `gpw2026/README.md` の bb 指定の節を参照。
