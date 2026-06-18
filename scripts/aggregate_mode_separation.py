#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モード分離実験の集計 (GPW2026)
================================================================================
問い: AllGrid が "正解" と判断しうる複数の候補手 (シードを変えると選択がばらつく)
      を、Proposed のクラスタリングは戦略的に別々のクラスタへ分離できているか。
      本来異なる正解が 1 クラスタに集約 (collapse) されると、代表点が 1 つしか出ず
      Proposed は複数の正解を同時に検討できなくなる。

定義 (ユーザー確定):
  - 正解集合 A      : AllGrid (A1) を R シード回し、選ばれた手 (candidate_idx) の集合。
  - モード          : A 内候補を shot_type でグループ化 (Draw/Hit/Peel/Freeze/...)。
                       m = AllGrid が選んだ異なる shot_type の数。m>=2 が多峰 = 主対象。
  - Proposed のクラスタリングは展開が simulateNoRand (決定的) なので局面ごと seed 非依存
    = 審判が採点したプールと同一。よって cluster_table は任意の 1 seed を使えばよい。

指標:
  - Mode Recall (主) : AllGrid が選んだ各 shot_type が Proposed の K 代表点に現れる割合。
                       2 モードが 1 クラスタに collapse すると代表点は片方しか出せず Recall<1。
  - Separation Rate  : AllGrid が選んだ「異なる shot_type の手のペア」が別クラスタにある割合。
  - Collapse 件数    : 異なる shot_type の選択手が同一クラスタに同居した件数。
  - AllGrid 多峰性    : 選択分布の distinct idx 数 / distinct type 数 / top1 頻度 / type エントロピー。
  - RandomK 対照     : 同じ K でランダム削減した場合の Mode Recall (seed 平均)。

入力ディレクトリ構成 (run_reinvest.sh 出力):
  <reinvest-dir>/A1/seed_*/reinvest_results*.csv   (AllGrid 選択分布)
  <reinvest-dir>/A2/seed_*/cluster_table*.csv      (Proposed クラスタ割当 = 権威マップ)
  <reinvest-dir>/A5/seed_*/cluster_table*.csv      (RandomK 代表; 任意)
  (任意) <referee-dir>/score_move_qtable*.csv       (q_ref で正解の質を検証)

使用例:
  python3 scripts/aggregate_mode_separation.py \
      --reinvest-dir experiments/reinvest/run_latest \
      --referee-dir  experiments/reinvest_referee \
      --out          experiments/mode_separation_out
================================================================================
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict


# ---------------------------------------------------------------------------
# CSV ローダ (stdlib のみ)
# ---------------------------------------------------------------------------

def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_arm_csvs(reinvest_dir, arm, filename_glob):
    """<reinvest-dir>/<arm>/seed_*/<filename_glob> を全部読み、行のリストを返す。"""
    rows = []
    pattern = os.path.join(reinvest_dir, arm, "seed_*", filename_glob)
    files = sorted(glob.glob(pattern))
    for fp in files:
        try:
            rows.extend(read_csv_rows(fp))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to read {fp}: {e}")
    return rows, files


def pos_key(row):
    return (int(row["game_id"]), int(row["end"]), int(row["shot_num"]))


def shot_type_from_label(label):
    """ラベル "Draw(CW,5)" -> "Draw"。cluster_table に shot_type が無い場合のフォールバック。"""
    if not label:
        return "?"
    return label.split("(")[0].strip().strip('"')


def entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# メイン集計
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="モード分離実験の集計")
    ap.add_argument("--reinvest-dir", required=True,
                    help="run_reinvest.sh の親ディレクトリ (A1/A2/A5 を含む)")
    ap.add_argument("--referee-dir", default=None,
                    help="(任意) score_move_qtable*.csv のディレクトリ")
    ap.add_argument("--out", default=None, help="出力ディレクトリ (省略時は reinvest-dir)")
    ap.add_argument("--allgrid-arm", default="A1")
    ap.add_argument("--proposed-arm", default="A2")
    ap.add_argument("--randomk-arm", default="A5")
    ap.add_argument("--min-count", type=int, default=1,
                    help="正解集合 A に含める最小選択回数 (default 1 = 1度でも選ばれた手)")
    ap.add_argument("--eps", type=float, default=0.5,
                    help="(審判使用時) 最良 q_ref からこの範囲内なら『真の好手』とみなす")
    ap.add_argument("--no-plot", action="store_true", help="プロットを生成しない")
    args = ap.parse_args()

    out_dir = args.out or args.reinvest_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. AllGrid 選択分布 (正解集合 A) ---
    ag_rows, ag_files = load_arm_csvs(args.reinvest_dir, args.allgrid_arm, "reinvest_results*.csv")
    if not ag_rows:
        print(f"[error] AllGrid ({args.allgrid_arm}) の reinvest_results が見つかりません: "
              f"{os.path.join(args.reinvest_dir, args.allgrid_arm)}")
        return
    print(f"[load] AllGrid: {len(ag_rows)} rows from {len(ag_files)} files")

    # 局面 -> {seed: best_idx}
    ag_choice = defaultdict(dict)          # key -> {seed: idx}
    ag_seeds = defaultdict(set)            # key -> set(seed)
    for r in ag_rows:
        try:
            idx = int(r["candidate_idx"])
        except (KeyError, ValueError):
            continue
        if idx < 0:
            continue
        k = pos_key(r)
        seed = r.get("seed", "?")
        ag_choice[k][seed] = idx
        ag_seeds[k].add(seed)

    # --- 2. Proposed クラスタ割当 (権威マップ; seed 非依存) ---
    pr_rows, pr_files = load_arm_csvs(args.reinvest_dir, args.proposed_arm, "cluster_table*.csv")
    if not pr_rows:
        print(f"[error] Proposed ({args.proposed_arm}) の cluster_table が見つかりません。"
              f" C++ を再ビルドして A2 を回しましたか?")
        return
    print(f"[load] Proposed cluster_table: {len(pr_rows)} rows from {len(pr_files)} files")

    # 局面 -> {candidate_idx: (cluster_id, is_rep, shot_type)}
    pr_map = defaultdict(dict)
    pr_rep_types = defaultdict(set)        # key -> set(shot_type of representatives)
    for r in pr_rows:
        k = pos_key(r)
        idx = int(r["candidate_idx"])
        cid = int(r["cluster_id"])
        is_rep = r["is_representative"] in ("1", "True", "true")
        st = (r.get("shot_type") or "").strip().strip('"') or shot_type_from_label(r.get("label", ""))
        pr_map[k][idx] = (cid, is_rep, st)
        if is_rep:
            pr_rep_types[k].add(st)

    # --- 3. RandomK 代表 (任意; seed 依存なので seed ごとに保持) ---
    rk_rows, rk_files = load_arm_csvs(args.reinvest_dir, args.randomk_arm, "cluster_table*.csv")
    rk_rep_types = defaultdict(lambda: defaultdict(set))   # key -> {seed: set(shot_type)}
    for r in rk_rows:
        k = pos_key(r)
        seed = r.get("seed", "?")
        if r["is_representative"] in ("1", "True", "true"):
            st = (r.get("shot_type") or "").strip().strip('"') or shot_type_from_label(r.get("label", ""))
            rk_rep_types[k][seed].add(st)
    if rk_files:
        print(f"[load] RandomK cluster_table: {len(rk_rows)} rows from {len(rk_files)} files")

    # --- 4. 審判 (任意): 局面 -> {candidate_idx: q_ref_mean} ---
    ref_q = defaultdict(dict)
    if args.referee_dir:
        ref_files = sorted(glob.glob(os.path.join(args.referee_dir, "score_move_qtable*.csv")))
        for fp in ref_files:
            for r in read_csv_rows(fp):
                try:
                    ref_q[pos_key(r)][int(r["candidate_idx"])] = float(r["q_ref_mean"])
                except (KeyError, ValueError):
                    continue
        if ref_files:
            print(f"[load] referee: {len(ref_files)} files")

    # --- 5. 局面ごとに指標を計算 ---
    per_pos = []
    for k in sorted(ag_choice.keys()):
        seeds = ag_seeds[k]
        R = len(seeds)
        # 選択頻度
        idx_count = defaultdict(int)
        for _seed, idx in ag_choice[k].items():
            idx_count[idx] += 1

        cmap = pr_map.get(k, {})

        def st_of(idx):
            if idx in cmap:
                return cmap[idx][2]
            return "?"

        def cl_of(idx):
            if idx in cmap:
                return cmap[idx][0]
            return None

        # 正解集合 A (min_count 以上選ばれた手)
        A_idx = [idx for idx, c in idx_count.items() if c >= args.min_count]
        if not A_idx:
            A_idx = list(idx_count.keys())

        # モード (shot_type) と頻度
        type_count = defaultdict(int)
        for idx in A_idx:
            type_count[st_of(idx)] += idx_count[idx]
        modes = set(type_count.keys())
        m = len(modes)

        # 多峰性統計
        top1_frac = (max(idx_count.values()) / R) if R else 0.0
        idx_entropy = entropy(list(idx_count.values()))
        type_entropy = entropy(list(type_count.values()))
        n_distinct_idx = len(idx_count)

        # Proposed Mode Recall
        rep_types = pr_rep_types.get(k, set())
        covered = modes & rep_types
        proposed_recall = (len(covered) / m) if m else float("nan")
        missed_modes = sorted(modes - rep_types)

        # 分離 / collapse: A 内の異なる shot_type のペア
        A_distinct = sorted(set(A_idx))
        pairs = 0
        separated = 0
        collapse_events = []
        for i in range(len(A_distinct)):
            for j in range(i + 1, len(A_distinct)):
                a, b = A_distinct[i], A_distinct[j]
                ta, tb = st_of(a), st_of(b)
                if ta == tb:
                    continue  # 同型 = 同モード, 対象外
                pairs += 1
                ca, cb = cl_of(a), cl_of(b)
                if ca is not None and cb is not None and ca == cb:
                    collapse_events.append(f"{ta}#{a}+{tb}#{b}@cl{ca}")
                else:
                    separated += 1
        separation_rate = (separated / pairs) if pairs else float("nan")
        collapse_count = len(collapse_events)

        # RandomK Mode Recall (seed 平均)
        rk_by_seed = rk_rep_types.get(k, {})
        rk_recalls = []
        for _seed, rts in rk_by_seed.items():
            if m:
                rk_recalls.append(len(modes & rts) / m)
        randomk_recall = (sum(rk_recalls) / len(rk_recalls)) if rk_recalls else float("nan")

        # 審判: A の各モードが真に好手か (q_ref が最良から eps 以内)
        ref_quality = ""
        if k in ref_q and ref_q[k]:
            qstar = max(ref_q[k].values())
            good_modes = set()
            for idx in A_idx:
                q = ref_q[k].get(idx)
                if q is not None and q >= qstar - args.eps:
                    good_modes.add(st_of(idx))
            ref_quality = f"{len(good_modes)}/{m}"

        per_pos.append({
            "game_id": k[0], "end": k[1], "shot_num": k[2],
            "R_seeds": R,
            "n_distinct_idx": n_distinct_idx,
            "num_modes": m,
            "modes": "|".join(sorted(modes)),
            "top1_frac": round(top1_frac, 3),
            "idx_entropy": round(idx_entropy, 3),
            "type_entropy": round(type_entropy, 3),
            "proposed_mode_recall": round(proposed_recall, 3) if not math.isnan(proposed_recall) else "",
            "missed_modes": "|".join(missed_modes),
            "separation_rate": round(separation_rate, 3) if not math.isnan(separation_rate) else "",
            "collapse_count": collapse_count,
            "collapse_events": ";".join(collapse_events),
            "randomk_mode_recall": round(randomk_recall, 3) if not math.isnan(randomk_recall) else "",
            "ref_good_modes": ref_quality,
        })

    # --- 6. 出力 CSV ---
    cols = ["game_id", "end", "shot_num", "R_seeds", "n_distinct_idx", "num_modes", "modes",
            "top1_frac", "idx_entropy", "type_entropy",
            "proposed_mode_recall", "missed_modes", "separation_rate",
            "collapse_count", "collapse_events", "randomk_mode_recall", "ref_good_modes"]
    per_pos_path = os.path.join(out_dir, "mode_separation_per_position.csv")
    with open(per_pos_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in per_pos:
            w.writerow(row)
    print(f"\n[out] per-position -> {per_pos_path}")

    # --- 7. サマリ (全体 + 多峰 m>=2 部分集合) ---
    def mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        return (sum(vals) / len(vals)) if vals else float("nan")

    def num_or_nan(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    multimodal = [p for p in per_pos if p["num_modes"] >= 2]
    lines = []
    lines.append("=" * 70)
    lines.append("モード分離実験 サマリ")
    lines.append("=" * 70)
    lines.append(f"総局面数               : {len(per_pos)}")
    lines.append(f"多峰局面 (m>=2)        : {len(multimodal)}  <- 本実験の主対象")
    lines.append(f"単峰局面 (m==1)        : {len(per_pos) - len(multimodal)}")
    lines.append("")
    lines.append(f"[AllGrid 多峰性] 平均 distinct idx     : "
                 f"{mean([p['n_distinct_idx'] for p in per_pos]):.2f}")
    lines.append(f"[AllGrid 多峰性] 平均 mode 数 (type)   : "
                 f"{mean([p['num_modes'] for p in per_pos]):.2f}")
    lines.append(f"[AllGrid 多峰性] 平均 top1 頻度        : "
                 f"{mean([p['top1_frac'] for p in per_pos]):.3f}")
    if multimodal:
        lines.append("")
        lines.append("--- 多峰局面 (m>=2) のみ ---")
        lines.append(f"Proposed Mode Recall (平均) : "
                     f"{mean([num_or_nan(p['proposed_mode_recall']) for p in multimodal]):.3f}  "
                     f"<- 主指標: Proposed が複数正解を検討できる割合")
        lines.append(f"RandomK  Mode Recall (平均) : "
                     f"{mean([num_or_nan(p['randomk_mode_recall']) for p in multimodal]):.3f}  "
                     f"<- 対照: 単なる削減")
        lines.append(f"Separation Rate (平均)      : "
                     f"{mean([num_or_nan(p['separation_rate']) for p in multimodal]):.3f}")
        n_collapse = sum(1 for p in multimodal if p["collapse_count"] > 0)
        lines.append(f"collapse が起きた多峰局面    : {n_collapse} / {len(multimodal)}")
        lines.append("")
        lines.append("--- collapse 詳細 (本来別の正解が 1 クラスタに集約) ---")
        for p in multimodal:
            if p["collapse_count"] > 0:
                lines.append(f"  g{p['game_id']} e{p['end']} s{p['shot_num']}: "
                             f"modes={p['modes']} missed={p['missed_modes']} "
                             f"collapse={p['collapse_events']}")
    else:
        lines.append("")
        lines.append("[note] 多峰局面が 0 件。既存 8 局面では AllGrid の選択が単峰に寄っている可能性。")
        lines.append("       -> シード数 R を増やす / 多峰局面を新規選定する を検討。")

    summary = "\n".join(lines)
    print("\n" + summary)
    summary_path = os.path.join(out_dir, "mode_separation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n[out] summary -> {summary_path}")

    # --- 8. プロット (任意) ---
    if not args.no_plot and multimodal:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [f"g{p['game_id']}\ne{p['end']}s{p['shot_num']}" for p in multimodal]
            prop = [num_or_nan(p["proposed_mode_recall"]) for p in multimodal]
            rand = [num_or_nan(p["randomk_mode_recall"]) for p in multimodal]
            x = range(len(multimodal))
            w = 0.38
            fig, ax = plt.subplots(figsize=(max(6, len(multimodal) * 1.1), 4.2))
            ax.bar([i - w / 2 for i in x], prop, w, label="Proposed", color="#2c7fb8")
            ax.bar([i + w / 2 for i in x], rand, w, label="RandomK", color="#c0c0c0")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Mode Recall (正解モード被覆率)")
            ax.set_title("多峰局面における Mode Recall: Proposed vs RandomK")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, fontsize=8)
            ax.axhline(1.0, color="green", lw=0.8, ls="--", alpha=0.6)
            ax.legend()
            fig.tight_layout()
            plot_path = os.path.join(out_dir, "mode_recall_proposed_vs_randomk.png")
            fig.savefig(plot_path, dpi=150)
            print(f"[out] plot -> {plot_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] plot skipped: {e}")


if __name__ == "__main__":
    main()
