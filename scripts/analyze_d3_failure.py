# -*- coding: utf-8 -*-
"""d3(深さ3)の敗因解剖。追加実験なしで既存データから多角的に分析する。

データ源:
  1. run50 の A1(=d3) と A1d1(=d1) の実走行 (同一50局面×同一5seed) -- 挙動比較の主データ
  2. 審判 qテーブル (q_ref, SD, score_hist=得点分布) -- 分かれた手の質・リスクの判定
  3. 自己対戦ゲームCSV (定数/noisy対決) -- 負け方の法医学

問い:
  Q1 前半はd1と同じ挙動か (残り手数別の選択一致率)
  Q2 手が分かれたとき、どちらの手が良いか (Δq_ref)
  Q3 d3が選ぶ手は博打か (SD・得点分布の裾)
  Q4 どういう手に変えるのか (ショット種別の遷移)
  Q5 対戦での負け方 (大敗か小差か / ハンマー条件)

Usage: python scripts/analyze_d3_failure.py
"""
import csv, glob, os, sys
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_board import draw_board
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

RUN50 = "reinvest_experiment/scorescreen/run50"
OUT = f"{RUN50}/d3_failure"
ARM_BASE = RUN50                                          # main() の引数で上書き
REFEREE_CSV = f"{RUN50}/referee/score_move_qtable.csv"
BATCH_CSV = "test_positions50/batch_0001.csv"
C_D1, C_D3 = "#009E73", "#D55E00"   # d1=緑, d3=橙 (CVD-safe)

BUCKETS = [(2, 4, "終盤 r=2-4"), (5, 8, "中盤 r=5-8"), (9, 16, "序盤 r=9+")]
def bucket(r):
    for lo, hi, name in BUCKETS:
        if lo <= r <= hi: return name
    return None  # r=1 は両者構造的に同一なので除外


def load_moves(arm):
    mv = {}
    for fp in glob.glob(f"{ARM_BASE}/{arm}/seed_*/reinvest_results.csv"):
        for row in csv.DictReader(open(fp, encoding="utf-8")):
            mv[(row["game_id"], row["end"], row["shot_num"], row["seed"])] = int(row["candidate_idx"])
    return mv


def load_referee():
    q = defaultdict(dict)
    for row in csv.DictReader(open(REFEREE_CSV, encoding="utf-8")):
        hist = {}
        if row["score_hist"]:
            for tok in row["score_hist"].split(";"):
                s, c = tok.split(":"); hist[int(s)] = int(c)
        n = sum(hist.values()) or 1
        q[(row["game_id"], row["end"], row["shot_num"])][int(row["candidate_idx"])] = dict(
            q=float(row["q_ref_mean"]), sd=float(row["q_ref_sd"]),
            typ=row["shot_type"].strip('"'), label=row["label"].strip('"'),
            p_big_win=sum(c for s, c in hist.items() if s >= 2) / n,
            p_big_loss=sum(c for s, c in hist.items() if s <= -2) / n)
    return q


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-d3", default="A1", help="d3側アームのディレクトリ名")
    ap.add_argument("--arm-d1", default="A1d1", help="d1側アームのディレクトリ名")
    ap.add_argument("--base", default=RUN50, help="アームディレクトリの親 (seed_* を含む階層の親)")
    ap.add_argument("--referee-csv", default=f"{RUN50}/referee/score_move_qtable.csv")
    ap.add_argument("--batch-csv", default="test_positions50/batch_0001.csv")
    ap.add_argument("--out-name", default="d3_failure", help="出力サブディレクトリ名 (--base配下)")
    args = ap.parse_args()
    global OUT, ARM_BASE, REFEREE_CSV, BATCH_CSV
    ARM_BASE, REFEREE_CSV, BATCH_CSV = args.base, args.referee_csv, args.batch_csv
    OUT = f"{args.base}/{args.out_name}"
    os.makedirs(OUT, exist_ok=True)

    d3 = load_moves(args.arm_d3)
    d1 = load_moves(args.arm_d1)
    ref = load_referee()
    rep = []
    def pr(s): rep.append(s); print(s)

    # ---- Q1: 一致率 / Q2: 分かれたときの質 / Q3: リスク / Q4: 種別遷移 ----
    stats = defaultdict(lambda: dict(n=0, agree=0, dq=[], d3sd=[], d1sd=[],
                                     d3bw=[], d1bw=[], d3bl=[], d1bl=[], trans=Counter()))
    for k, m3 in d3.items():
        if k not in d1: continue
        pos = k[:3]
        r = 16 - int(k[2])
        b = bucket(r)
        if b is None or pos not in ref: continue
        m1 = d1[k]
        st = stats[b]; st["n"] += 1
        if m3 == m1: st["agree"] += 1; continue
        c3, c1 = ref[pos].get(m3), ref[pos].get(m1)
        if not c3 or not c1: continue
        st["dq"].append(c3["q"] - c1["q"])           # >0 = d3の手が審判基準で良い
        st["d3sd"].append(c3["sd"]); st["d1sd"].append(c1["sd"])
        st["d3bw"].append(c3["p_big_win"]); st["d1bw"].append(c1["p_big_win"])
        st["d3bl"].append(c3["p_big_loss"]); st["d1bl"].append(c1["p_big_loss"])
        st["trans"][(c1["typ"], c3["typ"])] += 1

    pr("=" * 74)
    pr("Q1/Q2/Q3: 残り手数別 -- d3 と d1 の選択一致率と、分かれたときの手の質")
    pr("=" * 74)
    pr(f"{'フェーズ':>10} {'n':>4} {'一致率':>7} {'Δq(d3-d1)':>10} {'d3のSD':>7} {'d1のSD':>7} "
       f"{'d3大勝率':>8} {'d1大勝率':>8} {'d3大敗率':>8} {'d1大敗率':>8}")
    for _, _, b in BUCKETS:
        st = stats[b]
        if st["n"] == 0: continue
        agree = st["agree"] / st["n"]
        dq = np.mean(st["dq"]) if st["dq"] else float("nan")
        pr(f"{b:>10} {st['n']:>4} {agree:>7.1%} {dq:>+10.3f} "
           f"{np.mean(st['d3sd']):>7.2f} {np.mean(st['d1sd']):>7.2f} "
           f"{np.mean(st['d3bw']):>8.1%} {np.mean(st['d1bw']):>8.1%} "
           f"{np.mean(st['d3bl']):>8.1%} {np.mean(st['d1bl']):>8.1%}")
    pr("  (Δq<0 = 分かれた局面ではd1の手が審判基準で良い。大勝/大敗率 = その手の得点分布で±2点以上の確率)")

    pr("")
    pr("Q4: 手が分かれたとき、d1の手 → d3の手 のショット種別遷移 (終盤 r=2-4)")
    TYPE = {"0": "Pass", "1": "Draw", "2": "PreGuard", "3": "Hit", "4": "Freeze",
            "5": "Peel", "6": "ComeAround", "7": "PostGuard", "8": "DrawRaise",
            "9": "Takeout", "10": "Double"}
    for (t1, t3), c in stats["終盤 r=2-4"]["trans"].most_common(10):
        pr(f"   d1:{TYPE.get(t1, t1):>10} -> d3:{TYPE.get(t3, t3):>10}  x{c}")

    # ---- Q5: 対戦での負け方 ----
    pr("")
    pr("=" * 74)
    pr("Q5: 自己対戦での負け方 (d3視点, 定数対決+noisy対決 計100ゲーム)")
    pr("=" * 74)
    nets, e0_ham, e0_non = [], [], []
    for pat in ["reinvest_experiment/A7d3vsA7d1_*/selfplay_games.csv",
                "reinvest_experiment/A7d3n_vs_A7d1n_*/selfplay_games.csv"]:
        for fp in glob.glob(pat):
            for row in csv.DictReader(open(fp, encoding="utf-8")):
                nets.append(float(row["net_a"]))
                (e0_ham if int(float(row["a_hammer_end0"])) == 1 else e0_non).append(float(row["net_end0"]))
    nets = np.array(nets)
    pr(f"  総合: 平均net={nets.mean():+.2f}  中央値={np.median(nets):+.1f}")
    pr(f"  大敗(net<=-3): {(nets<=-3).mean():.1%}   小差負け(-2<=net<0): {((nets>=-2)&(nets<0)).mean():.1%}")
    pr(f"  大勝(net>=+3): {(nets>=+3).mean():.1%}   小差勝ち(0<net<=+2): {((nets<=+2)&(nets>0)).mean():.1%}")
    pr(f"  d3がハンマー時のend0: {np.mean(e0_ham):+.2f} (n={len(e0_ham)})   "
       f"d3が先攻時のend0: {np.mean(e0_non):+.2f} (n={len(e0_non)})")

    with open(f"{OUT}/d3_failure_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")

    # ---- 図1: 一致率とΔq ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    names = [b for _, _, b in BUCKETS]
    ag = [stats[b]["agree"] / stats[b]["n"] if stats[b]["n"] else 0 for b in names]
    ax1.bar(range(3), ag, 0.55, color="#0072B2", edgecolor="white")
    for i, v in enumerate(ag): ax1.text(i, v + 0.02, f"{v:.0%}", ha="center")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(names); ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("d3とd1の選択一致率"); ax1.set_title("Q1: どこで手が分かれるか")
    for s in ("top", "right"): ax1.spines[s].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)
    dq_data = [stats[b]["dq"] for b in names]
    bp = ax2.boxplot([d if d else [0] for d in dq_data], tick_labels=names, showmeans=True)
    ax2.axhline(0, color="0.5", ls="--", lw=1)
    ax2.set_ylabel("Δq = q_ref(d3の手) − q_ref(d1の手)")
    ax2.set_title("Q2: 分かれた手の質差 (負=d1が良い)")
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/d3_agreement_quality.png", dpi=150); plt.close(fig)

    # ---- 図2: 終盤で最も損した分岐の盤面 (d1の手 vs d3の手) ----
    board = {(r["match_id"], r["end"], r["shot_num"]): r
             for r in csv.DictReader(open(BATCH_CSV, newline="", encoding="utf-8"))}
    cases = []
    for k, m3 in d3.items():
        if k not in d1 or d1[k] == m3: continue
        pos = k[:3]; r = 16 - int(k[2])
        if not (2 <= r <= 4) or pos not in ref: continue
        c3, c1 = ref[pos].get(m3), ref[pos].get(d1[k])
        if c3 and c1: cases.append((c3["q"] - c1["q"], pos, c1, c3))
    cases.sort(key=lambda t: t[0])
    picks, seen = [], set()
    for dq, pos, c1, c3 in cases:
        if pos in seen: continue
        seen.add(pos); picks.append((dq, pos, c1, c3))
        if len(picks) == 4: break
    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 5.4))
    for ax, (dq, pos, c1, c3) in zip(np.atleast_1d(axes).ravel(), picks):
        g, e, s = pos
        row = board.get(pos)
        if row: draw_board(ax, row, "")
        ax.set_title(f"g{g} e{e} s{s} (残り{16-int(s)})\n"
                     f"d1: {c1['label']} (q={c1['q']:+.2f})\n"
                     f"d3: {c3['label']} (q={c3['q']:+.2f}, Δ={dq:+.2f})", fontsize=8.5)
    fig.suptitle("Q4補: 終盤でd3がd1と分かれて最も損した局面 (審判基準)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/d3_worst_deviations.png", dpi=150); plt.close(fig)
    print(f"\n[out] -> {OUT}/ (report.txt, d3_agreement_quality.png, d3_worst_deviations.png)")


if __name__ == "__main__":
    main()
