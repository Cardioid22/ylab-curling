# -*- coding: utf-8 -*-
"""「別の道」カタログ: 今すぐ有利になる手 A と、今の得点は低いが後の展開で同じように有利になる手 B が
別のクラスタに分かれている局面の一覧 (scripts/analyze_paths_and_spaces.py の paths_rows.csv で has_B=1 の局面)。

1局面 = 1ページ:
  (a) 局面  (b) 全候補の着地点 (灰) と A・B のクラスタ (★ = A, ▲ = B)  (c) 数値
  (d) A の結果盤面  (e) B の結果盤面  (f) 全候補の「今の得点」と審判値 q (A・B のクラスタを色付け)
Usage: python scripts/plot_paths_catalog.py gpw_experiment/cluster_vs_topk
出力: <dir>/paths_catalog/paths_catalog.pdf, index.csv
"""
import csv, glob, math, os, sys
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/scripts")
from plot_board import HC_X, HC_Y, R12, R8, R4, RBTN, SR, AREA_X, C0, C1
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False
GE = ROOT + "/gpw_experiment"
D = sys.argv[1]
OUT = D + "/paths_catalog"
CA, CB = "#d62728", "#7b3fb5"   # A / B のクラスタ色
SETUP = {"Draw", "Freeze", "PostGuard", "PreGuard", "Guard", "ComeAround"}

def key(r): return (int(r["game_id"]), int(r["end"]), int(r["shot_num"]))
R = [r for r in csv.DictReader(open(D + "/paths_rows.csv", encoding="utf-8")) if r["has_B"] == "1"]
keys = {(int(r["game"]), int(r["end"]), int(r["shot"])) for r in R}
Q = {}
for f in glob.glob(GE + "/referee/score_move_qtable_idx*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if key(r) in keys: Q[(key(r), int(r["candidate_idx"]))] = float(r["q_ref_mean"])
CT = defaultdict(list)
for r in csv.DictReader(open(f"{GE}/A9P5N/seed_42/cluster_table.csv", encoding="utf-8")):
    if key(r) in keys: CT[key(r)].append(r)
BATCH = {(int(r["match_id"]), int(r["end"]), int(r["shot_num"])): r
         for r in csv.DictReader(open(ROOT + "/test_positions500/batch_0001.csv", encoding="utf-8"))}

def export_dir(k):
    for sub in ("export", "export_rest"):
        d = f"{D}/{sub}/export_g{k[0]}_e{k[1]}_s{k[2]}"
        if os.path.exists(d + "/result_stones.csv"): return d
def dist(x, y): return math.hypot(x - HC_X, y - HC_Y)
def ranked(st_): return sorted([s for s, p in st_.items() if dist(*p) <= R12 + SR], key=lambda s: dist(*st_[s]))
def score_of(st_, me):
    rk = ranked(st_)
    if not rk: return 0
    n = 0
    for s in rk:
        if s[0] != rk[0][0]: break
        n += 1
    return n if rk[0][0] == me else -n

def house(ax, ylim, title):
    for rr, c in [(R12, "#cfe0f3"), (R8, "#ffffff"), (R4, "#f3cfcf"), (RBTN, "#ffffff")]:
        ax.add_patch(Circle((HC_X, HC_Y), rr, facecolor=c, edgecolor="#999", lw=0.7, zorder=1))
    ax.axhline(HC_Y, color="#ccc", lw=0.5, ls="--")
    ax.set_xlim(-AREA_X, AREA_X); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5)
def draw_stones(ax, st_, alpha=1.0, lw=0.8, z=3):
    for (t, i), (x, y) in st_.items():
        ax.add_patch(Circle((x, y), SR, facecolor=C0 if t == 0 else C1, edgecolor="#222", lw=lw, alpha=alpha, zorder=z))

def page(P, num):
    k, inp, res, me, cl, lab, q, imm = P["k"], P["inp"], P["res"], P["me"], P["cl"], P["lab"], P["q"], P["imm"]
    a, b = P["A"], P["B"]
    thrown = (me, k[2] // 2)
    ys = [p[1] for p in inp.values()] + [p[1] for c in (a, b) for p in res[c].values()]
    ylim = (max(31.5, min([HC_Y - R12 - 0.5] + [y - 0.4 for y in ys if y > 30])), HC_Y + R12 + 0.5)
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    ax = axes[0, 0]
    house(ax, ylim, f"(a) 局面  手番 = {'橙' if me == 0 else '青'} ({'後攻' if k[2] % 2 else '先攻'}, 第{k[2] + 1}投)\n"
                    f"数字 = ティーから近い順  今の得点 {score_of(inp, me):+d}")
    draw_stones(ax, inp)
    for n, s in enumerate(ranked(inp), 1):
        ax.text(*inp[s], str(n), ha="center", va="center", fontsize=6.5, color="w", weight="bold", zorder=4)
    ax = axes[0, 1]
    house(ax, ylim, "(b) 全候補の投げた石の止まる位置\n赤 = A のクラスタ  紫 = B のクラスタ  ★ = A  ▲ = B")
    draw_stones(ax, inp, alpha=0.3, lw=0.3)
    for c in range(len(cl)):
        p = res[c].get(thrown)
        if not p: continue
        col = CA if cl[c] == cl[a] else (CB if cl[c] == cl[b] else "#aaa")
        ax.plot(*p, "o", ms=4 if col != "#aaa" else 3, color=col, alpha=0.9, zorder=4)
    for c, m, col in ((a, "*", CA), (b, "^", CB)):
        p = res[c].get(thrown)
        if p: ax.plot(*p, m, ms=14 if m == "*" else 10, mfc=col, mec="k", mew=0.8, zorder=6)
    ax = axes[0, 2]; ax.axis("off")
    r = P["row"]
    L = [f"A (今すぐ有利): {lab[a]}",
         f"   審判値 {q[a]:+.2f}, 今の得点 {imm[a]:+d}, クラスタの大きさ {r['A_csize']}, 価値の順位 {r['A_value_rank_med']}",
         f"B (後の展開で有利): {lab[b]}",
         f"   審判値 {q[b]:+.2f}, 今の得点 {imm[b]:+d}, クラスタの大きさ {r['B_csize']}, 価値の順位 {r['B_value_rank_med']}",
         "",
         f"最良の審判値 {P['qs']:+.2f}。最良から 0.25 以内のクラスタ {r['n_nb_clusters']} 個 / 全 {r['K']} 個",
         f"提案の 4 手が A と B を両方含んだ seed: {r['proposal_shows_A_and_B_of5']}/5",
         f"区分: {r['stratum']} / 残り {r['phase_r']}",
         "",
         "審判値 = 手番側から見たエンド得点の期待値 (外乱込み, 残りの投球まで)",
         "今の得点 = 今エンドが終わったら何点か",
         "価値の順位 = クラスタ価値 (メンバーの推定値の平均) の順位 (5 seed の中央値)",
         "下段: 矢印 = 動いた石, × = 出た石, 白点 = 投げた石"]
    ax.text(0, 1, "\n".join(L), va="top", fontsize=8.2, transform=ax.transAxes)
    for ax, c, head in ((axes[1, 0], a, "(d) A: 今すぐ有利"), (axes[1, 1], b, "(e) B: 今の得点は低いが審判値は同等")):
        house(ax, ylim, f"{head}\n{lab[c]}  審判値 {q[c]:+.2f}  今の得点 {score_of(inp, me):+d}→{imm[c]:+d}")
        draw_stones(ax, inp, alpha=0.22, lw=0.3, z=2); draw_stones(ax, res[c])
        for s, (x0, y0) in inp.items():
            if s in res[c]:
                x1, y1 = res[c][s]
                if math.hypot(x1 - x0, y1 - y0) > 0.05:
                    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=8, color="k", lw=0.8, zorder=5))
            else:
                ax.plot(x0, y0, "x", color="k", ms=8, mew=1.5, zorder=5)
        p = res[c].get(thrown)
        if p: ax.plot(*p, "o", ms=3.2, color="w", mec="k", mew=0.6, zorder=6)
        else: ax.text(0.02, 0.02, "投げた石はプレー外", transform=ax.transAxes, fontsize=7)
    ax = axes[1, 2]
    import random
    rnd = random.Random(0)
    for c in range(len(cl)):
        col = CA if cl[c] == cl[a] else (CB if cl[c] == cl[b] else "#bbb")
        ax.plot(imm[c] + rnd.uniform(-0.18, 0.18), q[c], "o", ms=4, color=col, alpha=0.85, zorder=3 if col != "#bbb" else 2)
    ax.plot(imm[a], q[a], "*", ms=14, mfc=CA, mec="k", zorder=5); ax.plot(imm[b], q[b], "^", ms=10, mfc=CB, mec="k", zorder=5)
    ax.axhline(P["qs"] - 0.25, color="#888", lw=0.7, ls="--")
    ax.set_xlabel("今の得点 (投げた直後)", fontsize=8); ax.set_ylabel("審判値 q (エンドの得点の期待値)", fontsize=8)
    ax.set_title("(f) 全候補: 今の得点と審判値\n破線 = 最良から 0.25 以内", fontsize=8.5); ax.tick_params(labelsize=7)
    fig.suptitle(f"#{num:03d}  g{k[0]} e{k[1]} s{k[2]}", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig

def main():
    os.makedirs(OUT, exist_ok=True)
    Ps = []
    for r in R:
        k = (int(r["game"]), int(r["end"]), int(r["shot"])); d = export_dir(k)
        inp = {(int(x["team"]), int(x["stone_idx"])): (float(x["x"]), float(x["y"])) for x in csv.DictReader(open(d + "/input_board.csv", encoding="utf-8"))}
        res = defaultdict(dict)
        for x in csv.DictReader(open(d + "/result_stones.csv", encoding="utf-8")):
            res[int(x["candidate_idx"])][(int(x["team"]), int(x["stone_idx"]))] = (float(x["x"]), float(x["y"]))
        me = int(BATCH[k]["team"])
        cl = [0] * len(CT[k]); lab = {}
        for x in CT[k]: cl[int(x["candidate_idx"])] = int(x["cluster_id"]); lab[int(x["candidate_idx"])] = x["label"]
        q = [Q[(k, c)] for c in range(len(cl))]
        Ps.append(dict(k=k, row=r, inp=inp, res=res, me=me, cl=cl, lab=lab, q=q, qs=max(q),
                       imm=[score_of(res[c], me) for c in range(len(cl))], A=int(r["A_idx"]), B=int(r["B_idx"])))
    # 並び: B が「置く手」(ドロー/フリーズ/ガード) の局面を先に → B の審判値が A に近い (または上回る) 順
    Ps.sort(key=lambda P: (P["lab"][P["B"]].split("(")[0] not in SETUP, -(P["q"][P["B"]] - P["q"][P["A"]])))
    with open(OUT + "/index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["num", "game", "end", "shot", "stratum", "A_label", "A_q", "A_imm", "B_label", "B_q", "B_imm", "B_minus_A_q", "B_setup", "shows_A_and_B_of5"])
        for i, P in enumerate(Ps, 1):
            a, b = P["A"], P["B"]
            w.writerow([i, *P["k"], P["row"]["stratum"], P["lab"][a], f"{P['q'][a]:.3f}", P["imm"][a], P["lab"][b], f"{P['q'][b]:.3f}", P["imm"][b],
                        f"{P['q'][b] - P['q'][a]:+.3f}", int(P["lab"][b].split("(")[0] in SETUP), P["row"]["proposal_shows_A_and_B_of5"]])
    with PdfPages(OUT + "/paths_catalog.pdf") as pdf:
        # 先頭に一覧表 (1ページ 32 行)
        head = ["#", "局面", "区分", "A (今すぐ有利)", "A 審判値 / 今の得点", "B (後の展開で有利)", "B 審判値 / 今の得点", "B−A"]
        body = [[str(i), f"g{P['k'][0]} e{P['k'][1]} s{P['k'][2]}", P["row"]["stratum"], P["lab"][P["A"]],
                 f"{P['q'][P['A']]:+.2f} / {P['imm'][P['A']]:+d}", P["lab"][P["B"]], f"{P['q'][P['B']]:+.2f} / {P['imm'][P['B']]:+d}",
                 f"{P['q'][P['B']] - P['q'][P['A']]:+.2f}"] for i, P in enumerate(Ps, 1)]
        n_setup = sum(P["lab"][P["B"]].split("(")[0] in SETUP for P in Ps)
        for p0 in range(0, len(body), 32):
            fig = plt.figure(figsize=(11.69, 8.27))
            if p0 == 0:
                fig.text(0.03, 0.965, f"別の道: 今すぐ有利な A と、今の得点は 2 点以上低いが審判値は最良級の B が別クラスタ ({len(Ps)} 局面 / 非最終 468 局面)", fontsize=11)
                fig.text(0.03, 0.935, f"並び: B が置く手 (ドロー/フリーズ/ガード, {n_setup} 局面) を先に → B の審判値が A に近い・上回る順。"
                                      "最良級 = 最良の審判値から 0.25 以内。", fontsize=8)
            ax = fig.add_axes([0.03, 0.03, 0.94, 0.88]); ax.axis("off")
            t = ax.table(cellText=body[p0:p0 + 32], colLabels=head, loc="upper center", cellLoc="left",
                         colWidths=[0.03, 0.09, 0.09, 0.2, 0.1, 0.2, 0.1, 0.05])
            t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.25)
            pdf.savefig(fig); plt.close(fig)
        for i, P in enumerate(Ps, 1):
            fig = page(P, i); pdf.savefig(fig)
            if i <= 3: fig.savefig(f"{OUT}/sample_{i:03d}.png", dpi=100)
            plt.close(fig)
    print("pages", len(Ps), "B setup", sum(P["lab"][P["B"]].split("(")[0] in SETUP for P in Ps))

if __name__ == "__main__":
    main()
