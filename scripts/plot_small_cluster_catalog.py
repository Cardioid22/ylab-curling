# -*- coding: utf-8 -*-
"""最良手のクラスタが小さい局面 (大きさ 1〜2, 最終ショット除く) の図カタログ。

先生と「人間が見て意味のある局面」を選ぶための一覧。1局面 = 1ページ:
  上段 (a) 局面 (石の番号 = ティーから近い順, 赤丸 = 最良手の狙う石)
       (b) 全候補の着地点をクラスタ色で (★ = クラスタあり 4手, □ = クラスタなし 4手, ◎ = 最良手)
       (c) 数値 (5 seed の集計, regret)
  下段 (d) 最良手 (審判) の結果盤面, (e) クラスタありの 4手で最良の結果盤面, (f) クラスタなしの 4手で最良の結果盤面
クラスタは結果盤面 (外乱なし) で決まるので seed によらない。4手の選び方は事前推定 e_score (R_pre=5) が seed で揺れるので、
図は「代表 seed」(5 seed で最も多い勝ち負けパターンの最小 seed) のもの。

前提: scripts/analyze_cluster_vs_topk.py の screen_compare_rows.csv と、
各局面の --export-case 出力 (<dir>/export/export_g*_e*_s*/)。
Usage: python scripts/plot_small_cluster_catalog.py gpw_experiment/cluster_vs_topk
出力: <dir>/catalog/case_XX_g*_e*_s*.png, <dir>/catalog/catalog.pdf, <dir>/catalog/index.csv, <dir>/catalog/index.md
"""
import csv, glob, math, os, sys, statistics as st
from collections import defaultdict, Counter
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
OUT = D + "/catalog"
SEEDS = range(42, 47)
TAU = 0.25
WEIGHT_JA = {"TOUCH": "タッチ", "WEAK": "弱", "MIDDLE": "中", "STRONG": "強"}
TYPE_JA = {"Hit": "ヒット", "Draw": "ドロー", "Freeze": "フリーズ", "PostGuard": "ポストガード",
           "PreGuard": "プレガード", "Peel": "ピール", "ComeAround": "カムアラウンド", "Pass": "パス", "Guard": "ガード"}

def key(r): return (int(r["game_id"]), int(r["end"]), int(r["shot_num"]))

# ---- 局面の選択 (screen_compare_rows) ----
rows = defaultdict(dict)
for r in csv.DictReader(open(D + "/screen_compare_rows.csv", encoding="utf-8")):
    rows[(int(r["game"]), int(r["end"]), int(r["shot"]))][int(r["seed"])] = r
sel = [k for k, v in rows.items() if v[42]["phase_r"] != "r1" and int(v[42]["best_csize"]) <= 2]
missing = [k for k in sel if not os.path.exists(f"{D}/export/export_g{k[0]}_e{k[1]}_s{k[2]}/result_stones.csv")]
if missing: print(f"warning: export が無い局面 {len(missing)} 個は飛ばす (--export-case で作る)")
sel = [k for k in sel if k not in missing]
selset = set(sel)

Q = {}
for f in glob.glob(GE + "/referee/score_move_qtable_idx*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if key(r) in selset: Q[(key(r), int(r["candidate_idx"]))] = float(r["q_ref_mean"])
CT = defaultdict(dict)   # (k, seed) -> list of rows
for seed in SEEDS:
    tmp = defaultdict(list)
    for r in csv.DictReader(open(f"{GE}/A9P5N/seed_{seed}/cluster_table.csv", encoding="utf-8")):
        if key(r) in selset: tmp[key(r)].append(r)
    for k, v in tmp.items(): CT[k][seed] = v
BATCH = {}
for r in csv.DictReader(open(ROOT + "/test_positions500/batch_0001.csv", encoding="utf-8")):
    BATCH[(int(r["match_id"]), int(r["end"]), int(r["shot_num"]))] = r

# ---- 盤面ユーティリティ ----
def dist(x, y): return math.hypot(x - HC_X, y - HC_Y)
def in_house(x, y): return dist(x, y) <= R12 + SR
def ranked(stones):
    """stones: {(team, idx): (x, y)} -> ハウス内の石をティーから近い順に [(team, idx)]"""
    return sorted([s for s, p in stones.items() if in_house(*p)], key=lambda s: dist(*stones[s]))
def score_of(stones, me):
    """今エンドが終わったときの得点 (手番側から見て +/-)"""
    rk = ranked(stones)
    if not rk: return 0
    t0 = rk[0][0]; n = 0
    for s in rk:
        if s[0] != t0: break
        n += 1
    return n if t0 == me else -n

def load_case(k):
    g, e, s = k
    d = f"{D}/export/export_g{g}_e{e}_s{s}"
    inp = {(int(r["team"]), int(r["stone_idx"])): (float(r["x"]), float(r["y"]))
           for r in csv.DictReader(open(d + "/input_board.csv", encoding="utf-8"))}
    res = defaultdict(dict)
    for r in csv.DictReader(open(d + "/result_stones.csv", encoding="utf-8")):
        res[int(r["candidate_idx"])][(int(r["team"]), int(r["stone_idx"]))] = (float(r["x"]), float(r["y"]))
    cand = {int(r["candidate_idx"]): r for r in csv.DictReader(open(d + "/candidates.csv", encoding="utf-8"))}
    return inp, res, cand

def describe(label, inp, me):
    """ラベル -> 日本語の短い説明と、狙う石 (あれば)"""
    typ = label.split("(")[0]
    args = label[label.index("(") + 1:-1].split(",") if "(" in label else []
    ja = TYPE_JA.get(typ, typ)
    target = None
    if typ in ("Hit", "Freeze", "Peel", "PostGuard", "ComeAround") and len(args) >= 2 and args[1].lstrip("-").isdigit():
        n = int(args[1]); target = (n % 2, n // 2)
    txt = ja
    if typ == "Hit" and len(args) >= 3: txt += f"({WEIGHT_JA.get(args[2], args[2])})"
    if typ == "Draw" and len(args) >= 2: txt += f"({args[1]})"
    if target is not None and target in inp:
        who = "自" if target[0] == me else "相手"
        rk = ranked(inp)
        pos = f"No.{rk.index(target) + 1}" if target in rk else "ハウス外"
        txt += f" → {who}の石 [{pos}]"
    return txt, target

def outcome_type(inp, after, me):
    """結果盤面の型 (狙いのラベルではなく、実際に起きたこと)"""
    oo, om, _, _ = change(inp, after, me)
    moved = any(s in after and math.hypot(after[s][0] - p[0], after[s][1] - p[1]) > 0.05 for s, p in inp.items())
    if oo >= 2: return "相手石を2個以上出す"
    if oo == 1: return "相手石を1個出す"
    if om >= 1: return "自石だけ出す"
    if moved: return "石を動かす (除去なし)"
    return "置くだけ"

def side_effects(inp, after, me, target):
    """狙いのラベルでは分からない副作用: 狙った石が動いたか, 自石を押し込んだか (レイズ), 動いた/出た石の数"""
    moved = lambda s: s not in after or math.hypot(after[s][0] - inp[s][0], after[s][1] - inp[s][1]) > 0.05
    tmoved = None if target is None or target not in inp else moved(target)
    raise_ = any(s[0] == me and s in after and dist(*inp[s]) - dist(*after[s]) >= 0.3 for s in inp)
    return tmoved, raise_, sum(1 for s in inp if moved(s))

def change(inp, after, me):
    """結果盤面の要約: 出た石 (相手/自), 得点の変化"""
    op = 1 - me
    out_op = sum(1 for s in inp if s[0] == op and s not in after)
    out_me = sum(1 for s in inp if s[0] == me and s not in after)
    return out_op, out_me, score_of(inp, me), score_of(after, me)

# ---- 局面ごとの集計 ----
def per_position(k):
    inp, res, cand = load_case(k)
    b = BATCH[k]; me = int(b["team"]); op = 1 - me
    ct42 = CT[k][42]
    cl = {int(r["candidate_idx"]): int(r["cluster_id"]) for r in ct42}
    lab = {int(r["candidate_idx"]): r["label"] for r in ct42}
    csize = Counter(cl.values())
    q = {c: Q[(k, c)] for c in cl if (k, c) in Q}
    best = max(q, key=lambda c: (q[c], -c)); qs = q[best]
    per_seed = {}
    for seed in SEEDS:
        ct = CT[k][seed]
        ee = {int(r["candidate_idx"]): float(r["e_score"]) for r in ct}
        cv = [int(r["candidate_idx"]) for r in ct if r["is_representative"] == "1"]
        order = sorted(ee, key=lambda c: (-ee[c], c))
        tk = order[:len(cv)]
        cvb = max(cv, key=lambda c: q.get(c, -99)); tkb = max(tk, key=lambda c: q.get(c, -99))
        loss_cv, loss_tk = qs - q[cvb], qs - q[tkb]
        pat = "CV" if loss_cv < loss_tk - 1e-9 else ("TK" if loss_tk < loss_cv - 1e-9 else "=")
        per_seed[seed] = dict(ee=ee, cv=cv, tk=tk, cvb=cvb, tkb=tkb, loss_cv=loss_cv, loss_tk=loss_tk, pat=pat,
                              erank_best=order.index(best) + 1)
    pats = Counter(v["pat"] for v in per_seed.values())
    mode = max(pats, key=lambda p: (pats[p], {"CV": 2, "=": 1, "TK": 0}[p]))
    rep_seed = min(s for s in SEEDS if per_seed[s]["pat"] == mode)
    rs = rows[k]
    regm = lambda arm: st.mean(float(rs[s][f"reg_{arm}"]) for s in SEEDS if rs[s][f"reg_{arm}"] != "")
    desc, target = describe(lab[best], inp, me)
    out_op, out_me, sc0, sc1 = change(inp, res[best], me)
    # 狙う石の分類
    rk = ranked(inp)
    opp_rk = [s for s in rk if s[0] == op]
    if lab[best].startswith("Hit") and target is not None:
        if target[0] == op and target in rk and opp_rk and target != opp_rk[0]: group = "A: 相手のNo.1以外の石へのヒット"
        elif target[0] == op and opp_rk and target == opp_rk[0]: group = "B: 相手のいちばん内側の石へのヒット"
        else: group = "C: その他のヒット"
    else:
        group = "D: ヒット以外"
    return dict(k=k, inp=inp, res=res, me=me, op=op, cl=cl, lab=lab, csize=csize, q=q, best=best, qs=qs,
                per_seed=per_seed, pats=pats, rep_seed=rep_seed, desc=desc, target=target,
                out_op=out_op, out_me=out_me, sc0=sc0, sc1=sc1, group=group, otype=outcome_type(inp, res[best], me),
                fx=side_effects(inp, res[best], me, target),
                cat=rs[42]["category"], stratum=rs[42]["stratum"], phase=rs[42]["phase_r"],
                cv_has_best=sum(1 for s in SEEDS if best in per_seed[s]["cv"]),
                tk_has_best=sum(1 for s in SEEDS if best in per_seed[s]["tk"]),
                rescue=sum(int(rs[s]["rescue"]) for s in SEEDS), lost=sum(int(rs[s]["lost"]) for s in SEEDS),
                reg=dict((a, regm(a)) for a in ("A9P5N", "A8N", "A2N", "A1N")),
                erank_med=st.median(per_seed[s]["erank_best"] for s in SEEDS), N=len(cl), K=len(csize))

# ---- 描画 ----
def ylim_for(P):
    ys = [p[1] for p in P["inp"].values()] + [p[1] for c in (P["best"],) for p in P["res"][c].values()]
    lo = min([HC_Y - R12 - 0.5] + [y - 0.4 for y in ys if y > 30])
    return max(lo, 31.5), HC_Y + R12 + 0.5

def house(ax, P, title):
    for rr, c in [(R12, "#cfe0f3"), (R8, "#ffffff"), (R4, "#f3cfcf"), (RBTN, "#ffffff")]:
        ax.add_patch(Circle((HC_X, HC_Y), rr, facecolor=c, edgecolor="#999", lw=0.7, zorder=1))
    ax.axhline(HC_Y, color="#ccc", lw=0.5, ls="--")
    ax.set_xlim(-AREA_X, AREA_X); ax.set_ylim(*ylim_for(P))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.5)

def draw_stones(ax, st_, alpha=1.0, lw=0.8, z=3):
    for (t, i), (x, y) in st_.items():
        ax.add_patch(Circle((x, y), SR, facecolor=C0 if t == 0 else C1, edgecolor="#222", lw=lw, alpha=alpha, zorder=z))

def panel_input(ax, P):
    house(ax, P, f"(a) 局面  手番 = {'橙' if P['me'] == 0 else '青'}"
                 f" ({'後攻' if P['k'][2] % 2 == 1 else '先攻'}, 第{P['k'][2] + 1}投)\n数字 = ティーから近い順  赤丸 = 最良手の狙う石")
    draw_stones(ax, P["inp"])
    for n, s in enumerate(ranked(P["inp"]), 1):
        x, y = P["inp"][s]
        ax.text(x, y, str(n), ha="center", va="center", fontsize=6.5, color="w", weight="bold", zorder=4)
    if P["target"] in P["inp"]:
        ax.add_patch(Circle(P["inp"][P["target"]], SR * 1.9, facecolor="none", edgecolor="#d00", lw=1.8, zorder=5))

def land(P, c):
    s = (P["me"], P["k"][2] // 2)
    return P["res"][c].get(s)

def panel_map(ax, P, ps):
    house(ax, P, "(b) 全候補の着地点 (色 = クラスタ)\n★ = クラスタあり 4手  □ = クラスタなし 4手  ◎ = 最良手")
    draw_stones(ax, P["inp"], alpha=0.35, lw=0.3)
    cmap = plt.get_cmap("tab20")
    ee = ps["ee"]
    cids = sorted(set(P["cl"].values()), key=lambda c: -max(ee[i] for i in P["cl"] if P["cl"][i] == c))
    color = {c: cmap(i % 20) for i, c in enumerate(cids)}
    for c in P["cl"]:
        p = land(P, c)
        if p: ax.plot(*p, "o", ms=3.5, color=color[P["cl"][c]], alpha=0.85, zorder=4)
    for c in ps["cv"]:
        p = land(P, c)
        if p: ax.plot(*p, "*", ms=13, mfc=color[P["cl"][c]], mec="k", mew=0.8, zorder=6)
    for c in ps["tk"]:
        p = land(P, c)
        if p: ax.plot(*p, "s", ms=10, mfc="none", mec="k", mew=1.2, zorder=6)
    p = land(P, P["best"])
    if p: ax.plot(*p, "o", ms=16, mfc="none", mec="#d00", mew=1.8, zorder=7)
    else: ax.text(0.02, 0.02, "最良手の投石はシート外へ", transform=ax.transAxes, fontsize=7, color="#d00")

def panel_result(ax, P, c, title):
    house(ax, P, title)
    inp, after = P["inp"], P["res"][c]
    draw_stones(ax, inp, alpha=0.22, lw=0.3, z=2)
    draw_stones(ax, after)
    for s, (x0, y0) in inp.items():
        if s in after:
            x1, y1 = after[s]
            if math.hypot(x1 - x0, y1 - y0) > 0.05:
                ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=8, color="k", lw=0.8, zorder=5))
        else:
            ax.plot(x0, y0, "x", color="k", ms=8, mew=1.5, zorder=5)
    p = land(P, c)   # 投げた石 (白い点)
    if p: ax.plot(*p, "o", ms=3.2, color="w", mec="k", mew=0.6, zorder=6)
    else: ax.text(0.02, 0.02, "投げた石はプレー外", transform=ax.transAxes, fontsize=7)

def shot_title(P, c, head):
    desc, _ = describe(P["lab"][c], P["inp"], P["me"])
    oo, om, s0, s1 = change(P["inp"], P["res"][c], P["me"])
    return (f"{head}\n{desc}\n審判値 {P['q'][c]:+.2f} (最良との差 {P['qs'] - P['q'][c]:.2f})  クラスタの大きさ {P['csize'][P['cl'][c]]}\n"
            f"出した石 相手{oo}/自{om}  今の得点 {s0:+d}→{s1:+d}")

def panel_text(ax, P):
    ax.axis("off")
    ps = P["per_seed"]
    L = [f"{P['group']}  /  結果: {P['otype']}",
         f"区分: {P['cat']} / {P['stratum']} / 残り{P['phase']}",
         f"最良手の副作用: 狙った石が{'動く' if P['fx'][0] else ('動かない (別の石に当たる)' if P['fx'][0] is False else '-')},"
         f" 自石のレイズ{'あり' if P['fx'][1] else 'なし'}, 動いた石 {P['fx'][2]} 個",
         f"候補 {P['N']} 手 → クラスタ {P['K']} 個",
         f"最良手: {P['lab'][P['best']]}",
         f"  クラスタの大きさ {P['csize'][P['cl'][P['best']]]}, 推定順位 (中央値) {P['erank_med']:.0f} / {P['N']}",
         "",
         "5 seed の集計 (4手の選び方は seed で揺れる)",
         f"  最良手を残した: クラスタあり {P['cv_has_best']}/5, なし {P['tk_has_best']}/5",
         f"  4手の最良の比較: ありが良い {P['pats']['CV']}, 同じ {P['pats']['=']}, なしが良い {P['pats']['TK']}",
         f"  最良級(差≤{TAU})を あり だけ残した {P['rescue']}, なし だけ {P['lost']}",
         "",
         "実際の regret (5 seed 平均, 外乱込み, P=200)",
         f"  提案 {P['reg']['A9P5N']:.2f}  クラスタなし {P['reg']['A8N']:.2f}",
         f"  既存(medoid) {P['reg']['A2N']:.2f}  全探索 {P['reg']['A1N']:.2f}",
         "",
         f"図は seed {P['rep_seed']} (最も多いパターン)",
         "審判値 = 手番側から見たエンド得点の期待値",
         "下段: 矢印 = 動いた石, × = 出た石, 白点 = 投げた石"]
    ax.text(0.0, 1.0, "\n".join(L), va="top", ha="left", fontsize=8.2, transform=ax.transAxes)

def figure(P, num):
    ps = P["per_seed"][P["rep_seed"]]
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    panel_input(axes[0, 0], P)
    panel_map(axes[0, 1], P, ps)
    panel_text(axes[0, 2], P)
    panel_result(axes[1, 0], P, P["best"], shot_title(P, P["best"], "(d) 最良手 (審判)"))
    same_cv = " = 最良手" if ps["cvb"] == P["best"] else ""
    panel_result(axes[1, 1], P, ps["cvb"], shot_title(P, ps["cvb"], f"(e) クラスタあり 4手の最良{same_cv}"))
    same_tk = " = 最良手" if ps["tkb"] == P["best"] else ""
    panel_result(axes[1, 2], P, ps["tkb"], shot_title(P, ps["tkb"], f"(f) クラスタなし 4手の最良{same_tk}"))
    g, e, s = P["k"]
    fig.suptitle(f"#{num:02d}  g{g} e{e} s{s}", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig

def main():
    os.makedirs(OUT, exist_ok=True)
    Ps = [per_position(k) for k in sel]
    # 並び: グループ順 → 提案がクラスタなしより良かった順 (regret 差)
    Ps.sort(key=lambda P: (P["group"], P["reg"]["A9P5N"] - P["reg"]["A8N"]))
    with open(OUT + "/index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["num", "game", "end", "shot", "group", "category", "stratum", "phase_r", "N", "K", "best_label", "best_desc",
                    "best_csize", "best_erank_median", "q_best", "outcome_type", "target_moved", "raise_own", "n_moved", "out_opp", "out_me", "score_before", "score_after",
                    "cv_has_best_of5", "tk_has_best_of5", "cv_better_of5", "same_of5", "tk_better_of5", "rescue_of5", "lost_of5",
                    "reg_A9P5N", "reg_A8N", "reg_A2N", "reg_A1N", "rep_seed", "png"])
        for i, P in enumerate(Ps, 1):
            g, e, s = P["k"]
            png = f"case_{i:02d}_g{g}_e{e}_s{s}.png"
            w.writerow([i, g, e, s, P["group"], P["cat"], P["stratum"], P["phase"], P["N"], P["K"], P["lab"][P["best"]], P["desc"],
                        P["csize"][P["cl"][P["best"]]], P["erank_med"], f"{P['qs']:.3f}", P["otype"], "" if P["fx"][0] is None else int(P["fx"][0]), int(P["fx"][1]), P["fx"][2], P["out_op"], P["out_me"], P["sc0"], P["sc1"],
                        P["cv_has_best"], P["tk_has_best"], P["pats"]["CV"], P["pats"]["="], P["pats"]["TK"], P["rescue"], P["lost"],
                        *(f"{P['reg'][a]:.3f}" for a in ("A9P5N", "A8N", "A2N", "A1N")), P["rep_seed"], png])
    with open(OUT + "/index.md", "w", encoding="utf-8") as f:
        f.write("# 最良手のクラスタが小さい局面の一覧 (大きさ1〜2, 最終ショット除く)\n\n")
        f.write("並び: グループ → 提案 (ClusterValue+N) とクラスタなし (ScoreTopK) の regret 差が大きい順 (提案が良い順)。\n")
        f.write("「最良手を残した」は5 seed中の回数。regret は5 seed平均 (外乱込み, P=200, negamax木)。\n\n")
        cur = None
        for i, P in enumerate(Ps, 1):
            if P["group"] != cur:
                cur = P["group"]; n = sum(1 for Q_ in Ps if Q_["group"] == cur)
                f.write(f"\n## {cur} ({n} 局面)\n\n| # | 局面 | 区分 | 最良手 | 結果の型 | クラスタ | 推定順位 | 出した石(相手/自) | 得点 | 最良手を残した (あり/なし) | regret 提案/なし/既存 |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
            g, e, s = P["k"]
            f.write(f"| {i} | g{g} e{e} s{s} | {P['stratum']} | {P['desc']} | {P['otype']} | {P['csize'][P['cl'][P['best']]]} | {P['erank_med']:.0f}/{P['N']} | "
                    f"{P['out_op']}/{P['out_me']} | {P['sc0']:+d}→{P['sc1']:+d} | {P['cv_has_best']}/{P['tk_has_best']} | "
                    f"{P['reg']['A9P5N']:.2f}/{P['reg']['A8N']:.2f}/{P['reg']['A2N']:.2f} |\n")
    with PdfPages(OUT + "/catalog.pdf") as pdf:
        # 先頭に一覧表 (1ページ 32 行)
        head = ["#", "局面", "グループ", "最良手", "結果の型", "得点", "クラスタ", "推定順位", "最良手を残した\nあり/なし", "regret\n提案/なし/既存"]
        body = [[str(i), f"g{P['k'][0]} e{P['k'][1]} s{P['k'][2]}", P["group"][0], P["desc"], P["otype"], f"{P['sc0']:+d}→{P['sc1']:+d}",
                 str(P["csize"][P["cl"][P["best"]]]), f"{P['erank_med']:.0f}/{P['N']}", f"{P['cv_has_best']}/{P['tk_has_best']}",
                 f"{P['reg']['A9P5N']:.2f}/{P['reg']['A8N']:.2f}/{P['reg']['A2N']:.2f}"] for i, P in enumerate(Ps, 1)]
        for p0 in range(0, len(body), 32):
            fig = plt.figure(figsize=(11.69, 8.27))
            if p0 == 0:
                gc = Counter(P["group"] for P in Ps)
                fig.text(0.03, 0.965, f"最良手のクラスタが小さい局面 (大きさ 1〜2, 最終ショット除く): {len(Ps)} 局面 / 非最終 468 局面", fontsize=12)
                fig.text(0.03, 0.935, "  ".join(f"{g} ({n})" for g, n in sorted(gc.items()))
                         + "   並び: グループ → 提案がクラスタなしより良かった順。「最良手を残した」は 5 seed 中の回数。", fontsize=8)
            ax = fig.add_axes([0.03, 0.03, 0.94, 0.88]); ax.axis("off")
            rows_ = body[p0:p0 + 32]
            t = ax.table(cellText=rows_, colLabels=head, loc="upper center", cellLoc="left",
                         colWidths=[0.03, 0.09, 0.05, 0.25, 0.14, 0.06, 0.05, 0.07, 0.09, 0.13])
            t.auto_set_font_size(False); t.set_fontsize(7); t.scale(1, 1.25)
            pdf.savefig(fig); plt.close(fig)
        for i, P in enumerate(Ps, 1):
            fig = figure(P, i)
            g, e, s = P["k"]
            fig.savefig(f"{OUT}/case_{i:02d}_g{g}_e{e}_s{s}.png", dpi=110)
            pdf.savefig(fig); plt.close(fig)
    print("pages", len(Ps)); print(Counter(P["group"] for P in Ps)); print(Counter((P["group"][0], P["otype"]) for P in Ps))

def showcase(cases, out):
    """選んだ局面を1枚に並べる: 行 = 局面, 列 = (局面) (最良手の結果) (クラスタなし 4手の最良の結果)"""
    Ps = [per_position(k) for k in cases]
    fig, axes = plt.subplots(len(Ps), 3, figsize=(10.5, 4.7 * len(Ps)), squeeze=False)
    for i, P in enumerate(Ps):
        ps = P["per_seed"][P["rep_seed"]]
        g, e, s = P["k"]
        panel_input(axes[i, 0], P)
        axes[i, 0].set_title(f"g{g} e{e} s{s}  局面 (手番 = {'橙' if P['me'] == 0 else '青'}, "
                             f"{'後攻' if s % 2 == 1 else '先攻'}, 第{s + 1}投)\n数字 = ティーから近い順  赤丸 = 最良手の狙う石", fontsize=8.5)
        panel_result(axes[i, 1], P, P["best"], shot_title(P, P["best"], "最良手 (審判)")
                     + f"\n最良手を4手に残した: クラスタあり {P['cv_has_best']}/5, なし {P['tk_has_best']}/5")
        panel_result(axes[i, 2], P, ps["tkb"], shot_title(P, ps["tkb"], f"クラスタなし 4手の最良 (seed {P['rep_seed']})"))
    fig.tight_layout(h_pad=2.0); fig.savefig(out, dpi=130); plt.close(fig); print("saved", out)

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--showcase":
        # python scripts/plot_small_cluster_catalog.py <dir> --showcase <out.png> g,e,s [g,e,s ...]
        showcase([tuple(int(x) for x in a.split(",")) for a in sys.argv[4:]], sys.argv[3])
    else:
        main()
