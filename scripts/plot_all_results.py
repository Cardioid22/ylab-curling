# -*- coding: utf-8 -*-
"""
最近の実験結果をまとめて発表用グラフ化する。
出力: reinvest_experiment/figures/*.png
  fig1_regret_vs_budget      : 予算スイープ (efficiency曲線)  ← 主結果
  fig2_regret_bars_P200      : P=200 の手法別 regret (95%CI + 有意性)
  fig3_coverage_retention    : 削減後に最善手を残すか (A2 vs A5)
色は色覚安全 (Okabe-Ito系, 検証済) + マーカーで二次符号化。
"""
import csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#222222", "#555555", "#cccccc"

# 手法 = カテゴリカル。固定順・固定色・固定マーカー (順序を絶対に回さない)
ARMS = [
    ("A1", "AllGrid (総当たり)",          "#D55E00", "o"),
    ("A2", "Proposed distDelta (盤面類似)", "#0072B2", "s"),
    ("A5", "RandomK (ランダム削減)",       "#CC79A7", "^"),
    ("A7", "ScoreScreen (得点で絞る)",     "#009E73", "D"),
]
COL = {a: c for a, _, c, _ in ARMS}
MK  = {a: m for a, _, _, m in ARMS}
LAB = {a: l for a, l, _, _ in ARMS}
OUT = "reinvest_experiment/figures"

BUD = {
 50:  "reinvest_experiment/sweep/merged/P50/regret/reinvest_joined.csv",
 100: "reinvest_experiment/sweep/merged/P100/regret/reinvest_joined.csv",
 200: "reinvest_experiment/scorescreen/run50/regret/reinvest_joined.csv",
 500: "reinvest_experiment/sweep/merged/P500/regret/reinvest_joined.csv",
 1000:"reinvest_experiment/sweep/merged/P1000/regret/reinvest_joined.csv",
}

def pos_means(fp, arm, col="regret"):
    by = defaultdict(list)
    for r in csv.DictReader(open(fp, encoding="utf-8")):
        if r["arm"] != arm: continue
        try: by[(r["game_id"], r["end"], r["shot_num"])].append(float(r[col]))
        except (KeyError, ValueError): pass
    return {k: np.mean(v) for k, v in by.items()}

def boot_ci(x, n=10000, seed=7):
    x = np.asarray(x, float)
    if len(x) == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (n, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))

def recessive(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED); ax.grid(axis="y", color=GRID, alpha=0.5, lw=0.6)
    ax.set_axisbelow(True)

# ---------------- fig1: regret vs budget ----------------
def fig1():
    Ps = sorted(BUD)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    series = {}
    for a, lab, c, mk in ARMS:
        m, lo, hi = [], [], []
        for P in Ps:
            x = np.array(list(pos_means(BUD[P], a).values()))
            m.append(x.mean()); l, h = boot_ci(x); lo.append(l); hi.append(h)
        series[a] = m
        ax.fill_between(Ps, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(Ps, m, color=c, marker=mk, ms=8, lw=2, label=lab)
    ax.set_xscale("log"); ax.set_xticks(Ps); ax.set_xticklabels([str(p) for p in Ps])
    recessive(ax)
    ax.set_xlabel("playout 予算 P (対数軸, R=10固定・50局面×5seed)", color=INK)
    ax.set_ylabel("平均 regret = q* − 選んだ手のE[score]  (低いほど良い)", color=INK)
    ax.set_title("予算 vs 決定品質: 得点スクリーン(A7)は最小予算で最良・予算に鈍感", color=INK, fontsize=13)
    # A7@最小 が A1@最大 を下回る efficiency 注記
    a7lo = series["A7"][0]
    ax.axhline(a7lo, color=COL["A7"], ls="--", lw=1, alpha=0.6)
    ax.annotate(f"A7@P=50={a7lo:.2f}\nA1はP=1000でも{series['A1'][-1]:.2f}止まり\n→ 1/20以下の予算で上回る",
                xy=(Ps[0], a7lo), xytext=(130, a7lo-0.14), color=COL["A7"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=COL["A7"], alpha=0.7))
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout(); p = f"{OUT}/fig1_regret_vs_budget.png"; fig.savefig(p, dpi=160); plt.close(fig)
    print("saved", p)

# ---------------- fig2: regret bars P=200 + 有意性 ----------------
def fig2():
    fp = BUD[200]
    M = {a: pos_means(fp, a) for a, _, _, _ in ARMS}
    common = sorted(set.intersection(*[set(M[a]) for a, _, _, _ in ARMS]))
    arr = {a: np.array([M[a][k] for k in common]) for a in M}
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    xs = np.arange(len(ARMS))
    for i, (a, lab, c, mk) in enumerate(ARMS):
        m = arr[a].mean(); lo, hi = boot_ci(arr[a])
        ax.bar(i, m, 0.62, color=c, edgecolor="white", lw=1.2, zorder=2)
        ax.errorbar(i, m, yerr=[[m-lo], [hi-m]], color=INK, capsize=4, lw=1.2, zorder=3)
        ax.text(i, m+0.015, f"{m:.2f}", ha="center", va="bottom", color=INK, fontsize=10)
    recessive(ax)
    ax.set_xticks(xs); ax.set_xticklabels([a for a, _, _, _ in ARMS])
    ax.set_ylabel("平均 regret (低いほど良い)", color=INK)
    ax.set_title("P=200: 手法別 regret と有意性 (n=50局面)", color=INK, fontsize=13)
    ax.set_ylim(0, max(arr[a].mean() for a in arr)*1.5)
    # 有意性: 全6ペアの Wilcoxon(局面対応) → Holm補正 → 主要ペアを注記
    import itertools
    arms4 = [a for a,_,_,_ in ARMS]
    raw = {}
    for a,b in itertools.combinations(arms4,2):
        try: raw[(a,b)] = stats.wilcoxon(arr[a], arr[b]).pvalue
        except ValueError: raw[(a,b)] = 1.0
    items = sorted(raw.items(), key=lambda kv: kv[1]); m=len(items); pholm={}; prev=0.0
    for i,(k,p) in enumerate(items):
        v=min(1.0,(m-i)*p); v=max(v,prev); pholm[k]=v; prev=v
    def getp(a,b): return pholm.get((a,b), pholm.get((b,a),1.0))
    pairs = [("A7","A1"), ("A7","A2")]
    idx = {a:i for i,(a,_,_,_) in enumerate(ARMS)}
    ytop = max(arr[a].mean() for a in arr)
    for j,(a,b) in enumerate(pairs):
        p = getp(a,b); s = "**" if p<0.05 else ("*" if p<0.10 else "n.s.")
        y = ytop*1.10 + j*ytop*0.14
        x1,x2 = idx[a], idx[b]
        ax.plot([x1,x1,x2,x2], [y,y+0.01,y+0.01,y], color=MUTED, lw=1)
        ax.text((x1+x2)/2, y+0.012, f"{a} vs {b}: {s} (p_holm={p:.3f})", ha="center", va="bottom",
                color=MUTED, fontsize=8.5)
    fig.text(0.5, 0.01, "凡例: ** p_holm<0.05, * <0.10 (Wilcoxon符号順位・局面対応, Holm補正/全6ペア)",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=[0,0.03,1,1]); p = f"{OUT}/fig2_regret_bars_P200.png"; fig.savefig(p, dpi=160); plt.close(fig)
    print("saved", p)

# ---------------- fig3: coverage / retention (A2 vs A5) ----------------
def fig3():
    REF = "reinvest_experiment/scorescreen/run50/referee/score_move_qtable.csv"
    q = defaultdict(dict)
    for r in csv.DictReader(open(REF, encoding="utf-8")):
        q[(r["game_id"],r["end"],r["shot_num"])][int(r["candidate_idx"])] = float(r["q_ref_mean"])
    qstar = {k:max(v.values()) for k,v in q.items()}
    best  = {k:max(v,key=v.get) for k,v in q.items()}
    SRV = "reinvest_experiment/scorescreen/scorescreen"
    def reps(arm, seed):
        d = defaultdict(set)
        for s in ["bear","jaguar","tiger","lion"]:
            fp = f"{SRV}/run50_{s}/{arm}/seed_{seed}/cluster_table.csv"
            if not os.path.exists(fp): continue
            for r in csv.DictReader(open(fp, encoding="utf-8")):
                if r["is_representative"] in ("1","True","true"):
                    d[(r["game_id"],r["end"],r["shot_num"])].add(int(r["candidate_idx"]))
        return d
    def stat(arm, seeds):
        gap=[]; cov=[]
        for sd in seeds:
            for k,rep in reps(arm,sd).items():
                if k not in q: continue
                gap.append(qstar[k]-max(q[k][i] for i in rep if i in q[k]))
                cov.append(1 if best[k] in rep else 0)
        return np.array(gap), np.array(cov)
    g2,c2 = stat("A2",[42]); g5,c5 = stat("A5",[42,43,44,45,46])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    # 左: retention_gap (低いほど良い)
    ax=axes[0]
    for i,(a,gg) in enumerate([("A2",g2),("A5",g5)]):
        m=gg.mean(); lo,hi=boot_ci(gg)
        ax.bar(i,m,0.55,color=COL[a],edgecolor="white",lw=1.2,zorder=2)
        ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],color=INK,capsize=4,lw=1.2,zorder=3)
        ax.text(i,m+0.005,f"{m:.3f}",ha="center",va="bottom",color=INK,fontsize=10)
    recessive(ax); ax.set_xticks([0,1]); ax.set_xticklabels(["A2\ndistDelta","A5\nRandomK"])
    ax.set_ylabel("retention_gap = q* − 残したK候補内の最善 (低いほど良い)", color=INK, fontsize=9)
    ax.set_title("削減で失うE[score]", color=INK, fontsize=12)
    # 右: best手カバー率
    ax=axes[1]
    for i,(a,cc) in enumerate([("A2",c2),("A5",c5)]):
        m=cc.mean(); lo,hi=boot_ci(cc)
        ax.bar(i,m,0.55,color=COL[a],edgecolor="white",lw=1.2,zorder=2)
        ax.errorbar(i,m,yerr=[[max(0,m-lo)],[max(0,hi-m)]],color=INK,capsize=4,lw=1.2,zorder=3)
        ax.text(i,m+0.008,f"{m:.2f}",ha="center",va="bottom",color=INK,fontsize=10)
    recessive(ax); ax.set_xticks([0,1]); ax.set_xticklabels(["A2\ndistDelta","A5\nRandomK"])
    ax.set_ylabel("最善手カバー率 (高いほど良い)", color=INK, fontsize=9)
    ax.set_title("最善手を残せた割合", color=INK, fontsize=12)
    fig.suptitle("盤面類似クラスタ(A2)は「良い手の保持」でランダム削減(A5)に勝っていない",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0,0,1,0.95]); p=f"{OUT}/fig3_coverage_retention.png"; fig.savefig(p,dpi=160); plt.close(fig)
    print("saved", p)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3()
    print("done ->", OUT)
