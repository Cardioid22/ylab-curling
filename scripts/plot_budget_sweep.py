# -*- coding: utf-8 -*-
"""予算スイープの主結果図: regret vs playout予算 (log-x, アーム別)。
各予算の reinvest_joined.csv を読み、局面集約後の平均regret + ブートストラップ95%CI を描く。"""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

BUD = {
 50:  "reinvest_experiment/sweep/merged/P50/regret/reinvest_joined.csv",
 100: "reinvest_experiment/sweep/merged/P100/regret/reinvest_joined.csv",
 200: "reinvest_experiment/scorescreen/run50/regret/reinvest_joined.csv",
 500: "reinvest_experiment/sweep/merged/P500/regret/reinvest_joined.csv",
 1000:"reinvest_experiment/sweep/merged/P1000/regret/reinvest_joined.csv",
}
ARMS = [("A1","AllGrid (総当たり)","#d62728","o"),
        ("A2","Proposed distDelta (盤面類似)","#1f77b4","s"),
        ("A5","RandomK (ランダム削減)","#7f7f7f","^"),
        ("A7","ScoreScreen (得点で絞る)","#2ca02c","D")]
OUT = "reinvest_experiment/sweep/merged"

def pos_means(fp, arm):
    by=defaultdict(list)
    for r in csv.DictReader(open(fp,encoding="utf-8")):
        if r["arm"]!=arm: continue
        try: by[(r["game_id"],r["end"],r["shot_num"])].append(float(r["regret"]))
        except: pass
    return np.array([np.mean(v) for v in by.values()])

def boot_ci(x,n=10000,seed=1):
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(n,len(x)))
    b=x[idx].mean(1); return np.percentile(b,[2.5,97.5])

Ps=sorted(BUD)
fig,ax=plt.subplots(figsize=(8.5,5.6))
data={}
for arm,label,c,mk in ARMS:
    means=[]; los=[]; his=[]
    for P in Ps:
        x=pos_means(BUD[P],arm); m=x.mean(); lo,hi=boot_ci(x)
        means.append(m); los.append(m-lo); his.append(hi-m)
    data[arm]=means
    ax.errorbar(Ps,means,yerr=[los,his],label=label,color=c,marker=mk,
                ms=7,lw=2,capsize=3,alpha=0.9)

ax.set_xscale("log")
ax.set_xticks(Ps); ax.set_xticklabels([str(p) for p in Ps])
ax.set_xlabel("playout 予算 P  (対数軸, R=10固定・50局面×5seed)")
ax.set_ylabel("平均 regret  (= q* − 選んだ手のE[score],  低いほど良い)")
ax.set_title("予算 vs 決定品質: 得点スクリーン(A7)は最小予算で最良・予算に鈍感")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right", framealpha=0.95)

# 効率の目安: A7@最小予算 の水平線 と A1 が到達できていないこと
a7_50=data["A7"][0]
ax.axhline(a7_50, color="#2ca02c", ls="--", lw=1, alpha=0.6)
ax.annotate(f"A7@P=50 = {a7_50:.2f}\n(A1はP=1000でも {data['A1'][-1]:.2f} 止まり\n→ A7は1/20以下の予算で上回る)",
            xy=(50,a7_50), xytext=(120, a7_50-0.16), fontsize=9,
            color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", alpha=0.7))
fig.tight_layout()
os.makedirs(OUT,exist_ok=True)
p=os.path.join(OUT,"regret_vs_budget.png")
fig.savefig(p,dpi=150,bbox_inches="tight"); print("saved",p)

# 表もCSVで
with open(os.path.join(OUT,"regret_vs_budget.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["P"]+[a for a,_,_,_ in ARMS])
    for i,P in enumerate(Ps):
        w.writerow([P]+[f"{data[a][i]:.4f}" for a,_,_,_ in ARMS])
print("saved",os.path.join(OUT,"regret_vs_budget.csv"))
