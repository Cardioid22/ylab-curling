# -*- coding: utf-8 -*-
"""
MCTS の木の深さの価値: 「depth-1 の手」と「depth-k(=min(r,3)) の手」の得点期待値差 Δ。
  depth-1 の手 = argmax q_ref (審判)  … depth-1 MCTS が収束する手。追加実行不要。
  depth-k の手 = AllGrid depth-k MCTS の選択手 (要・実行)。
  Δ = q*(=max q_ref) − q_ref(depth-k の手)   … 深く読むと選ぶ手の得点期待値がどれだけ動くか。
  ※ q_ref は depth-1 に整合的なので「どちらが強いか」ではなく「差の大きさ」を見る指標。

残り手数 r 別に集計し、Δが大きい/小さい局面を盤面図にプロットする。

入力:
  --referee-csv : depth 実験の審判 (q_ref)
  --treedepth-dir : experiments/treedepth (d2/seed_*/reinvest_results.csv, d3/... を含む)
  --batch-csv : test_positions_depth/batch_0001.csv (盤面図用)
使用例:
  python scripts/analyze_tree_depth.py \
    --referee-csv reinvest_experiment/depth/referee/score_move_qtable.csv \
    --treedepth-dir experiments/treedepth \
    --batch-csv test_positions_depth/batch_0001.csv \
    --out reinvest_experiment/depth/treedepth_analysis
"""
import argparse, csv, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_board import draw_board, C0, C1
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

def load_referee(fp):
    q = defaultdict(dict)
    for r in csv.DictReader(open(fp, encoding="utf-8")):
        q[(r["game_id"], r["end"], r["shot_num"])][int(r["candidate_idx"])] = float(r["q_ref_mean"])
    return q

def load_depth_moves(base, d):
    # base/d{d}/seed_*/reinvest_results*.csv -> pos -> [best_idx per seed]
    mv = defaultdict(list)
    for fp in glob.glob(f"{base}/d{d}/seed_*/reinvest_results*.csv") + glob.glob(f"{base}/d{d}/reinvest_results*.csv"):
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            try: idx = int(r["candidate_idx"])
            except (KeyError, ValueError): continue
            if idx < 0: continue
            mv[(r["game_id"], r["end"], r["shot_num"])].append(idx)
    return mv

def boot_ci(x, n=10000, seed=5):
    x=np.asarray(x,float)
    if len(x)==0: return (np.nan,np.nan)
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),(n,len(x)))
    return tuple(np.percentile(x[idx].mean(1),[2.5,97.5]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--referee-csv", required=True)
    ap.add_argument("--treedepth-dir", required=True)
    ap.add_argument("--batch-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=3, help="大/小 Δ の盤面図を各何局面")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    q = load_referee(args.referee_csv)
    qstar = {k: max(v.values()) for k, v in q.items()}
    d2 = load_depth_moves(args.treedepth_dir, 2)
    d3 = load_depth_moves(args.treedepth_dir, 3)
    board = {(r["match_id"], r["end"], r["shot_num"]): r
             for r in csv.DictReader(open(args.batch_csv, newline="", encoding="utf-8"))}

    per = []  # (r, delta, key)
    for k in q:
        g, e, s = k; r = 16 - int(s)
        if r <= 1:
            per.append((r, 0.0, k)); continue          # 最終手は自明にΔ=0
        moves = d2[k] if r == 2 else d3[k]              # depth-k の手 (seed群)
        if not moves: continue
        deltas = [qstar[k] - q[k].get(m, qstar[k]) for m in moves]
        per.append((r, float(np.mean(deltas)), k))

    # r 別集計
    print(f"{'残りr':>5} {'n':>3} {'Δ(平均)':>10} {'95%CI':>18}")
    print("-"*42)
    rows=[]
    for r in sorted(set(p[0] for p in per)):
        ds=[p[1] for p in per if p[0]==r]; lo,hi=boot_ci(ds)
        print(f"{r:>5} {len(ds):>3} {np.mean(ds):>10.3f}   [{lo:>5.3f},{hi:>5.3f}]")
        rows.append([r,len(ds),round(np.mean(ds),4),round(lo,4),round(hi,4)])
    with open(os.path.join(args.out,"tree_depth_curve.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["remaining","n","delta_mean","ci_lo","ci_hi"]); w.writerows(rows)

    # Δ vs r 曲線
    fig,ax=plt.subplots(figsize=(7,5))
    R=[x[0] for x in rows]; M=[x[2] for x in rows]
    lo=[x[2]-x[3] for x in rows]; hi=[x[4]-x[2] for x in rows]
    ax.errorbar(R,M,yerr=[lo,hi],marker="o",ms=8,lw=2,capsize=4,color="#009E73")
    ax.axhline(0,color="0.6",ls="--",lw=1); ax.set_xticks(R)
    ax.set_xlabel("残り手数 r"); ax.set_ylabel("Δ = q* − q_ref(depth-k の手)  [点]")
    ax.set_title("木の深さの価値: depth-1 と depth-k で選ぶ手の得点期待値差\n(小さいほど depth-1 で十分)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out,"tree_depth_curve.png"),dpi=150); plt.close(fig)

    # 大Δ / 小Δ 局面の盤面図 (r>=2 のみ)
    cand=[p for p in per if p[0]>=2]
    cand.sort(key=lambda p:-p[1])
    picks=[("大Δ", cand[:args.top]), ("小Δ", cand[-args.top:][::-1])]
    per_pos_csv=[]
    for label, group in picks:
        fig,axes=plt.subplots(1,len(group),figsize=(3.0*len(group),4.6))
        if len(group)==1: axes=[axes]
        for ax,(r,delta,k) in zip(axes,group):
            g,e,s=k; row=board.get(k)
            if row is None: ax.axis("off"); continue
            draw_board(ax,row,f"g{g} e{e} s{s} (残り{r})\nΔ={delta:.2f}点")
            per_pos_csv.append([g,e,s,r,round(delta,3),label])
        fig.suptitle(f"{label} の局面 (木の深さが{'効く' if label=='大Δ' else '効かない'})", fontsize=12)
        fig.tight_layout(rect=[0,0,1,0.94])
        fig.savefig(os.path.join(args.out,f"boards_{'large' if label=='大Δ' else 'small'}_delta.png"),dpi=150)
        plt.close(fig)
    with open(os.path.join(args.out,"tree_depth_per_position.csv"),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["game_id","end","shot_num","remaining","delta","group"])
        for r in sorted(per,key=lambda p:-p[1]):
            w.writerow([r[2][0],r[2][1],r[2][2],r[0],round(r[1],4),""])
    print(f"\n[out] -> {args.out}/ (tree_depth_curve.png, boards_large/small_delta.png, *.csv)")

if __name__=="__main__":
    main()
