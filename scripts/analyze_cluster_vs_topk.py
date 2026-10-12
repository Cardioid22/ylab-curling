# -*- coding: utf-8 -*-
"""クラスタあり (ClusterValue+N の採用代表 4 手) vs クラスタなし (同じ事前推定 e_score の上位 4 手) のスクリーン比較。

A9P5N の cluster_table (500局面 x 5seed) と審判 Q テーブルから、同じ推定値のもとで
「クラスタで選んだ 4 手」と「推定値上位 4 手 (= ScoreTopK の規則)」が残す候補をオフラインで比べる。
実際の regret (joined_noisy7.csv) も行に付ける。

Usage: python scripts/analyze_cluster_vs_topk.py gpw_experiment/cluster_vs_topk
出力: <out>/screen_compare_rows.csv (局面 x seed), 標準出力に層別の要約
"""
import csv, glob, os, sys, statistics as st
from collections import defaultdict, Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GE = ROOT + "/gpw_experiment"
OUT = sys.argv[1]
TAU = 0.25

def key(r): return (int(r["game_id"]), int(r["end"]), int(r["shot_num"]))

# 審判
Q = {}; QSD = {}; QLAB = {}
for f in glob.glob(GE + "/referee/score_move_qtable_idx*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        k = key(r); c = int(r["candidate_idx"])
        Q[(k, c)] = float(r["q_ref_mean"]); QSD[(k, c)] = float(r["q_ref_sd"]); QLAB[(k, c)] = r["label"]
qstar = defaultdict(lambda: -1e9)
for (k, c), q in Q.items(): qstar[k] = max(qstar[k], q)

cat = {}
for r in csv.DictReader(open(ROOT + "/test_positions500/categories.csv", encoding="utf-8")):
    cat[key(r)] = (r["category"], r["stratum"])
feat = {}
for r in csv.DictReader(open(GE + "/features500_A9P5N/position_features.csv", encoding="utf-8")):
    feat[key(r)] = r

# 実際の regret (外乱込み 7 手法)
REG = defaultdict(dict)
for r in csv.DictReader(open(GE + "/regret500/joined_noisy7.csv", encoding="utf-8")):
    REG[(key(r), int(r["seed"]))][r["arm"]] = float(r["regret"])

def typ(label): return label.split("(")[0]

rows = []
for seed in range(42, 47):
    ct = defaultdict(list)
    for r in csv.DictReader(open(f"{GE}/A9P5N/seed_{seed}/cluster_table.csv", encoding="utf-8")):
        ct[key(r)].append(r)
    for k, cands in ct.items():
        qs = qstar[k]
        if qs < -1e8: continue
        c2cl = {int(r["candidate_idx"]): int(r["cluster_id"]) for r in cands}
        csize = Counter(c2cl.values())
        e = {int(r["candidate_idx"]): float(r["e_score"]) for r in cands}
        lab = {int(r["candidate_idx"]): r["label"] for r in cands}
        cv = [int(r["candidate_idx"]) for r in cands if r["is_representative"] == "1"]
        order = sorted(e, key=lambda c: (-e[c], c))
        erank = {c: i + 1 for i, c in enumerate(order)}
        topk = order[:len(cv)]
        q = lambda c: Q.get((k, c), float("nan"))
        near = {c for c in e if q(c) >= qs - TAU}
        cv_q = max(q(c) for c in cv); tk_q = max(q(c) for c in topk)
        tk_ncl = len({c2cl[c] for c in topk})
        cv_near = [c for c in cv if c in near]; tk_near = [c for c in topk if c in near]
        rescued = [c for c in cv_near if c not in topk]  # クラスタで拾った near-best
        rg = REG.get((k, seed), {})
        best_c = max(e, key=lambda c: q(c))
        rows.append(dict(
            game=k[0], end=k[1], shot=k[2], seed=seed, category=cat[k][0], stratum=cat[k][1],
            phase_r=feat[k]["phase_r"] if k in feat else "", N=len(e), K_clusters=len(csize),
            largest_frac=max(csize.values()) / len(e), q_star=qs,
            cv_qmax=cv_q, tk_qmax=tk_q, cv_screen_loss=qs - cv_q, tk_screen_loss=qs - tk_q,
            tk_n_clusters=tk_ncl, tk_dup=int(tk_ncl < len(topk)),
            cv_types=len({typ(lab[c]) for c in cv}), tk_types=len({typ(lab[c]) for c in topk}),
            cv_has_near=int(bool(cv_near)), tk_has_near=int(bool(tk_near)),
            rescue=int(bool(cv_near) and not tk_near), lost=int(bool(tk_near) and not cv_near),
            rescued_labels="|".join(lab[c] for c in rescued),
            rescued_erank="|".join(str(erank[c]) for c in rescued),
            rescued_csize="|".join(str(csize[c2cl[c]]) for c in rescued),
            best_label=lab[best_c], best_erank=erank[best_c], best_csize=csize[c2cl[best_c]],
            tk_labels="|".join(lab[c] for c in topk), cv_labels="|".join(lab[c] for c in cv),
            reg_A9P5N=rg.get("A9P5N", ""), reg_A8N=rg.get("A8N", ""), reg_A1N=rg.get("A1N", ""), reg_A2N=rg.get("A2N", ""),
        ))

os.makedirs(OUT, exist_ok=True)
with open(OUT + "/screen_compare_rows.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def summ(sub, name):
    n = len(sub)
    m = lambda f: sum(r[f] for r in sub) / n
    regd = [r["reg_A9P5N"] - r["reg_A8N"] for r in sub if r["reg_A9P5N"] != "" and r["reg_A8N"] != ""]
    print(f"{name:22s} n={n:4d} tk_dup={m('tk_dup'):.2f} tk_ncl={m('tk_n_clusters'):.2f} "
          f"types cv/tk={m('cv_types'):.2f}/{m('tk_types'):.2f} near cv/tk={m('cv_has_near'):.2f}/{m('tk_has_near'):.2f} "
          f"rescue={m('rescue'):.3f} lost={m('lost'):.3f} SL cv/tk={m('cv_screen_loss'):.3f}/{m('tk_screen_loss'):.3f} "
          f"reg(A9P5N-A8N)={st.mean(regd) if regd else float('nan'):+.3f}")

print("== 全体 (局面 x seed)")
summ(rows, "all")
print("== 層別 (stratum)")
for s in sorted({r["stratum"] for r in rows}): summ([r for r in rows if r["stratum"] == s], s)
print("== phase_r")
for s in sorted({r["phase_r"] for r in rows}): summ([r for r in rows if r["phase_r"] == s], s)
