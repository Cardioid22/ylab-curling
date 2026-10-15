# -*- coding: utf-8 -*-
"""GPW2026 本論文 (5章・6.1節) に載せる数値を、載せる手法だけで計算し直す。

5章: 決定的評価 4 手法 (A1, A2, A5, A9P5) と外乱込み評価 5 手法 (A1N, A2N, A5N, A5N4, A9P5N) の
     regret (局面ごとに 5 seed 平均 → 500 局面で平均)。9 手法で Friedman 検定 → 全 36 対の Wilcoxon + Holm 補正。
     クラスタなし (A8N) / ベイズ TS (A12N) / PW (A11) は論文に載せないので、検定の対象からも外す。
6.1: A9P5N の cluster_table (seed 42; クラスタは外乱なしの結果盤面で決まるので seed によらない) から、
     最終ショットを除く局面でのクラスタの大きさと、クラスタ内の手の種類・回転の混在。
Usage: python scripts/gpw_paper_numbers.py   -> gpw_experiment/paper_numbers/numbers.txt
"""
import csv, glob, itertools, os, re, statistics as st
from collections import defaultdict, Counter
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GE = ROOT + "/gpw_experiment"
OUT = GE + "/paper_numbers"
ARMS = ["A1", "A2", "A5", "A9P5", "A1N", "A2N", "A5N", "A5N4", "A9P5N"]
SRC = {a: GE + "/regret500/joined_500full11.csv" for a in ("A1", "A2", "A5", "A9P5")}
SRC.update({a: GE + "/regret500/joined_noisy7.csv" for a in ("A1N", "A2N", "A5N", "A5N4", "A9P5N")})

def key(r): return (int(r["game_id"]), int(r["end"]), int(r["shot_num"]))

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, m, run = [0.0] * len(ps), len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj

def main():
    os.makedirs(OUT, exist_ok=True)
    L = []
    # ---- 5章: regret ----
    per = {a: defaultdict(list) for a in ARMS}
    for a in ARMS:
        for r in csv.DictReader(open(SRC[a], encoding="utf-8")):
            if r["arm"] == a: per[a][key(r)].append(float(r["regret"]))
    pos = sorted(set.intersection(*(set(per[a]) for a in ARMS)))
    assert all(len(per[a][k]) == 5 for a in ARMS for k in pos)
    M = {a: np.array([st.mean(per[a][k]) for k in pos]) for a in ARMS}
    L.append(f"== 5章: regret (局面 {len(pos)} × 5 seed, P=200, 局面ごとに seed 平均)")
    for a in ARMS: L.append(f"  {a:6s} {M[a].mean():.3f}")
    fr = stats.friedmanchisquare(*[M[a] for a in ARMS])
    L.append(f"  Friedman ({len(ARMS)} 手法): chi2={fr.statistic:.1f} p={fr.pvalue:.1e}")
    pairs = list(itertools.combinations(ARMS, 2))
    raw = [stats.wilcoxon(M[a], M[b]).pvalue for a, b in pairs]
    adj = holm(raw)
    L.append("  全 36 対の Wilcoxon (Holm 補正)。勝敗 = 左の手法の regret が小さい局面数 - 大きい局面数")
    for (a, b), p, q in sorted(zip(pairs, raw, adj), key=lambda t: t[2]):
        d = M[a] - M[b]
        L.append(f"  {a:6s} vs {b:6s}  diff={d.mean():+.3f}  win={int((d < 0).sum())}-{int((d > 0).sum())}  p_raw={p:.1e}  p_holm={q:.1e}")
    # ---- 6.1: クラスタの性質 ----
    rows = defaultdict(list)
    for r in csv.DictReader(open(GE + "/A9P5N/seed_42/cluster_table.csv", encoding="utf-8")):
        if int(r["shot_num"]) != 15: rows[key(r)].append(r)
    Ns, Ks, maxshare, single, nclust = [], [], [], 0, 0
    multi, mixtype, mixrot = 0, 0, 0
    for k, rs in rows.items():
        cl = defaultdict(list)
        for r in rs: cl[int(r["cluster_id"])].append(r)
        Ns.append(len(rs)); Ks.append(len(cl))
        maxshare.append(max(len(v) for v in cl.values()) / len(rs))
        for v in cl.values():
            nclust += 1
            if len(v) == 1: single += 1; continue
            multi += 1
            types = {r["label"].split("(")[0] for r in v}
            rots = {m.group(1) for r in v for m in [re.search(r"\((CW|CCW)", r["label"])] if m}
            mixtype += len(types) >= 2; mixrot += len(rots) >= 2
    L.append(f"\n== 6.1: クラスタの性質 (最終ショットを除く {len(rows)} 局面, A9P5N seed 42 の cluster_table)")
    L.append(f"  候補数 N: 中央値 {st.median(Ns):.0f}, 平均 {st.mean(Ns):.1f}   クラスタ数 K: 中央値 {st.median(Ks):.0f}, 平均 {st.mean(Ks):.1f}")
    L.append(f"  最大クラスタが候補に占める割合: 中央値 {st.median(maxshare):.2f}, 平均 {st.mean(maxshare):.2f}")
    L.append(f"  1 手だけのクラスタ: {single}/{nclust} = {single / nclust:.2f}")
    L.append(f"  2 手以上のクラスタ {multi} 個のうち, 種類 (Hit/Draw/...) の違う手を含む {mixtype / multi:.2f}, 回転の違う手を含む {mixrot / multi:.2f}")
    # ---- 6.3: 提案の 4 手に最良手 (審判) が残る率。最良手のクラスタが小さい (1〜2 手) 局面とそれ以外 ----
    Q = defaultdict(dict)
    for f in glob.glob(GE + "/referee/score_move_qtable_idx*.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8")): Q[key(r)][int(r["candidate_idx"])] = float(r["q_ref_mean"])
    small = {(int(r["game"]), int(r["end"]), int(r["shot"]))
             for r in csv.DictReader(open(GE + "/cluster_vs_topk/catalog/index.csv", encoding="utf-8"))}
    kept = defaultdict(lambda: [0, 0])
    for seed in range(42, 47):
        reps, keys = defaultdict(set), set()
        for r in csv.DictReader(open(f"{GE}/A9P5N/seed_{seed}/cluster_table.csv", encoding="utf-8")):
            if int(r["shot_num"]) == 15: continue
            keys.add(key(r))
            if r["is_representative"] == "1": reps[key(r)].add(int(r["candidate_idx"]))
        for k in keys:
            best = max(Q[k], key=lambda c: (Q[k][c], -c))
            for g in ("small" if k in small else "rest", "all"):
                kept[g][0] += best in reps[k]; kept[g][1] += 1
    L.append("\n== 6.3: 提案 (A9P5N) の 4 手に最良手が残る率 (局面 × 5 seed)")
    for g in ("small", "rest", "all"):
        L.append(f"  {g:5s} {kept[g][0]}/{kept[g][1]} = {kept[g][0] / kept[g][1]:.3f}")
    open(OUT + "/numbers.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
