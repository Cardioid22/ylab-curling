# -*- coding: utf-8 -*-
"""視覚的評価とボロノイ節のための2つの分析 (最終ショットを除く 468 局面, 既存データのみ)。

(1) 別の道 (paths): 「A で今すぐ有利になる」手と、「B でも後の展開で同じように有利になる」手が
    別のクラスタに分かれているか。審判値 q が最良から tau 以内のクラスタ (= 最良級のクラスタ) を集め、
    各クラスタの最良手について「今の得点」(今エンドが終わったら何点か) を結果盤面から数える。
    A = 最良級クラスタのうち今の得点が最大のもの、B = 今の得点が A より 2 点以上低いのに q が最良級のもの。
    提案 (ClusterValue+N) の 4 手が A と B の両方を含むか、前回の手法 (medoid) の代表手が最良級か、も数える。
(2) 行動空間 vs 結果盤面空間 (spaces): VOOT のように行動のパラメータ (vx, vy, rot) の近さで分けた場合と、
    結果盤面の近さ (distDelta) で分けた場合とで、価値 q がどれだけ「まとまる」か。
    - 最近傍: 各候補に最も近い候補との |Δq| (行動空間の最近傍 / 結果盤面空間の最近傍)
    - 分割: 同じ数 K のセルに分けたときの η² = 1 − (セル内平方和 / 全平方和)。
      結果盤面クラスタ (提案と同じ) / 行動空間の k-means のボロノイセル / 行動空間でランダムな母点のボロノイセル (VOO 型) /
      同じ大きさのランダム分割
入力: gpw_experiment/A9P5N/seed_*/cluster_table.csv, referee Q, cluster_vs_topk/export*/ (--export-case の出力)
Usage: python scripts/analyze_paths_and_spaces.py gpw_experiment/cluster_vs_topk
出力: <dir>/paths_rows.csv, <dir>/space_rows.csv, <dir>/paths_and_spaces_summary.txt
"""
import csv, glob, math, os, sys, statistics as st
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GE = ROOT + "/gpw_experiment"
D_OUT = sys.argv[1]
TAU = 0.25
SEEDS = range(42, 47)
HX, HY, R12, SR = 0.0, 38.405, 1.829, 0.145
RNG = np.random.default_rng(0)
N_RAND = 20

def key(r): return (int(r["game_id"]), int(r["end"]), int(r["shot_num"]))

rows = list(csv.DictReader(open(D_OUT + "/screen_compare_rows.csv", encoding="utf-8")))
meta = {(int(r["game"]), int(r["end"]), int(r["shot"])): r for r in rows if r["seed"] == "42"}
nf = {k for k, r in meta.items() if r["phase_r"] != "r1"}

def export_dir(k):
    for sub in ("export", "export_rest"):
        d = f"{D_OUT}/{sub}/export_g{k[0]}_e{k[1]}_s{k[2]}"
        if os.path.exists(d + "/result_stones.csv"): return d
    return None
pos = sorted(k for k in nf if export_dir(k))
if len(pos) < len(nf): print(f"warning: export が無い局面 {len(nf) - len(pos)} 個は飛ばす")
posset = set(pos)

Q = {}
for f in glob.glob(GE + "/referee/score_move_qtable_idx*.csv"):
    for r in csv.DictReader(open(f, encoding="utf-8")):
        if key(r) in posset: Q[(key(r), int(r["candidate_idx"]))] = float(r["q_ref_mean"])
CT = defaultdict(dict)
for seed in SEEDS:
    tmp = defaultdict(list)
    for r in csv.DictReader(open(f"{GE}/A9P5N/seed_{seed}/cluster_table.csv", encoding="utf-8")):
        if key(r) in posset: tmp[key(r)].append(r)
    for k, v in tmp.items(): CT[k][seed] = v
BATCH = {}
for r in csv.DictReader(open(ROOT + "/test_positions500/batch_0001.csv", encoding="utf-8")):
    BATCH[(int(r["match_id"]), int(r["end"]), int(r["shot_num"]))] = r

def score_of(stones, me):
    """今エンドが終わったときの得点 (手番側から見て +/-)"""
    d = sorted((math.hypot(x - HX, y - HY), t) for (t, i), (x, y) in stones.items() if math.hypot(x - HX, y - HY) <= R12 + SR)
    if not d: return 0
    n = 0
    for _, t in d:
        if t != d[0][1]: break
        n += 1
    return n if d[0][1] == me else -n

def eta2(q, labels):
    q = np.asarray(q); tot = ((q - q.mean()) ** 2).sum()
    if tot <= 1e-12: return float("nan")
    w = sum(((q[labels == c] - q[labels == c].mean()) ** 2).sum() for c in np.unique(labels))
    return 1.0 - w / tot

def voronoi(X, sites):
    return np.argmin(((X[:, None, :] - X[None, sites, :]) ** 2).sum(-1), axis=1)

path_rows, space_rows = [], []
for k in pos:
    d = export_dir(k)
    me = int(BATCH[k]["team"])
    inp = {(int(r["team"]), int(r["stone_idx"])): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(d + "/input_board.csv", encoding="utf-8"))}
    res = defaultdict(dict)
    for r in csv.DictReader(open(d + "/result_stones.csv", encoding="utf-8")):
        res[int(r["candidate_idx"])][(int(r["team"]), int(r["stone_idx"]))] = (float(r["x"]), float(r["y"]))
    cand = {int(r["candidate_idx"]): r for r in csv.DictReader(open(d + "/candidates.csv", encoding="utf-8"))}
    Dm = np.array([[float(x) for x in line.strip().rstrip(",").split(",")] for line in open(d + "/distance_matrix.csv", encoding="utf-8") if line.strip()])
    np.fill_diagonal(Dm, 0.0)
    ct42 = CT[k][42]
    idx = [int(r["candidate_idx"]) for r in ct42]
    N = len(idx)
    assert Dm.shape == (N, N) and sorted(idx) == list(range(N)), (k, Dm.shape, N)
    cl = np.array([0] * N); lab = {}
    for r in ct42:
        cl[int(r["candidate_idx"])] = int(r["cluster_id"]); lab[int(r["candidate_idx"])] = r["label"]
    q = np.array([Q[(k, i)] for i in range(N)])
    qs = q.max()
    imm = np.array([score_of(res[i], me) for i in range(N)])
    imm0 = score_of(inp, me)

    # ---- (1) 別の道 ----
    clusters = defaultdict(list)
    for i in range(N): clusters[cl[i]].append(i)
    def medoid(m): return min(m, key=lambda i: Dm[i, m].sum())
    nb = []   # 最良級クラスタ: (cluster, 最良手, q, 今の得点, medoid が最良級か)
    for c, m in clusters.items():
        b = max(m, key=lambda i: q[i])
        if q[b] >= qs - TAU: nb.append((c, b, q[b], imm[b], q[medoid(m)] >= qs - TAU))
    A = max(nb, key=lambda t: (t[3], t[2]))
    Bs = [t for t in nb if t[0] != A[0] and t[3] <= A[3] - 2]
    B = max(Bs, key=lambda t: t[2]) if Bs else None
    show_both = []; rankA = []; rankB = []
    for seed in SEEDS:
        cv = {}
        for r in CT[k][seed]: cv[cl[int(r["candidate_idx"])]] = float(r["cluster_value"])
        order = sorted(cv, key=lambda c: -cv[c])
        rankA.append(order.index(A[0]) + 1)
        if B: rankB.append(order.index(B[0]) + 1)
        reps_cl = {cl[int(r["candidate_idx"])] for r in CT[k][seed] if r["is_representative"] == "1"}
        reps_nb = [int(r["candidate_idx"]) for r in CT[k][seed] if r["is_representative"] == "1" and q[int(r["candidate_idx"])] >= qs - TAU]
        show_both.append(int(B is not None and any(cl[i] == A[0] for i in reps_nb) and any(cl[i] == B[0] for i in reps_nb)))
    path_rows.append(dict(
        game=k[0], end=k[1], shot=k[2], stratum=meta[k]["stratum"], phase_r=meta[k]["phase_r"], N=N, K=len(clusters),
        q_star=round(qs, 3), score_now=imm0, n_nb_clusters=len(nb),
        n_nb_medoid_ok=sum(t[4] for t in nb),
        A_idx=A[1], A_cluster=A[0], A_label=lab[A[1]], A_q=round(A[2], 3), A_imm=A[3], A_csize=len(clusters[A[0]]),
        B_idx=B[1] if B else "", B_cluster=B[0] if B else "", B_label=lab[B[1]] if B else "", B_q=round(B[2], 3) if B else "", B_imm=B[3] if B else "",
        B_csize=len(clusters[B[0]]) if B else "", has_B=int(B is not None),
        proposal_shows_A_and_B_of5=sum(show_both),
        A_value_rank_med=st.median(rankA), B_value_rank_med=st.median(rankB) if rankB else "",
    ))

    # ---- (2) 行動空間 vs 結果盤面空間 ----
    vx = np.array([float(cand[i]["vx"]) for i in range(N)]); vy = np.array([float(cand[i]["vy"]) for i in range(N)])
    rot = np.array([float(cand[i]["rot"]) for i in range(N)])
    z = lambda a: (a - a.mean()) / (a.std() if a.std() > 1e-9 else 1.0)
    X = np.stack([z(vx), z(vy), rot], axis=1)          # 回転の違い = 標準偏差 1 つ分
    Da = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1); np.fill_diagonal(Da, np.inf)
    Dr = Dm.copy(); np.fill_diagonal(Dr, np.inf)
    nn_a = Da.argmin(1); nn_r = Dr.argmin(1)
    K = len(clusters)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X).labels_ if K < N else np.arange(N)
    # 頑健な尺度 (四分位範囲) でも試す: 強いヒットの大きな vy に標準偏差が引っ張られ、ドローの強さの違いが潰れるのを避ける
    iqr = lambda a: max(np.subtract(*np.percentile(a, [75, 25])), 1e-3)
    X2 = np.stack([vx / iqr(vx), vy / iqr(vy), rot], axis=1)
    km2 = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X2).labels_ if K < N else np.arange(N)
    e_vor = st.mean(eta2(q, voronoi(X, RNG.choice(N, K, replace=False))) for _ in range(N_RAND))
    sizes = [len(m) for m in clusters.values()]
    def rand_part():
        perm = RNG.permutation(N); lab_ = np.empty(N, int); p = 0
        for c, s in enumerate(sizes): lab_[perm[p:p + s]] = c; p += s
        return lab_
    e_rand = st.mean(eta2(q, rand_part()) for _ in range(N_RAND))
    space_rows.append(dict(
        game=k[0], end=k[1], shot=k[2], stratum=meta[k]["stratum"], N=N, K=K,
        nn_action_dq=round(float(np.median(np.abs(q - q[nn_a]))), 4),
        nn_result_dq=round(float(np.median(np.abs(q - q[nn_r]))), 4),
        nn_action_bigjump=round(float(np.mean(np.abs(q - q[nn_a]) > 0.5)), 4),
        nn_result_bigjump=round(float(np.mean(np.abs(q - q[nn_r]) > 0.5)), 4),
        nn_action_diff_cluster=round(float(np.mean(cl != cl[nn_a])), 4),
        eta2_result=round(eta2(q, cl), 4), eta2_action_kmeans=round(eta2(q, km), 4), eta2_action_kmeans_iqr=round(eta2(q, km2), 4),
        eta2_action_voronoi=round(e_vor, 4), eta2_random=round(e_rand, 4),
    ))

for name, rr in (("paths_rows.csv", path_rows), ("space_rows.csv", space_rows)):
    with open(f"{D_OUT}/{name}", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rr[0].keys())); w.writeheader(); w.writerows(rr)

out = []
P = lambda s: (print(s), out.append(s))
n = len(path_rows)
P(f"局面数 {n} (最終ショット除く, export のある局面)。tau={TAU}")
P("== (1) 別の道")
P(f"最良級クラスタが 2 個以上: {sum(r['n_nb_clusters'] >= 2 for r in path_rows)}/{n}")
P(f"「今すぐ有利な A」と「今の得点は 2 点以上低いが最良級の B」が別クラスタにある: {sum(r['has_B'] for r in path_rows)}/{n}")
hb = [r for r in path_rows if r["has_B"]]
if hb:
    P(f"  そのうち提案の 4 手が A と B の両方を含む (5 seed 平均): {st.mean(r['proposal_shows_A_and_B_of5'] for r in hb) / 5:.2f}")
    P(f"  クラスタ価値の順位 (中央値の中央値): A {st.median(r['A_value_rank_med'] for r in hb)}, B {st.median(r['B_value_rank_med'] for r in hb)} "
      f"(クラスタ数の中央値 {st.median(r['K'] for r in hb)}); B が価値の上位 4 に入る局面 {sum(r['B_value_rank_med'] <= 4 for r in hb)}/{len(hb)}, 上位 8 {sum(r['B_value_rank_med'] <= 8 for r in hb)}/{len(hb)}")
    P(f"  B の種類: " + str(sorted(((t, sum(1 for r in hb if r['B_label'].split('(')[0] == t)) for t in {r['B_label'].split('(')[0] for r in hb}), key=lambda x: -x[1])))
nbm = [(r["n_nb_medoid_ok"], r["n_nb_clusters"]) for r in path_rows]
P(f"最良級クラスタの medoid (前回の手法の代表手) も最良級である率: {sum(a for a, b in nbm) / sum(b for a, b in nbm):.2f}")
P("== (2) 行動空間 vs 結果盤面空間 (局面ごとの値の中央値)")
for f_ in ("nn_action_dq", "nn_result_dq", "nn_action_bigjump", "nn_result_bigjump", "nn_action_diff_cluster",
           "eta2_result", "eta2_action_kmeans", "eta2_action_kmeans_iqr", "eta2_action_voronoi", "eta2_random"):
    P(f"  {f_:24s} {st.median(r[f_] for r in space_rows if not math.isnan(r[f_])):.3f}")
try:
    from scipy.stats import wilcoxon
    for a, b in (("nn_action_dq", "nn_result_dq"), ("eta2_result", "eta2_action_kmeans"), ("eta2_result", "eta2_action_kmeans_iqr"), ("eta2_result", "eta2_action_voronoi"), ("eta2_result", "eta2_random")):
        pr = [(r[a], r[b]) for r in space_rows if not (math.isnan(r[a]) or math.isnan(r[b]))]
        P(f"  {a} vs {b}: {sum(x > y for x, y in pr)} 勝 {sum(x < y for x, y in pr)} 敗, Wilcoxon p={wilcoxon([x for x, _ in pr], [y for _, y in pr]).pvalue:.1e}")
except ImportError:
    pass
open(f"{D_OUT}/paths_and_spaces_summary.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
