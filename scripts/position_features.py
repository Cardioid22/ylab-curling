#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""局面ごとの特徴量テーブル (GPW2026 本論文: 「どんな局面でクラスタリングが有効か」)
================================================================================
1 局面 = 1 行で、以下を横に並べた position_features.csv を作る。

  A. 盤面特徴   (batch_*.csv)            : 石数・ハウス内/ガード・No.1・密集度・ハンマー・進行度
  B. 候補集合特徴 (審判 Q テーブル)       : 候補数・q*・多峰性 (q* 近傍の候補数/戦術種別数)・リスク水準
  C. クラスタ構造 (ClusterValue の cluster_table) : クラスタ数・巨大クラスタ率・singleton 率・
                                            η²(真値 q のクラスタ間分散比)・妥当性 ρ・被覆 (screen loss)
                                            + リスク統合クラスタ価値 (μ_c − λσ_c) の妥当性
  D. 成果       (aggregate_reinvest.py の reinvest_joined.csv) : アーム別 regret とその差分

C は seed ごとに計算して平均 (クラスタ構造自体は決定的なので seed 不変、妥当性/被覆は seed 依存)。
cluster_sd / cluster_value_risk 列が無い旧 cluster_table でも e_score/e_sd から同じ式で再計算する。

Usage:
  python scripts/position_features.py \
      --positions-dir test_positions50 \
      --referee-csv reinvest_experiment/scorescreen/run50/referee/score_move_qtable.csv \
      --cluster-dir reinvest_experiment/run50v2/A9 \
      --medoid-dir  reinvest_experiment/run50v2/A2 \
      --joined reinvest_experiment/run50v2/regret/reinvest_joined.csv \
      --arms A1,A2,A9 --risk-lambda 0.5 \
      --out reinvest_experiment/run50v2/features
--referee-csv はディレクトリでもよい (score_move_qtable*.csv を全部読む)。
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HC_X, HC_Y, HR, SR = 0.0, 38.405, 1.829, 0.145
HOG_Y = HC_Y - 6.401      # ホッグライン (ティーの 6.401m 手前)
FRONT_Y = HC_Y - HR       # ハウス前端
KEY = ["game_id", "end", "shot_num"]

HIT_TYPES = {"Hit", "Peel", "Takeout", "Double"}
DRAW_TYPES = {"Draw", "Freeze", "ComeAround", "DrawRaise"}
GUARD_TYPES = {"PreGuard", "PostGuard"}


def label_type(label: str) -> str:
    p = label.find("(")
    return label if p < 0 else label[:p]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def spearman(a, b, min_n):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < min_n or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    r = stats.spearmanr(a, b).statistic
    return float(r) if not np.isnan(r) else float("nan")


# ============================================================================
# A. 盤面特徴
# ============================================================================
def board_features(row: dict) -> dict:
    team = int(row["team"]); opp = 1 - team

    def stones(t):
        out = []
        for i in range(8):
            if int(row[f"t{t}s{i}_inplay"]) == 1:
                out.append((float(row[f"t{t}s{i}_x"]), float(row[f"t{t}s{i}_y"])))
        return out

    my, op = stones(team), stones(opp)
    allst = [(x, y, 1) for x, y in my] + [(x, y, 0) for x, y in op]
    shot = int(row["shot_num"]); r = 16 - shot

    d = lambda x, y: math.hypot(x - HC_X, y - HC_Y)
    inh = lambda x, y: d(x, y) <= HR + SR
    ingz = lambda x, y: (not inh(x, y)) and (HOG_Y <= y < FRONT_Y)

    f = dict(
        game_id=int(row["match_id"]), end=int(row["end"]), shot_num=shot, team=team,
        remaining=r,
        has_hammer=int(shot % 2 == 1),           # 奇数ショットを投げる側が最終ショット (ハンマー) を持つ
        phase_shot=("early" if shot <= 5 else "mid" if shot <= 11 else "late"),
        phase_r=("r1" if r <= 1 else "late_r2-4" if r <= 4 else "mid_r5-8" if r <= 8 else "early_r9+"),
    )
    n = len(allst)
    f["n_stones"] = n
    f["n_stones_bin"] = ("0-1" if n <= 1 else "2-4" if n <= 4 else "5-8" if n <= 8 else "9-12" if n <= 12 else "13-16")
    f["n_my"] = len(my); f["n_op"] = len(op)
    f["n_my_house"] = sum(inh(x, y) for x, y in my)
    f["n_op_house"] = sum(inh(x, y) for x, y in op)
    f["n_house"] = f["n_my_house"] + f["n_op_house"]
    f["house_diff"] = f["n_my_house"] - f["n_op_house"]
    f["n_my_guard"] = sum(ingz(x, y) for x, y in my)
    f["n_op_guard"] = sum(ingz(x, y) for x, y in op)
    f["n_guard"] = f["n_my_guard"] + f["n_op_guard"]
    f["n_center_guard"] = sum(1 for x, y, _ in allst if ingz(x, y) and abs(x) < 0.5)
    f["center_lane_blocked"] = int(any(abs(x) < 0.3 and HOG_Y <= y < HC_Y for x, y, _ in allst))
    f["n_front_house"] = sum(1 for x, y, _ in allst if inh(x, y) and y < HC_Y)
    f["n_back_house"] = sum(1 for x, y, _ in allst if inh(x, y) and y >= HC_Y)
    f["n_within_0.61"] = sum(1 for x, y, _ in allst if d(x, y) <= 0.61)   # 4ft 円
    f["n_within_1.22"] = sum(1 for x, y, _ in allst if d(x, y) <= 1.22)   # 8ft 円

    # No.1 / No.2 と現在の仮得点 (ハウス内のみ対象)
    hs = sorted([(d(x, y), m) for x, y, m in allst if inh(x, y)])
    if hs:
        f["d_no1"] = hs[0][0]; f["no1_mine"] = hs[0][1]
        f["d_no2"] = hs[1][0] if len(hs) > 1 else float("nan")
        f["no1_no2_gap"] = (hs[1][0] - hs[0][0]) if len(hs) > 1 else float("nan")
        owner = hs[0][1]; cnt = 0
        for _, m in hs:
            if m == owner:
                cnt += 1
            else:
                break
        f["counting_my"] = cnt if owner == 1 else 0
        f["counting_op"] = cnt if owner == 0 else 0
        f["score_now"] = cnt if owner == 1 else -cnt
    else:
        f["d_no1"] = float("nan"); f["no1_mine"] = -1
        f["d_no2"] = float("nan"); f["no1_no2_gap"] = float("nan")
        f["counting_my"] = 0; f["counting_op"] = 0; f["score_now"] = 0

    # 密集度
    pts = np.array([(x, y) for x, y, _ in allst])
    if n >= 2:
        D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        f["min_pair_dist"] = float(D.min())
        f["mean_nn_dist"] = float(D.min(axis=1).mean())
        f["x_spread"] = float(pts[:, 0].std()); f["y_spread"] = float(pts[:, 1].std())
    else:
        f["min_pair_dist"] = float("nan"); f["mean_nn_dist"] = float("nan")
        f["x_spread"] = float("nan"); f["y_spread"] = float("nan")
    hp = np.array([(x, y) for x, y, _ in allst if inh(x, y)])
    if len(hp) >= 2:
        D = np.sqrt(((hp[:, None, :] - hp[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        f["house_mean_nn_dist"] = float(D.min(axis=1).mean())
    else:
        f["house_mean_nn_dist"] = float("nan")
    f["house_density"] = f["n_house"] / (math.pi * (HR + SR) ** 2)
    return f


# ============================================================================
# B. 候補集合特徴 (審判)
# ============================================================================
def hist_stats(hist: str):
    """'s:c;s:c;...' -> (entropy_bits, p_negative, p_ge2)"""
    tot = 0; items = []
    for tok in (hist or "").split(";"):
        if ":" not in tok:
            continue
        s, c = tok.split(":"); s = int(s); c = int(c)
        items.append((s, c)); tot += c
    if tot == 0:
        return float("nan"), float("nan"), float("nan")
    p = np.array([c / tot for _, c in items])
    ent = float(-(p * np.log2(p)).sum())
    p_neg = sum(c for s, c in items if s < 0) / tot
    p_ge2 = sum(c for s, c in items if s >= 2) / tot
    return ent, p_neg, p_ge2


def referee_features(rows: list, lam: float) -> dict:
    q = np.array([fnum(r["q_ref_mean"]) for r in rows])
    s = np.array([fnum(r.get("q_ref_sd", "nan")) for r in rows])
    types = [label_type(r["label"]) for r in rows]
    n = len(q)
    f = {"n_cand": n}
    if n == 0:
        return f
    o = np.argsort(-q)
    qs = q[o]
    f["q_star"] = float(qs[0]); f["q_2nd"] = float(qs[1]) if n > 1 else float("nan")
    f["q_gap12"] = float(qs[0] - qs[1]) if n > 1 else float("nan")
    f["q_median"] = float(np.median(q)); f["q_mean"] = float(q.mean()); f["q_min"] = float(q.min())
    f["q_range"] = float(q.max() - q.min())
    f["q_iqr"] = float(np.percentile(q, 75) - np.percentile(q, 25))
    f["q_star_minus_median"] = f["q_star"] - f["q_median"]
    f["n_near_best_025"] = int((q >= qs[0] - 0.25).sum())
    f["n_near_best_050"] = int((q >= qs[0] - 0.5).sum())
    f["frac_near_best_050"] = f["n_near_best_050"] / n
    f["q_sd_mean"] = float(np.nanmean(s)); f["q_sd_median"] = float(np.nanmedian(s))
    f["q_sd_best"] = float(s[o[0]])
    f["best_type"] = types[o[0]]
    tset = sorted(set(types)); f["n_types"] = len(tset)
    f["frac_hit"] = sum(t in HIT_TYPES for t in types) / n
    f["frac_draw"] = sum(t in DRAW_TYPES for t in types) / n
    f["frac_guard"] = sum(t in GUARD_TYPES for t in types) / n
    tb = {}
    for t, qq in zip(types, q):
        tb[t] = max(tb.get(t, -1e9), qq)
    tbv = sorted(tb.values(), reverse=True)
    f["best_type_margin"] = float(tbv[0] - tbv[1]) if len(tbv) > 1 else float("nan")
    f["n_types_near_best_050"] = int(sum(v >= qs[0] - 0.5 for v in tbv))
    f["best_type_is_hit"] = int(types[o[0]] in HIT_TYPES)
    # 先読み価値 (審判が q_immediate_mean を出している場合のみ)
    if "q_immediate_mean" in rows[0] and rows[0]["q_immediate_mean"] not in ("", None):
        qi = np.array([fnum(r["q_immediate_mean"]) for r in rows])
        f["lookahead_gain_best"] = float(q[o[0]] - qi[o[0]])
        f["lookahead_gain_mean"] = float(np.nanmean(q - qi))
    ent, pneg, pge2 = hist_stats(rows[o[0]].get("score_hist", ""))
    f["best_hist_entropy"] = ent; f["best_p_negative"] = pneg; f["best_p_ge2"] = pge2
    # リスク調整の真値 (λ 指定時)
    if lam > 0 and not np.all(np.isnan(s)):
        qa = q - lam * s
        oa = np.argsort(-qa)
        f[f"qadj_star_l{lam:g}"] = float(qa[oa[0]])
        f[f"qadj_best_same_as_q_best_l{lam:g}"] = int(oa[0] == o[0])
        f[f"qadj_best_type_l{lam:g}"] = types[oa[0]]
    return f


# ============================================================================
# C. クラスタ構造 (cluster_table 1 seed 分)
# ============================================================================
def cluster_features(rows: list, qmap: dict, lam: float, with_value: bool) -> dict:
    clusters = defaultdict(list); reps = set()
    e, sd, cval_col = {}, {}, {}
    for r in rows:
        ci = int(r["candidate_idx"]); cid = int(r["cluster_id"])
        clusters[cid].append(ci)
        if str(r.get("is_representative")) in ("1", "True", "true"):
            reps.add(ci)
        if with_value:
            e[ci] = fnum(r.get("e_score")); sd[ci] = fnum(r.get("e_sd"))
            if r.get("cluster_value") not in ("", None):
                cval_col[cid] = fnum(r["cluster_value"])
    N = sum(len(v) for v in clusters.values()); K = len(clusters)
    f = {"n_cand_ct": N, "K_clusters": K, "n_reps": len(reps)}
    if N == 0 or K == 0:
        return f
    sizes = np.array([len(v) for v in clusters.values()], float)
    f["largest_frac"] = float(sizes.max() / N)
    f["n_singleton"] = int((sizes == 1).sum()); f["singleton_frac"] = f["n_singleton"] / K
    p = sizes / N
    f["size_entropy_norm"] = float(-(p * np.log(p)).sum() / math.log(K)) if K > 1 else float("nan")
    f["mean_cluster_size"] = float(N / K)
    # 戦術種別の純度 (サイズ加重)
    types = {int(r["candidate_idx"]): (r.get("shot_type") or label_type(r.get("label", ""))) for r in rows}
    pur = 0.0
    for mem in clusters.values():
        cnt = defaultdict(int)
        for m in mem:
            cnt[types[m]] += 1
        pur += max(cnt.values())
    f["type_purity"] = pur / N

    # 真値 q によるクラスタ間分散比 η² と クラスタ内 SD
    qc = {cid: [qmap[m][0] for m in mem if m in qmap] for cid, mem in clusters.items()}
    allq = [v for vs in qc.values() for v in vs]
    if len(allq) >= 3:
        gm = float(np.mean(allq)); sst = float(sum((v - gm) ** 2 for v in allq))
        ssb = float(sum(len(vs) * (np.mean(vs) - gm) ** 2 for vs in qc.values() if vs))
        f["eta2_q"] = ssb / sst if sst > 0 else float("nan")
        wsd = [(len(vs), float(np.std(vs))) for vs in qc.values() if len(vs) >= 2]
        f["within_sd_q"] = (sum(n_ * s_ for n_, s_ in wsd) / sum(n_ for n_, _ in wsd)) if wsd else float("nan")
        means = [np.mean(vs) for vs in qc.values() if vs]
        f["between_range_q"] = float(max(means) - min(means))

    # 被覆: 審判最良手は代表に残っているか / 代表集合の最良 q (= スクリーン後に到達可能な上限)
    best_cid = None
    if qmap:
        best_idx = max(qmap, key=lambda i: qmap[i][0]); q_star = qmap[best_idx][0]
        f["best_is_rep"] = int(best_idx in reps)
        best_cid = next((cid for cid, mem in clusters.items() if best_idx in mem), None)
        f["best_cluster_selected"] = int(any(m in reps for m in clusters[best_cid])) if best_cid is not None else 0
        f["best_cluster_size"] = len(clusters[best_cid]) if best_cid is not None else float("nan")
        repq = [qmap[m][0] for m in reps if m in qmap]
        f["rep_q_max"] = float(max(repq)) if repq else float("nan")
        f["screen_loss"] = float(q_star - max(repq)) if repq else float("nan")
        # 最良クラスタ (真値平均) が採用されたか
        true_means = {cid: np.mean(vs) for cid, vs in qc.items() if vs}
        if true_means:
            top_true = max(true_means, key=true_means.get)
            f["true_top_cluster_selected"] = int(any(m in reps for m in clusters[top_true]))

    if with_value and e:
        # 妥当性: 候補単位 (e_score vs q) / クラスタ単位 (cluster_value vs 真値平均)
        idx = [i for i in e if i in qmap and not np.isnan(e[i])]
        f["rho_cand"] = spearman([e[i] for i in idx], [qmap[i][0] for i in idx], 5)
        mu = {}; sig = {}
        for cid, mem in clusters.items():
            es = [e[m] for m in mem if m in e and not np.isnan(e[m])]
            if not es:
                continue
            m_ = float(np.mean(es)); mu[cid] = m_
            v = float(np.mean([sd[m] ** 2 + (e[m] - m_) ** 2 for m in mem if m in e and not np.isnan(e[m])]))
            sig[cid] = math.sqrt(max(0.0, v))
        cids = [cid for cid in mu if cid in qc and qc[cid]]
        est = [cval_col.get(cid, mu[cid]) for cid in cids]
        tru = [float(np.mean(qc[cid])) for cid in cids]
        f["rho_cluster"] = spearman(est, tru, 4)
        if not (np.isnan(f["rho_cluster"]) or np.isnan(f["rho_cand"])):
            f["rho_gain"] = f["rho_cluster"] - f["rho_cand"]
        else:
            f["rho_gain"] = float("nan")
        f["cluster_sd_mean"] = float(np.mean(list(sig.values()))) if sig else float("nan")
        # 最良手のクラスタは価値順で何位か
        if qmap and best_cid is not None and best_cid in mu:
            order = sorted(mu, key=lambda c: -mu[c])
            f["best_cluster_rank"] = order.index(best_cid) + 1
        if lam > 0:
            estr = [mu[cid] - lam * sig[cid] for cid in cids]
            trur = [float(np.mean([qmap[m][0] - lam * qmap[m][1] for m in clusters[cid] if m in qmap])) for cid in cids]
            f[f"rho_cluster_risk_l{lam:g}"] = spearman(estr, trur, 4)
            ea = [e[i] - lam * sd[i] for i in idx]; ta = [qmap[i][0] - lam * qmap[i][1] for i in idx]
            f[f"rho_cand_risk_l{lam:g}"] = spearman(ea, ta, 5)
            # λ を入れると採用クラスタ上位K (K=n_reps) がどれだけ入れ替わるか
            k = max(1, len(reps))
            top0 = set(sorted(cids, key=lambda c: -mu[c])[:k])
            top1 = set(sorted(cids, key=lambda c: -(mu[c] - lam * sig[c]))[:k])
            f[f"topk_overlap_l{lam:g}"] = len(top0 & top1) / k
    return f


# ============================================================================
def load_referee(path: Path) -> dict:
    files = [path] if path.is_file() else sorted(path.glob("score_move_qtable*.csv"))
    if not files:
        sys.exit(f"referee not found: {path}")
    by_pos = defaultdict(list)
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                by_pos[(int(r["game_id"]), int(r["end"]), int(r["shot_num"]))].append(r)
    return by_pos


def load_cluster_dir(d: Path) -> dict:
    """{(pos): {seed: [rows]}}"""
    out = defaultdict(lambda: defaultdict(list))
    for fp in sorted(d.glob("seed_*/cluster_table*.csv")):
        seed = fp.parent.name.replace("seed_", "")
        with open(fp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[(int(r["game_id"]), int(r["end"]), int(r["shot_num"]))][seed].append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions-dir", type=Path, required=True)
    ap.add_argument("--referee-csv", type=Path, required=True, help="score_move_qtable.csv またはそのディレクトリ")
    ap.add_argument("--cluster-dir", type=Path, default=None, help="ClusterValue 系アームの dir (seed_*/cluster_table.csv, e_score あり)")
    ap.add_argument("--medoid-dir", type=Path, default=None, help="Proposed(A2) の dir (medoid 代表の被覆を md_* 列に)")
    ap.add_argument("--joined", type=Path, default=None, help="aggregate_reinvest.py の reinvest_joined.csv")
    ap.add_argument("--arms", default="A1,A2,A9", help="joined から regret を取るアーム (カンマ区切り; 基準→提案 の順。差分 d_後_前 を作る)")
    ap.add_argument("--risk-lambda", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lam = args.risk_lambda

    # A
    board = []
    for fp in sorted(args.positions_dir.glob("batch_*.csv")):
        with open(fp, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                board.append(board_features(r))
    dfA = pd.DataFrame(board)
    print(f"[A] board features: {len(dfA)} positions, {dfA.shape[1]} cols", file=sys.stderr)

    # B
    ref = load_referee(args.referee_csv)
    dfB = pd.DataFrame([dict(zip(KEY, k), **referee_features(v, lam)) for k, v in ref.items()])
    print(f"[B] referee features: {len(dfB)} positions", file=sys.stderr)
    qmaps = {k: {int(r["candidate_idx"]): (fnum(r["q_ref_mean"]), fnum(r.get("q_ref_sd", "nan"))) for r in v}
             for k, v in ref.items()}

    # C
    frames = [dfA, dfB]
    if args.cluster_dir:
        ct = load_cluster_dir(args.cluster_dir)
        rows = []
        for k, seeds in ct.items():
            per = [cluster_features(v, qmaps.get(k, {}), lam, True) for v in seeds.values()]
            dfp = pd.DataFrame(per)
            agg = dfp.mean(numeric_only=True).to_dict()
            agg["n_seeds_ct"] = len(per)
            if "rho_gain" in dfp:
                agg["rho_gain_sd"] = float(dfp["rho_gain"].std()) if len(dfp) > 1 else float("nan")
            rows.append(dict(zip(KEY, k), **agg))
        dfC = pd.DataFrame(rows)
        print(f"[C] cluster features: {len(dfC)} positions from {args.cluster_dir}", file=sys.stderr)
        frames.append(dfC)
    if args.medoid_dir:
        ct = load_cluster_dir(args.medoid_dir)
        rows = []
        keep = ("screen_loss", "best_is_rep", "rep_q_max", "best_cluster_selected")
        for k, seeds in ct.items():
            per = [cluster_features(v, qmaps.get(k, {}), 0.0, False) for v in seeds.values()]
            dfp = pd.DataFrame(per).mean(numeric_only=True)
            rows.append(dict(zip(KEY, k), **{f"md_{c}": v for c, v in dfp.items() if c in keep}))
        dfM = pd.DataFrame(rows)
        print(f"[C'] medoid coverage: {len(dfM)} positions from {args.medoid_dir}", file=sys.stderr)
        frames.append(dfM)

    # D
    if args.joined:
        j = pd.read_csv(args.joined)
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        j = j[j["arm"].isin(arms)]
        metric_cols = [c for c in ("regret", "regret_adj") if c in j.columns]
        piv = j.groupby(KEY + ["arm"])[metric_cols].agg(["mean", "std", "size"])
        piv.columns = ["_".join(c) for c in piv.columns]
        piv = piv.reset_index()
        wide = None
        for a in arms:
            sub = piv[piv["arm"] == a].drop(columns=["arm"])
            ren = {}
            for m in metric_cols:
                ren[f"{m}_mean"] = f"{m}_{a}"; ren[f"{m}_std"] = f"{m}_{a}_sd"; ren[f"{m}_size"] = f"n_seeds_{a}"
            sub = sub.rename(columns=ren)
            # n_seeds は metric ごとに同じなので 1 つに畳む
            dup = [c for c in sub.columns if c == f"n_seeds_{a}"]
            if len(dup) > 1:
                sub = sub.loc[:, ~sub.columns.duplicated()]
            wide = sub if wide is None else wide.merge(sub, on=KEY, how="outer")
        if wide is not None:
            # 差分は「後ろのアーム − 前のアーム」(--arms を 基準→提案 の順に並べる想定; 例 A1,A2,A9 → d_A9_A1, d_A9_A2, d_A2_A1)
            for m in metric_cols:
                for i, a in enumerate(arms):
                    for b in arms[:i]:
                        ca, cb = f"{m}_{a}", f"{m}_{b}"
                        if ca in wide and cb in wide:
                            nm = f"d_{a}_{b}" if m == "regret" else f"dadj_{a}_{b}"
                            wide[nm] = wide[ca] - wide[cb]     # 正 = a の regret が大きい (= a が悪い)
            frames.append(wide)
            print(f"[D] outcomes: arms={arms}, {len(wide)} positions", file=sys.stderr)

    df = frames[0]
    for f_ in frames[1:]:
        df = df.merge(f_, on=KEY, how="left")
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "position_features.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}: {len(df)} rows x {df.shape[1]} cols", file=sys.stderr)

    # 列辞書
    dic = args.out / "feature_dictionary.md"
    with open(dic, "w", encoding="utf-8") as fh:
        fh.write(FEATURE_DICT)
        fh.write("\n\n## この出力に含まれる列\n\n")
        for c in df.columns:
            fh.write(f"- `{c}`\n")
    print(f"Wrote {dic}", file=sys.stderr)


FEATURE_DICT = """# position_features.csv 列の意味

## A. 盤面 (batch_*.csv 由来)
- `remaining` = 16 − shot_num (残り手数, 投げる手を含む)。`phase_shot` early(≤5)/mid(6-11)/late(≥12)。
  `phase_r` early_r9+/mid_r5-8/late_r2-4/r1 (cluster_switch 分析と同じ区分)。
- `has_hammer`: 奇数ショットを投げる側 = 最終ショットを持つ側。
- `n_*_house`: ハウス内 (ティーから 1.829+0.145m 以内)。`n_*_guard`: ハウス外でホッグ〜ハウス前端の間。
- `n_center_guard`: |x|<0.5 のガード。`center_lane_blocked`: |x|<0.3 かつティー手前に石 (センター経路が塞がれている)。
- `d_no1/no1_mine/d_no2/no1_no2_gap`: ハウス内 No.1/No.2 石 (no1_mine=1 なら自分の石, -1 はハウス空)。
- `counting_my/counting_op/score_now`: 今エンドが終わった場合の仮得点 (投げる側視点)。
- `n_within_0.61/1.22`: 4ft/8ft 円内の石数。`min_pair_dist/mean_nn_dist/house_mean_nn_dist`: 密集度。
- `x_spread/y_spread`: 石座標の標準偏差。`house_density`: ハウス内石数/面積。

## B. 候補集合 (審判 Q テーブル由来。q = 単一エンド純得点期待値 (K=200, 実行不確実性込み))
- `n_cand`: 候補数 N。`q_star`: 最良候補の q。`q_gap12`: 1位−2位。`q_star_minus_median`: 最良と中央値の差 (=良手の希少さ)。
- `n_near_best_025/050`: q* から 0.25/0.5 点以内の候補数 (**多峰性**: 正解が一意か)。`n_types_near_best_050`: その戦術種別数。
- `q_sd_mean/median/best`: 審判 SD (実行リスクの水準)。`best_type`, `best_type_is_hit`, `best_type_margin` (戦術種別間の最良差)。
- `frac_hit/draw/guard`: 候補プール内の種別比率。`lookahead_gain_*`: q − 即時評価 (審判に q_immediate がある場合のみ)。
- `best_hist_entropy/best_p_negative/best_p_ge2`: 最良手の得点分布のエントロピー/失点確率/2点以上確率。
- `qadj_star_l{λ}`, `qadj_best_same_as_q_best_l{λ}`: リスク調整真値 q−λs の最良と、その手が q 最良と同じか。

## C. クラスタ構造 (ClusterValue の cluster_table 由来; seed 平均)
- `K_clusters`, `largest_frac` (最大クラスタの占有率 = **巨大クラスタ退化**), `n_singleton/singleton_frac`, `size_entropy_norm`, `type_purity`。
- `eta2_q`: 真値 q の分散のうちクラスタ間で説明される割合 (**クラスタが価値的に意味ある区分か**)。`within_sd_q`: クラスタ内の q の SD。
- `rho_cand/rho_cluster/rho_gain`: 候補単位/クラスタ単位の推定値 vs 真値の Spearman ρ とその差 (分散削減効果)。
- `best_is_rep`: 審判最良手が代表手に残ったか。`best_cluster_selected`: 最良手のクラスタが採用されたか。`best_cluster_rank`: 価値順位。
- `rep_q_max`, `screen_loss` = q* − max(代表手の q) (**スクリーンで失った上限**; regret = screen_loss + 探索損)。
- `true_top_cluster_selected`: 真値平均が最良のクラスタが採用されたか。`cluster_sd_mean`: プールSD σ_c の平均。
- `rho_cluster_risk_l{λ}` / `rho_cand_risk_l{λ}`: リスク調整値 (μ−λσ) の妥当性。`topk_overlap_l{λ}`: λ で上位K採用クラスタがどれだけ変わるか。
- `md_*`: Proposed(A2, medoid 代表) の同指標。

## D. 成果 (reinvest_joined.csv 由来)
- `regret_<arm>` (seed 平均), `regret_<arm>_sd`, `n_seeds_<arm>`。`d_A_B` = regret_A − regret_B (**正なら A が悪い**)。
- `regret_adj_<arm>`, `dadj_A_B`: aggregate_reinvest.py --risk-lambda 指定時のリスク調整 regret とその差。
"""

if __name__ == "__main__":
    main()
