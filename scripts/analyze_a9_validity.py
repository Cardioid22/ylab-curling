# -*- coding: utf-8 -*-
"""A9のクラスタ価値(cluster_value)が審判の真値をどれだけ言い当てているかを5seed分検証する。
GPWアブスト用の妥当性の数値根拠。

出力: reinvest_experiment/run50v2/A9/validity/
  - a9_validity_report.txt (seed別+統合のSpearman rho)
  - a9_validity_summary.csv
"""
import csv, glob, os
from collections import defaultdict
import numpy as np
from scipy import stats

RUN50V2 = "reinvest_experiment/run50v2"
REFEREE = "reinvest_experiment/scorescreen/run50/referee/score_move_qtable.csv"
OUT = f"{RUN50V2}/A9/validity"
os.makedirs(OUT, exist_ok=True)


def load_referee():
    q = defaultdict(dict)
    for r in csv.DictReader(open(REFEREE, encoding="utf-8")):
        q[(r["game_id"], r["end"], r["shot_num"])][int(r["candidate_idx"])] = float(r["q_ref_mean"])
    return q


def main():
    ref = load_referee()
    rows_out = []
    all_rho_cand, all_rho_clus = [], []

    for fp in sorted(glob.glob(f"{RUN50V2}/A9/seed_*/cluster_table.csv")):
        seed = fp.split("seed_")[1].split(os.sep)[0].split("/")[0]
        per_pos = defaultdict(list)
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            per_pos[(r["game_id"], r["end"], r["shot_num"])].append(r)

        cand_rho, clus_rho = [], []
        for key, rows in per_pos.items():
            q = ref.get(key)
            if not q:
                continue
            es, qs = [], []
            for r in rows:
                ci = int(r["candidate_idx"])
                if r["e_score"] and ci in q:
                    es.append(float(r["e_score"])); qs.append(q[ci])
            if len(es) >= 5 and len(set(es)) > 1 and len(set(qs)) > 1:
                v = stats.spearmanr(es, qs).statistic
                if not np.isnan(v):
                    cand_rho.append(v)

            clusters = defaultdict(list); cval = {}
            for r in rows:
                cid = int(r["cluster_id"]); ci = int(r["candidate_idx"])
                if ci in q:
                    clusters[cid].append(q[ci])
                if r["cluster_value"]:
                    cval[cid] = float(r["cluster_value"])
            est, true = [], []
            for cid, mem in clusters.items():
                if cid in cval and mem:
                    est.append(cval[cid]); true.append(np.mean(mem))
            if len(est) >= 4 and len(set(est)) > 1 and len(set(true)) > 1:
                v = stats.spearmanr(est, true).statistic
                if not np.isnan(v):
                    clus_rho.append(v)

        rows_out.append([seed, len(cand_rho), np.median(cand_rho), len(clus_rho), np.median(clus_rho)])
        all_rho_cand.extend(cand_rho)
        all_rho_clus.extend(clus_rho)

    report = []
    report.append("=" * 70)
    report.append("A9 クラスタ価値の妥当性検証 (5シード)")
    report.append("=" * 70)
    report.append(f"{'seed':>6} {'n(候補)':>8} {'候補ρ中央値':>12} {'n(クラスタ)':>10} {'クラスタρ中央値':>14}")
    for row in rows_out:
        report.append(f"{row[0]:>6} {row[1]:>8} {row[2]:>+12.3f} {row[3]:>10} {row[4]:>+14.3f}")
    report.append("-" * 70)
    report.append(f"統合(全seedプール): 候補レベルρ中央値={np.median(all_rho_cand):+.3f} (n={len(all_rho_cand)})")
    report.append(f"統合(全seedプール): クラスタレベルρ中央値={np.median(all_rho_clus):+.3f} (n={len(all_rho_clus)})")
    report.append(f"クラスタレベルの方が候補レベルより高精度 (エリア平均=分散削減) か: "
                 f"{'YES' if np.median(all_rho_clus) > np.median(all_rho_cand) else 'NO'}")

    text = "\n".join(report)
    print(text)
    with open(f"{OUT}/a9_validity_report.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(f"{OUT}/a9_validity_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "n_cand", "cand_rho_median", "n_cluster", "cluster_rho_median"])
        w.writerows(rows_out)
    print(f"\n[out] -> {OUT}/")


if __name__ == "__main__":
    main()
