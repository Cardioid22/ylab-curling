# -*- coding: utf-8 -*-
"""深さ3(d3)と深さ1(d1)で選ぶ手が分かれるとき、それがクラスタ(戦術)単位で見て
「同じエリア内の微調整」なのか「別エリアへの乗り換え」なのかを分析する。GPWアブスト用。

使用データ:
  - d3の選択手: A1 (AllGrid depth3, 修正木) reinvest_results.csv
  - d1の選択手: A1d1 (AllGrid depth1, 修正木) reinvest_results.csv
  - クラスタ割当: A9 (ClusterValue) cluster_table.csv (同一seedのdistDeltaクラスタリング結果)
  - 審判: q_ref (クラスタ価値の高低比較・大きな損失の有無判定用)

Usage:
  python scripts/analyze_cluster_switch.py \
    --d3-dir reinvest_experiment/run50v2/A1 \
    --d1-dir reinvest_experiment/run50v2/A1d1 \
    --d1-extra reinvest_experiment/run50_A1d1fix/seed_42 \
    --cluster-dir reinvest_experiment/run50v2/A9 \
    --referee-csv reinvest_experiment/scorescreen/run50/referee/score_move_qtable.csv \
    --out reinvest_experiment/run50v2/cluster_switch
"""
import argparse, csv, glob, os
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "MS Gothic"; plt.rcParams["axes.unicode_minus"] = False

TYPE = {"0": "Pass", "1": "Draw", "2": "PreGuard", "3": "Hit", "4": "Freeze",
        "5": "Peel", "6": "ComeAround", "7": "PostGuard", "8": "DrawRaise",
        "9": "Takeout", "10": "Double"}
BUCKETS = [(2, 4, "終盤 r=2-4"), (5, 8, "中盤 r=5-8"), (9, 16, "序盤 r=9+")]


def bucket(r):
    for lo, hi, name in BUCKETS:
        if lo <= r <= hi:
            return name
    return None


def load_moves_by_seed(dir_pat, extra_dir=None):
    """dir/seed_*/reinvest_results.csv -> {seed: {(g,e,s): cand_idx}}"""
    out = defaultdict(dict)
    for fp in glob.glob(f"{dir_pat}/seed_*/reinvest_results.csv"):
        seed = fp.split("seed_")[1].split(os.sep)[0].split("/")[0]
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            out[seed][(r["game_id"], r["end"], r["shot_num"])] = int(r["candidate_idx"])
    if extra_dir:
        fp = f"{extra_dir}/reinvest_results.csv"
        if os.path.exists(fp):
            seed = extra_dir.rstrip("/").split("seed_")[-1]
            for r in csv.DictReader(open(fp, encoding="utf-8")):
                out.setdefault(seed, {})[(r["game_id"], r["end"], r["shot_num"])] = int(r["candidate_idx"])
    return out


def load_clusters_by_seed(dir_pat):
    """dir/seed_*/cluster_table.csv -> {seed: {(g,e,s): {cand_idx: cid}}}, size, shot_type"""
    clus = defaultdict(lambda: defaultdict(dict))
    csize = defaultdict(lambda: defaultdict(dict))
    ctype = defaultdict(lambda: defaultdict(dict))
    for fp in glob.glob(f"{dir_pat}/seed_*/cluster_table.csv"):
        seed = fp.split("seed_")[1].split(os.sep)[0].split("/")[0]
        for r in csv.DictReader(open(fp, encoding="utf-8")):
            key = (r["game_id"], r["end"], r["shot_num"])
            cid = int(r["cluster_id"]); ci = int(r["candidate_idx"])
            clus[seed][key][ci] = cid
            csize[seed][key][cid] = csize[seed][key].get(cid, 0) + 1
            ctype[seed][key][ci] = r["shot_type"].strip('"')
    return clus, csize, ctype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3-dir", required=True)
    ap.add_argument("--d1-dir", required=True)
    ap.add_argument("--d1-extra", default=None, help="d1側の追加1シードディレクトリ (dir/seed_XX形式でない場合)")
    ap.add_argument("--cluster-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d3 = load_moves_by_seed(args.d3_dir)
    d1 = load_moves_by_seed(args.d1_dir, args.d1_extra)
    clus, csize, ctype = load_clusters_by_seed(args.cluster_dir)

    seeds = sorted(set(d3) & set(d1) & set(clus))
    print(f"共通シード: {seeds}")

    stats = defaultdict(lambda: dict(n=0, agree=0, same_cluster=0, diff_cluster=0,
                                     sizes_from=[], sizes_to=[], trans=Counter()))
    for seed in seeds:
        for key, m3 in d3[seed].items():
            if key not in d1[seed] or key not in clus[seed]:
                continue
            m1 = d1[seed][key]
            r = 16 - int(key[2])
            b = bucket(r)
            if b is None:
                continue
            st = stats[b]
            st["n"] += 1
            if m3 == m1:
                st["agree"] += 1
                continue
            c = clus[seed][key]
            if m3 not in c or m1 not in c:
                continue
            cid3, cid1 = c[m3], c[m1]
            sizes = csize[seed][key]
            types = ctype[seed][key]
            if cid3 == cid1:
                st["same_cluster"] += 1
            else:
                st["diff_cluster"] += 1
                st["sizes_from"].append(sizes.get(cid1, 1))
                st["sizes_to"].append(sizes.get(cid3, 1))
                st["trans"][(types.get(m1, "?"), types.get(m3, "?"))] += 1

    rep = []
    def pr(s): rep.append(s); print(s)

    pr("=" * 78)
    pr(f"深さで手が変わるときのクラスタ単位の乗り換え分析 (seed: {seeds})")
    pr("=" * 78)
    pr(f"{'フェーズ':>10} {'n':>4} {'一致率':>7} {'同一クラスタ内':>10} {'別クラスタへ乗換':>12} {'乗換率':>7}")
    for _, _, b in BUCKETS:
        st = stats[b]
        if st["n"] == 0:
            continue
        tot_diff = st["same_cluster"] + st["diff_cluster"]
        switch_rate = st["diff_cluster"] / tot_diff if tot_diff else float("nan")
        pr(f"{b:>10} {st['n']:>4} {st['agree']/st['n']:>7.1%} "
           f"{st['same_cluster']:>10} {st['diff_cluster']:>12} {switch_rate:>7.1%}")

    pr("")
    pr("--- 終盤(r=2-4)で別クラスタへ乗り換えた場合のクラスタサイズ (密度チェック) ---")
    st = stats["終盤 r=2-4"]
    if st["sizes_from"]:
        pr(f"  乗換元サイズ: 中央値={np.median(st['sizes_from']):.0f} 平均={np.mean(st['sizes_from']):.1f} "
           f"(サイズ1の割合={sum(1 for s in st['sizes_from'] if s==1)}/{len(st['sizes_from'])})")
        pr(f"  乗換先サイズ: 中央値={np.median(st['sizes_to']):.0f} 平均={np.mean(st['sizes_to']):.1f} "
           f"(サイズ1の割合={sum(1 for s in st['sizes_to'] if s==1)}/{len(st['sizes_to'])})")
        pr("  -> サイズ1でない = 複数候補を含む戦術グループ間の乗り換えであり、個別候補の入替ではない")

    pr("")
    pr("--- 終盤(r=2-4)のショット種別遷移 (d1のクラスタ代表種別 -> d3のクラスタ代表種別) ---")
    for (t1, t3), c in stats["終盤 r=2-4"]["trans"].most_common(12):
        pr(f"   {TYPE.get(t1, t1):>10} -> {TYPE.get(t3, t3):>10}  x{c}")

    with open(f"{args.out}/cluster_switch_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")

    # 図: フェーズ別 同一クラスタ/別クラスタ 積み上げ棒
    names = [b for _, _, b in BUCKETS]
    same = [stats[b]["same_cluster"] for b in names]
    diff = [stats[b]["diff_cluster"] for b in names]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    ax.bar(x, same, 0.55, label="同一クラスタ内の微調整", color="#0072B2")
    ax.bar(x, diff, 0.55, bottom=same, label="別クラスタへの乗り換え", color="#D55E00")
    for i in range(3):
        tot = same[i] + diff[i]
        if tot:
            ax.text(i, tot + 0.5, f"{diff[i]/tot:.0%}が乗換", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("手が分かれた局面数 (d3 vs d1)")
    ax.set_title("深さで手が変わるとき、クラスタ(戦術)ごと乗り換わるか")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}/cluster_switch.png", dpi=150)
    plt.close(fig)

    # 図2: 残り手数別の d3/d1 選択一致率 (「後半で著しく変わる」の直接根拠)
    agree_rate = [stats[b]["agree"] / stats[b]["n"] if stats[b]["n"] else 0 for b in names]
    ns = [stats[b]["n"] for b in names]
    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    ax2.bar(x, agree_rate, 0.55, color="#009E73")
    for i, (v, n) in enumerate(zip(agree_rate, ns)):
        ax2.text(i, v + 0.02, f"{v:.0%}\n(n={n})", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("depth3とdepth1の選択一致率")
    ax2.set_title("残り手数別: 深さで選ぶ手はどこで変わるか")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{args.out}/agreement_by_phase.png", dpi=150)
    plt.close(fig2)
    print(f"\n[out] -> {args.out}/ (cluster_switch_report.txt, cluster_switch.png, agreement_by_phase.png)")


if __name__ == "__main__":
    main()
