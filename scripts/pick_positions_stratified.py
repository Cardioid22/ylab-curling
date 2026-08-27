#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""層化 (非ランダム) の局面追加 — 500局面化 (GPW2026 本論文)
================================================================================
run200 で薄かった層を意図的に厚くする。既存セット (test_positions200) を先頭にそのまま残し
(global index 保存 = 既存結果を流用できる)、プールから割当表 (quota) に従って局面を足す。

層 (stratum) の定義 (盤面だけから決まるもの; 審判に依存しない):
  empty_early   : 石 0-1 個, shot<=3        「空のハウス」(ユーザー仮説: クラスタリング不要のはず)
  sparse_early  : 石 2-4 個, shot<=5
  <cat>_mid     : shot 6-11, cat = takeout/freeze/draw/crowded/other (pick_more_positions.py の label_of と同じ判定)
  <cat>_late    : shot 12-14 (shot 15 = 最終ショットは除外: 距離関数が潰れる退化層で、run200 に既に 32 局面ある)

制約: 既存セットと game_id を重複させない、1 game 1 局面、空でない (empty_early を除く)。

Usage:
  python scripts/pick_positions_stratified.py \
      --src-dir clustered_ayumu/test_positions_20260417_055725 \
      --existing-dir test_positions200 --out-dir test_positions500 --seed 31 \
      [--quota "empty_early=40,sparse_early=30,takeout_mid=40,takeout_late=40,freeze_mid=30,freeze_late=30,draw_mid=20,draw_late=20,crowded_mid=25,crowded_late=25"]
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

# pick_more_positions.py と同じ判定を再利用
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_more_positions import parse, label_of  # noqa: E402

DEFAULT_QUOTA = ("empty_early=40,sparse_early=30,"
                 "takeout_mid=40,takeout_late=40,freeze_mid=30,freeze_late=30,"
                 "draw_mid=20,draw_late=20,crowded_mid=25,crowded_late=25")


def stratum_of(f) -> str | None:
    s, n = f["shot"], f["n"]
    if s >= 15:
        return None                      # 最終ショットは除外
    if n <= 1 and s <= 3:
        return "empty_early"
    if 2 <= n <= 4 and s <= 5:
        return "sparse_early"
    if s <= 5:
        return None                      # 序盤で石が多い局面はプールにほぼ無い / 対象外
    ph = "mid" if s <= 11 else "late"
    return f"{label_of(f)}_{ph}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", type=Path, required=True)
    ap.add_argument("--existing-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quota", default=DEFAULT_QUOTA)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    quota = {}
    for tok in args.quota.split(","):
        k, v = tok.split("="); quota[k.strip()] = int(v)

    # 既存
    exist_rows, exist_gids, header = [], set(), None
    with open(args.existing_dir / "batch_0001.csv", newline="") as fh:
        rd = csv.DictReader(fh); header = rd.fieldnames
        for row in rd:
            exist_rows.append(row); exist_gids.add(int(row["match_id"]))
    print(f"existing: {len(exist_rows)} positions ({len(exist_gids)} games)", file=sys.stderr)

    # プール → 層
    pool = defaultdict(list)
    n_pool = 0
    for fp in sorted(args.src_dir.glob("batch_*.csv")):
        with open(fp, newline="") as fh:
            for row in csv.DictReader(fh):
                n_pool += 1
                f = parse(row)
                if f["gid"] in exist_gids:
                    continue
                st = stratum_of(f)
                if st is None:
                    continue
                pool[st].append(f)
    print(f"pool: {n_pool} positions; per-stratum available (after excluding existing games):", file=sys.stderr)
    for k in sorted(pool):
        print(f"  {k:<16} {len(pool[k]):5d}" + ("   <- quota %d" % quota[k] if k in quota else ""), file=sys.stderr)
    missing = [k for k in quota if k not in pool]
    if missing:
        print(f"WARNING: strata with no candidates: {missing}", file=sys.stderr)

    rng = random.Random(args.seed)
    selected, used_gids = [], set()
    short = {}
    for k, q in quota.items():
        cands = pool.get(k, [])
        rng.shuffle(cands)
        got = 0
        for f in cands:
            if got >= q:
                break
            if f["gid"] in used_gids:
                continue
            selected.append((k, f)); used_gids.add(f["gid"]); got += 1
        if got < q:
            short[k] = (got, q)
    if short:
        print(f"WARNING: quota not met: {short}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "batch_0001.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header); w.writeheader()
        for row in exist_rows:
            w.writerow(row)
        for _, f in selected:
            w.writerow(f["raw"])
    # categories.csv: 既存は既存のものを引き継ぎ (無ければ label), 新規は層名
    with open(args.out_dir / "categories.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "stratum", "game_id", "end", "shot_num", "team", "n_stones", "my_in_house", "op_in_house", "origin"])
        ex_lab = {}
        p = args.existing_dir / "categories.csv"
        if p.exists():
            for r in csv.DictReader(open(p, newline="")):
                ex_lab[(int(r["game_id"]), int(r["end"]), int(r["shot_num"]))] = r["category"]
        for row in exist_rows:
            f = parse(row)
            w.writerow([ex_lab.get((f["gid"], f["end"], f["shot"]), label_of(f)), stratum_of(f) or "r1/other",
                        f["gid"], f["end"], f["shot"], f["team"], f["n"], len(f["my_h"]), len(f["op_h"]), "existing"])
        for k, f in selected:
            w.writerow([label_of(f), k, f["gid"], f["end"], f["shot"], f["team"], f["n"], len(f["my_h"]), len(f["op_h"]), "new"])
    print(f"\nWrote {len(exist_rows) + len(selected)} positions ({len(exist_rows)} existing + {len(selected)} new) -> {args.out_dir}", file=sys.stderr)
    by = defaultdict(int)
    for k, _ in selected:
        by[k] += 1
    print("new positions by stratum:", file=sys.stderr)
    for k in quota:
        print(f"  {k:<16} {by[k]:3d} / {quota[k]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
