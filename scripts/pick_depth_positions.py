#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
「深読みの価値」実験用の局面選定。
残り手数 r=1..4 (= shot_num 15/14/13/12) の局面を各 per-r 個、プールから任意抽出。
制約: n_stones >= min_stones (空盤面なし), game_id 重複なし。固定seedで再現可能。

Usage:
  python scripts/pick_depth_positions.py \
    --src-dir clustered_ayumu/test_positions_20260417_055725 \
    --out-dir test_positions_depth --per-r 13 --min-stones 2 --seed 23
"""
from __future__ import annotations
import argparse, csv, sys, random
from collections import defaultdict
from pathlib import Path

def n_stones(row):
    return sum(int(row[f't{t}s{i}_inplay']) for t in (0,1) for i in range(8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--per-r', type=int, default=13)   # 残り手数ごとの局面数
    ap.add_argument('--min-stones', type=int, default=2)
    ap.add_argument('--seed', type=int, default=23)
    args = ap.parse_args()

    header=None; rows=[]
    for fp in sorted(args.src_dir.glob('batch_*.csv')):
        with open(fp, newline='') as f:
            rd=csv.DictReader(f)
            if header is None: header=rd.fieldnames
            rows.extend(list(rd))
    print(f'pool: {len(rows)} positions', file=sys.stderr)

    # shot_num -> 候補 (n>=min, 石あり)
    by_shot=defaultdict(list)
    for r in rows:
        s=int(r['shot_num'])
        if s in (12,13,14,15) and n_stones(r)>=args.min_stones:
            by_shot[s].append(r)

    rng=random.Random(args.seed)
    picked=[]; used=set()
    for s in (15,14,13,12):   # r=1,2,3,4
        cand=by_shot[s][:]; rng.shuffle(cand)
        k=0
        for r in cand:
            gid=int(r['match_id'])
            if gid in used: continue
            picked.append(r); used.add(gid); k+=1
            if k>=args.per_r: break
        print(f'  shot={s} (残り{16-s}手): {k} 局面', file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir/'batch_0001.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=header); w.writeheader()
        for r in picked: w.writerow(r)
    with open(args.out_dir/'categories.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['remaining','game_id','end','shot_num','team','n_stones'])
        for r in picked:
            w.writerow([16-int(r['shot_num']), r['match_id'], r['end'], r['shot_num'],
                        r['team'], n_stones(r)])
    print(f'\nWrote {len(picked)} positions -> {args.out_dir}/batch_0001.csv', file=sys.stderr)

if __name__=='__main__':
    sys.exit(main())
