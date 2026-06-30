#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テスト局面の追加選定 (50局面化)
================================================================================
現在の8局面 (test_positions_categorized8) とは異なる盤面を追加し、計50局面にする。
論文用に選定バイアスを避けるため、カテゴリ決め打ちではなく
「石数 × 進行フェーズ」で層化した多様サンプリングを行う (固定seedで再現可能)。

制約 (ユーザー指定):
  - 現8局面と異なる盤面 (game_id を重複させない)
  - ストーンが無い(空)盤面は選ばない → n_stones >= MIN_STONES(既定2)

出力: <out-dir>/batch_0001.csv (既存8 + 新規42 = 50) と categories.csv (参考ラベル付き)

Usage:
  python scripts/pick_more_positions.py \
    --src-dir clustered_ayumu/test_positions_20260417_055725 \
    --existing-dir test_positions_categorized8 \
    --out-dir test_positions50 \
    --n-new 42 --min-stones 2 --seed 17
================================================================================
"""
from __future__ import annotations
import argparse, csv, math, sys
from collections import defaultdict
from pathlib import Path
import random

HC_X, HC_Y, HR, SR = 0.0, 38.405, 1.829, 0.145
FRONT = HC_Y - HR


def in_house(x, y):
    return (x - HC_X) ** 2 + (y - HC_Y) ** 2 <= (HR + SR) ** 2


def dtee(x, y):
    return math.hypot(x - HC_X, y - HC_Y)


def parse(row):
    team = int(row['team']); opp = 1 - team

    def st(t):
        out = []
        for i in range(8):
            if int(row[f't{t}s{i}_inplay']) == 1:
                out.append((float(row[f't{t}s{i}_x']), float(row[f't{t}s{i}_y'])))
        return out

    my, op = st(team), st(opp)
    allst = [(x, y, 'my') for x, y in my] + [(x, y, 'op') for x, y in op]
    return dict(
        gid=int(row['match_id']), end=int(row['end']), shot=int(row['shot_num']),
        team=team, my=my, op=op, n=len(allst),
        my_h=[(x, y) for x, y in my if in_house(x, y)],
        op_h=[(x, y) for x, y in op if in_house(x, y)],
        nearest=min(allst, key=lambda s: dtee(s[0], s[1])) if allst else None,
        raw=row,
    )


# 参考ラベル用 (pick_categorized_positions.py と同じ判定)
def label_of(f):
    cats = []
    if f['op_h'] and f['nearest'] and f['nearest'][2] == 'op':
        cats.append(('takeout', 1.0 / (dtee(*f['nearest'][:2]) + 0.1) + 0.5 * len(f['op_h'])))
    if f['op_h']:
        tgt = min(f['op_h'], key=lambda s: dtee(s[0], s[1]))
        if not any(abs(x - tgt[0]) < 0.3 and 0 < (tgt[1] - y) < 0.6 for x, y in f['my'] + f['op']):
            cats.append(('freeze', 1.0 / (dtee(*tgt) + 0.1)))
    nd = dtee(*f['nearest'][:2]) if f['nearest'] else 99
    if nd > 0.5 and f['n'] >= 1:
        cats.append(('draw', nd - 0.1 * f['n']))
    if f['n'] >= 6:
        cats.append(('crowded', float(f['n'])))
    if not cats:
        return 'other'
    return max(cats, key=lambda t: t[1])[0]


def stratum(f):
    n = f['n']
    nb = '2-4' if n <= 4 else '5-8' if n <= 8 else '9-12' if n <= 12 else '13-16'
    s = f['shot']
    ph = 'early' if s <= 5 else 'mid' if s <= 11 else 'late'
    return (nb, ph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', type=Path, required=True)
    ap.add_argument('--existing-dir', type=Path, required=True,
                    help='現8局面のディレクトリ (重複除外用)')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--n-new', type=int, default=42)
    ap.add_argument('--min-stones', type=int, default=2)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()

    # 既存局面 (除外用 game_id と、出力先頭に残すための raw)
    exist_rows = []
    exist_gids = set()
    exist_keys = set()
    header = None
    ex_csv = args.existing_dir / 'batch_0001.csv'
    with open(ex_csv, newline='') as f:
        rd = csv.DictReader(f); header = rd.fieldnames
        for row in rd:
            exist_rows.append(row)
            exist_gids.add(int(row['match_id']))
            exist_keys.add((int(row['match_id']), int(row['end']), int(row['shot_num'])))
    print(f'existing: {len(exist_rows)} positions, gids={sorted(exist_gids)}', file=sys.stderr)

    # プール読み込み + フィルタ
    feats = []
    for fp in sorted(args.src_dir.glob('batch_*.csv')):
        with open(fp, newline='') as f:
            rd = csv.DictReader(f)
            if header is None: header = rd.fieldnames
            for row in rd:
                feats.append(parse(row))
    print(f'pool: {len(feats)} positions', file=sys.stderr)

    cand = [f for f in feats
            if f['n'] >= args.min_stones                 # 空盤面/ほぼ空を除外
            and f['gid'] not in exist_gids               # 現8と異なる game
            and (f['gid'], f['end'], f['shot']) not in exist_keys]
    print(f'candidates after filter (n>={args.min_stones}, exclude existing): {len(cand)}',
          file=sys.stderr)

    # 層化 (石数bin × フェーズ) → 各 game は1局面まで → round-robin で多様に
    rng = random.Random(args.seed)
    rng.shuffle(cand)
    strata = defaultdict(list)
    for f in cand:
        strata[stratum(f)].append(f)
    keys = sorted(strata.keys())
    for k in keys:
        rng.shuffle(strata[k])

    selected = []
    used_gids = set()
    # round-robin: 各層から1つずつ、game_id 重複を避けて n-new 個集める
    progress = True
    while len(selected) < args.n_new and progress:
        progress = False
        for k in keys:
            if len(selected) >= args.n_new:
                break
            bucket = strata[k]
            while bucket:
                f = bucket.pop()
                if f['gid'] in used_gids:
                    continue
                selected.append(f)
                used_gids.add(f['gid'])
                progress = True
                break

    if len(selected) < args.n_new:
        print(f'WARNING: only {len(selected)}/{args.n_new} new positions found', file=sys.stderr)

    # 出力: 既存8 + 新規 を1ファイルに
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / 'batch_0001.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in exist_rows:
            w.writerow(row)
        for ft in selected:
            w.writerow(ft['raw'])

    # categories.csv (既存はそのまま読み直し、新規はラベル計算)
    lab_csv = args.out_dir / 'categories.csv'
    with open(lab_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'game_id', 'end', 'shot_num', 'team',
                    'n_stones', 'my_in_house', 'op_in_house', 'origin'])
        # 既存
        ex_lab = {}
        ex_lab_csv = args.existing_dir / 'categories.csv'
        if ex_lab_csv.exists():
            for r in csv.DictReader(open(ex_lab_csv, newline='')):
                ex_lab[(int(r['game_id']), int(r['end']), int(r['shot_num']))] = r['category']
        for row in exist_rows:
            ft = parse(row)
            cat = ex_lab.get((ft['gid'], ft['end'], ft['shot']), label_of(ft))
            w.writerow([cat, ft['gid'], ft['end'], ft['shot'], ft['team'], ft['n'],
                        len(ft['my_h']), len(ft['op_h']), 'existing'])
        for ft in selected:
            w.writerow([label_of(ft), ft['gid'], ft['end'], ft['shot'], ft['team'], ft['n'],
                        len(ft['my_h']), len(ft['op_h']), 'new'])

    total = len(exist_rows) + len(selected)
    print(f'\nWrote {total} positions ({len(exist_rows)} existing + {len(selected)} new) '
          f'-> {out_csv}', file=sys.stderr)
    # 層別の内訳
    by_stra = defaultdict(int)
    for ft in selected:
        by_stra[stratum(ft)] += 1
    print('new positions by stratum (n_stones × phase):', file=sys.stderr)
    for k in sorted(by_stra):
        print(f'  {k}: {by_stra[k]}', file=sys.stderr)
    by_lab = defaultdict(int)
    for ft in selected:
        by_lab[label_of(ft)] += 1
    print('new positions by reference label:', file=sys.stderr)
    for k in sorted(by_lab):
        print(f'  {k}: {by_lab[k]}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
