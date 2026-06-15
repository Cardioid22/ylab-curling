#!/usr/bin/env python3
"""
カテゴリラベルの審判検証

審判(score_move)の Q テーブルと、意図したカテゴリ(categories.csv)を突き合わせ、
各局面で「意図した手タイプが実際に最善(または最善に近い)か」を判定する。

カテゴリ → 期待する手タイプ(label接頭辞):
  guard   -> PreGuard     (自分がガードを置くのが最善か)
  takeout -> Hit
  draw    -> Draw
  freeze  -> Freeze
  crowded -> (任意。石数で定義されるカテゴリなので手タイプ不問 → 常にOK)

Usage:
  python scripts/validate_categories.py \
    --qtable depth3_experiment/<dir>/score_move_qtable.csv \
    --categories test_positions_categorized10/categories.csv
"""
from __future__ import annotations
import argparse, csv, re, sys
from collections import defaultdict

EXPECT = {
    'guard':   'PreGuard',
    'takeout': 'Hit',
    'draw':    'Draw',
    'freeze':  'Freeze',
    'crowded': None,   # 手タイプ不問
}

def prefix(label):
    m = re.match(r'^([A-Za-z]+)', label.strip().strip('"'))
    return m.group(1) if m else '?'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qtable', required=True)
    ap.add_argument('--categories', required=True)
    args = ap.parse_args()

    # categories.csv
    cat = {}
    with open(args.categories) as f:
        for r in csv.DictReader(f):
            cat[(int(r['game_id']), int(r['end']), int(r['shot_num']))] = r['category']

    # qtable: 局面ごとに候補をまとめる
    perpos = defaultdict(list)  # key -> [(q, label, type_prefix)]
    with open(args.qtable) as f:
        for r in csv.DictReader(f):
            key = (int(r['game_id']), int(r['end']), int(r['shot_num']))
            perpos[key].append((float(r['q_ref_mean']), r['label'].strip('"'), prefix(r['label'])))

    print(f"{'category':9} {'gid':>5} {'best move':24} {'Qbest':>7} | "
          f"{'intended-type best':22} {'Qint':>7} {'gap':>6}  判定")
    print('-'*100)
    n_ok = n_total = 0
    for key, cands in sorted(perpos.items(), key=lambda kv: kv[0]):
        c = cat.get(key, '?')
        cands.sort(reverse=True)  # Qで降順
        qbest, lbest, tbest = cands[0]
        exp = EXPECT.get(c)
        if exp is None:
            # crowded: 手タイプ不問。常にOK
            print(f"{c:9} {key[0]:>5} {lbest:24} {qbest:>7.3f} | {'(任意)':22} {'':>7} {'':>6}  OK(石数)")
            n_ok += 1; n_total += 1
            continue
        # 意図タイプの中での最良
        same = [(q, l) for q, l, t in cands if t == exp]
        n_total += 1
        if same:
            qint, lint = max(same)
            gap = qbest - qint
            ok = (tbest == exp)
            verdict = 'OK' if ok else (f'△近い(gap={gap:.2f})' if gap <= 0.3 else 'NG(別タイプが最善)')
            if ok or gap <= 0.3: n_ok += 1
            print(f"{c:9} {key[0]:>5} {lbest:24} {qbest:>7.3f} | {lint:22} {qint:>7.3f} {gap:>6.3f}  {verdict}")
        else:
            print(f"{c:9} {key[0]:>5} {lbest:24} {qbest:>7.3f} | {'(該当手なし)':22} {'':>7} {'':>6}  NG(候補に無)")

    print('-'*100)
    print(f"検証OK (最善 or gap<=0.3): {n_ok}/{n_total}")
    print("\n△/NG の局面は categories.csv の意図と審判が食い違う。")
    print("→ pick_categorized_positions.py の shortlist から差し替えるか、カテゴリ定義を見直す。")
    return 0

if __name__ == '__main__':
    sys.exit(main())
