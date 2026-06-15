#!/usr/bin/env python3
"""
カテゴリ別テスト局面選定 (再投資実験用)

5カテゴリ × 2局面 = 10局面を、構造的ヒューリスティクスで元プールから選ぶ:
  guard    : ガードを打つべき場面 (早期shot & 盤面が疎)
  takeout  : 相手石を弾き出すべき場面 (相手石がハウス内 & 最近接が相手)
  draw     : ハウス中央を狙うべき場面 (ティー近傍が空いている)
  freeze   : 自/相手石付近にフリーズすべき場面 (相手石がハウス内 & 前方に空き)
  crowded  : 石が6個以上残っている場面

⚠️ 構造選定は「候補の絞り込み」まで。各局面で「意図した手が実際に最善か」は
   審判 (score_move の Q_ref) で検証し、外れたら shortlist から差し替えること。

Usage:
  python scripts/pick_categorized_positions.py \
    --src-dir clustered_ayumu/test_positions_20260417_055725 \
    --out-dir test_positions_categorized10 \
    --seed 17
"""
from __future__ import annotations
import argparse, csv, glob, math, random, sys
from pathlib import Path

HC_X, HC_Y, HR, SR = 0.0, 38.405, 1.829, 0.145
FRONT = HC_Y - HR  # ハウス手前端 ~36.576

def in_house(x, y): return (x-HC_X)**2 + (y-HC_Y)**2 <= (HR+SR)**2
def guard_zone(x, y): return (not in_house(x, y)) and (FRONT-5.0 < y < FRONT) and abs(x-HC_X) < 2.0
def dtee(x, y): return math.hypot(x-HC_X, y-HC_Y)

def parse(row):
    team = int(row['team']); opp = 1-team
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

# --- カテゴリ別スコア (高いほど好適。Noneなら非該当) ---
def score_guard(f):
    if f['shot'] > 3 or f['n'] > 2: return None
    # 早く・疎なほど良い。相手が1石ハウス内だとガードで蓋する動機が明確
    s = (3 - f['shot']) + (2 - f['n']) * 0.5 + (1.0 if f['op_h'] else 0.0)
    return s

def score_takeout(f):
    if not f['op_h'] or not f['nearest'] or f['nearest'][2] != 'op': return None
    # 相手のショットストーンがティーに近いほどテイクアウト価値が明確
    nd = dtee(f['nearest'][0], f['nearest'][1])
    return 1.0/(nd+0.1) + 0.5*len(f['op_h'])

def score_draw(f):
    nd = dtee(f['nearest'][0], f['nearest'][1]) if f['nearest'] else 99
    if nd <= 0.5 or f['n'] < 1: return None  # ティー近傍が埋まっていない
    # 中央が空いていてドローで取れる。混みすぎは除く
    return nd - 0.1*f['n']

def score_freeze(f):
    if not f['op_h']: return None
    # 相手のハウス内ショットストーンに前方空きがあるとフリーズ価値が高い
    tgt = min(f['op_h'], key=lambda s: dtee(s[0], s[1]))
    nd = dtee(tgt[0], tgt[1])
    front_clear = not any(abs(x-tgt[0]) < 0.3 and 0 < (tgt[1]-y) < 0.6 for x, y in f['my']+f['op'])
    if not front_clear: return None
    return 1.0/(nd+0.1) + 0.3*len(f['op_h'])

def score_crowded(f):
    if f['n'] < 6: return None
    return float(f['n'])

CATEGORIES = [
    ('guard',   score_guard),
    ('takeout', score_takeout),
    ('draw',    score_draw),
    ('freeze',  score_freeze),
    ('crowded', score_crowded),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--per-cat', type=int, default=2)
    ap.add_argument('--shortlist', type=int, default=5)
    ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args()

    header = None
    feats = []
    for fp in sorted(args.src_dir.glob('batch_*.csv')):
        with open(fp, newline='') as f:
            rd = csv.DictReader(f)
            if header is None: header = rd.fieldnames
            for row in rd:
                feats.append(parse(row))
    if not feats:
        sys.exit(f'no positions under {args.src_dir}')
    print(f'loaded {len(feats)} positions', file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(feats)  # タイブレークを seed で決定的に散らす

    used_gids = set()
    selected = []  # (category, feat)
    shortlists = {}
    for cat, scorer in CATEGORIES:
        scored = [(scorer(f), f) for f in feats]
        scored = [(s, f) for s, f in scored if s is not None]
        scored.sort(key=lambda t: t[0], reverse=True)
        shortlists[cat] = scored[:args.shortlist]
        picked = 0
        for s, f in scored:
            if f['gid'] in used_gids: continue
            selected.append((cat, f))
            used_gids.add(f['gid'])
            picked += 1
            if picked >= args.per_cat: break
        if picked < args.per_cat:
            print(f'WARNING: {cat} only found {picked}/{args.per_cat}', file=sys.stderr)

    # 出力
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / 'batch_0001.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for cat, ft in selected:
            w.writerow(ft['raw'])
    # ラベル対応表も出す
    label_csv = args.out_dir / 'categories.csv'
    with open(label_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'game_id', 'end', 'shot_num', 'team', 'n_stones',
                    'my_in_house', 'op_in_house'])
        for cat, ft in selected:
            w.writerow([cat, ft['gid'], ft['end'], ft['shot'], ft['team'], ft['n'],
                        len(ft['my_h']), len(ft['op_h'])])

    # レポート
    print(f'\nWrote {len(selected)} positions to {out_csv}', file=sys.stderr)
    print(f'Labels: {label_csv}\n', file=sys.stderr)
    print(f"{'category':9} {'gid':>5} {'e':>2} {'s':>3} {'team':>4} {'n':>3} "
          f"{'myH':>3} {'opH':>3}", file=sys.stderr)
    print('-'*44, file=sys.stderr)
    for cat, ft in selected:
        print(f"{cat:9} {ft['gid']:>5} {ft['end']:>2} {ft['shot']:>3} {ft['team']:>4} "
              f"{ft['n']:>3} {len(ft['my_h']):>3} {len(ft['op_h']):>3}", file=sys.stderr)
    return 0

if __name__ == '__main__':
    sys.exit(main())
