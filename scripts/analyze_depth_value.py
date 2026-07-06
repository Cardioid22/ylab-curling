# -*- coding: utf-8 -*-
"""
「深読みの価値」曲線: 残り手数 r ごとに、先読みの価値 = q* − q_ref(GREEDY) を集計。
  GREEDY(先読みなし) = argmax q_immediate_mean (外乱込み・即時盤面評価)
  q_ref = 続きをロールアウトで読んだ最終エンドスコア期待値 (続き固定)
  q*     = max_candidate q_ref (読んだ場合の最善)
  value_of_lookahead = q* − q_ref(GREEDYの手)  … 即時最善で選ぶと最終得点をどれだけ損するか

入力: 拡張審判 score_move_qtable*.csv (列に q_immediate_mean を含む)。categories.csv は remaining ラベル用(任意)。
Usage: python scripts/analyze_depth_value.py --referee-csv <glob> [--out DIR]
"""
import argparse, csv, glob, os
from collections import defaultdict
import numpy as np

def boot_ci(x, n=10000, seed=3):
    x=np.asarray(x,float)
    if len(x)==0: return (float('nan'),)*2
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),(n,len(x)))
    return tuple(np.percentile(x[idx].mean(1),[2.5,97.5]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--referee-csv', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-plot', action='store_true')
    args=ap.parse_args()

    rows=[]
    for fp in sorted(glob.glob(args.referee_csv)):
        rows += list(csv.DictReader(open(fp, newline='', encoding='utf-8')))
    if not rows or 'q_immediate_mean' not in rows[0]:
        print('[error] q_immediate_mean 列が無い。新ビルドで審判を回したか確認。'); return

    pos=defaultdict(list)
    for r in rows:
        try:
            pos[(r['game_id'],r['end'],int(r['shot_num']))].append(
                (int(r['candidate_idx']), float(r['q_ref_mean']), float(r['q_immediate_mean'])))
        except (KeyError,ValueError): continue

    per=[]  # (remaining, value_of_lookahead, greedy_is_best, qstar, qref_greedy)
    for (g,e,s), cands in pos.items():
        r = 16 - s
        qref={i:q for i,q,_ in cands}; qimm={i:m for i,_,m in cands}
        qstar=max(qref.values()); best=max(qref,key=qref.get)
        greedy=max(qimm,key=qimm.get)             # 先読みなしが選ぶ手
        val=qstar - qref[greedy]                  # 先読みの価値
        per.append((r, val, 1 if greedy==best else 0, qstar, qref[greedy]))

    print(f"{'残りr':>5} {'n':>3} {'先読みの価値(平均)':>16} {'95%CI':>20} {'GREEDY=最善率':>12}")
    print("-"*64)
    out_rows=[]
    for r in sorted(set(p[0] for p in per)):
        vals=[p[1] for p in per if p[0]==r]
        best=[p[2] for p in per if p[0]==r]
        lo,hi=boot_ci(vals)
        print(f"{r:>5} {len(vals):>3} {np.mean(vals):>16.3f}   [{lo:>6.3f},{hi:>6.3f}]   {np.mean(best):>12.2f}")
        out_rows.append([r,len(vals),round(np.mean(vals),4),round(lo,4),round(hi,4),round(np.mean(best),3)])

    print("\n解釈: 値が小さい→先読みしても得点はほぼ変わらない=深読み不要。r=1(最終手)は~0が健全性チェック。")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out,'depth_value_curve.csv'),'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['remaining','n','value_mean','ci_lo','ci_hi','greedy_is_best_rate'])
            w.writerows(out_rows)
        if not args.no_plot:
            try:
                import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
                plt.rcParams['font.family']='MS Gothic'; plt.rcParams['axes.unicode_minus']=False
                R=[x[0] for x in out_rows]; M=[x[2] for x in out_rows]
                lo=[x[2]-x[3] for x in out_rows]; hi=[x[4]-x[2] for x in out_rows]
                fig,ax=plt.subplots(figsize=(7,5))
                ax.errorbar(R,M,yerr=[lo,hi],marker='o',ms=8,lw=2,capsize=4,color='#d62728')
                ax.axhline(0,color='0.5',ls='--',lw=1)
                ax.set_xticks(R); ax.set_xlabel('残り手数 r (=16 − shot_num)')
                ax.set_ylabel('先読みの価値  = q* − q_ref(GREEDY)  [点]')
                ax.set_title('深読みの価値 vs 終端までの距離\n(小さいほど「即時最善で十分＝深読み不要」)')
                ax.grid(alpha=0.3); fig.tight_layout()
                p=os.path.join(args.out,'depth_value_curve.png'); fig.savefig(p,dpi=150)
                print('saved',p)
            except Exception as ex: print('[warn] plot skipped:',ex)
        print('saved',os.path.join(args.out,'depth_value_curve.csv'))

if __name__=='__main__':
    main()
