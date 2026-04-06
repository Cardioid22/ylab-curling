"""
Transformer評価関数を使ったクラスタリング実験
20盤面に対して: 全候補をTransformerで評価した最良手 vs クラスタリングで絞った候補の最良手
Exact Match / Same Type / Same Cluster / ScoreDiff を比較
"""
import sys, os, json, glob
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Jiritsukun-Jr_GAT2025', 'jiritsu'))
import net

Y_CENTER = 38.405
HOUSE_RAD = 1.829
device = torch.device('cpu')


class TransformerEvaluator:
    def __init__(self):
        jiritsu_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Jiritsukun-Jr_GAT2025', 'jiritsu')
        self.model = net.Net_transformer(dim=256, head_num=8).to(device)
        self.model.load_state_dict(torch.load(
            os.path.join(jiritsu_dir, 'model_transformer_15.pth'), map_location=device))
        self.model.eval()
        self.src_mask = nn.Transformer.generate_square_subsequent_mask(16)[:16, 1:]

    def evaluate(self, result_stones, score_dist=2, remain_end=3):
        """result_stones: [{"team":0/1, "x":float, "y":float}, ...]"""
        sheet = np.zeros((15, 5))
        idx = 0
        for st in result_stones:
            if idx >= 15:
                break
            x, y = st['x'], st['y']
            # 場外チェック
            if x < -2.375 or x > 2.375 or y > 40.234 or y < 32.004:
                continue
            dist = np.sqrt(x**2 + (y - Y_CENTER)**2)
            team_val = 1 if st['team'] == 0 else -1
            sheet[idx, 0] = x
            sheet[idx, 1] = y
            sheet[idx, 2] = dist
            sheet[idx, 3] = 1
            sheet[idx, 4] = 1 if dist < HOUSE_RAD else 0
            idx += 1

        for i in range(idx, 15):
            sheet[i, 2] = 150

        sheet = sheet[np.argsort(sheet[:, 2])]

        for i in range(15):
            if sheet[i, 2] >= 100:
                sheet[i, 2] = 0

        stone_num = 0
        for i in range(15):
            if sheet[i, 3] != 0:
                stone_num += 1
            else:
                break

        d1 = np.zeros((1, 15, 5))
        d1[0] = sheet
        # d2: 9次元 = score_dist(5) + remain_end(4)
        d2 = np.zeros((1, 9))
        d2[0, min(max(score_dist, 0), 4)] = 1
        d2[0, 5 + min(max(remain_end, 0), 3)] = 1
        d3 = np.zeros((1, 16))
        d3[0] = np.append(self.src_mask[stone_num] != 0, False)

        d1t = torch.from_numpy(d1).float().to(device)
        d2t = torch.from_numpy(d2).float().to(device)
        d3t = torch.from_numpy(d3).bool().to(device)

        with torch.no_grad():
            predict = self.model(d1t, d2t, d3t).cpu().numpy()[0]

        scores = np.arange(-5, 6)
        expected_score = float(np.sum(scores * predict))
        return expected_score


def main():
    evaluator = TransformerEvaluator()
    print("Transformer (Net_transformer + model_transformer_15.pth) loaded.")

    json_dir = os.path.join(os.path.dirname(__file__),
                            '..', 'build', 'Release', 'experiments', 'pool_clustering_results')
    json_files = sorted(glob.glob(os.path.join(json_dir, 'detail_*_r20.json')))
    print(f"Found {len(json_files)} detail JSON files.\n")

    results = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        state_name = data['state_name']
        candidates = data['candidates']
        k = data['k']
        n = data['n_candidates']

        # 全候補をTransformerで評価
        transformer_scores = []
        for c in candidates:
            score = evaluator.evaluate(c['result_stones'])
            transformer_scores.append(score)

        # 歩方式: 全候補の中で最良手
        pool_best_idx = int(np.argmax(transformer_scores))
        pool_best_score = transformer_scores[pool_best_idx]
        pool_best = candidates[pool_best_idx]

        # DC方式: メドイドの中で最良手
        medoid_indices = [c['index'] for c in candidates if c['is_medoid']]
        medoid_scores = [(idx, transformer_scores[idx]) for idx in medoid_indices]
        dc_best_idx, dc_best_score = max(medoid_scores, key=lambda x: x[1])
        dc_best = candidates[dc_best_idx]

        # 静的評価（evaluateBoard）での最良手（JSONのscoreフィールド）
        static_scores = [c['score'] for c in candidates]
        static_best_idx = int(np.argmax(static_scores))

        # クラスタ判定
        pool_cluster = pool_best['cluster_id']
        dc_cluster = dc_best['cluster_id']

        exact_match = (pool_best_idx == dc_best_idx)
        same_type = (pool_best['type'] == dc_best['type'])
        same_cluster = (pool_cluster == dc_cluster)
        score_diff = dc_best_score - pool_best_score

        # Transformer vs Static の最良手一致
        transformer_vs_static = (pool_best_idx == static_best_idx)

        results.append({
            'state': state_name,
            'n': n,
            'k': k,
            'pool_best': pool_best['label'],
            'pool_best_type': pool_best['type'],
            'pool_best_tscore': pool_best_score,
            'dc_best': dc_best['label'],
            'dc_best_type': dc_best['type'],
            'dc_best_tscore': dc_best_score,
            'exact_match': exact_match,
            'same_type': same_type,
            'same_cluster': same_cluster,
            'score_diff': score_diff,
            'transformer_vs_static': transformer_vs_static,
            'static_best': candidates[static_best_idx]['label'],
        })

        status = "EXACT" if exact_match else ("SameCluster" if same_cluster else ("SameType" if same_type else "MISS"))
        print(f"[{state_name}] N={n} K={k}")
        print(f"  Pool(Transformer): {pool_best['label']} ({pool_best['type']}, tscore={pool_best_score:+.3f})")
        print(f"  DC(Transformer):   {dc_best['label']} ({dc_best['type']}, tscore={dc_best_score:+.3f}) → {status}")
        print(f"  Static best:       {candidates[static_best_idx]['label']} (static_score={static_scores[static_best_idx]:.0f})"
              f" {'= Transformer' if transformer_vs_static else '≠ Transformer'}")
        print()

    # ========== サマリー ==========
    print("=" * 60)
    print("  Summary (20 boards, retention=20%)")
    print("=" * 60)

    n_total = len(results)
    n_exact = sum(1 for r in results if r['exact_match'])
    n_same_type = sum(1 for r in results if r['same_type'])
    n_same_cluster = sum(1 for r in results if r['same_cluster'])
    n_trans_vs_static = sum(1 for r in results if r['transformer_vs_static'])
    avg_diff = np.mean([abs(r['score_diff']) for r in results])

    print(f"\n  Transformer evaluation (Pool vs DeltaClustered):")
    print(f"    Exact Match:  {n_exact}/{n_total} ({100*n_exact/n_total:.0f}%)")
    print(f"    Same Type:    {n_same_type}/{n_total} ({100*n_same_type/n_total:.0f}%)")
    print(f"    Same Cluster: {n_same_cluster}/{n_total} ({100*n_same_cluster/n_total:.0f}%)")
    print(f"    Avg |ScoreDiff|: {avg_diff:.4f}")

    print(f"\n  Transformer vs Static (同じ最良手を選ぶか):")
    print(f"    一致: {n_trans_vs_static}/{n_total} ({100*n_trans_vs_static/n_total:.0f}%)")

    # MISS cases detail
    misses = [r for r in results if not r['same_cluster']]
    if misses:
        print(f"\n  MISS cases ({len(misses)}):")
        for r in misses:
            print(f"    {r['state']}: Pool={r['pool_best']}({r['pool_best_type']}) DC={r['dc_best']}({r['dc_best_type']})")


if __name__ == '__main__':
    main()
