"""
Jiritsukun-Jrのルールベース評価関数を使ったクラスタリング実験（200盤面）
evaluateBoard()の代わりにcalc_score_rulebaseを使用
"""
import sys, os, json, glob
import numpy as np

Y_CENTER = 38.405
X_CENTER = 0.0
STONE_RAD = 0.145
HOUSE_RAD = 1.829


class RulebaseEvaluator:
    """Jiritsukun-Jrのcalc_score_rulebase を移植"""

    def _build_sheet(self, result_stones, my_team=0):
        """JSON形式の石データをJiritsukun-Jr形式のsheetに変換
        sheet: (15, 7) = [x, y, dist, exists, in_house, team(+1/-1), guard_flag]
        """
        sheet = np.zeros((15, 7))
        idx = 0
        for st in result_stones:
            if idx >= 15:
                break
            x, y = st['x'], st['y']
            # 場外チェック
            if x < -2.375 or x > 2.375 or y > 40.234 or y < 32.004:
                continue
            dist = np.sqrt(x**2 + (y - Y_CENTER)**2)
            # ハウス外かつティーより奥は無視（Jiritsukun-Jrの処理）
            if dist > HOUSE_RAD + STONE_RAD and y > Y_CENTER:
                continue
            team_val = 1 if st['team'] == my_team else -1
            sheet[idx, 0] = x
            sheet[idx, 1] = y
            sheet[idx, 2] = dist
            sheet[idx, 3] = 1  # exists
            sheet[idx, 5] = team_val
            idx += 1

        for i in range(idx, 15):
            sheet[i, 2] = 150

        # ティーからの距離でソート
        sheet = sheet[np.argsort(sheet[:, 2])]

        # ハウス内フラグ
        for i in range(15):
            if sheet[i, 2] < HOUSE_RAD and sheet[i, 5] != 0:
                sheet[i, 4] = 1
            elif sheet[i, 2] >= 100:
                sheet[i, 2] = 0

        return sheet

    def calc_score(self, sheet):
        """カーリングの公式スコア（No.1チームの連続石数）"""
        score = 0
        if sheet[0, 5] == 0:
            return 0
        No1 = sheet[0, 5]
        for i in range(15):
            if sheet[i, 2] > HOUSE_RAD + STONE_RAD or sheet[i, 5] == 0:
                break
            if No1 == sheet[i, 5]:
                score += 1
            else:
                break
        return score * No1

    def evaluate(self, result_stones, player=1, shot_num=4):
        """
        Jiritsukun-Jrのcalc_score_rulebaseを移植
        result_stones: [{"team":0/1, "x":float, "y":float}, ...]
        player: 1=自分の手番, -1=相手の手番
        返り値: 自分視点のスコア（大きいほど良い）
        """
        sheet = self._build_sheet(result_stones)

        alpha = 1.83 + STONE_RAD
        beta = 1.22
        gamma = 1.83 + STONE_RAD
        a = 0.915
        b = 1.83
        w1 = 1
        w2 = 1
        mu = 0.1 if player == -1 else 0.2

        # 基本スコア
        e = float(self.calc_score(sheet))
        e *= 1

        num_enemy_stone = 0

        for i in range(14):
            if sheet[i, 5] == 0:
                break

            # ks: No.1からの相手石の数に基づく係数
            ns = 0
            for j in range(i):
                if sheet[j, 5] == -1:
                    ns += 1
                else:
                    break
            ks = 1.0 / (1 + ns)

            # hx: x方向の位置価値
            hx = 0.0
            if player == -1:
                if abs(sheet[i, 0] - X_CENTER) < alpha:
                    hx = 1 - abs(sheet[i, 0] - X_CENTER) / alpha
            else:
                if sheet[i, 2] < alpha:
                    hx = abs(sheet[i, 0] - X_CENTER) / alpha

            # hy: y方向の位置価値
            hy = 0.0
            y_shifted = -(Y_CENTER - sheet[i, 1]) + 4.88
            if 4.88 - beta < y_shifted < 4.88:
                hy = (y_shifted - 4.88 + beta)**2 / (beta**2)
            elif 4.88 < y_shifted < 4.88 + gamma:
                hy = 1 - (y_shifted - 4.88)**2 / (gamma**2)

            js = w1 * ks + w2 * hx * hy

            # ds: 味方石との密集度
            ds = 1.0
            Tmax = 0.01

            if sheet[i, 5] == -1 and sheet[i, 2] < HOUSE_RAD:
                num_enemy_stone += 1

            for j in range(15):
                if sheet[j, 5] == 0:
                    break
                if i != j and sheet[i, 5] == sheet[j, 5]:
                    d = np.linalg.norm(sheet[i, :2] - sheet[j, :2])
                    if d < 1e-6:
                        continue
                    D2 = a + b * abs(sheet[i, 1] - sheet[j, 1]) / d
                    dist_val = 1.0
                    if d < D2:
                        dist_val = 0.5 + 0.5 * d * d / (D2 * D2)
                    if dist_val < ds:
                        ds = dist_val
                    # ガード検出
                    dy = sheet[j, 1] - sheet[i, 1]
                    if dy > 8 * STONE_RAD and dy < HOUSE_RAD:
                        T = 0.0
                        dx = abs(sheet[i, 0] - sheet[j, 0])
                        if dx < 1 * STONE_RAD:
                            T = 1.0
                        elif dx < alpha:
                            T = 1.0 / (dx + 1)
                        if T > Tmax:
                            Tmax = T

            N = js / ds
            e += mu * N * sheet[i, 5] * Tmax

        if player == 1:
            e -= num_enemy_stone * 0.5

        return e


def main():
    evaluator = RulebaseEvaluator()
    print("Jiritsukun-Jr Rulebase Evaluator loaded.")

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

        # 全候補をルールベース評価
        rb_scores = []
        for c in candidates:
            score = evaluator.evaluate(c['result_stones'], player=1)
            rb_scores.append(score)

        # 歩方式: 全候補の中で最良手
        pool_best_idx = int(np.argmax(rb_scores))
        pool_best_score = rb_scores[pool_best_idx]
        pool_best = candidates[pool_best_idx]

        # DC方式: メドイドの中で最良手
        medoid_indices = [c['index'] for c in candidates if c['is_medoid']]
        medoid_scores = [(idx, rb_scores[idx]) for idx in medoid_indices]
        dc_best_idx, dc_best_score = max(medoid_scores, key=lambda x: x[1])
        dc_best = candidates[dc_best_idx]

        # 静的評価（evaluateBoard）での最良手
        static_scores = [c['score'] for c in candidates]
        static_best_idx = int(np.argmax(static_scores))

        # クラスタ判定
        pool_cluster = pool_best['cluster_id']
        dc_cluster = dc_best['cluster_id']

        exact_match = (pool_best_idx == dc_best_idx)
        same_type = (pool_best['type'] == dc_best['type'])
        same_cluster = (pool_cluster == dc_cluster)
        score_diff = dc_best_score - pool_best_score

        rb_vs_static = (pool_best_idx == static_best_idx)

        results.append({
            'state': state_name, 'n': n, 'k': k,
            'pool_best': pool_best['label'], 'pool_type': pool_best['type'],
            'pool_score': pool_best_score,
            'dc_best': dc_best['label'], 'dc_type': dc_best['type'],
            'dc_score': dc_best_score,
            'exact': exact_match, 'same_type': same_type,
            'same_cluster': same_cluster, 'score_diff': score_diff,
            'rb_vs_static': rb_vs_static,
            'static_best': candidates[static_best_idx]['label'],
        })

    # ========== サマリー ==========
    print("=" * 60)
    n_total = len(results)
    n_exact = sum(1 for r in results if r['exact'])
    n_same_type = sum(1 for r in results if r['same_type'])
    n_same_cluster = sum(1 for r in results if r['same_cluster'])
    n_rb_vs_static = sum(1 for r in results if r['rb_vs_static'])
    avg_diff = np.mean([abs(r['score_diff']) for r in results])
    max_diff = max(abs(r['score_diff']) for r in results)

    print(f"  Rulebase evaluation ({n_total} boards, retention=20%)")
    print(f"=" * 60)
    print(f"  Exact Match:    {n_exact}/{n_total} ({100*n_exact/n_total:.0f}%)")
    print(f"  Same Type:      {n_same_type}/{n_total} ({100*n_same_type/n_total:.0f}%)")
    print(f"  Same Cluster:   {n_same_cluster}/{n_total} ({100*n_same_cluster/n_total:.0f}%)")
    print(f"  Avg |ScoreDiff|: {avg_diff:.4f}")
    print(f"  Max |ScoreDiff|: {max_diff:.4f}")
    print(f"\n  Rulebase vs Static (同じ最良手を選ぶか):")
    print(f"  一致: {n_rb_vs_static}/{n_total} ({100*n_rb_vs_static/n_total:.0f}%)")

    # MISS詳細
    misses = [r for r in results if not r['same_cluster']]
    if misses:
        print(f"\n  MISS cases ({len(misses)}):")
        for r in misses[:10]:
            print(f"    {r['state']}: Pool={r['pool_best']}({r['pool_type']},score={r['pool_score']:.2f})"
                  f" DC={r['dc_best']}({r['dc_type']},score={r['dc_score']:.2f})"
                  f" diff={r['score_diff']:.4f}")
        if len(misses) > 10:
            print(f"    ... and {len(misses)-10} more")

    # 盤面ごとの詳細（最初の20件）
    print(f"\n  Detail (first 20):")
    for r in results[:20]:
        status = "EXACT" if r['exact'] else ("SameClust" if r['same_cluster'] else ("SameType" if r['same_type'] else "MISS"))
        print(f"    {r['state']:<20} Pool={r['pool_best']:<28} DC={r['dc_best']:<28} {status} diff={r['score_diff']:+.3f}")


if __name__ == '__main__':
    main()
