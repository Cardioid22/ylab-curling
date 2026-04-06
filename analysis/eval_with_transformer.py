"""
Jiritsukun-JrのTransformerモデルを評価関数として使うテスト
盤面を入力 → エンド終了時のスコア分布を予測 → 期待スコアを返す
"""
import sys
import os
import json
import numpy as np
import torch

# Jiritsukun-Jrのnet.pyを読み込む
JIRITSU_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Jiritsukun-Jr_GAT2025', 'jiritsu')
sys.path.insert(0, JIRITSU_DIR)
import net

device = torch.device('cpu')
Y_CENTER = 38.405
STONE_RAD = 0.145
HOUSE_RAD = 1.829


class TransformerEvaluator:
    def __init__(self):
        self.model = net.Net(dim=256, head_num=8).to(device)
        checkpoint = torch.load(os.path.join(JIRITSU_DIR, 'basemodel_9.pth'), map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.src_mask = torch.nn.Transformer.generate_square_subsequent_mask(16)[:16, 1:]

        # 勝率テーブル（Jiritsukun-Jrから）
        self.wtable = [
            [0.919, 0.771, 0.609, 0.340, 0.162, 0.034, 0.015],
            [0.946, 0.794, 0.557, 0.279, 0.122, 0.021, 0.014],
            [0.962, 0.881, 0.677, 0.260, 0.042, 0.011, 0.011],
            [1.000, 1.000, 1.000, 0.220, 0.000, 0.000, 0.000]
        ]

    def board_to_features(self, stones, score_dist=2, remain_end=3, shot_num=15):
        """盤面をTransformerの入力特徴量に変換
        stones: [(team, x, y), ...] team: 0=my, 1=opp
        Jiritsukun-Jrのcreate_sheet_data + calc_score_modelbaseと同じ変換
        """
        # sheet: [x, y, dist, exists, in_house, team, guard_flag]
        sheet = np.zeros((15, 7))
        idx = 0
        for team, x, y in stones:
            if idx >= 15:
                break
            dist = np.sqrt(x**2 + (y - Y_CENTER)**2)
            sheet[idx, 0] = x
            sheet[idx, 1] = y
            sheet[idx, 2] = dist
            sheet[idx, 3] = 1  # 存在フラグ
            sheet[idx, 5] = 1 if team == 0 else -1  # 1=my, -1=opp
            idx += 1

        # 存在しない石は距離を大きく
        for i in range(idx, 15):
            sheet[i, 2] = 150

        # ティーからの距離でソート
        sheet = sheet[np.argsort(sheet[:, 2])]

        # ハウス内フラグ + 距離リセット（Jiritsukun-Jrと同じ処理）
        for i in range(15):
            if sheet[i, 2] < HOUSE_RAD and sheet[i, 5] != 0:
                sheet[i, 4] = 1
            elif sheet[i, 2] >= 100:
                sheet[i, 2] = 0

        # guard flag (column 6): Jiritsukun-Jrのcreate_sheet_dataのfull=1モードと同じ
        flag = False
        if sheet[0, 5] != 0:
            player = sheet[0, 5]
            for i in range(1, 15):
                if sheet[i, 5] == 0:
                    break
                if player != sheet[i, 5]:
                    flag = True
                if flag:
                    sheet[i, 6] = 1

        # 特徴量構築（d1: 石情報の最初5列 = [x, y, dist, exists, in_house]）
        d1 = np.zeros((1, 15, 5))
        d2 = np.zeros((1, 24))
        d3 = np.zeros((1, 16))

        score_dist_clamped = min(max(score_dist, 0), 4)
        remain_end_clamped = min(max(remain_end, 0), 3)

        score_dist_oh = np.zeros(5)
        score_dist_oh[score_dist_clamped] = 1
        remain_end_oh = np.zeros(4)
        remain_end_oh[remain_end_clamped] = 1
        # shot_num: Jiritsukun-Jrではshot_num_onehot[14]=1とハードコード
        shot_num_oh = np.zeros(15)
        shot_num_oh[min(shot_num, 14)] = 1

        d2[0] = np.concatenate([score_dist_oh, remain_end_oh, shot_num_oh])
        d1[0] = sheet[:, :5]  # [x, y, dist, exists, in_house]

        stone_num = 0
        for i in range(15):
            if sheet[i, 3] != 0:
                stone_num += 1
            else:
                break
        d3[0] = np.append(self.src_mask[stone_num] != 0, False)

        return d1, d2, d3

    def evaluate(self, stones, score_dist=2, remain_end=3, shot_num=15):
        """盤面の期待スコアを返す（my_team視点）"""
        d1, d2, d3 = self.board_to_features(stones, score_dist, remain_end, shot_num)
        d1 = torch.from_numpy(d1).float().to(device)
        d2 = torch.from_numpy(d2).float().to(device)
        d3 = torch.from_numpy(d3).bool().to(device)

        with torch.no_grad():
            predict = self.model(d1, d2, d3).cpu().numpy()[0]

        # スコア分布 (-5〜+5) → 期待スコア
        scores = np.arange(-5, 6)  # [-5, -4, ..., 4, 5]
        expected_score = np.sum(scores * predict)
        return expected_score, predict

    def evaluate_with_winrate(self, stones, score_dist=2, remain_end=3, shot_num=15):
        """盤面の期待勝率を返す"""
        d1, d2, d3 = self.board_to_features(stones, score_dist, remain_end, shot_num)
        d1 = torch.from_numpy(d1).float().to(device)
        d2 = torch.from_numpy(d2).float().to(device)
        d3 = torch.from_numpy(d3).bool().to(device)

        with torch.no_grad():
            predict = self.model(d1, d2, d3).cpu().numpy()[0]

        # 勝率計算
        winrate = 0.0
        for score_idx in range(11):
            score = score_idx - 5
            wr = self.calc_winrate(remain_end, score_dist, score, 1)
            winrate += wr * predict[score_idx]
        return winrate, predict

    def calc_winrate(self, rest_end, score_dist, score, player):
        score_dist_after = int(np.clip(score_dist + 1 + score, 0, 6))
        if rest_end == 0:
            if score_dist_after > 3:
                return 1.0
            elif score_dist_after == 3:
                return 0.22 if score > 0 else 0.78
            else:
                return 0.0
        else:
            wt = np.array(self.wtable)
            if player == 1:
                wt = 1 - wt[:, ::-1]
            if score > 0:
                wt = 1 - wt[:, ::-1]
                return wt[3 - rest_end, 6 - score_dist_after]
            else:
                return wt[3 - rest_end, 6 - score_dist_after]


# ========== テスト ==========
if __name__ == '__main__':
    evaluator = TransformerEvaluator()
    print("Transformer model loaded successfully!")

    # テスト盤面: opp2_my1（自分1石+相手2石）
    stones = [
        (0, -0.5, Y_CENTER + 0.3),   # my stone
        (1, 0.3, Y_CENTER - 0.2),    # opp stone
        (1, -0.1, Y_CENTER + 0.8),   # opp stone
    ]

    expected_score, dist = evaluator.evaluate(stones, score_dist=2, remain_end=3, shot_num=4)
    print(f"\nBoard: opp2_my1")
    print(f"Expected score: {expected_score:.3f}")
    print(f"Score distribution: {dict(zip(range(-5,6), [f'{p:.3f}' for p in dist]))}")

    winrate, _ = evaluator.evaluate_with_winrate(stones, score_dist=2, remain_end=3, shot_num=4)
    print(f"Win rate: {winrate:.3f}")

    # detail JSONがあれば候補手ごとの評価も行う
    json_path = os.path.join(os.path.dirname(__file__),
                             '..', 'build', 'Release', 'experiments',
                             'pool_clustering_results', 'detail_opp2_my1_r20.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)

        print(f"\n=== Evaluating {len(data['candidates'])} candidates with Transformer ===")
        results = []
        for c in data['candidates']:
            stones_after = [(s['team'], s['x'], s['y']) for s in c['result_stones']]
            exp_score, _ = evaluator.evaluate(stones_after, score_dist=2, remain_end=3, shot_num=5)
            results.append((c['index'], c['label'], c['type'], c['cluster_id'], exp_score))

        # ソートして上位10件
        results.sort(key=lambda x: -x[4])
        print(f"\nTop 10 candidates by Transformer expected score:")
        for idx, label, typ, cid, score in results[:10]:
            print(f"  [{idx}] {label:<30} type={typ:<6} cluster={cid:>2} score={score:+.3f}")

        # Pool bestとDC bestの比較
        pool_best = max(results, key=lambda x: x[4])
        dc_medoids = [c for c in data['candidates'] if c['is_medoid']]
        dc_results = [(r[0], r[1], r[2], r[3], r[4]) for r in results if any(m['index'] == r[0] for m in dc_medoids)]
        dc_best = max(dc_results, key=lambda x: x[4]) if dc_results else None

        print(f"\nPool best (Transformer): [{pool_best[0]}] {pool_best[1]} (score={pool_best[4]:+.3f})")
        if dc_best:
            print(f"DC best (Transformer):   [{dc_best[0]}] {dc_best[1]} (score={dc_best[4]:+.3f})")
            print(f"Same shot: {pool_best[0] == dc_best[0]}")
            print(f"Same cluster: {pool_best[3] == dc_best[3]}")
