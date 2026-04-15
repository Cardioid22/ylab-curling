#!/usr/bin/env python3
"""
クラスタリング有効性実験 (Python版)

AllGrid(全候補評価) vs Clustered(クラスタリング後評価) の比較
評価関数: Jiritsukun-Jr の Transformer モデル
シミュレーション: jiritsu_server 経由

使い方:
  1. jiritsu_server を起動: ./jiritsu_server.exe 0
  2. 実行: python clustering_experiment.py --sim-port 0 --jiritsu-dir /path/to/jiritsu
"""

import socket
import json
import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys
import csv
import argparse
from typing import List, Dict, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ============================================================
# 定数
# ============================================================
Y_CENTER = 38.405
X_CENTER = 0.0
HOUSE_RAD = 1.829
STONE_RAD = 0.145

# DrawPos 座標 (歩の定跡位置)
R_S = 0.9  # Small ring radius
R_L = HOUSE_RAD - 2 * STONE_RAD  # ≈1.54
SQRT2_INV = 1.0 / np.sqrt(2.0)

DRAW_POSITIONS = {
    'TEE': (X_CENTER, Y_CENTER),
    'S0':  (X_CENTER, Y_CENTER + 4 * STONE_RAD),
    'S2':  (X_CENTER + R_S, Y_CENTER),
    'S4':  (X_CENTER, Y_CENTER + 2 * STONE_RAD),
    'S6':  (X_CENTER - R_S, Y_CENTER),
    'L0':  (X_CENTER, Y_CENTER + R_L),
    'L1':  (X_CENTER + R_L * SQRT2_INV, Y_CENTER + R_L * SQRT2_INV),
    'L2':  (X_CENTER + R_L, Y_CENTER),
    'L3':  (X_CENTER + R_L * SQRT2_INV, Y_CENTER - R_L * SQRT2_INV),
    'L4':  (X_CENTER, Y_CENTER - HOUSE_RAD),
    'L5':  (X_CENTER - R_L * SQRT2_INV, Y_CENTER - R_L * SQRT2_INV),
    'L6':  (X_CENTER - R_L, Y_CENTER),
    'L7':  (X_CENTER - R_L * SQRT2_INV, Y_CENTER + R_L * SQRT2_INV),
    'G3':  (X_CENTER + 0.6, Y_CENTER - 2.17),
    'G4':  (X_CENTER, Y_CENTER - 3.0),
    'G5':  (X_CENTER - 0.6, Y_CENTER - 2.17),
}

SHEET_BOUNDS = {'x_min': -2.375, 'x_max': 2.375, 'y_min': 32.004, 'y_max': 40.234}


# ============================================================
# シミュレータクライアント
# ============================================================
class SimulatorClient:
    def __init__(self, port):
        self.port = port
        self._connect()

    def _connect(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((socket.gethostname(), 10000 + self.port))

    def simulate(self, stones_dict, target_x, target_y, rot):
        """ショットをシミュレーション。rot: 1(CCW) or -1(CW)"""
        msg = dict(stones_dict)
        msg["shot"] = [target_x, target_y, rot]
        try:
            self.s.send((json.dumps(msg) + '\n').encode('utf-8'))
            response = self.s.recv(4096)
            return json.loads(response.decode('utf-8'))
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as e:
            print(f"\n  [WARN] Connection lost, reconnecting... ({e})")
            self._connect()
            self.s.send((json.dumps(msg) + '\n').encode('utf-8'))
            response = self.s.recv(4096)
            return json.loads(response.decode('utf-8'))

    def close(self):
        self.s.close()


# ============================================================
# Transformer 評価器
# ============================================================
class TransformerEvaluator:
    def __init__(self, jiritsu_dir):
        sys.path.insert(0, jiritsu_dir)
        import net
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = net.Net(dim=256, head_num=8).to(self.device)
        checkpoint = torch.load(
            os.path.join(jiritsu_dir, 'basemodel_9.pth'),
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.src_mask = nn.Transformer.generate_square_subsequent_mask(16)[:16, 1:]

        # 勝率テーブル
        wt = [[0.959,0.939,0.919,0.771,0.609,0.340,0.162,0.034,0.015,0.010,0.005],
              [0.989,0.969,0.946,0.794,0.557,0.279,0.122,0.021,0.014,0.009,0.004],
              [1.000,0.999,0.962,0.881,0.677,0.260,0.042,0.011,0.011,0.001,0.000],
              [1.000,1.000,1.000,1.000,1.000,0.220,0.000,0.000,0.000,0.000,0.000]]
        self.wtable_1st = np.array(wt)
        self.wtable_2nd = 1 - self.wtable_1st[:, ::-1]

    def evaluate(self, sheet_data, player, remain_end, score_dist):
        """盤面を評価してスコア(低い=有利)を返す"""
        d1 = np.zeros((1, 15, 5))
        d2 = np.zeros((1, 24))
        d3 = np.zeros((1, 16))

        score_dist_oh = np.zeros(5)
        score_dist_oh[int(np.clip(score_dist, 0, 4))] = 1
        remain_end_oh = np.zeros(4)
        remain_end_oh[int(np.clip(remain_end, 0, 3))] = 1
        shot_num_oh = np.zeros(15)
        shot_num_oh[14] = 1
        d2[0] = np.concatenate([score_dist_oh, remain_end_oh, shot_num_oh])

        stone_num = 0
        for i in range(15):
            if sheet_data[i, 3] != 0:
                stone_num += 1
            else:
                break
        d1[0] = sheet_data
        d3[0] = np.append(self.src_mask[stone_num] != 0, False)

        d1_t = torch.from_numpy(d1).float().to(self.device)
        d2_t = torch.from_numpy(d2).float().to(self.device)
        d3_t = torch.from_numpy(d3).bool().to(self.device)

        with torch.no_grad():
            predict = self.model(d1_t, d2_t, d3_t).cpu().numpy()[0]

        # 勝率に基づくスコア計算
        e = 0.0
        for score_idx in range(11):
            e -= self._calc_winrate(remain_end, score_dist, score_idx - 5, 1) * predict[score_idx]
        return e

    def _calc_winrate(self, rest_end, score_dist, score, player):
        score_dist_after = int(np.clip(score_dist + 3 + score, 0, 10))
        if rest_end == 0 and score_dist_after != 5:
            return 1.0 if score_dist_after > 5 else 0.0
        if score > 0:
            return self.wtable_1st[3 - rest_end, 10 - score_dist_after]
        elif score == 0:
            if player == 1:
                return self.wtable_2nd[3 - rest_end, 10 - score_dist_after]
            else:
                return self.wtable_1st[3 - rest_end, 10 - score_dist_after]
        else:
            return self.wtable_2nd[3 - rest_end, 10 - score_dist_after]


# ============================================================
# 候補手生成 (大渡さんの genRootVMove に準拠)
# ============================================================
class CandidateGenerator:
    def generate(self, stones_on_sheet, my_team):
        """候補手を生成。各候補は (type, label, target_x, target_y, rot, target_stone_idx)"""
        candidates = []
        opp_team = 1 - my_team

        # 1. 戦略的ドロー (DrawPos: TEE + S0-S6 + L0-L7 = 13箇所)
        draw_keys = ['TEE','S0','S2','S4','S6','L0','L1','L2','L3','L4','L5','L6','L7']
        for key in draw_keys:
            tx, ty = DRAW_POSITIONS[key]
            for rot in [1, -1]:
                rot_name = "CCW" if rot == 1 else "CW"
                candidates.append({
                    'type': 'Draw', 'label': f'Draw({rot_name},{key})',
                    'target_x': tx, 'target_y': ty, 'rot': rot,
                    'target_stone_idx': -1
                })

        # 2. プレガード (G3, G4, G5)
        for key in ['G3', 'G4', 'G5']:
            tx, ty = DRAW_POSITIONS[key]
            for rot in [1, -1]:
                rot_name = "CCW" if rot == 1 else "CW"
                candidates.append({
                    'type': 'PreGuard', 'label': f'PreGuard({rot_name},{key})',
                    'target_x': tx, 'target_y': ty, 'rot': rot,
                    'target_stone_idx': -1
                })

        # 3. ヒット (相手石へ)
        for stone in stones_on_sheet:
            if stone['team'] != opp_team:
                continue
            sx, sy = stone['x'], stone['y']
            for rot in [1, -1]:
                rot_name = "CCW" if rot == 1 else "CW"
                candidates.append({
                    'type': 'Hit', 'label': f'Hit({rot_name},{stone["idx"]})',
                    'target_x': sx, 'target_y': -sy, 'rot': rot,  # 負y → ヒット速度
                    'target_stone_idx': stone['idx']
                })

        # 4. フリーズ (ハウス内相手石)
        for stone in stones_on_sheet:
            if stone['team'] != opp_team:
                continue
            dist = np.sqrt(stone['x']**2 + (stone['y'] - Y_CENTER)**2)
            if dist > HOUSE_RAD + STONE_RAD:
                continue
            sx, sy = stone['x'], stone['y']
            freeze_y = sy - 2 * STONE_RAD - 0.02
            for rot in [1, -1]:
                rot_name = "CCW" if rot == 1 else "CW"
                candidates.append({
                    'type': 'Freeze', 'label': f'Freeze({rot_name},{stone["idx"]})',
                    'target_x': sx, 'target_y': freeze_y, 'rot': rot,
                    'target_stone_idx': stone['idx']
                })

        # 5. パス
        candidates.append({
            'type': 'Pass', 'label': 'Pass',
            'target_x': 0.0, 'target_y': 0.01, 'rot': 1,
            'target_stone_idx': -1
        })

        return candidates


# ============================================================
# デルタ距離関数
# ============================================================
def get_zone(x, y):
    dist = np.sqrt(x**2 + (y - Y_CENTER)**2)
    if dist <= HOUSE_RAD:
        return 0  # ハウス内
    if y < Y_CENTER - HOUSE_RAD and y > Y_CENTER - 3 * HOUSE_RAD:
        return 1  # ガードゾーン
    return 2  # 遠方


def evaluate_board(stones_on_sheet):
    """盤面スコア (team0視点: 正=有利)"""
    in_house = []
    for s in stones_on_sheet:
        dist = np.sqrt(s['x']**2 + (s['y'] - Y_CENTER)**2)
        if dist <= HOUSE_RAD + STONE_RAD:
            in_house.append((dist, s['team']))
    if not in_house:
        return 0.0
    in_house.sort(key=lambda x: x[0])
    scoring_team = in_house[0][1]
    score = sum(1 for d, t in in_house if t == scoring_team)
    # (最初に相手チームの石が出たら止まるロジック)
    score = 0
    for d, t in in_house:
        if t == scoring_team:
            score += 1
        else:
            break
    return score if scoring_team == 0 else -score


def dist_delta(input_stones, result_a, result_b):
    """デルタ距離関数 v2"""
    MOVE_TH = 0.01
    PEN_EXIST = 30.0
    PEN_ZONE = 12.0
    NEW_W = 4.0
    MOVED_W = 2.0
    PEN_INTERACT = 15.0
    INTERACT_TH = 0.03
    SCORE_W = 20.0      # 8→20: スコア差をより重視
    PROX_W = 5.0

    # 入力石のindexセット
    input_idx = {s['idx'] for s in input_stones}

    distance = 0.0
    max_disp_a = 0.0
    max_disp_b = 0.0
    new_stone_a = None
    new_stone_b = None

    # 全indexを収集
    all_idx = set()
    for s in input_stones + result_a + result_b:
        all_idx.add(s['idx'])

    for idx in all_idx:
        in_inp = any(s['idx'] == idx for s in input_stones)
        s_a = next((s for s in result_a if s['idx'] == idx), None)
        s_b = next((s for s in result_b if s['idx'] == idx), None)
        in_a = s_a is not None
        in_b = s_b is not None

        if in_inp:
            s_inp = next(s for s in input_stones if s['idx'] == idx)
            if in_a and in_b:
                dxa = s_a['x'] - s_inp['x']
                dya = s_a['y'] - s_inp['y']
                dxb = s_b['x'] - s_inp['x']
                dyb = s_b['y'] - s_inp['y']
                ma = np.sqrt(dxa**2 + dya**2)
                mb = np.sqrt(dxb**2 + dyb**2)
                max_disp_a = max(max_disp_a, ma)
                max_disp_b = max(max_disp_b, mb)
                if ma < MOVE_TH and mb < MOVE_TH:
                    continue
                dd = np.sqrt((dxa-dxb)**2 + (dya-dyb)**2)
                distance += MOVED_W * dd
                if get_zone(s_a['x'], s_a['y']) != get_zone(s_b['x'], s_b['y']):
                    distance += PEN_ZONE
            elif in_a != in_b:
                distance += PEN_EXIST
        else:
            if in_a and in_b:
                new_stone_a = s_a
                new_stone_b = s_b
                dd = np.sqrt((s_a['x']-s_b['x'])**2 + (s_a['y']-s_b['y'])**2)
                distance += NEW_W * dd
                if get_zone(s_a['x'], s_a['y']) != get_zone(s_b['x'], s_b['y']):
                    distance += PEN_ZONE
            elif in_a != in_b:
                distance += PEN_EXIST

    if (max_disp_a > INTERACT_TH) != (max_disp_b > INTERACT_TH):
        distance += PEN_INTERACT

    # 近接度
    if new_stone_a and new_stone_b:
        def min_prox(new_s, result):
            mn = 1e9
            for s in result:
                if s['idx'] == new_s['idx']:
                    continue
                d = np.sqrt((new_s['x']-s['x'])**2 + (new_s['y']-s['y'])**2)
                mn = min(mn, d)
            return mn
        pa = min_prox(new_stone_a, result_a)
        pb = min_prox(new_stone_b, result_b)
        if pa < 100 or pb < 100:
            distance += PROX_W * abs(pa - pb)

    # スコア差
    distance += SCORE_W * abs(evaluate_board(result_a) - evaluate_board(result_b))

    # No.1石チーム差
    def closest_team(stones):
        best_d, best_t = 1e9, -1
        for s in stones:
            d = np.sqrt(s['x']**2 + (s['y'] - Y_CENTER)**2)
            if d < best_d:
                best_d, best_t = d, s['team']
        return best_t
    ta = closest_team(result_a)
    tb = closest_team(result_b)
    if ta >= 0 and tb >= 0 and ta != tb:
        distance += 25.0  # 10→25: No.1石チームが変わるのは決定的な差

    return distance


# ============================================================
# 盤面状態管理
# ============================================================
def build_stones_dict(stones_on_sheet):
    """石リスト → シミュレータ用dict"""
    # ティーからの距離でソート
    sorted_stones = sorted(stones_on_sheet,
        key=lambda s: np.sqrt(s['x']**2 + (s['y'] - Y_CENTER)**2))
    d = {"result": [0, 0]}
    for i in range(15):
        if i < len(sorted_stones):
            s = sorted_stones[i]
            team_val = 1 if s['team'] == 0 else -1  # team0→1, team1→-1
            d[f"stone{i+1}"] = [s['x'], s['y'], team_val]
        else:
            d[f"stone{i+1}"] = [0, 0, 0]
    return d, sorted_stones


def parse_result_stones(shot_result, sorted_input, current_team):
    """シミュレーション結果から石リストを再構築"""
    stones = []
    # 既存石 (stone1-14)
    for i in range(14):
        x = shot_result[f"stone{i+1}"][0]
        y = shot_result[f"stone{i+1}"][1]
        if i < len(sorted_input) and sorted_input[i]['team'] >= 0:
            team = sorted_input[i]['team']
        else:
            continue
        # 場外判定
        if x < -2.375 or x > 2.375 or y > 40.38 or y < 32.004:
            continue
        dist = np.sqrt(x**2 + (y - Y_CENTER)**2)
        if dist > HOUSE_RAD + STONE_RAD and y > Y_CENTER:
            continue
        stones.append({'x': x, 'y': y, 'team': team, 'idx': sorted_input[i]['idx']})
    # 新石 (stone16)
    s16 = shot_result.get("stone16")
    if s16 and s16[0] is not None and s16[1] is not None:
        x16 = float(s16[0])
        y16 = float(s16[1])
        if -2.375 <= x16 <= 2.375 and 32.004 <= y16 <= 40.38:
            new_idx = max((s['idx'] for s in stones), default=-1) + 1
            stones.append({'x': x16, 'y': y16, 'team': current_team, 'idx': new_idx})
    return stones


def build_sheet_data2(sorted_input, shot_result, player):
    """Transformer入力用の (15,5) 配列を構築 (jiritsu_ver3 create_sheet_data2 相当)"""
    sheet = np.zeros((15, 5))
    for i in range(14):
        sheet[i, 0] = shot_result[f"stone{i+1}"][0]
        sheet[i, 1] = shot_result[f"stone{i+1}"][1]
        sheet[i, 2] = np.sqrt((sheet[i, 0] - X_CENTER)**2 + (sheet[i, 1] - Y_CENTER)**2)
        if i >= len(sorted_input) or sorted_input[i].get('team', -1) < 0:
            sheet[i, 2] = 150
            sheet[i, 3] = 0
        else:
            t = sorted_input[i]['team']
            sheet[i, 3] = (1 if t == 0 else -1) * player
        if sheet[i, 2] < HOUSE_RAD + STONE_RAD:
            sheet[i, 4] = 1
    # stone16 (新石)
    sheet[14, 0] = shot_result["stone16"][0]
    sheet[14, 1] = shot_result["stone16"][1]
    sheet[14, 2] = np.sqrt((sheet[14, 0] - X_CENTER)**2 + (sheet[14, 1] - Y_CENTER)**2)
    sheet[14, 3] = player
    if sheet[14, 2] < HOUSE_RAD + STONE_RAD:
        sheet[14, 4] = 1
    sheet = sheet[np.argsort(sheet[:, 2])]
    for i in range(15):
        if (sheet[i, 0] < -2.375 or sheet[i, 0] > 2.375 or
            sheet[i, 1] > 40.38 or sheet[i, 1] < 32.004 or sheet[i, 2] > 100):
            sheet[i, :] = 0
        elif sheet[i, 2] > HOUSE_RAD + STONE_RAD and sheet[i, 1] > Y_CENTER:
            sheet[i, :] = 0
        else:
            sheet[i, 1] = sheet[i, 1] - Y_CENTER  # Transformer用にy座標をオフセット
    return sheet


# ============================================================
# メイン実験クラス
# ============================================================
class ClusteringExperiment:
    def __init__(self, sim_port, jiritsu_dir):
        self.sim = SimulatorClient(sim_port)
        self.evaluator = TransformerEvaluator(jiritsu_dir)
        self.generator = CandidateGenerator()

    def generate_test_positions(self, num_games):
        """gPolicy/Transformerで自己対戦してテスト盤面を収集"""
        print(f"\n=== Phase 1: Generating test positions ({num_games} games) ===")
        records = []

        for game in range(num_games):
            print(f"  Game {game}...", end='', flush=True)
            stones = []
            end_scores = [[0]*10, [0]*10]
            first_team = 0
            count = 0

            for end in range(8):
                stones = []
                next_idx = 0
                for shot in range(16):
                    current_team = first_team if shot % 2 == 0 else 1 - first_team

                    # この盤面を記録
                    records.append({
                        'game_id': game, 'end': end, 'shot_num': shot,
                        'current_team': current_team,
                        'first_team': first_team,
                        'stones': [s.copy() for s in stones],
                        'score_diff': sum(end_scores[current_team]) - sum(end_scores[1 - current_team]),
                    })
                    count += 1

                    # 簡易手選択: ドロー + ヒットから最良を選択
                    candidates = self.generator.generate(stones, current_team)
                    best_score = 1e9
                    best_cand = candidates[0]
                    for c in candidates[:10]:  # 速度のため上位10候補のみ評価
                        stones_dict, sorted_inp = build_stones_dict(stones)
                        result = self.sim.simulate(stones_dict, c['target_x'], c['target_y'], c['rot'])
                        sheet = build_sheet_data2(sorted_inp, result, 1)
                        remain_end = min(3, 7 - end)
                        sd = int(np.clip(records[-1]['score_diff'] + 2, 0, 4))
                        score = self.evaluator.evaluate(sheet, 1, remain_end, sd)
                        if score < best_score:
                            best_score = score
                            best_cand = c

                    # 最良手を実行
                    stones_dict, sorted_inp = build_stones_dict(stones)
                    result = self.sim.simulate(
                        stones_dict, best_cand['target_x'], best_cand['target_y'], best_cand['rot'])
                    stones = parse_result_stones(result, sorted_inp, current_team)
                    next_idx = max((s['idx'] for s in stones), default=-1) + 1

                # エンド終了: スコア計算
                board_score = evaluate_board(stones)
                if board_score > 0:
                    end_scores[0][end] = int(board_score)
                    first_team = 0  # 得点チームが先攻
                elif board_score < 0:
                    end_scores[1][end] = int(-board_score)
                    first_team = 1

            print(f" {count} positions")

        print(f"  Total: {len(records)} test positions")
        return records

    def evaluate_position(self, record, retention_pct):
        """1盤面について AllGrid vs Clustered を比較"""
        stones = record['stones']
        current_team = record['current_team']
        remain_end = min(3, 7 - record['end'])
        sd = int(np.clip(record['score_diff'] + 2, 0, 4))

        # 候補手生成 (Pass除外: 石がある場面でPassは非合理的)
        all_candidates = self.generator.generate(stones, current_team)
        candidates = [c for c in all_candidates if c['type'] != 'Pass']
        if not candidates:
            candidates = all_candidates  # 候補がなければPassも含める
        n = len(candidates)
        if n == 0:
            return None

        stones_dict, sorted_inp = build_stones_dict(stones)

        # 全候補をシミュレーション + Transformer評価
        t0 = time.time()
        scores = []
        result_states = []
        for c in candidates:
            sd_copy = dict(stones_dict)
            result = self.sim.simulate(sd_copy, c['target_x'], c['target_y'], c['rot'])
            result_stones = parse_result_stones(result, sorted_inp, current_team)
            result_states.append(result_stones)

            sheet = build_sheet_data2(sorted_inp, result, 1)
            score = self.evaluator.evaluate(sheet, 1, remain_end, sd)
            scores.append(score)

        allgrid_best = int(np.argmin(scores))  # 低い=有利
        allgrid_time = time.time() - t0

        # クラスタリング
        t1 = time.time()
        n_clusters = max(1, n * retention_pct // 100)

        # 距離テーブル構築
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_delta(stones, result_states[i], result_states[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # 階層的クラスタリング (平均連結法)
        if n > 1:
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='average')
            labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        else:
            labels = np.array([1])

        # メドイド計算
        medoids = []
        for c_id in range(1, n_clusters + 1):
            members = np.where(labels == c_id)[0]
            if len(members) == 0:
                continue
            if len(members) == 1:
                medoids.append(int(members[0]))
                continue
            best_m, best_sum = -1, 1e18
            for m in members:
                s = sum(dist_matrix[m, o] for o in members if o != m)
                if s < best_sum:
                    best_sum, best_m = s, int(m)
            medoids.append(best_m)

        # メドイドの中から最良手
        clustered_best = medoids[0] if medoids else 0
        clustered_best_score = scores[clustered_best]
        for m in medoids:
            if scores[m] < clustered_best_score:
                clustered_best_score = scores[m]
                clustered_best = m

        clustered_time = time.time() - t1

        # AllGrid最良手のクラスタ
        ag_cluster = int(labels[allgrid_best])
        cl_cluster = int(labels[clustered_best])

        return {
            'game_id': record['game_id'],
            'end': record['end'],
            'shot_num': record['shot_num'],
            'n_candidates': n,
            'n_clustered': n_clusters,
            'allgrid_best_label': candidates[allgrid_best]['label'],
            'allgrid_best_type': candidates[allgrid_best]['type'],
            'allgrid_best_score': scores[allgrid_best],
            'allgrid_time': allgrid_time,
            'clustered_best_label': candidates[clustered_best]['label'],
            'clustered_best_type': candidates[clustered_best]['type'],
            'clustered_best_score': scores[clustered_best],
            'clustered_time': clustered_time,
            'exact_match': allgrid_best == clustered_best,
            'same_cluster': ag_cluster == cl_cluster,
            'same_type': candidates[allgrid_best]['type'] == candidates[clustered_best]['type'],
            'score_diff': scores[allgrid_best] - scores[clustered_best],
        }

    def run(self, num_games, retention_rates):
        records = self.generate_test_positions(num_games)

        for ret in retention_rates:
            print(f"\n=== Retention={ret}% ===")
            results = []
            t_start = time.time()

            for i, rec in enumerate(records):
                r = self.evaluate_position(rec, ret)
                if r is None:
                    continue
                results.append(r)
                status = "EXACT" if r['exact_match'] else ("SameClus" if r['same_cluster'] else "DIFF")
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{len(records)}] E{rec['end']}S{rec['shot_num']} "
                          f"{status} AG={r['allgrid_best_label']} CL={r['clustered_best_label']} "
                          f"{r['allgrid_time']:.1f}s")

            total_time = time.time() - t_start
            self._print_summary(results, ret, total_time)
            self._export_csv(results, ret)

    def _print_summary(self, results, ret, total_time):
        N = len(results)
        if N == 0:
            return
        exact = sum(1 for r in results if r['exact_match'])
        same_c = sum(1 for r in results if r['same_cluster'])
        same_t = sum(1 for r in results if r['same_type'])
        avg_sd = np.mean([abs(r['score_diff']) for r in results])

        print(f"\n{'='*50}")
        print(f"  Retention={ret}% | N={N} | Time={total_time:.0f}s")
        print(f"{'='*50}")
        print(f"  Exact Match:  {exact}/{N} ({100*exact/N:.1f}%)")
        print(f"  Same Cluster: {same_c}/{N} ({100*same_c/N:.1f}%)")
        print(f"  Same Type:    {same_t}/{N} ({100*same_t/N:.1f}%)")
        print(f"  Avg|ScoreDiff|: {avg_sd:.4f}")

        print(f"\n  By Shot Number:")
        print(f"  Shot |  N | Exact% | SameClus% | SameType%")
        print(f"  -----|----|---------|-----------|---------")
        for s in range(16):
            rs = [r for r in results if r['shot_num'] == s]
            if not rs:
                continue
            n = len(rs)
            e = sum(1 for r in rs if r['exact_match'])
            sc = sum(1 for r in rs if r['same_cluster'])
            st = sum(1 for r in rs if r['same_type'])
            print(f"  {s:4d} | {n:2d} | {100*e/n:6.1f}% | {100*sc/n:8.1f}% | {100*st/n:8.1f}%")

        print(f"\n  By End Number:")
        print(f"  End  |  N | Exact% | SameClus% | SameType%")
        print(f"  -----|----|---------|-----------|---------")
        for e in range(8):
            rs = [r for r in results if r['end'] == e]
            if not rs:
                continue
            n = len(rs)
            ex = sum(1 for r in rs if r['exact_match'])
            sc = sum(1 for r in rs if r['same_cluster'])
            st = sum(1 for r in rs if r['same_type'])
            print(f"  {e:4d} | {n:2d} | {100*ex/n:6.1f}% | {100*sc/n:8.1f}% | {100*st/n:8.1f}%")

    def _export_csv(self, results, ret):
        os.makedirs('experiment_results', exist_ok=True)
        # タイムスタンプ付きファイル名（上書き防止）
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'experiment_results/clustering_python_ret{ret}_{ts}.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  Exported: {path}")


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Clustering Effectiveness Experiment (Python)')
    parser.add_argument('--sim-port', type=int, default=0, help='Simulator port offset')
    parser.add_argument('--jiritsu-dir', type=str, required=True,
                        help='Path to Jiritsukun-Jr jiritsu/ directory')
    parser.add_argument('--num-games', type=int, default=10, help='Number of self-play games')
    parser.add_argument('--retention', type=str, default='10,20', help='Retention rates (comma-separated)')
    args = parser.parse_args()

    rates = [int(x) for x in args.retention.split(',')]

    print("=== Clustering Effectiveness Experiment (Python) ===")
    print(f"  sim_port={args.sim_port}")
    print(f"  jiritsu_dir={args.jiritsu_dir}")
    print(f"  num_games={args.num_games}")
    print(f"  retention={rates}")

    exp = ClusteringExperiment(args.sim_port, args.jiritsu_dir)
    exp.run(args.num_games, rates)
    print("\n=== Done ===")


if __name__ == '__main__':
    main()
