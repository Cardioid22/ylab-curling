#pragma once
#ifndef _MCTS_SHARED_H_
#define _MCTS_SHARED_H_

// 深さ1・深さ3 MCTS 実験で共通に使う関数群
//   - 距離関数 (distDelta) とクラスタリング
//   - ε-greedy ロールアウト
//   - UCB1 スコア
//   - テスト盤面 CSV ローダ / サンプリング
//
// depth1_mcts_experiment / depth_n_mcts_experiment の両方から利用する

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include <vector>
#include <set>
#include <string>
#include <random>
#include <optional>

namespace dc = digitalcurling3;

namespace mcts_shared {

// ========== 定数 ==========
inline constexpr float kHouseCenterX = 0.0f;
inline constexpr float kHouseCenterY = 38.405f;
inline constexpr float kHouseRadius = 1.829f;
inline constexpr float kStoneRadius = 0.145f;

// ========== 盤面評価 ==========

// カーリングの得点ルール準拠: ハウス内でティーに最も近い石のチームが得点
// my_team 視点の連続値（正=自分有利、負=相手有利）
float evaluateEndScore(const dc::GameState& state, dc::Team my_team);

// team 0 視点の盤面スコア（距離関数内部で使う）
float evaluateBoard(const dc::GameState& state);

// 石の位置ゾーン (0:ハウス内, 1:ガード, 2:遠方)
int getZone(const std::optional<dc::Transform>& stone);

// ========== Delta距離関数 ==========

// 入力盤面との差分に注目した距離関数
// 新石位置差、既存石移動差、ゾーン差、インタラクション差、スコア差等の重み付き和
float distDelta(const dc::GameState& input,
                const dc::GameState& a,
                const dc::GameState& b);

// 候補手同士の距離行列を構築
std::vector<std::vector<float>> makeDistanceTableDelta(
    const dc::GameState& input_state,
    const std::vector<dc::GameState>& result_states);

// ========== 階層的クラスタリング (平均連結法) ==========

std::vector<std::set<int>> runClustering(
    const std::vector<std::vector<float>>& dist_table,
    int n_desired_clusters);

// 各クラスタから最小距離合計のメドイドを選出
std::vector<int> calculateMedoids(
    const std::vector<std::vector<float>>& dist_table,
    const std::vector<std::set<int>>& clusters);

// ========== ロールアウト (ε-greedy グリッドポリシー) ==========

// initial_shot を適用した後、残りショットをε-greedyで消化
// root_team 視点のエンドスコアを返す（連続値）
// rng はスレッドローカルで渡す
double rollout(
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    const dc::GameState& state,
    const ShotInfo& initial_shot,
    int remaining_shots,
    dc::Team root_team,
    std::mt19937& rng,
    double epsilon = 0.3);

// initial_shot なしで、state の時点からそのまま最後までロールアウト
// gen: 各手番で賢いロールアウト候補 (generateRolloutCandidates) を生成するのに使う
double rolloutFromState(
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    const dc::GameState& state,
    int remaining_shots,
    dc::Team root_team,
    std::mt19937& rng,
    double epsilon = 0.3);

// ========== UCB1 ==========

double ucb1Score(double mean, int visits, int total_visits, double c);

// ========== テスト盤面ローダ ==========

struct TestPositionRecord {
    int game_id;
    int end;
    int shot_num;
    dc::Team current_team;
    dc::GameState state;
};

// batch_*.csv 形式のディレクトリから局面を読み込む
// max_n <= 0 なら全件
std::vector<TestPositionRecord> loadTestPositionsFromCSV(
    const std::string& dir,
    const dc::GameSetting& game_setting,
    int max_n = -1);

// シードを使った非復元ランダムサンプリング (n 個)
std::vector<TestPositionRecord> sampleTestPositions(
    const std::vector<TestPositionRecord>& all,
    int n,
    uint64_t seed);

}  // namespace mcts_shared

#endif  // _MCTS_SHARED_H_
