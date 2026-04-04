#pragma once
#ifndef _DEPTH1_MCTS_EXPERIMENT_H_
#define _DEPTH1_MCTS_EXPERIMENT_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include <vector>
#include <set>
#include <string>

namespace dc = digitalcurling3;

// 深さ1フラットMC実験
// 歩(Ayumu)のじりつくん方式に近い設計:
//   - 木なし（ルート直下のみ）
//   - UCB1で有望な手を適応的に再サンプリング
//   - 評価は連続値（スコア差、二値化しない）
class Depth1MctsExperiment {
public:
    Depth1MctsExperiment(dc::GameSetting const& game_setting);
    void run();

private:
    dc::GameSetting game_setting_;
    std::vector<std::string> test_state_names_;

    // ロールアウト用（ShotGeneratorポリシー）
    std::unique_ptr<ShotGenerator> rollout_generator_;
    std::vector<Position> rollout_grid_;

    static constexpr float kHouseCenterX = 0.0f;
    static constexpr float kHouseCenterY = 38.405f;
    static constexpr float kHouseRadius = 1.829f;

    // UCB1の腕（= 1つの候補手）
    struct Arm {
        int candidate_idx;
        std::string label;
        std::string type;
        ShotInfo shot;
        double total_reward = 0.0;
        double total_reward_sq = 0.0;  // 分散計算用
        int visits = 0;

        double mean() const { return visits > 0 ? total_reward / visits : 0.0; }
        double variance() const {
            if (visits < 2) return 0.0;
            double m = mean();
            return total_reward_sq / visits - m * m;
        }
    };

    // テスト盤面生成
    std::vector<dc::GameState> createTestStates();

    // ロールアウト: 候補手を打った後、ランダムポリシーで1エンド終了まで
    double rollout(SimulatorWrapper& sim, const dc::GameState& state,
                   const ShotInfo& shot, int remaining_shots);

    // 盤面のエンドスコア評価（ハウス内の石の配置から得点を計算、連続値）
    float evaluateEndScore(const dc::GameState& state, dc::Team my_team) const;

    // UCB1スコア
    double ucb1Score(const Arm& arm, int total_visits, double c = 1.41) const;

    // フラットMCの実行: 指定された腕に対してbudget回のロールアウトを配分
    int runFlatMC(std::vector<Arm>& arms, SimulatorWrapper& sim,
                  const dc::GameState& state, int budget, int remaining_shots);

    // Delta距離関数（pool_clustering_experimentと同じ）
    int getZone(const std::optional<dc::Transform>& stone) const;
    float evaluateBoard(const dc::GameState& state) const;
    float distDelta(const dc::GameState& input, const dc::GameState& a, const dc::GameState& b) const;
    std::vector<std::vector<float>> makeDistanceTableDelta(
        const dc::GameState& input_state,
        const std::vector<dc::GameState>& result_states);
    std::vector<std::set<int>> runClustering(
        const std::vector<std::vector<float>>& dist_table, int n_desired_clusters);
    std::vector<int> calculateMedoids(
        const std::vector<std::vector<float>>& dist_table,
        const std::vector<std::set<int>>& clusters);
};

#endif
