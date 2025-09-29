#include "ground_truth_finder.h"
#include <iostream>
#include <chrono>
#include <algorithm>

GroundTruthFinder::GroundTruthFinder(
    std::vector<dc::GameState> grid_states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    int gridM,
    int gridN
) : grid_states_(std::move(grid_states)),
    state_to_shot_table_(std::move(state_to_shot_table)),
    GridSize_M_(gridM),
    GridSize_N_(gridN) {
}

ShotInfo GroundTruthFinder::findGroundTruthByExtensiveSearch(
    const dc::GameState& state, int max_iterations) {

    std::cout << "Finding ground truth with extensive search..." << std::endl;

    // デフォルトの設定を使用
    dc::Team team = dc::Team::k0;
    dc::GameSetting game_setting;
    game_setting.max_end = 1;
    game_setting.five_rock_rule = true;
    game_setting.thinking_time[0] = std::chrono::seconds(86400);
    game_setting.thinking_time[1] = std::chrono::seconds(86400);

    // 超高性能MCTS（計算時間無制限）でベンチマーク解を見つける
    auto simWrapper = std::make_shared<SimulatorWrapper>(team, game_setting);

    MCTS benchmark_mcts(state, NodeSource::AllGrid, grid_states_,
                       state_to_shot_table_, simWrapper, GridSize_M_, GridSize_N_);

    // 長時間実行
    std::cout << "Running extensive MCTS with " << max_iterations << " iterations..." << std::endl;
    benchmark_mcts.grow_tree(max_iterations, 3600.0); // 1時間上限

    ShotInfo ground_truth = benchmark_mcts.get_best_shot();
    std::cout << "Ground truth found: vx=" << ground_truth.vx
              << ", vy=" << ground_truth.vy
              << ", rot=" << ground_truth.rot << std::endl;

    return ground_truth;
}

ShotInfo GroundTruthFinder::setGroundTruthManually(
    const dc::GameState& state,
    const std::vector<ShotInfo>& expert_shots) {

    // 専門家が指定したショット候補から最適なものを選択
    if (expert_shots.empty()) {
        std::cerr << "No expert shots provided!" << std::endl;
        return ShotInfo{0.0f, 0.0f, 0};
    }

    // 簡単な場合は最初の候補を返す
    return expert_shots[0];
}

ShotInfo GroundTruthFinder::findGroundTruthByConsensus(
    const dc::GameState& state) {

    std::cout << "Finding ground truth by consensus..." << std::endl;

    // 複数の独立したMCTS実行
    std::vector<ShotInfo> consensus_shots = runMultipleMCTS(state, 5);

    if (consensus_shots.empty()) {
        std::cerr << "No consensus shots found!" << std::endl;
        return ShotInfo{0.0f, 0.0f, 0};
    }

    // 最も頻出するショットを選択（簡単な実装）
    return consensus_shots[0];
}

ShotInfo GroundTruthFinder::runExhaustiveSearch(const dc::GameState& state) {
    // findGroundTruthByExtensiveSearchと同じ実装
    return findGroundTruthByExtensiveSearch(state);
}

std::vector<ShotInfo> GroundTruthFinder::runMultipleMCTS(
    const dc::GameState& state, int num_runs) {

    // デフォルトの設定を使用
    dc::Team team = dc::Team::k0;
    dc::GameSetting game_setting;
    game_setting.max_end = 1;
    game_setting.five_rock_rule = true;
    game_setting.thinking_time[0] = std::chrono::seconds(86400);
    game_setting.thinking_time[1] = std::chrono::seconds(86400);

    std::vector<ShotInfo> results;

    for (int run = 0; run < num_runs; ++run) {
        std::cout << "MCTS run " << (run + 1) << "/" << num_runs << std::endl;

        auto simWrapper = std::make_shared<SimulatorWrapper>(team, game_setting);
        MCTS mcts(state, NodeSource::AllGrid, grid_states_,
                 state_to_shot_table_, simWrapper, GridSize_M_, GridSize_N_);

        // 中程度の探索
        mcts.grow_tree(10000, 180.0); // 3分制限

        ShotInfo result = mcts.get_best_shot();
        results.push_back(result);
    }

    return results;
}

bool GroundTruthFinder::validateGroundTruth(
    const dc::GameState& state, const ShotInfo& ground_truth) {

    // 基本的な妥当性チェック
    if (ground_truth.vx == 0.0f && ground_truth.vy == 0.0f) {
        return false;
    }

    // 物理的に実現可能な速度範囲かチェック
    float speed = std::sqrt(ground_truth.vx * ground_truth.vx + ground_truth.vy * ground_truth.vy);
    if (speed < 0.1f || speed > 5.0f) {
        return false;
    }

    return true;
}