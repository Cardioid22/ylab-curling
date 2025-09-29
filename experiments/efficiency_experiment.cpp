#include "efficiency_experiment.h"
#include <iostream>
#include <iomanip>

EfficiencyExperiment::EfficiencyExperiment(
    std::vector<dc::GameState> grid_states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    int gridM,
    int gridN
) : grid_states_(std::move(grid_states)),
    state_to_shot_table_(std::move(state_to_shot_table)),
    GridSize_M_(gridM),
    GridSize_N_(gridN) {

    initializeSimulators();
}

void EfficiencyExperiment::initializeSimulators() {
    // デフォルトの設定を使用
    dc::Team team = dc::Team::k0;
    dc::GameSetting game_setting;
    game_setting.max_end = 1;
    game_setting.five_rock_rule = true;
    game_setting.thinking_time[0] = std::chrono::seconds(86400);
    game_setting.thinking_time[1] = std::chrono::seconds(86400);

    simulator_clustered_ = std::make_shared<SimulatorWrapper>(team, game_setting);
    simulator_allgrid_ = std::make_shared<SimulatorWrapper>(team, game_setting);

    std::cout << "Simulators initialized for efficiency experiment" << std::endl;
}

EfficiencyExperiment::ExperimentResult EfficiencyExperiment::runSingleExperiment(
    const dc::GameState& test_state,
    const ShotInfo& ground_truth,
    int max_iterations,
    double max_time) {

    ExperimentResult result;
    result.test_state = test_state;
    result.ground_truth = ground_truth;

    std::cout << "\n=== Running Single Experiment ===" << std::endl;
    std::cout << "Test state: end=" << test_state.end
              << ", shot=" << test_state.shot << std::endl;
    std::cout << "Ground truth: vx=" << std::fixed << std::setprecision(6)
              << ground_truth.vx << ", vy=" << ground_truth.vy
              << ", rot=" << ground_truth.rot << std::endl;

    // クラスタリング版実行
    std::cout << "\n--- Starting Clustered MCTS ---" << std::endl;
    MCTS_WithTracking mcts_clustered(test_state, NodeSource::Clustered,
                                   grid_states_, state_to_shot_table_,
                                   simulator_clustered_, GridSize_M_, GridSize_N_);
    mcts_clustered.setGroundTruth(ground_truth);
    auto clustered_result = mcts_clustered.trackGroundTruthDiscovery(max_iterations, max_time);

    std::cout << "\n--- Starting AllGrid MCTS ---" << std::endl;
    // 全グリッド版実行
    MCTS_WithTracking mcts_allgrid(test_state, NodeSource::AllGrid,
                                 grid_states_, state_to_shot_table_,
                                 simulator_allgrid_, GridSize_M_, GridSize_N_);
    mcts_allgrid.setGroundTruth(ground_truth);
    auto allgrid_result = mcts_allgrid.trackGroundTruthDiscovery(max_iterations, max_time);

    // 結果統合
    result.clustered_discovery_iterations = clustered_result.convergence_iteration;
    result.allgrid_discovery_iterations = allgrid_result.convergence_iteration;
    result.clustered_success = clustered_result.success;
    result.allgrid_success = allgrid_result.success;
    result.clustered_final_score = clustered_result.final_score;
    result.allgrid_final_score = allgrid_result.final_score;

    if (result.clustered_success && result.allgrid_success) {
        result.efficiency_ratio = static_cast<double>(result.clustered_discovery_iterations) /
                                result.allgrid_discovery_iterations;
    } else {
        result.efficiency_ratio = -1.0; // 失敗を示す
    }

    result.clustered_score_history = clustered_result.score_history;
    result.allgrid_score_history = allgrid_result.score_history;

    std::cout << "\n=== Experiment Results ===" << std::endl;
    std::cout << "Clustered: " << (result.clustered_success ? "SUCCESS" : "FAILED")
              << " (iterations: " << result.clustered_discovery_iterations << ")" << std::endl;
    std::cout << "AllGrid: " << (result.allgrid_success ? "SUCCESS" : "FAILED")
              << " (iterations: " << result.allgrid_discovery_iterations << ")" << std::endl;
    std::cout << "Efficiency ratio: " << result.efficiency_ratio << std::endl;

    return result;
}

std::vector<EfficiencyExperiment::ExperimentResult> EfficiencyExperiment::runBatchExperiment(
    const std::vector<dc::GameState>& test_states,
    int trials_per_state) {

    std::cout << "\n=== Starting Batch Experiment ===" << std::endl;
    std::cout << "Test states: " << test_states.size() << std::endl;
    std::cout << "Trials per state: " << trials_per_state << std::endl;

    GroundTruthFinder truth_finder(grid_states_, state_to_shot_table_, GridSize_M_, GridSize_N_);
    std::vector<ExperimentResult> all_results;

    for (size_t state_idx = 0; state_idx < test_states.size(); ++state_idx) {
        const auto& state = test_states[state_idx];

        std::cout << "\n--- Processing test state " << (state_idx + 1)
                  << "/" << test_states.size() << " ---" << std::endl;

        // 正解手を決定
        std::cout << "Finding ground truth..." << std::endl;
        ShotInfo ground_truth = truth_finder.findGroundTruthByExtensiveSearch(state);

        // 各状態に対して複数回実験実行
        for (int trial = 0; trial < trials_per_state; ++trial) {
            std::cout << "\nTrial " << (trial + 1) << "/" << trials_per_state << std::endl;

            auto result = runSingleExperiment(state, ground_truth);
            all_results.push_back(result);

            // 進捗保存（5回ごと）
            if ((trial + 1) % 5 == 0) {
                std::cout << "Progress: " << (trial + 1) << "/" << trials_per_state
                          << " trials completed for state " << (state_idx + 1) << std::endl;
            }
        }
    }

    std::cout << "\n=== Batch Experiment Completed ===" << std::endl;
    std::cout << "Total experiments: " << all_results.size() << std::endl;

    return all_results;
}