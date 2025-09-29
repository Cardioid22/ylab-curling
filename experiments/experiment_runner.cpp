#include "experiment_runner.h"
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <chrono>

ExperimentRunner::ExperimentRunner(
    std::vector<dc::GameState> grid_states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    int gridM,
    int gridN
) : grid_states_(std::move(grid_states)),
    state_to_shot_table_(std::move(state_to_shot_table)),
    GridSize_M_(gridM),
    GridSize_N_(gridN) {

    experiment_ = std::make_unique<EfficiencyExperiment>(
        grid_states_, state_to_shot_table_, GridSize_M_, GridSize_N_);
    analyzer_ = std::make_unique<StatisticalAnalysis>();
    truth_finder_ = std::make_unique<GroundTruthFinder>(
        grid_states_, state_to_shot_table_, GridSize_M_, GridSize_N_);

    std::cout << "ExperimentRunner initialized for grid " << GridSize_M_
              << "x" << GridSize_N_ << std::endl;
}

void ExperimentRunner::runEfficiencyExperiment(const ExperimentConfig& config) {
    std::cout << "\n=== Starting Clustering Efficiency Experiment ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Max iterations: " << config.max_iterations << std::endl;
    std::cout << "  Max time: " << config.max_time << "s" << std::endl;
    std::cout << "  Trials per state: " << config.trials_per_state << std::endl;
    std::cout << "  Convergence threshold: " << config.convergence_threshold << std::endl;

    // 実験フォルダ作成
    std::string experiment_folder = createExperimentFolder();
    std::cout << "Results will be saved to: " << experiment_folder << std::endl;

    // テスト用の多様な盤面を生成
    std::vector<dc::GameState> test_states = generateTestStates();
    std::cout << "Generated " << test_states.size() << " test states" << std::endl;

    std::vector<EfficiencyExperiment::ExperimentResult> all_results;

    for (size_t state_idx = 0; state_idx < test_states.size(); ++state_idx) {
        const auto& state = test_states[state_idx];

        std::cout << "\n--- Processing test state " << (state_idx + 1)
                  << "/" << test_states.size() << " ---" << std::endl;
        std::cout << "State info: end=" << state.end << ", shot=" << state.shot << std::endl;

        // 正解手を決定
        std::cout << "Finding ground truth..." << std::endl;
        ShotInfo ground_truth = truth_finder_->findGroundTruthByExtensiveSearch(
            state, config.ground_truth_iterations);

        // 各状態に対して複数回実験実行
        for (int trial = 0; trial < config.trials_per_state; ++trial) {
            std::cout << "\nTrial " << (trial + 1) << "/" << config.trials_per_state << std::endl;

            auto result = experiment_->runSingleExperiment(
                state, ground_truth, config.max_iterations, config.max_time);
            all_results.push_back(result);

            // 進捗保存（5回ごと）
            if ((trial + 1) % 5 == 0) {
                exportIntermediateResults(all_results, state_idx, trial, experiment_folder);
            }
        }
    }

    // 統計分析
    std::cout << "\n=== Analyzing Results ===" << std::endl;
    auto stats = analyzer_->analyzeResults(all_results);

    // 結果出力
    exportAllResults(all_results, stats, experiment_folder);

    std::cout << "\n=== Experiment Completed ===" << std::endl;
    std::cout << "Results saved to: " << experiment_folder << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "  Mean efficiency ratio: " << stats.mean_efficiency_ratio << std::endl;
    std::cout << "  Success rate: " << stats.successful_experiments
              << "/" << stats.total_experiments
              << " (" << (100.0 * stats.successful_experiments / stats.total_experiments) << "%)" << std::endl;
    std::cout << "  P-value: " << stats.p_value << std::endl;

    if (stats.mean_efficiency_ratio < 1.0 && stats.p_value < 0.05) {
        std::cout << "*** SIGNIFICANT IMPROVEMENT DETECTED! ***" << std::endl;
        std::cout << "Clustering MCTS shows statistically significant efficiency gains." << std::endl;
    }
}

void ExperimentRunner::runSingleStateExperiment(
    const dc::GameState& test_state,
    const ExperimentConfig& config) {

    std::cout << "\n=== Single State Experiment ===" << std::endl;

    // 実験フォルダ作成
    std::string experiment_folder = createExperimentFolder() + "_single_state/";
    std::filesystem::create_directories(experiment_folder);

    // 正解手を決定
    std::cout << "Finding ground truth..." << std::endl;
    ShotInfo ground_truth = truth_finder_->findGroundTruthByExtensiveSearch(
        test_state, config.ground_truth_iterations);

    std::vector<EfficiencyExperiment::ExperimentResult> results;

    // 複数回実験実行
    for (int trial = 0; trial < config.trials_per_state; ++trial) {
        std::cout << "\nTrial " << (trial + 1) << "/" << config.trials_per_state << std::endl;

        auto result = experiment_->runSingleExperiment(
            test_state, ground_truth, config.max_iterations, config.max_time);
        results.push_back(result);
    }

    // 統計分析
    auto stats = analyzer_->analyzeResults(results);

    // 結果出力
    exportAllResults(results, stats, experiment_folder);

    std::cout << "\nSingle state experiment completed. Results saved to: " << experiment_folder << std::endl;
}

std::vector<dc::GameState> ExperimentRunner::generateTestStates() {
    std::vector<dc::GameState> states;

    // デフォルトのゲーム設定を使用
    dc::GameSetting game_setting;
    game_setting.max_end = 1;
    game_setting.five_rock_rule = true;
    game_setting.thinking_time[0] = std::chrono::seconds(86400);
    game_setting.thinking_time[1] = std::chrono::seconds(86400);

    // 1. ゲーム開始時の状態（空の盤面）
    dc::GameState initial_state(game_setting);
    initial_state.end = 1;
    initial_state.shot = 0;
    initial_state.hammer = dc::Team::k0;
    // stonesは全てstd::nulloptのまま（空の盤面）
    states.push_back(initial_state);

    // 2. ガードストーンありの状態（2手目）
    dc::GameState guard_state(game_setting);
    guard_state.end = 1;
    guard_state.shot = 2;
    guard_state.hammer = dc::Team::k0;
    // Team 0の最初のストーンをガード位置に配置
    guard_state.stones[0][0] = dc::Transform(dc::Vector2(0.5f, 36.5f), 0.0f);
    // Team 1の最初のストーンをガード位置に配置
    guard_state.stones[1][0] = dc::Transform(dc::Vector2(-0.3f, 36.8f), 0.0f);
    states.push_back(guard_state);

    // 3. ハウス内にストーンがある中盤状態（4手目）
    dc::GameState house_state(game_setting);
    house_state.end = 1;
    house_state.shot = 4;
    house_state.hammer = dc::Team::k0;
    // 既存のガードストーン
    house_state.stones[0][0] = dc::Transform(dc::Vector2(0.5f, 36.5f), 0.0f);
    house_state.stones[1][0] = dc::Transform(dc::Vector2(-0.3f, 36.8f), 0.0f);
    // ハウス内のストーン
    house_state.stones[0][1] = dc::Transform(dc::Vector2(0.2f, 38.6f), 0.0f);  // ハウス内
    house_state.stones[1][1] = dc::Transform(dc::Vector2(-0.4f, 38.2f), 0.0f); // ハウス内
    states.push_back(house_state);

    // 4. 複雑な中盤状態（6手目）
    dc::GameState complex_state(game_setting);
    complex_state.end = 1;
    complex_state.shot = 6;
    complex_state.hammer = dc::Team::k0;
    // 既存のストーン配置
    complex_state.stones[0][0] = dc::Transform(dc::Vector2(0.5f, 36.5f), 0.0f);
    complex_state.stones[1][0] = dc::Transform(dc::Vector2(-0.3f, 36.8f), 0.0f);
    complex_state.stones[0][1] = dc::Transform(dc::Vector2(0.2f, 38.6f), 0.0f);
    complex_state.stones[1][1] = dc::Transform(dc::Vector2(-0.4f, 38.2f), 0.0f);
    complex_state.stones[0][2] = dc::Transform(dc::Vector2(1.0f, 37.5f), 0.0f);  // サイドガード
    complex_state.stones[1][2] = dc::Transform(dc::Vector2(0.0f, 38.405f), 0.0f); // ボタン（中心）
    states.push_back(complex_state);

    // 5. 終盤の重要な局面（14手目、最後から2手）
    dc::GameState endgame_state(game_setting);
    endgame_state.end = 1;
    endgame_state.shot = 14;
    endgame_state.hammer = dc::Team::k0;
    // 複雑なハウス内配置
    endgame_state.stones[0][0] = dc::Transform(dc::Vector2(0.3f, 36.2f), 0.0f);   // ガード
    endgame_state.stones[1][0] = dc::Transform(dc::Vector2(-0.5f, 36.8f), 0.0f);  // ガード
    endgame_state.stones[0][1] = dc::Transform(dc::Vector2(0.6f, 38.8f), 0.0f);   // ハウス内
    endgame_state.stones[1][1] = dc::Transform(dc::Vector2(-0.3f, 38.5f), 0.0f);  // ハウス内
    endgame_state.stones[0][2] = dc::Transform(dc::Vector2(-0.1f, 38.405f), 0.0f); // ボタン近く
    endgame_state.stones[1][2] = dc::Transform(dc::Vector2(0.2f, 38.6f), 0.0f);   // ハウス内
    endgame_state.stones[0][3] = dc::Transform(dc::Vector2(0.8f, 37.9f), 0.0f);   // バック
    endgame_state.stones[1][3] = dc::Transform(dc::Vector2(-0.6f, 38.1f), 0.0f);  // ハウス内
    endgame_state.stones[0][4] = dc::Transform(dc::Vector2(1.2f, 37.0f), 0.0f);   // アウトサイド
    endgame_state.stones[1][4] = dc::Transform(dc::Vector2(0.4f, 38.0f), 0.0f);   // ハウス内
    endgame_state.stones[0][5] = dc::Transform(dc::Vector2(-0.8f, 37.5f), 0.0f);  // サイド
    endgame_state.stones[1][5] = dc::Transform(dc::Vector2(0.1f, 38.9f), 0.0f);   // ハウス奥
    endgame_state.stones[0][6] = dc::Transform(dc::Vector2(0.0f, 38.0f), 0.0f);   // ハウス内
    endgame_state.stones[1][6] = dc::Transform(dc::Vector2(-0.2f, 38.7f), 0.0f);  // ハウス内
    states.push_back(endgame_state);

    // 6. 特殊なタクティカル状況（カムアラウンド/テイクアウトが重要）
    dc::GameState tactical_state(game_setting);
    tactical_state.end = 1;
    tactical_state.shot = 8;
    tactical_state.hammer = dc::Team::k1; // Team 1がハンマー
    // 戦術的に興味深い配置
    tactical_state.stones[0][0] = dc::Transform(dc::Vector2(0.0f, 37.0f), 0.0f);   // センターガード
    tactical_state.stones[1][0] = dc::Transform(dc::Vector2(1.5f, 38.405f), 0.0f); // ハウス右側
    tactical_state.stones[0][1] = dc::Transform(dc::Vector2(-1.0f, 38.2f), 0.0f);  // ハウス左側
    tactical_state.stones[1][1] = dc::Transform(dc::Vector2(0.1f, 38.405f), 0.0f); // ボタン近く
    tactical_state.stones[0][2] = dc::Transform(dc::Vector2(0.5f, 39.0f), 0.0f);   // バックストーン
    tactical_state.stones[1][2] = dc::Transform(dc::Vector2(-0.3f, 38.6f), 0.0f);  // ハウス内
    tactical_state.stones[0][3] = dc::Transform(dc::Vector2(0.8f, 36.5f), 0.0f);   // サイドガード
    tactical_state.stones[1][3] = dc::Transform(dc::Vector2(-0.7f, 38.8f), 0.0f);  // ハウス奥
    states.push_back(tactical_state);

    std::cout << "Generated " << states.size() << " test states with realistic stone positions:" << std::endl;
    for (size_t i = 0; i < states.size(); ++i) {
        int stone_count = 0;
        for (size_t team = 0; team < 2; ++team) {
            for (size_t idx = 0; idx < 8; ++idx) {
                if (states[i].stones[team][idx]) stone_count++;
            }
        }
        std::cout << "  State " << i << ": Shot " << states[i].shot
                  << ", " << stone_count << " stones on ice" << std::endl;
    }

    return states;
}

void ExperimentRunner::exportIntermediateResults(
    const std::vector<EfficiencyExperiment::ExperimentResult>& results,
    size_t state_idx,
    int trial,
    const std::string& folder) {

    std::string filename = folder + "intermediate_results_state" +
                          std::to_string(state_idx) + "_trial" + std::to_string(trial) + ".csv";

    analyzer_->exportDetailedResults(results, filename);
    std::cout << "Intermediate results saved: " << filename << std::endl;
}

void ExperimentRunner::exportAllResults(
    const std::vector<EfficiencyExperiment::ExperimentResult>& results,
    const StatisticalAnalysis::StatisticalResult& stats,
    const std::string& base_folder) {

    std::string folder = base_folder.empty() ? createExperimentFolder() : base_folder;

    if (!base_folder.empty()) {
        // フォルダが既に存在している場合は追加作成しない
    } else {
        std::filesystem::create_directories(folder);
    }

    std::cout << "\nExporting results to: " << folder << std::endl;

    // 1. 統計サマリー
    analyzer_->exportStatisticalSummary(stats, folder + "statistical_summary.csv");

    // 2. 全結果の詳細データ
    analyzer_->exportDetailedResults(results, folder + "detailed_results.csv");

    // 3. 効率比の分布ヒストグラム
    analyzer_->exportEfficiencyHistogram(results, folder + "efficiency_histogram.csv");

    // 4. 学習曲線の比較
    analyzer_->exportLearningCurves(results, folder + "learning_curves.csv");

    // 5. 成功率の分析
    analyzer_->exportSuccessRateAnalysis(results, folder + "success_rate_analysis.csv");

    std::cout << "All results exported successfully!" << std::endl;
}

std::string ExperimentRunner::createExperimentFolder() {
    std::string timestamp = getCurrentTimestamp();
    std::string folder = "efficiency_experiment_" + timestamp + "/";
    std::filesystem::create_directories(folder);
    return folder;
}

std::string ExperimentRunner::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
    return ss.str();
}