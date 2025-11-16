#include "agreement_experiment.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>

AgreementExperiment::AgreementExperiment(
    dc::Team team,
    dc::GameSetting game_setting,
    int grid_m,
    int grid_n,
    std::vector<dc::GameState> grid_states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    std::shared_ptr<SimulatorWrapper> simulator_clustered,
    std::shared_ptr<SimulatorWrapper> simulator_allgrid
) : team_(team),
    game_setting_(game_setting),
    grid_m_(grid_m),
    grid_n_(grid_n),
    grid_states_(grid_states),
    state_to_shot_table_(state_to_shot_table),
    simulator_clustered_(simulator_clustered),
    simulator_allgrid_(simulator_allgrid)
{
    std::cout << "[AgreementExperiment] Initialized with "
              << grid_m_ << "x" << grid_n_ << " grid ("
              << (grid_m_ * grid_n_) << " total positions)\n";
}

int AgreementExperiment::calculateFullExplorationIterations(int max_depth) {
    int grid_size = grid_m_ * grid_n_;
    int total = 0;

    for (int depth = 0; depth <= max_depth; ++depth) {
        int iterations_at_depth = std::pow(grid_size, depth);
        total += iterations_at_depth;
        std::cout << "  Depth " << depth << ": " << iterations_at_depth << " iterations\n";
    }

    std::cout << "  Total iterations for depth " << max_depth
              << ": " << total << "\n";

    return total;
}

std::vector<int> AgreementExperiment::generateClusteredIterationCounts(int depth) {
    // Clustered MCTS has log2(16) = 4, so search tree is smaller
    // Full exploration to depth 3: 1 + 4 + 16 + 64 = 85
    int clustered_full = calculateFullExplorationIterations(depth);

    // Range of iteration counts to test
    std::vector<int> counts;

    // Start with small iteration counts
    counts.push_back(clustered_full / 10);
    counts.push_back(clustered_full / 5);
    counts.push_back(clustered_full / 2);
    // Full exploration
    counts.push_back(clustered_full);
    // Larger conuts
    counts.push_back(clustered_full * 2);
    counts.push_back(clustered_full * 5);
    counts.push_back(clustered_full * 10);

    return counts;
}

MCTSRunResult AgreementExperiment::runAllGridMCTS(const dc::GameState& state, int iterations) {
    std::cout << "  [AllGrid MCTS] Running " << iterations << " iterations...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create MCTS with AllGrid node source
    MCTS mcts(
        state,
        NodeSource::AllGrid,
        grid_states_,
        state_to_shot_table_,
        simulator_allgrid_,
        grid_m_,
        grid_n_
    );

    // Run MCTS
    mcts.grow_tree(iterations, 3600.0);  // 1 hour timeout

    // Get best shot
    ShotInfo best_shot = mcts.get_best_shot();
    double win_rate = mcts.get_best_shot_winrate();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Debug: Print best shot details
    std::cout << "  [DEBUG] Best shot: vx=" << best_shot.vx
              << ", vy=" << best_shot.vy
              << ", rot=" << best_shot.rot << "\n";

    // Find which grid ID this shot corresponds to (closest match)
    int selected_grid_id = -1;
    double min_diff = 1e9;

    for (const auto& [grid_id, shot] : state_to_shot_table_) {
        double diff_vx = std::abs(shot.vx - best_shot.vx);
        double diff_vy = std::abs(shot.vy - best_shot.vy);
        double total_diff = diff_vx + diff_vy;

        if (total_diff < min_diff) {
            min_diff = total_diff;
            selected_grid_id = grid_id;
        }
    }

    if (selected_grid_id != -1) {
        const ShotInfo& closest_shot = state_to_shot_table_.at(selected_grid_id);
        std::cout << "  [DEBUG] Closest grid: " << selected_grid_id
                  << " (diff: " << std::fixed << std::setprecision(4) << min_diff << ")\n";
        std::cout << "  [DEBUG] Closest shot: vx=" << closest_shot.vx
                  << ", vy=" << closest_shot.vy
                  << ", rot=" << closest_shot.rot << "\n";
    } else {
        std::cout << "  [ERROR] No grid found in state_to_shot_table!" << "\n";
    }

    // Export MCTS details
    mcts.report_rollout_result();

    MCTSRunResult result;
    result.selected_grid_id = selected_grid_id;
    result.win_rate = win_rate;
    result.iterations = iterations;
    result.elapsed_time_sec = elapsed.count();
    result.node_source = NodeSource::AllGrid;

    std::cout << "  [AllGrid MCTS] Selected grid ID: " << selected_grid_id
              << " (win rate: " << std::fixed << std::setprecision(3) << win_rate
              << ", time: " << std::setprecision(2)
              << elapsed.count() << "s)\n";

    return result;
}

MCTSRunResult AgreementExperiment::runClusteredMCTS(const dc::GameState& state, int iterations) {
    std::cout << "    [Clustered MCTS] Running " << iterations << " iterations...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create MCTS with Clustered node source
    MCTS mcts(
        state,
        NodeSource::Clustered,
        grid_states_,
        state_to_shot_table_,
        simulator_clustered_,
        grid_m_,
        grid_n_
    );

    // Run MCTS
    mcts.grow_tree(iterations, 3600.0);  // 1 hour timeout

    // Get best shot
    ShotInfo best_shot = mcts.get_best_shot();
    double win_rate = mcts.get_best_shot_winrate();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Debug: Print best shot details
    std::cout << "    [DEBUG] Best shot: vx=" << best_shot.vx
              << ", vy=" << best_shot.vy
              << ", rot=" << best_shot.rot << "\n";

    // Find which grid ID this shot corresponds to (closest match)
    int selected_grid_id = -1;
    double min_diff = 1e9;

    for (const auto& [grid_id, shot] : state_to_shot_table_) {
        double diff_vx = std::abs(shot.vx - best_shot.vx);
        double diff_vy = std::abs(shot.vy - best_shot.vy);
        double total_diff = diff_vx + diff_vy;

        if (total_diff < min_diff) {
            min_diff = total_diff;
            selected_grid_id = grid_id;
        }
    }

    if (selected_grid_id != -1) {
        const ShotInfo& closest_shot = state_to_shot_table_.at(selected_grid_id);
        std::cout << "    [DEBUG] Closest grid: " << selected_grid_id
                  << " (diff: " << std::fixed << std::setprecision(4) << min_diff << ")\n";
        std::cout << "    [DEBUG] Closest shot: vx=" << closest_shot.vx
                  << ", vy=" << closest_shot.vy
                  << ", rot=" << closest_shot.rot << "\n";
    } else {
        std::cout << "    [ERROR] No grid found in state_to_shot_table!" << "\n";
    }

    MCTSRunResult result;
    result.selected_grid_id = selected_grid_id;
    result.win_rate = win_rate;
    result.iterations = iterations;
    result.elapsed_time_sec = elapsed.count();
    result.node_source = NodeSource::Clustered;

    std::cout << "    [Clustered MCTS] Selected grid ID: " << selected_grid_id
              << " (win rate: " << std::fixed << std::setprecision(3) << win_rate
              << ", time: " << std::setprecision(2)
              << elapsed.count() << "s)\n";

    return result;
}

AgreementResult AgreementExperiment::runSingleTest(const TestState& test_state) {
    std::cout << "\n===========================================\n";
    std::cout << "Test ID: " << test_state.test_id << "\n";
    std::cout << "Description: " << test_state.description << "\n";
    std::cout << "Shot: " << test_state.state.shot << "\n";
    std::cout << "===========================================\n";

    AgreementResult result;
    result.test_id = test_state.test_id;
    result.test_description = test_state.description;
    result.shot_number = test_state.state.shot;

    int test_depth = 3;

    // Step 1: Run AllGrid MCTS with full exploration
    std::cout << "\n[Step 1] Running AllGrid MCTS (Ground Truth)\n";
	int allgrid_iterations = calculateFullExplorationIterations(test_depth); // X-depth full exploration
    result.allgrid_result = runAllGridMCTS(test_state.state, allgrid_iterations);

    std::cout << "\n[Ground Truth] AllGrid selected: Grid ID "
              << result.allgrid_result.selected_grid_id << "\n";

    // Step 2: Run Clustered MCTS with various iteration counts
    std::cout << "\n[Step 2] Running Clustered MCTS with various iterations\n";
    result.clustered_iterations_tested = generateClusteredIterationCounts(test_depth);

    for (int iterations : result.clustered_iterations_tested) {
        MCTSRunResult clustered_res = runClusteredMCTS(test_state.state, iterations);
        result.clustered_results.push_back(clustered_res);

        // Check agreement
        bool agrees = (clustered_res.selected_grid_id == result.allgrid_result.selected_grid_id);
        result.agreement_flags.push_back(agrees);

        std::cout << "      Iterations: " << std::setw(6) << iterations
                  << " | Selected: Grid " << std::setw(2) << clustered_res.selected_grid_id
                  << " | Agreement: " << (agrees ? "YES" : "NO") << "\n";
    }

    // Calculate overall agreement rate
    result.overall_agreement_rate = calculateAgreementRate(result.agreement_flags);

    std::cout << "\n[Summary] Agreement Rate: "
              << std::fixed << std::setprecision(1)
              << result.overall_agreement_rate << "%\n";
    std::cout << "===========================================\n";

    return result;
}

double AgreementExperiment::calculateAgreementRate(const std::vector<bool>& agreement_flags) {
    if (agreement_flags.empty()) return 0.0;

    int agreements = 0;
    for (bool flag : agreement_flags) {
        if (flag) agreements++;
    }

    return (static_cast<double>(agreements) / agreement_flags.size()) * 100.0;
}

void AgreementExperiment::runExperiment(int num_test_patterns_per_type) {
    std::cout << "\n=========================================\n";
    std::cout << "AGREEMENT EXPERIMENT: Clustered vs AllGrid\n";
    std::cout << "=========================================\n";
    std::cout << "Grid size: " << grid_m_ << "x" << grid_n_ << "\n";
    std::cout << "Test patterns per type: " << num_test_patterns_per_type << "\n";
    std::cout << "=========================================\n\n";

    // Generate test states using ClusteringValidation
    ClusteringValidation validator(team_);
    std::vector<TestState> test_states = validator.generateTestStates(num_test_patterns_per_type);

    std::cout << "Generated " << test_states.size() << " test states\n\n";

    // Run experiment for each test state
    for (const auto& test_state : test_states) {
        AgreementResult result = runSingleTest(test_state);
        results_.push_back(result);
    }

    // Print final summary
    printSummary();
}

void AgreementExperiment::printSummary() {
    std::cout << "\n\n";
    std::cout << "=========================================\n";
    std::cout << "FINAL SUMMARY\n";
    std::cout << "=========================================\n";
    std::cout << "Total tests: " << results_.size() << "\n";

    // Calculate average agreement rate for each iteration count
    if (!results_.empty() && !results_[0].clustered_iterations_tested.empty()) {
        std::cout << "\nAverage Agreement Rate by Iteration Count:\n";
        std::cout << "-------------------------------------------\n";

        for (size_t i = 0; i < results_[0].clustered_iterations_tested.size(); ++i) {
            int iterations = results_[0].clustered_iterations_tested[i];
            double total_agreement = 0.0;

            for (const auto& result : results_) {
                if (i < result.agreement_flags.size() && result.agreement_flags[i]) {
                    total_agreement += 1.0;
                }
            }

            double avg_agreement = (total_agreement / results_.size()) * 100.0;

            std::cout << "  Iterations: " << std::setw(6) << iterations
                      << " | Agreement: " << std::fixed << std::setprecision(1)
                      << std::setw(5) << avg_agreement << "%\n";
        }
    }

    std::cout << "=========================================\n";
}

void AgreementExperiment::exportSummaryToFile(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    // Write summary header
    file << "=========================================\n";
    file << "FINAL SUMMARY\n";
    file << "=========================================\n";
    file << "Grid size: " << grid_m_ << "x" << grid_n_ << "\n";
    file << "Total tests: " << results_.size() << "\n";
    file << "\n";

    // Calculate and write average agreement rate for each iteration count
    if (!results_.empty() && !results_[0].clustered_iterations_tested.empty()) {
        file << "Average Agreement Rate by Iteration Count:\n";
        file << "-------------------------------------------\n";

        for (size_t i = 0; i < results_[0].clustered_iterations_tested.size(); ++i) {
            int iterations = results_[0].clustered_iterations_tested[i];
            double total_agreement = 0.0;
            int total_tests = 0;

            for (const auto& result : results_) {
                if (i < result.agreement_flags.size()) {
                    total_tests++;
                    if (result.agreement_flags[i]) {
                        total_agreement += 1.0;
                    }
                }
            }

            double avg_agreement = total_tests > 0 ? (total_agreement / total_tests) * 100.0 : 0.0;

            file << "  Iterations: " << std::setw(6) << iterations
                 << " | Agreement: " << std::fixed << std::setprecision(1)
                 << std::setw(5) << avg_agreement << "%"
                 << " (" << static_cast<int>(total_agreement) << "/" << total_tests << ")\n";
        }
    }

    file << "=========================================\n";
    file << "\n";

    // Write detailed breakdown by test case
    file << "Detailed Breakdown by Test Case:\n";
    file << "-------------------------------------------\n";

    for (const auto& result : results_) {
        file << "\nTest " << result.test_id << ": " << result.test_description << "\n";
        file << "  AllGrid (Ground Truth): Grid " << result.allgrid_result.selected_grid_id
             << " (WinRate: " << std::fixed << std::setprecision(3) << result.allgrid_result.win_rate
             << ", Time: " << std::setprecision(2) << result.allgrid_result.elapsed_time_sec << "s)\n";

        for (size_t i = 0; i < result.clustered_results.size(); ++i) {
            const auto& clustered_res = result.clustered_results[i];
            bool agrees = i < result.agreement_flags.size() ? result.agreement_flags[i] : false;

            file << "  Clustered (Iter " << std::setw(6) << clustered_res.iterations << "): Grid "
                 << std::setw(2) << clustered_res.selected_grid_id
                 << " (WinRate: " << std::fixed << std::setprecision(3) << clustered_res.win_rate
                 << ", Time: " << std::setprecision(2) << clustered_res.elapsed_time_sec << "s)"
                 << " | Agreement: " << (agrees ? "YES" : "NO") << "\n";
        }
    }

    file.close();
    std::cout << "\nSummary exported to: " << filename << "\n";
}

void AgreementExperiment::exportResultsToCSV(const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    // Header
    file << "TestID,Description,Shot,Method,Iterations,SelectedGridID,WinRate,ElapsedTime,Agreement\n";

    for (const auto& result : results_) {
        // AllGrid result
        file << result.test_id << ","
             << result.test_description << ","
             << result.shot_number << ","
             << "AllGrid,"
             << result.allgrid_result.iterations << ","
             << result.allgrid_result.selected_grid_id << ","
             << std::fixed << std::setprecision(6) << result.allgrid_result.win_rate << ","
             << result.allgrid_result.elapsed_time_sec << ","
             << "N/A\n";

        // Clustered results
        for (size_t i = 0; i < result.clustered_results.size(); ++i) {
            const auto& clustered_res = result.clustered_results[i];
            bool agrees = result.agreement_flags[i];

            file << result.test_id << ","
                 << result.test_description << ","
                 << result.shot_number << ","
                 << "Clustered,"
                 << clustered_res.iterations << ","
                 << clustered_res.selected_grid_id << ","
                 << std::fixed << std::setprecision(6) << clustered_res.win_rate << ","
                 << clustered_res.elapsed_time_sec << ","
                 << (agrees ? "YES" : "NO") << "\n";
        }
    }

    file.close();
    std::cout << "\nResults exported to: " << filename << "\n";
}
