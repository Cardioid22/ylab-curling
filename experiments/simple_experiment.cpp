#include "simple_experiment.h"
#include "../src/simulator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

SimpleExperiment::SimpleExperiment(
    std::vector<dc::GameState> grid_states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    int gridM,
    int gridN
) : grid_states_(std::move(grid_states)),
    state_to_shot_table_(std::move(state_to_shot_table)),
    GridSize_M_(gridM),
    GridSize_N_(gridN) {
}

ShotInfo SimpleExperiment::findIdealShot(
    const dc::GameState& state,
    int max_iterations) {
   
    std::cout << "=== Finding Ideal Shot ===" << std::endl;
    std::cout << "State: end=" << state.end << ", shot=" << state.shot << std::endl;
    std::cout << "Running benchmark MCTS with " << max_iterations << " iterations..." << std::endl;

    // Benchmark MCTS (AllGrid, long-running)
    dc::Team team = dc::Team::k0;
    dc::GameSetting game_setting;
    game_setting.max_end = 1;
    game_setting.five_rock_rule = true;

    auto simWrapper = std::make_shared<SimulatorWrapper>(team, game_setting);
    int num_rollout_simulations = 10;  // Number of simulations per rollout

    MCTS benchmark_mcts(state, NodeSource::AllGrid, grid_states_,
                       state_to_shot_table_, simWrapper, GridSize_M_, GridSize_N_, 4, num_rollout_simulations);
    benchmark_mcts.grow_tree(max_iterations, 3600.0);

    ShotInfo ideal = benchmark_mcts.get_best_shot();

    std::cout << "Ideal shot found: vx=" << ideal.vx
              << ", vy=" << ideal.vy
              << ", rot=" << ideal.rot << std::endl;

    return ideal;
}

bool SimpleExperiment::isIdealShot(const ShotInfo& shot, const ShotInfo& ideal, double tolerance) const {
    return std::abs(shot.vx - ideal.vx) < tolerance &&
           std::abs(shot.vy - ideal.vy) < tolerance &&
           shot.rot == ideal.rot;
}

ExperimentResult SimpleExperiment::runSingleComparison(
    const dc::GameState& state,
    const ShotInfo& ideal_shot,
    int max_iterations) {

    ExperimentResult result;
    result.end = state.end;
    result.shot = state.shot;
    result.ideal_shot = ideal_shot;
    result.clustered_found = false;
    result.allgrid_found = false;
    result.clustered_iterations = -1;
    result.allgrid_iterations = -1;

    std::cout << "\n=== Running Comparison Experiment ===" << std::endl;
    std::cout << "State: end=" << state.end << ", shot=" << state.shot << std::endl;
    std::cout << "Ideal shot: vx=" << ideal_shot.vx
              << ", vy=" << ideal_shot.vy
              << ", rot=" << ideal_shot.rot << std::endl;

    dc::Team team = dc::Team::k0;
    dc::GameSetting game_setting;
    game_setting.max_end = 1;
    game_setting.five_rock_rule = true;

    // Clustered MCTS
    std::cout << "\n--- Clustered MCTS ---" << std::endl;
    auto sim_clustered = std::make_shared<SimulatorWrapper>(team, game_setting);
    int num_rollout_simulations = 10;  // Number of simulations per rollout
    MCTS mcts_clustered(state, NodeSource::Clustered, grid_states_,
                       state_to_shot_table_, sim_clustered, GridSize_M_, GridSize_N_, 4, num_rollout_simulations);
    int repeat = 3;
    for (int iter = 1; iter <= repeat; ++iter) {
        mcts_clustered.grow_tree(max_iterations, 3600); 

        ShotInfo current_best = mcts_clustered.get_best_shot();

        if (isIdealShot(current_best, ideal_shot)) {
            result.clustered_iterations = iter;
            result.clustered_found = true;
            std::cout << "Found ideal shot at iteration " << iter << std::endl;
            break;
        }

        if (iter % 500 == 0) {
            std::cout << "Iteration " << iter << "/" << repeat
                      << " (best: vx=" << current_best.vx << ")" << std::endl;
        }
    }

    if (!result.clustered_found) {
        std::cout << "Ideal shot NOT found within " << max_iterations << " iterations" << std::endl;
    }

    // AllGrid MCTS
    std::cout << "\n--- AllGrid MCTS ---" << std::endl;
    auto sim_allgrid = std::make_shared<SimulatorWrapper>(team, game_setting);
    MCTS mcts_allgrid(state, NodeSource::AllGrid, grid_states_,
                     state_to_shot_table_, sim_allgrid, GridSize_M_, GridSize_N_, 4, num_rollout_simulations);

    for (int iter = 1; iter <= repeat; ++iter) {
        mcts_allgrid.grow_tree(max_iterations, 3600);  // 1 iteration at a time

        ShotInfo current_best = mcts_allgrid.get_best_shot();

        if (isIdealShot(current_best, ideal_shot)) {
            result.allgrid_iterations = iter;
            result.allgrid_found = true;
            std::cout << "Found ideal shot at iteration " << iter << std::endl;
            break;
        }

        if (iter % 500 == 0) {
            std::cout << "Iteration " << iter << "/" << repeat
                      << " (best: vx=" << current_best.vx << ")" << std::endl;
        }
    }

    if (!result.allgrid_found) {
        std::cout << "Ideal shot NOT found within " << max_iterations << " iterations" << std::endl;
    }

    // Calculate comparison result
    if (result.clustered_found && result.allgrid_found) {
        result.iteration_ratio = static_cast<double>(result.clustered_iterations) /
                                static_cast<double>(result.allgrid_iterations);

        std::cout << "\n=== Comparison Result ===" << std::endl;
        std::cout << "Clustered: " << result.clustered_iterations << " iterations" << std::endl;
        std::cout << "AllGrid: " << result.allgrid_iterations << " iterations" << std::endl;
        std::cout << "Ratio (Clustered/AllGrid): " << result.iteration_ratio << std::endl;

        if (result.iteration_ratio < 1.0) {
            double reduction = (1.0 - result.iteration_ratio) * 100.0;
            std::cout << "Clustered is " << reduction << "% more efficient" << std::endl;
        }
    } else {
        result.iteration_ratio = -1.0;
        std::cout << "\n=== Comparison Failed ===" << std::endl;
        std::cout << "One or both methods did not find the ideal shot" << std::endl;
    }

    return result;
}

std::vector<dc::GameState> SimpleExperiment::generateTestStates(int num_states) {
    std::vector<dc::GameState> states;

    dc::GameSetting game_setting;
    game_setting.max_end = 1;
    game_setting.five_rock_rule = true;

    // Initial state
    dc::GameState initial_state(game_setting);
    initial_state.end = 1;
    initial_state.shot = 0;
    initial_state.hammer = dc::Team::k0;
    states.push_back(initial_state);

    if (num_states <= 1) return states;

    // With guard stone (shot 2)
    dc::GameState state2(game_setting);
    state2.end = 1;
    state2.shot = 2;
    state2.hammer = dc::Team::k0;
    state2.stones[0][0] = dc::Transform(dc::Vector2(0.3f, 37.0f), 0.0f);
    states.push_back(state2);

    if (num_states <= 2) return states;

    // Mid-game (shot 4)
    dc::GameState state3(game_setting);
    state3.end = 1;
    state3.shot = 4;
    state3.hammer = dc::Team::k0;
    state3.stones[0][0] = dc::Transform(dc::Vector2(0.3f, 37.0f), 0.0f);
    state3.stones[1][0] = dc::Transform(dc::Vector2(-0.4f, 37.2f), 0.0f);
    state3.stones[0][1] = dc::Transform(dc::Vector2(0.2f, 38.5f), 0.0f);
    states.push_back(state3);

    if (num_states <= 3) return states;

    // Complex mid-game (shot 6)
    dc::GameState state4(game_setting);
    state4.end = 1;
    state4.shot = 6;
    state4.hammer = dc::Team::k0;
    state4.stones[0][0] = dc::Transform(dc::Vector2(0.3f, 37.0f), 0.0f);
    state4.stones[1][0] = dc::Transform(dc::Vector2(-0.4f, 37.2f), 0.0f);
    state4.stones[0][1] = dc::Transform(dc::Vector2(0.2f, 38.5f), 0.0f);
    state4.stones[1][1] = dc::Transform(dc::Vector2(-0.3f, 38.3f), 0.0f);
    state4.stones[0][2] = dc::Transform(dc::Vector2(0.8f, 37.5f), 0.0f);
    states.push_back(state4);

    if (num_states <= 4) return states;

    // End-game (shot 8)
    dc::GameState state5(game_setting);
    state5.end = 1;
    state5.shot = 8;
    state5.hammer = dc::Team::k0;
    state5.stones[0][0] = dc::Transform(dc::Vector2(0.3f, 37.0f), 0.0f);
    state5.stones[1][0] = dc::Transform(dc::Vector2(-0.4f, 37.2f), 0.0f);
    state5.stones[0][1] = dc::Transform(dc::Vector2(0.2f, 38.5f), 0.0f);
    state5.stones[1][1] = dc::Transform(dc::Vector2(-0.3f, 38.3f), 0.0f);
    state5.stones[0][2] = dc::Transform(dc::Vector2(0.8f, 37.5f), 0.0f);
    state5.stones[1][2] = dc::Transform(dc::Vector2(0.0f, 38.405f), 0.0f);
    state5.stones[0][3] = dc::Transform(dc::Vector2(1.0f, 37.8f), 0.0f);
    states.push_back(state5);

    return states;
}

std::vector<ExperimentResult> SimpleExperiment::runBatchExperiment(
    int num_states,
    int max_iterations) {

    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Batch Experiment" << std::endl;
    std::cout << "Number of test states: " << num_states << std::endl;
    std::cout << "Max iterations per test: " << max_iterations << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::vector<dc::GameState> test_states = generateTestStates(num_states);
    std::vector<ExperimentResult> results;

    for (size_t i = 0; i < test_states.size(); ++i) {
        std::cout << "\n\n******** Test State " << (i + 1) << "/" << test_states.size()
                  << " ********" << std::endl;

        // Find ideal shot
        ShotInfo ideal = findIdealShot(test_states[i], 100000);

        // Run comparison experiment
        ExperimentResult result = runSingleComparison(test_states[i], ideal, max_iterations);
        results.push_back(result);
    }

    return results;
}

void SimpleExperiment::saveResults(
    const std::vector<ExperimentResult>& results,
    const std::string& filename) {

    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Header
    file << "End,Shot,Ideal_vx,Ideal_vy,Ideal_rot,"
         << "Clustered_Iterations,Clustered_Found,"
         << "AllGrid_Iterations,AllGrid_Found,"
         << "Iteration_Ratio\n";

    // Data
    for (const auto& result : results) {
        file << result.end << ","
             << result.shot << ","
             << std::fixed << std::setprecision(6)
             << result.ideal_shot.vx << ","
             << result.ideal_shot.vy << ","
             << result.ideal_shot.rot << ","
             << result.clustered_iterations << ","
             << (result.clustered_found ? 1 : 0) << ","
             << result.allgrid_iterations << ","
             << (result.allgrid_found ? 1 : 0) << ","
             << result.iteration_ratio << "\n";
    }

    file.close();

    std::cout << "\n\nResults saved to: " << filename << std::endl;

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Experiment Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    int total = results.size();
    int both_found = 0;
    double total_ratio = 0.0;

    for (const auto& result : results) {
        if (result.clustered_found && result.allgrid_found) {
            both_found++;
            total_ratio += result.iteration_ratio;
        }
    }

    std::cout << "Total tests: " << total << std::endl;
    std::cout << "Both methods found ideal shot: " << both_found << "/" << total << std::endl;

    if (both_found > 0) {
        double avg_ratio = total_ratio / both_found;
        std::cout << "\nAverage iteration ratio (Clustered/AllGrid): " << avg_ratio << std::endl;

        if (avg_ratio < 1.0) {
            double reduction = (1.0 - avg_ratio) * 100.0;
            std::cout << "Average reduction: " << reduction << "%" << std::endl;
            std::cout << "\n*** Clustering is MORE efficient ***" << std::endl;
        } else {
            double increase = (avg_ratio - 1.0) * 100.0;
            std::cout << "Average increase: " << increase << "%" << std::endl;
            std::cout << "\n*** Clustering is LESS efficient ***" << std::endl;
        }
    } else {
        std::cout << "\nNo valid comparisons (ideal shot not found by both methods)" << std::endl;
    }

    std::cout << "========================================" << std::endl;
}
