#include "timing_benchmark.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <random>
#include <filesystem>

namespace dc = digitalcurling3;

TimingBenchmark::TimingBenchmark(
    dc::Team team,
    dc::GameSetting const& game_setting,
    std::shared_ptr<SimulatorWrapper> sim_wrapper,
    int gridM,
    int gridN,
    std::vector<ShotInfo> const& shot_data,
    std::unordered_map<int, ShotInfo> const& state_to_shot_table
)
    : team_(team),
      game_setting_(game_setting),
      sim_wrapper_(sim_wrapper),
      gridM_(gridM),
      gridN_(gridN),
      shot_data_(shot_data),
      state_to_shot_table_(state_to_shot_table)
{}

std::vector<dc::GameState> TimingBenchmark::createTestStates() {
    // Create test states at various game phases using ClusteringValidation
    ClusteringValidation validation(team_);
    auto test_patterns = validation.generateTestStates(1); // 1 variation per pattern

    std::vector<dc::GameState> states;
    for (const auto& ts : test_patterns) {
        states.push_back(ts.state);
    }
    return states;
}

void TimingBenchmark::benchmarkSingleSimulation(TimingResult& result, int num_trials) {
    std::cout << "\n--- Benchmarking Single Simulation ---\n";

    dc::GameState initial_state(game_setting_);
    std::vector<double> times;

    // Warm up
    for (int i = 0; i < 3; ++i) {
        sim_wrapper_->run_single_simulation(initial_state, shot_data_[0]);
    }

    // Measure
    for (int trial = 0; trial < num_trials; ++trial) {
        int shot_idx = trial % shot_data_.size();
        auto start = std::chrono::high_resolution_clock::now();
        sim_wrapper_->run_single_simulation(initial_state, shot_data_[shot_idx]);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    result.avg_single_sim_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    result.min_single_sim_time_ms = *std::min_element(times.begin(), times.end());
    result.max_single_sim_time_ms = *std::max_element(times.begin(), times.end());
    result.single_sim_count = num_trials;

    std::cout << "  Average: " << result.avg_single_sim_time_ms << " ms\n";
    std::cout << "  Min:     " << result.min_single_sim_time_ms << " ms\n";
    std::cout << "  Max:     " << result.max_single_sim_time_ms << " ms\n";
}

void TimingBenchmark::benchmarkGridPresim(TimingResult& result, int num_trials) {
    std::cout << "\n--- Benchmarking Grid Pre-simulation (" << gridM_ << "x" << gridN_ << ") ---\n";

    dc::GameState initial_state(game_setting_);
    int S = gridM_ * gridN_;
    std::vector<double> times;

    for (int trial = 0; trial < num_trials; ++trial) {
        std::vector<dc::GameState> grid_states(S);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < S; ++i) {
            grid_states[i] = sim_wrapper_->run_single_simulation(initial_state, shot_data_[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    result.grid_presim_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    result.grid_size = S;

    std::cout << "  Grid size: " << S << " positions\n";
    std::cout << "  Average time: " << result.grid_presim_time_ms << " ms\n";
}

void TimingBenchmark::benchmarkClustering(TimingResult& result, int num_trials) {
    std::cout << "\n--- Benchmarking ClusteringV2 ---\n";

    dc::GameState initial_state(game_setting_);
    int S = gridM_ * gridN_;

    // Pre-simulate grid states
    std::vector<dc::GameState> grid_states(S);
    for (int i = 0; i < S; ++i) {
        grid_states[i] = sim_wrapper_->run_single_simulation(initial_state, shot_data_[i]);
    }

    std::vector<double> times;
    int cluster_num = 4;

    for (int trial = 0; trial < num_trials; ++trial) {
        auto start = std::chrono::high_resolution_clock::now();
        ClusteringV2 algo(cluster_num, grid_states, gridM_, gridN_, team_);
        auto recommended = algo.getRecommendedStates();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    result.avg_clustering_time_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    result.clustering_count = num_trials;

    std::cout << "  Cluster num: " << cluster_num << "\n";
    std::cout << "  Average time: " << result.avg_clustering_time_ms << " ms\n";
}

void TimingBenchmark::benchmarkRollout(TimingResult& result, int num_trials) {
    std::cout << "\n--- Benchmarking Rollout (varying max_rollout_shots) ---\n";

    // Use initial game state (properly initialized)
    dc::GameState test_state(game_setting_);

    // Test rollout depth limits: -1 (full game), 16, 32, 64, 128
    std::vector<int> depth_limits = {16, 32, 64, 128, -1};

    int original_max_rollout = sim_wrapper_->max_rollout_shots;

    for (int depth : depth_limits) {
        sim_wrapper_->max_rollout_shots = depth;
        std::vector<double> times;

        for (int trial = 0; trial < num_trials; ++trial) {
            int shot_idx = trial % shot_data_.size();
            auto start = std::chrono::high_resolution_clock::now();
            sim_wrapper_->run_multiple_simulations_with_random_policy(
                test_state, shot_data_[shot_idx], 1  // 1 simulation
            );
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(ms);
        }

        double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        TimingResult::RolloutDepthResult rdr;
        rdr.max_shots = depth;
        rdr.avg_time_ms = avg;
        rdr.sample_count = num_trials;
        result.rollout_by_depth.push_back(rdr);

        std::cout << "  max_rollout_shots=" << (depth == -1 ? -1 : depth)
                  << ": avg " << avg << " ms per rollout (1 sim)\n";
    }

    // Full game rollout stats
    sim_wrapper_->max_rollout_shots = -1;
    std::vector<double> full_times;
    for (int trial = 0; trial < std::min(num_trials, 5); ++trial) {
        auto start = std::chrono::high_resolution_clock::now();
        sim_wrapper_->run_multiple_simulations_with_random_policy(
            test_state, shot_data_[0], 1
        );
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        full_times.push_back(ms);
    }
    result.avg_rollout_time_ms = std::accumulate(full_times.begin(), full_times.end(), 0.0) / full_times.size();

    // Restore
    sim_wrapper_->max_rollout_shots = original_max_rollout;
}

void TimingBenchmark::benchmarkMCTSIteration(TimingResult& result) {
    std::cout << "\n--- Benchmarking MCTS Depth-1 Iterations ---\n";

    // Tournament settings
    double total_time_s = 219.0;
    int total_shots = 80;   // 10 ends × 8 shots per team
    double per_shot_budget_s = total_time_s / total_shots;

    std::cout << "  Tournament: " << total_time_s << "s total, "
              << total_shots << " shots, "
              << per_shot_budget_s << "s per shot avg\n";

    dc::GameState initial_state(game_setting_);
    int S = gridM_ * gridN_;

    // Pre-simulate grid states
    std::vector<dc::GameState> grid_states(S);
    for (int i = 0; i < S; ++i) {
        grid_states[i] = sim_wrapper_->run_single_simulation(initial_state, shot_data_[i]);
    }

    int cluster_num = 4;
    int original_max_rollout = sim_wrapper_->max_rollout_shots;

    // Test different combinations of max_rollout_shots and num_rollout_sims
    std::vector<int> rollout_depths = {16, 32, 64, -1};
    std::vector<int> rollout_sim_counts = {1, 2, 5, 10};

    for (int max_shots : rollout_depths) {
        for (int num_sims : rollout_sim_counts) {
            sim_wrapper_->max_rollout_shots = max_shots;

            // Run MCTS depth 1 with time limit and count iterations
            int max_iter = 10000; // large number, will be time-limited
            double time_limit = 5.0; // 5 seconds max for measurement

            auto start = std::chrono::high_resolution_clock::now();

            MCTS mcts(initial_state, NodeSource::Clustered, grid_states,
                     state_to_shot_table_, sim_wrapper_, gridM_, gridN_,
                     cluster_num, num_sims);
            mcts.grow_tree(max_iter, time_limit);

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(end - start).count();

            int total_rollouts = mcts.get_total_rollout_count();
            double total_rollout_time = mcts.get_total_rollout_time();
            double total_clustering_time = mcts.get_total_clustering_time();
            int total_clustering_count = mcts.get_total_clustering_count();

            // Estimate time per iteration
            double avg_iter_time_ms = (total_rollouts > 0)
                ? (elapsed_s * 1000.0) / total_rollouts
                : 0.0;

            // Available time = per_shot_budget - grid_presim_time
            double available_time_s = per_shot_budget_s - (result.grid_presim_time_ms / 1000.0);
            int max_iters = (avg_iter_time_ms > 0)
                ? static_cast<int>(available_time_s * 1000.0 / avg_iter_time_ms)
                : 0;

            TimingResult::MCTSTimingResult mtr;
            mtr.max_rollout_shots = max_shots;
            mtr.num_rollout_sims = num_sims;
            mtr.avg_iter_time_ms = avg_iter_time_ms;
            mtr.max_iters_in_budget = max_iters;
            mtr.per_shot_budget_s = per_shot_budget_s;
            result.mcts_timing.push_back(mtr);

            std::cout << "  rollout_depth=" << (max_shots == -1 ? -1 : max_shots)
                      << ", rollout_sims=" << num_sims
                      << ": " << total_rollouts << " iters in " << std::fixed << std::setprecision(2) << elapsed_s << "s"
                      << " (avg " << std::setprecision(1) << avg_iter_time_ms << " ms/iter)"
                      << " -> max ~" << max_iters << " iters in " << std::setprecision(2) << per_shot_budget_s << "s budget"
                      << " (rollout: " << std::setprecision(1) << total_rollout_time * 1000.0 << "ms"
                      << ", clustering: " << total_clustering_time * 1000.0 << "ms"
                      << " x" << total_clustering_count << ")"
                      << "\n";
        }
    }

    sim_wrapper_->max_rollout_shots = original_max_rollout;
}

TimingResult TimingBenchmark::runBenchmark(int num_trials) {
    std::cout << "\n============================================\n";
    std::cout << "    TIMING BENCHMARK\n";
    std::cout << "    Grid: " << gridM_ << "x" << gridN_
              << " (" << gridM_ * gridN_ << " positions)\n";
    std::cout << "    Trials per test: " << num_trials << "\n";
    std::cout << "    Tournament budget: 219s / 80 shots = 2.74s per shot\n";
    std::cout << "============================================\n";

    TimingResult result;

    benchmarkSingleSimulation(result, num_trials);
    benchmarkGridPresim(result, num_trials);
    benchmarkClustering(result, num_trials);
    benchmarkRollout(result, num_trials);
    benchmarkMCTSIteration(result);

    // Summary
    std::cout << "\n============================================\n";
    std::cout << "    SUMMARY\n";
    std::cout << "============================================\n";
    std::cout << "  Single sim:     " << std::fixed << std::setprecision(2) << result.avg_single_sim_time_ms << " ms\n";
    std::cout << "  Grid presim:    " << result.grid_presim_time_ms << " ms (" << result.grid_size << " positions)\n";
    std::cout << "  Clustering:     " << result.avg_clustering_time_ms << " ms\n";
    std::cout << "  Full rollout:   " << result.avg_rollout_time_ms << " ms (1 sim, full game)\n";

    double budget_ms = 2740.0;  // 2.74s
    double overhead_ms = result.grid_presim_time_ms + result.avg_clustering_time_ms;
    double remaining_ms = budget_ms - overhead_ms;
    int max_single_sims = static_cast<int>(remaining_ms / result.avg_single_sim_time_ms);

    std::cout << "\n  Per-shot budget:       " << budget_ms << " ms\n";
    std::cout << "  Overhead (grid+clust): " << std::setprecision(1) << overhead_ms << " ms\n";
    std::cout << "  Remaining for search:  " << remaining_ms << " ms\n";
    std::cout << "  Max depth-1 sims:      " << max_single_sims << " single simulations\n";

    std::cout << "\n  Best MCTS depth-1 configs:\n";
    for (const auto& mtr : result.mcts_timing) {
        if (mtr.max_iters_in_budget >= 1) {
            std::cout << "    rollout_depth=" << (mtr.max_rollout_shots == -1 ? -1 : mtr.max_rollout_shots)
                      << " sims=" << mtr.num_rollout_sims
                      << " -> ~" << mtr.max_iters_in_budget << " iters/shot ("
                      << std::setprecision(1) << mtr.avg_iter_time_ms << " ms/iter)\n";
        }
    }

    return result;
}

void TimingBenchmark::exportResults(const TimingResult& result, const std::string& output_dir) {
    std::filesystem::create_directories(output_dir);

    // Export main timing results
    {
        std::string path = output_dir + "/timing_summary.csv";
        std::ofstream f(path);
        f << "metric,value,unit\n";
        f << "single_sim_avg," << result.avg_single_sim_time_ms << ",ms\n";
        f << "single_sim_min," << result.min_single_sim_time_ms << ",ms\n";
        f << "single_sim_max," << result.max_single_sim_time_ms << ",ms\n";
        f << "grid_presim," << result.grid_presim_time_ms << ",ms\n";
        f << "grid_size," << result.grid_size << ",positions\n";
        f << "clustering_avg," << result.avg_clustering_time_ms << ",ms\n";
        f << "full_rollout_avg," << result.avg_rollout_time_ms << ",ms\n";
        f.close();
        std::cout << "Saved: " << path << "\n";
    }

    // Export rollout depth results
    {
        std::string path = output_dir + "/rollout_by_depth.csv";
        std::ofstream f(path);
        f << "max_rollout_shots,avg_time_ms,sample_count\n";
        for (const auto& r : result.rollout_by_depth) {
            f << r.max_shots << "," << r.avg_time_ms << "," << r.sample_count << "\n";
        }
        f.close();
        std::cout << "Saved: " << path << "\n";
    }

    // Export MCTS timing results
    {
        std::string path = output_dir + "/mcts_depth1_timing.csv";
        std::ofstream f(path);
        f << "max_rollout_shots,num_rollout_sims,avg_iter_time_ms,max_iters_in_budget,per_shot_budget_s\n";
        for (const auto& m : result.mcts_timing) {
            f << m.max_rollout_shots << ","
              << m.num_rollout_sims << ","
              << m.avg_iter_time_ms << ","
              << m.max_iters_in_budget << ","
              << m.per_shot_budget_s << "\n";
        }
        f.close();
        std::cout << "Saved: " << path << "\n";
    }
}
