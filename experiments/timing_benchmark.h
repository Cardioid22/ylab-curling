#pragma once
#ifndef _TIMING_BENCHMARK_H_
#define _TIMING_BENCHMARK_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/simulator.h"
#include "../src/clustering-v2.h"
#include "../src/mcts.h"
#include "../src/structure.h"
#include "clustering_validation.h"
#include <chrono>
#include <vector>
#include <string>
#include <memory>

namespace dc = digitalcurling3;

struct TimingResult {
    // Single simulation timing
    double avg_single_sim_time_ms;
    double min_single_sim_time_ms;
    double max_single_sim_time_ms;
    int single_sim_count;

    // Clustering timing
    double avg_clustering_time_ms;
    int clustering_count;

    // Rollout timing (full game)
    double avg_rollout_time_ms;         // full game rollout (1 sim)
    int avg_rollout_shot_count;         // average shots per full rollout

    // Rollout timing with depth limits
    struct RolloutDepthResult {
        int max_shots;                  // depth limit
        double avg_time_ms;             // average time per rollout
        int sample_count;
    };
    std::vector<RolloutDepthResult> rollout_by_depth;

    // MCTS iteration timing (depth 1)
    struct MCTSTimingResult {
        int max_rollout_shots;          // rollout depth limit
        int num_rollout_sims;           // number of rollout simulations
        double avg_iter_time_ms;        // average time per MCTS iteration
        int max_iters_in_budget;        // max iterations possible in per-shot budget
        double per_shot_budget_s;       // time budget per shot
    };
    std::vector<MCTSTimingResult> mcts_timing;

    // Grid pre-simulation timing
    double grid_presim_time_ms;
    int grid_size;
};

class TimingBenchmark {
public:
    TimingBenchmark(
        dc::Team team,
        dc::GameSetting const& game_setting,
        std::shared_ptr<SimulatorWrapper> sim_wrapper,
        int gridM,
        int gridN,
        std::vector<ShotInfo> const& shot_data,
        std::unordered_map<int, ShotInfo> const& state_to_shot_table
    );

    // Run complete benchmark
    TimingResult runBenchmark(int num_trials = 20);

    // Export results
    void exportResults(const TimingResult& result, const std::string& output_dir);

private:
    dc::Team team_;
    dc::GameSetting game_setting_;
    std::shared_ptr<SimulatorWrapper> sim_wrapper_;
    int gridM_, gridN_;
    std::vector<ShotInfo> shot_data_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;

    // Benchmark individual components
    void benchmarkSingleSimulation(TimingResult& result, int num_trials);
    void benchmarkClustering(TimingResult& result, int num_trials);
    void benchmarkRollout(TimingResult& result, int num_trials);
    void benchmarkMCTSIteration(TimingResult& result);
    void benchmarkGridPresim(TimingResult& result, int num_trials);

    // Create test game states at different phases
    std::vector<dc::GameState> createTestStates();
};

#endif // _TIMING_BENCHMARK_H_
