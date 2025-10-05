#pragma once
#ifndef _SIMPLE_EXPERIMENT_H_
#define _SIMPLE_EXPERIMENT_H_

#include "../src/structure.h"
#include "../src/mcts.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include <vector>
#include <string>

namespace dc = digitalcurling3;

// Simple experiment result structure
struct ExperimentResult {
    // Board state info
    int end;
    int shot;

    // Ideal shot (benchmark)
    ShotInfo ideal_shot;

    // Clustered MCTS results
    int clustered_iterations;      // Iterations to reach ideal shot
    bool clustered_found;          // Found or not

    // AllGrid MCTS results
    int allgrid_iterations;        // Iterations to reach ideal shot
    bool allgrid_found;            // Found or not

    // Comparison result
    double iteration_ratio;        // clustered / allgrid
};

// Simple experiment runner class
class SimpleExperiment {
public:
    SimpleExperiment(
        std::vector<dc::GameState> grid_states,
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        int gridM,
        int gridN
    );

    // Find ideal shot (long-running MCTS)
    ShotInfo findIdealShot(
        const dc::GameState& state,
        int max_iterations = 10000
    );

    // Run comparison experiment on a single board state
    ExperimentResult runSingleComparison(
        const dc::GameState& state,
        const ShotInfo& ideal_shot,
        int max_iterations = 5000
    );

    // Run experiment on multiple board states
    std::vector<ExperimentResult> runBatchExperiment(
        int num_states = 5,
        int max_iterations = 5000
    );

    // Save results to CSV
    void saveResults(
        const std::vector<ExperimentResult>& results,
        const std::string& filename
    );

private:
    std::vector<dc::GameState> grid_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;
    int GridSize_M_;
    int GridSize_N_;

    // Check if ideal shot is reached
    bool isIdealShot(const ShotInfo& shot, const ShotInfo& ideal, double tolerance = 0.01) const;

    // Generate test board states
    std::vector<dc::GameState> generateTestStates(int num_states);
};

#endif // _SIMPLE_EXPERIMENT_H_
