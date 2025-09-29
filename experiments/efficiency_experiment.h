#pragma once
#ifndef _EFFICIENCY_EXPERIMENT_H_
#define _EFFICIENCY_EXPERIMENT_H_

#include "mcts_with_tracking.h"
#include "ground_truth_finder.h"
#include "../src/structure.h"
#include "../src/simulator.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dc = digitalcurling3;

class EfficiencyExperiment {
public:
    struct ExperimentResult {
        int clustered_discovery_iterations;
        int allgrid_discovery_iterations;
        double efficiency_ratio;

        bool clustered_success;
        bool allgrid_success;

        std::vector<double> clustered_score_history;
        std::vector<double> allgrid_score_history;

        dc::GameState test_state;
        ShotInfo ground_truth;

        double clustered_final_score;
        double allgrid_final_score;
    };

    EfficiencyExperiment(
        std::vector<dc::GameState> grid_states,
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        int gridM,
        int gridN
    );

    ExperimentResult runSingleExperiment(
        const dc::GameState& test_state,
        const ShotInfo& ground_truth,
        int max_iterations = 10000,
        double max_time = 300.0
    );

    std::vector<ExperimentResult> runBatchExperiment(
        const std::vector<dc::GameState>& test_states,
        int trials_per_state = 20
    );

private:
    std::vector<dc::GameState> grid_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;
    int GridSize_M_;
    int GridSize_N_;

    std::shared_ptr<SimulatorWrapper> simulator_clustered_;
    std::shared_ptr<SimulatorWrapper> simulator_allgrid_;

    void initializeSimulators();
};

#endif // _EFFICIENCY_EXPERIMENT_H_