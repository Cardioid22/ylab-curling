#pragma once
#ifndef _EXPERIMENT_RUNNER_H_
#define _EXPERIMENT_RUNNER_H_

#include "efficiency_experiment.h"
#include "statistical_analysis.h"
#include "ground_truth_finder.h"
#include "../src/structure.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include <vector>
#include <string>

namespace dc = digitalcurling3;

struct ExperimentConfig {
    double convergence_threshold = 0.8;
    int max_iterations = 10000;
    double max_time = 300.0;
    int trials_per_state = 20;
    int ground_truth_iterations = 100000;
};

class ExperimentRunner {
public:
    ExperimentRunner(
        std::vector<dc::GameState> grid_states,
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        int gridM,
        int gridN
    );

    void runEfficiencyExperiment(const ExperimentConfig& config = ExperimentConfig{});

    void runSingleStateExperiment(
        const dc::GameState& test_state,
        const ExperimentConfig& config = ExperimentConfig{}
    );

    void exportAllResults(
        const std::vector<EfficiencyExperiment::ExperimentResult>& results,
        const StatisticalAnalysis::StatisticalResult& stats,
        const std::string& base_folder = ""
    );

private:
    std::vector<dc::GameState> grid_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;
    int GridSize_M_;
    int GridSize_N_;

    std::unique_ptr<EfficiencyExperiment> experiment_;
    std::unique_ptr<StatisticalAnalysis> analyzer_;
    std::unique_ptr<GroundTruthFinder> truth_finder_;

    std::vector<dc::GameState> generateTestStates();

    void exportIntermediateResults(
        const std::vector<EfficiencyExperiment::ExperimentResult>& results,
        size_t state_idx,
        int trial,
        const std::string& folder
    );

    std::string createExperimentFolder();
    std::string getCurrentTimestamp();
};

#endif // _EXPERIMENT_RUNNER_H_