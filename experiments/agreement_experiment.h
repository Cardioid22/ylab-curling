#pragma once
#ifndef _AGREEMENT_EXPERIMENT_H_
#define _AGREEMENT_EXPERIMENT_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/structure.h"
#include "../src/simulator.h"
#include "../src/mcts.h"
#include "clustering_validation.h"
#include <vector>
#include <string>
#include <memory>

namespace dc = digitalcurling3;

// Result from a single MCTS run
struct MCTSRunResult {
    int selected_grid_id;      // Selected grid position ID
    double win_rate;           // Win rate of selected shot
    int iterations;            // Number of iterations performed
    double elapsed_time_sec;   // Time taken
    NodeSource node_source;    // Clustered or AllGrid
};

// Result comparing Clustered vs AllGrid MCTS
struct AgreementResult {
    int test_id;                    // Test state ID
    std::string test_description;   // Description of test state
    int shot_number;                // Shot number of the state

    // AllGrid MCTS result (Ground Truth)
    MCTSRunResult allgrid_result;

    // Clustered MCTS results (various iteration counts)
    std::vector<MCTSRunResult> clustered_results;
    std::vector<int> clustered_iterations_tested;  // Iteration counts tested

    // Agreement analysis
    std::vector<bool> agreement_flags;  // Does clustered match allgrid?
    double overall_agreement_rate;      // Percentage of agreement
};

// Main experiment class for comparing Clustered vs AllGrid MCTS
class AgreementExperiment {
public:
    AgreementExperiment(
        dc::Team team,
        dc::GameSetting game_setting,
        int grid_m,
        int grid_n,
        std::vector<dc::GameState> grid_states,
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        std::shared_ptr<SimulatorWrapper> simulator_clustered,
        std::shared_ptr<SimulatorWrapper> simulator_allgrid
    );

    // Run the complete experiment
    void runExperiment(int num_test_patterns_per_type = 1);

    // Export results to CSV
    void exportResultsToCSV(const std::string& filename);

private:
    dc::Team team_;
    dc::GameSetting game_setting_;
    int grid_m_;
    int grid_n_;
    std::vector<dc::GameState> grid_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;
    std::shared_ptr<SimulatorWrapper> simulator_clustered_;
    std::shared_ptr<SimulatorWrapper> simulator_allgrid_;

    std::vector<AgreementResult> results_;

    // Calculate total iterations needed to fully explore depth d
    // For 16 grid: depth 3 = 1 + 16 + 16^2 + 16^3 = 4369
    int calculateFullExplorationIterations(int max_depth);

    // Run AllGrid MCTS to find ground truth
    MCTSRunResult runAllGridMCTS(const dc::GameState& state, int iterations);

    // Run Clustered MCTS with specific iteration count
    MCTSRunResult runClusteredMCTS(const dc::GameState& state, int iterations);

    // Run experiment for a single test state
    AgreementResult runSingleTest(const TestState& test_state);

    // Generate Clustered MCTS iteration counts to test
    std::vector<int> generateClusteredIterationCounts();

    // Calculate agreement rate
    double calculateAgreementRate(const std::vector<bool>& agreement_flags);

    // Print summary to console
    void printSummary();
};

#endif // _AGREEMENT_EXPERIMENT_H_
