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
    std::vector<std::vector<int>> cluster_table;  // Cluster ID -> State IDs mapping (only for Clustered)
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
    std::vector<bool> agreement_flags;           // Does clustered match allgrid? (exact match)
    std::vector<bool> cluster_agreement_flags;   // Is allgrid's shot in clustered's cluster?
    double overall_agreement_rate;               // Percentage of exact agreement
    double overall_cluster_agreement_rate;       // Percentage of cluster-based agreement
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
    void runExperiment(int num_test_patterns_per_type = 1, int test_depth = 1);

    // Export results to CSV
    void exportResultsToCSV(const std::string& filename);

    // Export summary to separate file
    void exportSummaryToFile(const std::string& filename);

    // Generate filename with grid size, depth, and cluster info
    std::string generateFilename(const std::string& prefix, const std::string& extension, int depth) const;

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
    AgreementResult runSingleTest(const TestState& test_state, int test_depth);

    // Generate Clustered MCTS iteration counts to test
    std::vector<int> generateClusteredIterationCounts(int dephth);

    // Calculate agreement rate
    double calculateAgreementRate(const std::vector<bool>& agreement_flags);

    // Check if allgrid's selected shot is in the same cluster as clustered's selected shot
    bool checkClusterMembership(
        int allgrid_grid_id,
        int clustered_grid_id,
        const std::vector<std::vector<int>>& cluster_table
    );

    // Print summary to console
    void printSummary();
};

#endif // _AGREEMENT_EXPERIMENT_H_
