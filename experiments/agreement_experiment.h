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

// 1つのショットを複数回シミュレーションした結果
struct ShotSimulationResult {
    int shot_id;                          // グリッドID
    std::vector<float> final_scores;      // 各シミュレーションの最終得点
    float mean_score;                     // 平均得点
    float std_score;                      // 得点の標準偏差
};

// クラスタメンバーの詳細分析結果
struct ClusterMemberAnalysis {
    int cluster_id;                                    // クラスタID
    int allgrid_selected_shot_id;                      // AllGridが選んだ正解手のID
    int clustered_selected_shot_id;                    // Clusteredが選んだ手のID
    std::vector<int> member_shot_ids;                  // クラスタ内の全メンバーのショットID
    std::vector<ShotSimulationResult> member_results;  // 各メンバーのシミュレーション結果
    float cluster_score_variance;                      // クラスタ内の得点分散
    float cluster_mean_score;                          // クラスタ全体の平均スコア
    bool contains_allgrid_shot;                        // AllGridの選んだ手を含むか
    bool contains_clustered_shot;                      // Clusteredの選んだ手を含むか
};

// 全クラスタの分析結果
struct AllClusterAnalysis {
    std::vector<ClusterMemberAnalysis> all_clusters;   // 全クラスタの分析
    int best_cluster_id;                               // 最高平均スコアのクラスタID
    int allgrid_cluster_id;                            // AllGridの手が属するクラスタID
    int clustered_cluster_id;                          // Clusteredの手が属するクラスタID
};

// ベストショット比較結果
struct BestShotComparison {
    int best_overall_shot_id;                          // 全体で最高平均スコアのショットID
    float best_overall_mean_score;                     // 全体での最高平均スコア
    int allgrid_shot_id;                               // AllGridが選んだショットID
    float allgrid_mean_score;                          // AllGridの手の平均スコア
    int clustered_shot_id;                             // Clusteredが選んだショットID
    float clustered_mean_score;                        // Clusteredの手の平均スコア
    std::vector<ShotSimulationResult> all_shot_results; // 全ショットの結果
};

// Result from a single MCTS run
struct MCTSRunResult {
    int selected_grid_id;      // Selected grid position ID
    double win_rate;           // Win rate of selected shot
    int iterations;            // Number of iterations performed
    double elapsed_time_sec;   // Time taken
    NodeSource node_source;    // Clustered or AllGrid
    std::vector<std::vector<int>> cluster_table;  // Cluster ID -> State IDs mapping (only for Clustered)
    float silhouette_score = -1.0f;  // Silhouette score for clustering quality (only for Clustered)
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

    // Cluster member analysis (only performed when cluster agrees)
    std::vector<ClusterMemberAnalysis> cluster_member_analyses;  // One per clustered result that agrees

    // All cluster analysis (新機能: 全クラスタの詳細分析)
    std::vector<AllClusterAnalysis> all_cluster_analyses;  // イテレーション毎の全クラスタ分析

    // Best shot comparison (新機能: 最良ショット比較)
    std::vector<BestShotComparison> best_shot_comparisons;  // イテレーション毎のベストショット比較
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
        std::shared_ptr<SimulatorWrapper> simulator_allgrid,
        int cluster_num,  // Number of clusters for Clustered MCTS
        int simulations_per_shot = 10  // Number of simulations per shot for cluster analysis
    );

    // Run the complete experiment
    void runExperiment(int num_test_patterns_per_type = 1, int test_depth = 1);

    // Export results to CSV
    void exportResultsToCSV(const std::string& filename);

    // Export summary to separate file
    void exportSummaryToFile(const std::string& filename);

    // Generate filename with grid size, depth, cluster info, and test case count
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
    int cluster_num_;  // Number of clusters for Clustered MCTS
    int simulations_per_shot_;  // Number of simulations per shot for cluster analysis

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

    // Analyze cluster members when cluster agreement is found
    ClusterMemberAnalysis analyzeClusterMembers(
        const dc::GameState& initial_state,
        int allgrid_shot_id,
        int clustered_shot_id,
        const std::vector<std::vector<int>>& cluster_table
    );

    // Analyze a single cluster
    ClusterMemberAnalysis analyzeSingleCluster(
        const dc::GameState& initial_state,
        int cluster_id,
        const std::vector<int>& member_ids,
        int allgrid_shot_id,
        int clustered_shot_id
    );

    // Analyze all clusters
    AllClusterAnalysis analyzeAllClusters(
        const dc::GameState& initial_state,
        int allgrid_shot_id,
        int clustered_shot_id,
        const std::vector<std::vector<int>>& cluster_table
    );

    // Generate best shot comparison
    BestShotComparison generateBestShotComparison(
        const dc::GameState& initial_state,
        int allgrid_shot_id,
        int clustered_shot_id
    );

    // Simulate a single shot multiple times and return statistics
    ShotSimulationResult simulateShotMultipleTimes(
        const dc::GameState& initial_state,
        const ShotInfo& shot,
        int shot_id,
        int num_simulations
    );

    // Play out to the end of the game using random policy
    float playoutToEnd(dc::GameState state);

    // Statistical helper functions
    float calculateMean(const std::vector<float>& values);
    float calculateStdDev(const std::vector<float>& values, float mean);
    float calculateVariance(const std::vector<float>& values);

    // Print summary to console
    void printSummary();
};

#endif // _AGREEMENT_EXPERIMENT_H_
