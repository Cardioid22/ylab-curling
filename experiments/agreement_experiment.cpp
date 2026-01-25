#include "../experiments/agreement_experiment.h"
#include "../src/clustering-v2.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <map>
#include <optional> // Added for std::optional

AgreementExperiment::AgreementExperiment(
    dc::Team team,
    dc::GameSetting game_setting,
    int grid_m,
    int grid_n,
    std::vector<dc::GameState> grid_states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    std::shared_ptr<SimulatorWrapper> simulator_clustered,
    std::shared_ptr<SimulatorWrapper> simulator_allgrid,
    int cluster_num,
    int simulations_per_shot
) : team_(team),
    game_setting_(game_setting),
    grid_m_(grid_m),
    grid_n_(grid_n),
    grid_states_(grid_states),
    state_to_shot_table_(state_to_shot_table),
    simulator_clustered_(simulator_clustered),
    simulator_allgrid_(simulator_allgrid),
    cluster_num_(cluster_num),
    simulations_per_shot_(simulations_per_shot)
{
    std::cout << "[AgreementExperiment] Initialized with "
              << grid_m_ << "x" << grid_n_ << " grid ("
              << (grid_m_ * grid_n_) << " total positions)\n";
    std::cout << "[AgreementExperiment] Using " << cluster_num_ << " clusters\n";
    std::cout << "[AgreementExperiment] Simulations per shot for cluster analysis: " << simulations_per_shot_ << "\n";
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
    counts.push_back(clustered_full * 0.1);
    counts.push_back(clustered_full * 0.15);
    counts.push_back(clustered_full * 0.2);
    counts.push_back(clustered_full * 0.25);
    counts.push_back(clustered_full * 0.3);
    //counts.push_back(clustered_full * 0.5);
    // Full exploration
    //counts.push_back(clustered_full);
    
    //counts.push_back(clustered_full * 2);
    //counts.push_back(clustered_full * 5);
    //counts.push_back(clustered_full * 10);

    return counts;
}

MCTSRunResult AgreementExperiment::runAllGridMCTS(const dc::GameState& state, int iterations) {
    std::cout << "  [AllGrid MCTS] Running " << iterations << " iterations...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Create MCTS with AllGrid node source
    int num_rollout_simulations = 10;  // Number of simulations per rollout
    MCTS mcts(
        state,
        NodeSource::AllGrid,
        grid_states_,
        state_to_shot_table_,
        simulator_allgrid_,
        grid_m_,
        grid_n_,
        cluster_num_,  // Pass cluster_num (not used for AllGrid, but required)
        num_rollout_simulations
    );

    // Run MCTS
    mcts.grow_tree(iterations, 10800.0);  // 3 hour timeout

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

    // Get rollout timing from MCTS
    double rollout_time = mcts.get_total_rollout_time();
    int rollout_count = mcts.get_total_rollout_count();

    MCTSRunResult result;
    result.selected_grid_id = selected_grid_id;
    result.win_rate = win_rate;
    result.iterations = iterations;
    result.elapsed_time_sec = elapsed.count();
    result.node_source = NodeSource::AllGrid;
    result.rollout_time_sec = rollout_time;  // Store rollout time
    result.rollout_count = rollout_count;  // Store rollout count

    std::cout << "  [AllGrid MCTS] Selected grid ID: " << selected_grid_id
              << " (win rate: " << std::fixed << std::setprecision(3) << win_rate
              << ", time: " << std::setprecision(2)
              << elapsed.count() << "s, rollouts: " << rollout_time << "s)\n";

    return result;
}

MCTSRunResult AgreementExperiment::runClusteredMCTS(const dc::GameState& state, int iterations) {
    std::cout << "    [Clustered MCTS] Running " << iterations << " iterations...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get cluster information before running MCTS
    auto clustering_start = std::chrono::high_resolution_clock::now();
    ClusteringV2 clustering(cluster_num_, grid_states_, grid_m_, grid_n_, team_);
    std::vector<std::vector<int>> cluster_table = clustering.getClusterIdTable();
    float silhouette_score = clustering.evaluateClusteringQuality();
    auto clustering_end = std::chrono::high_resolution_clock::now();
    double clustering_time = std::chrono::duration<double>(clustering_end - clustering_start).count();

    // Create MCTS with Clustered node source
    int num_rollout_simulations = 10;  // Number of simulations per rollout
    MCTS mcts(
        state,
        NodeSource::Clustered,
        grid_states_,
        state_to_shot_table_,
        simulator_clustered_,
        grid_m_,
        grid_n_,
        cluster_num_,  // Pass configurable cluster_num
        num_rollout_simulations,
        std::nullopt // Explicitly pass nullopt for optional argument
    );

    // Run MCTS (Initial Clustering)
    mcts.grow_tree(iterations, 10800.0);  // 3 hour timeout

    // Get best shot from initial clustering
    ShotInfo best_shot = mcts.get_best_shot();
    double win_rate = mcts.get_best_shot_winrate();

    // --- Zoom-In Phase ---
    std::cout << "    [Zoom-In] Identifying target cluster...\n";
    
    // Find grid ID of the best shot
    int best_grid_id = -1;
    double min_diff_for_id = 1e9;
    for (const auto& [grid_id, shot] : state_to_shot_table_) {
        double diff = std::abs(shot.vx - best_shot.vx) + std::abs(shot.vy - best_shot.vy);
        if (diff < min_diff_for_id) {
            min_diff_for_id = diff;
            best_grid_id = grid_id;
        }
    }

    // Find which cluster contains this grid ID
    int target_cluster_id = -1;
    if (best_grid_id != -1) {
        for (size_t i = 0; i < cluster_table.size(); ++i) {
            // Check if best_grid_id is in this cluster
            for (int member_id : cluster_table[i]) {
                if (member_id == best_grid_id) {
                    target_cluster_id = i;
                    break;
                }
            }
            if (target_cluster_id != -1) break;
        }
    }

    // Run Zoom-In if cluster found
    if (target_cluster_id != -1) {
        std::vector<ShotInfo> zoomin_candidates;
        for (int member_id : cluster_table[target_cluster_id]) {
            if (state_to_shot_table_.count(member_id)) {
                zoomin_candidates.push_back(state_to_shot_table_.at(member_id));
            }
        }

        std::cout << "    [Zoom-In] Focusing on Cluster " << target_cluster_id 
                  << " with " << zoomin_candidates.size() << " candidates. Running MCTS...\n";
        for(size_t i=0; i<zoomin_candidates.size(); ++i) {
            std::cout << "      Candidate " << i << ": vx=" << zoomin_candidates[i].vx 
                      << ", vy=" << zoomin_candidates[i].vy << "\n";
        }

        // Create Zoom-In MCTS
        MCTS mcts_zoomin(
            state,
            NodeSource::Specified,
            grid_states_,
            state_to_shot_table_,
            simulator_clustered_,
            grid_m_,
            grid_n_,
            cluster_num_,
            num_rollout_simulations,
            std::make_optional(zoomin_candidates) // Use make_optional
        );

        // Run Zoom-In MCTS
        mcts_zoomin.grow_tree(iterations, 10800.0);

        // Update best shot with Zoom-In result
        best_shot = mcts_zoomin.get_best_shot();
        win_rate = mcts_zoomin.get_best_shot_winrate();
    } else {
        std::cout << "    [Zoom-In] WARNING: Could not identify target cluster. Using initial result.\n";
    }
    // ---------------------

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Get rollout timing from MCTS (Note: Tracking initial MCTS timing)
    double rollout_time = mcts.get_total_rollout_time();
    int rollout_count = mcts.get_total_rollout_count();

    // Get clustering timing from MCTS (internal clustering during tree expansion)
    double mcts_clustering_time = mcts.get_total_clustering_time();
    int mcts_clustering_count = mcts.get_total_clustering_count();

    // Total clustering time = pre-MCTS clustering + MCTS internal clustering
    double total_clustering_time = clustering_time + mcts_clustering_time;

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
    result.cluster_table = cluster_table;  // Store cluster information
    result.silhouette_score = silhouette_score;  // Store silhouette score
    result.clustering_time_sec = total_clustering_time;  // Store total clustering time (pre-MCTS + MCTS internal)
    result.rollout_time_sec = rollout_time;  // Store rollout time
    result.rollout_count = rollout_count;  // Store rollout count

    std::cout << "    [Clustered MCTS] Selected grid ID: " << selected_grid_id
              << " (win rate: " << std::fixed << std::setprecision(3) << win_rate
              << ", time: " << std::setprecision(2)
              << elapsed.count() << "s, silhouette: " << std::setprecision(4) << silhouette_score
              << ", clustering: " << std::setprecision(2) << total_clustering_time << "s"
              << " [pre: " << clustering_time << "s + mcts: " << mcts_clustering_time
              << "s (" << mcts_clustering_count << " ops)]"
              << ", rollouts: " << rollout_time << "s)\n";

    return result;
}

AgreementResult AgreementExperiment::runSingleTest(const TestState& test_state, int test_depth) {
    auto test_start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\n===========================================\n";
    std::cout << "Test ID: " << test_state.test_id << "\n";
    std::cout << "Description: " << test_state.description << "\n";
    std::cout << "Shot: " << test_state.state.shot << "\n";
    std::cout << "Test Depth: " << test_depth << "\n";
    std::cout << "===========================================\n";

    AgreementResult result;
    result.test_id = test_state.test_id;
    result.test_description = test_state.description;
    result.shot_number = test_state.state.shot;

    // Step 1: Run AllGrid MCTS with full exploration
    std::cout << "\n[Step 1] Running AllGrid MCTS (Ground Truth)\n";
    auto allgrid_start = std::chrono::high_resolution_clock::now();
	int allgrid_iterations = calculateFullExplorationIterations(test_depth); // X-depth full exploration
    result.allgrid_result = runAllGridMCTS(test_state.state, allgrid_iterations);
    auto allgrid_end = std::chrono::high_resolution_clock::now();
    result.allgrid_time_sec = std::chrono::duration<double>(allgrid_end - allgrid_start).count();

    std::cout << "\n[Ground Truth] AllGrid selected: Grid ID "
              << result.allgrid_result.selected_grid_id
              << " (Time: " << std::fixed << std::setprecision(2) << result.allgrid_time_sec << "s)\n";

    // Step 2: Run Clustered MCTS with various iteration counts
    std::cout << "\n[Step 2] Running Clustered MCTS with various iterations\n";
    auto clustered_start = std::chrono::high_resolution_clock::now();
    result.clustered_iterations_tested = generateClusteredIterationCounts(test_depth);

    // Calculate target iteration count (1/10 of AllGrid iterations)
    int target_iteration_for_analysis = allgrid_iterations / 10;

    for (int iterations : result.clustered_iterations_tested) {
        MCTSRunResult clustered_res = runClusteredMCTS(test_state.state, iterations);
        result.clustered_results.push_back(clustered_res);

        // Check exact agreement
        bool agrees = (clustered_res.selected_grid_id == result.allgrid_result.selected_grid_id);
        result.agreement_flags.push_back(agrees);

        // Check cluster-based agreement
        bool cluster_agrees = checkClusterMembership(
            result.allgrid_result.selected_grid_id,
            clustered_res.selected_grid_id,
            clustered_res.cluster_table
        );
        result.cluster_agreement_flags.push_back(cluster_agrees);

        std::cout << "      Iterations: " << std::setw(6) << iterations
                  << " | Selected: Grid " << std::setw(2) << clustered_res.selected_grid_id
                  << " | Exact: " << (agrees ? "YES" : "NO")
                  << " | Cluster: " << (cluster_agrees ? "YES" : "NO") << "\n";

        // Analyze cluster members when:
        // 1. Iterations match the target (1/10 of AllGrid)
        // 2. Simulations per shot is enabled
        // We analyze regardless of agreement to investigate failures
        if (iterations == target_iteration_for_analysis && simulations_per_shot_ > 0) {
            std::cout << "      [Target iteration count reached: analyzing cluster members (Agreement: " 
                      << (cluster_agrees ? "YES" : "NO") << ")]\n";
            auto analysis_start = std::chrono::high_resolution_clock::now();

            ClusterMemberAnalysis member_analysis = analyzeClusterMembers(
                test_state.state,
                result.allgrid_result.selected_grid_id,
                clustered_res.selected_grid_id,
                clustered_res.cluster_table
            );
            result.cluster_member_analyses.push_back(member_analysis);

            // 全クラスタ分析を実行
            std::cout << "      [Analyzing all clusters]\n";
            AllClusterAnalysis all_cluster_analysis = analyzeAllClusters(
                test_state.state,
                result.allgrid_result.selected_grid_id,
                clustered_res.selected_grid_id,
                clustered_res.cluster_table
            );
            result.all_cluster_analyses.push_back(all_cluster_analysis);

            // ベストショット比較を生成
            std::cout << "      [Generating best shot comparison]\n";
            BestShotComparison best_shot_comp = generateBestShotComparison(
                test_state.state,
                result.allgrid_result.selected_grid_id,
                clustered_res.selected_grid_id
            );
            result.best_shot_comparisons.push_back(best_shot_comp);

            auto analysis_end = std::chrono::high_resolution_clock::now();
            result.cluster_analysis_time_sec += std::chrono::duration<double>(analysis_end - analysis_start).count();
        }
    }

    auto clustered_end = std::chrono::high_resolution_clock::now();
    result.clustered_total_time_sec = std::chrono::duration<double>(clustered_end - clustered_start).count();

    // Calculate overall agreement rates
    result.overall_agreement_rate = calculateAgreementRate(result.agreement_flags);
    result.overall_cluster_agreement_rate = calculateAgreementRate(result.cluster_agreement_flags);

    // Calculate total test time
    auto test_end_time = std::chrono::high_resolution_clock::now();
    result.total_test_time_sec = std::chrono::duration<double>(test_end_time - test_start_time).count();

    std::cout << "\n[Summary] Exact Agreement Rate: "
              << std::fixed << std::setprecision(1)
              << result.overall_agreement_rate << "%\n";
    std::cout << "[Summary] Cluster Agreement Rate: "
              << std::fixed << std::setprecision(1)
              << result.overall_cluster_agreement_rate << "%\n";
    std::cout << "\n[Timing Breakdown]\n";
    std::cout << "  AllGrid MCTS:       " << std::setw(8) << std::fixed << std::setprecision(2)
              << result.allgrid_time_sec << "s ("
              << std::setw(5) << std::setprecision(1) << (result.allgrid_time_sec / result.total_test_time_sec * 100) << "%)\n";
    std::cout << "  Clustered MCTS:     " << std::setw(8) << result.clustered_total_time_sec << "s ("
              << std::setw(5) << (result.clustered_total_time_sec / result.total_test_time_sec * 100) << "%)\n";
    std::cout << "  Cluster Analysis:   " << std::setw(8) << result.cluster_analysis_time_sec << "s ("
              << std::setw(5) << (result.cluster_analysis_time_sec / result.total_test_time_sec * 100) << "%)\n";
    double other_time = result.total_test_time_sec - result.allgrid_time_sec - result.clustered_total_time_sec;
    std::cout << "  Other:              " << std::setw(8) << other_time << "s ("
              << std::setw(5) << (other_time / result.total_test_time_sec * 100) << "%)\n";
    std::cout << "  Total:              " << std::setw(8) << result.total_test_time_sec << "s\n";
    std::cout << "===========================================\n";

    return result;
}

bool AgreementExperiment::checkClusterMembership(
    int allgrid_grid_id,
    int clustered_grid_id,
    const std::vector<std::vector<int>>& cluster_table) {

    // Find which cluster contains the clustered_grid_id
    int clustered_cluster_id = -1;
    for (size_t cluster_id = 0; cluster_id < cluster_table.size(); ++cluster_id) {
        for (int state_id : cluster_table[cluster_id]) {
            if (state_id == clustered_grid_id) {
                clustered_cluster_id = cluster_id;
                break;
            }
        }
        if (clustered_cluster_id != -1) break;
    }

    // If we couldn't find the cluster, return false
    if (clustered_cluster_id == -1) {
        std::cerr << "      [WARNING] Could not find cluster for grid ID "
                  << clustered_grid_id << "\n";
        return false;
    }

    // Check if allgrid_grid_id is in the same cluster
    const auto& cluster = cluster_table[clustered_cluster_id];
    bool found = std::find(cluster.begin(), cluster.end(), allgrid_grid_id) != cluster.end();

    if (found) {
        std::cout << "      [DEBUG] Grid " << allgrid_grid_id
                  << " found in cluster " << clustered_cluster_id
                  << " (size: " << cluster.size() << ")\n";
    }

    return found;
}

double AgreementExperiment::calculateAgreementRate(const std::vector<bool>& agreement_flags) {
    if (agreement_flags.empty()) return 0.0;

    int agreements = 0;
    for (bool flag : agreement_flags) {
        if (flag) agreements++;
    }

    return (static_cast<double>(agreements) / agreement_flags.size()) * 100.0;
}

void AgreementExperiment::runExperiment(int num_test_patterns_per_type, int test_depth) {
    std::cout << "\n=========================================\n";
    std::cout << "AGREEMENT EXPERIMENT: Clustered vs AllGrid\n";
    std::cout << "=========================================\n";
    std::cout << "Grid size: " << grid_m_ << "x" << grid_n_ << " (total: " << (grid_m_ * grid_n_) << ")\n";
    std::cout << "Test depth: " << test_depth << "\n";
    std::cout << "Number of clusters: " << cluster_num_ << "\n";
    std::cout << "Test patterns per type: " << num_test_patterns_per_type << "\n";
    std::cout << "=========================================\n\n";

    // Generate test states using ClusteringValidation
    ClusteringValidation validator(team_);
    std::vector<TestState> test_states = validator.generateTestStates(num_test_patterns_per_type);

    std::cout << "Generated " << test_states.size() << " test states\n\n";

    // Run experiment for each test state
    for (const auto& test_state : test_states) {
        AgreementResult result = runSingleTest(test_state, test_depth);
        results_.push_back(result);
    }

    // Print final summary
    printSummary();
}

void AgreementExperiment::runSingleTestById(int test_id, int num_test_patterns_per_type, int test_depth) {
    std::cout << "\n=========================================\n";
    std::cout << "SINGLE TEST EXECUTION (Test ID: " << test_id << ")\n";
    std::cout << "=========================================\n";
    std::cout << "Grid size: " << grid_m_ << "x" << grid_n_ << " (total: " << (grid_m_ * grid_n_) << ")\n";
    std::cout << "Test depth: " << test_depth << "\n";
    std::cout << "Number of clusters: " << cluster_num_ << "\n";
    std::cout << "Test patterns per type: " << num_test_patterns_per_type << "\n";
    std::cout << "=========================================\n\n";

    // Generate all test states (deterministic with fixed seed)
    ClusteringValidation validator(team_);
    std::vector<TestState> test_states = validator.generateTestStates(num_test_patterns_per_type);

    // Validate test ID
    if (test_id < 0 || test_id >= static_cast<int>(test_states.size())) {
        std::cerr << "[ERROR] Invalid test ID: " << test_id << "\n";
        std::cerr << "Valid range: 0 to " << (test_states.size() - 1) << "\n";
        return;
    }

    std::cout << "Total available tests: " << test_states.size() << "\n";
    std::cout << "Running test ID: " << test_id << "\n\n";

    // Run single test
    const auto& test_state = test_states[test_id];
    AgreementResult result = runSingleTest(test_state, test_depth);
    results_.push_back(result);

    std::cout << "\n[Test " << test_id << " Complete]\n";
    std::cout << "=========================================\n";
}

void AgreementExperiment::printSummary() {
    std::cout << "\n\n";
    std::cout << "=========================================\n";
    std::cout << "FINAL SUMMARY\n";
    std::cout << "=========================================\n";
    std::cout << "Total tests: " << results_.size() << "\n";

    // Calculate average silhouette score across all tests
    if (!results_.empty()) {
        double total_silhouette = 0.0;
        int silhouette_count = 0;

        for (const auto& result : results_) {
            for (const auto& clustered_res : result.clustered_results) {
                if (clustered_res.silhouette_score >= 0.0f) {
                    total_silhouette += clustered_res.silhouette_score;
                    silhouette_count++;
                }
            }
        }

        if (silhouette_count > 0) {
            double avg_silhouette = total_silhouette / silhouette_count;
            std::cout << "Average Silhouette Score: " << std::fixed << std::setprecision(4)
                      << avg_silhouette << " (across " << silhouette_count << " clustered runs)\n";
        }
    }

    // Calculate average agreement rates for each iteration count
    if (!results_.empty() && !results_[0].clustered_iterations_tested.empty()) {
        std::cout << "\nAverage Agreement Rate by Iteration Count:\n";
        std::cout << "-------------------------------------------\
";
        std::cout << "Iterations |  Exact  | Cluster\n";
        std::cout << "-----------+---------+--------\n";

        for (size_t i = 0; i < results_[0].clustered_iterations_tested.size(); ++i) {
            int iterations = results_[0].clustered_iterations_tested[i];
            double total_exact_agreement = 0.0;
            double total_cluster_agreement = 0.0;

            for (const auto& result : results_) {
                if (i < result.agreement_flags.size() && result.agreement_flags[i]) {
                    total_exact_agreement += 1.0;
                }
                if (i < result.cluster_agreement_flags.size() && result.cluster_agreement_flags[i]) {
                    total_cluster_agreement += 1.0;
                }
            }

            double avg_exact_agreement = (total_exact_agreement / results_.size()) * 100.0;
            double avg_cluster_agreement = (total_cluster_agreement / results_.size()) * 100.0;

            std::cout << std::setw(10) << iterations << " | "
                      << std::fixed << std::setprecision(1)
                      << std::setw(6) << avg_exact_agreement << "% | "
                      << std::setw(6) << avg_cluster_agreement << "%\n";
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

    // Calculate average silhouette score
    if (!results_.empty()) {
        double total_silhouette = 0.0;
        int silhouette_count = 0;

        for (const auto& result : results_) {
            for (const auto& clustered_res : result.clustered_results) {
                if (clustered_res.silhouette_score >= 0.0f) {
                    total_silhouette += clustered_res.silhouette_score;
                    silhouette_count++;
                }
            }
        }

        if (silhouette_count > 0) {
            double avg_silhouette = total_silhouette / silhouette_count;
            file << "Average Silhouette Score: " << std::fixed << std::setprecision(4)
                 << avg_silhouette << " (across " << silhouette_count << " clustered runs)\n";
        }
    }
    file << "\n";

    // Calculate and write average agreement rates for each iteration count
    if (!results_.empty() && !results_[0].clustered_iterations_tested.empty()) {
        file << "Average Agreement Rate by Iteration Count:\n";
        file << "-------------------------------------------\
";
        file << "Iterations |  Exact Agreement  | Cluster Agreement\n";
        file << "-----------+-------------------+------------------\n";

        for (size_t i = 0; i < results_[0].clustered_iterations_tested.size(); ++i) {
            int iterations = results_[0].clustered_iterations_tested[i];
            double total_exact_agreement = 0.0;
            double total_cluster_agreement = 0.0;
            int total_tests = 0;

            for (const auto& result : results_) {
                if (i < result.agreement_flags.size()) {
                    total_tests++;
                    if (result.agreement_flags[i]) {
                        total_exact_agreement += 1.0;
                    }
                    if (i < result.cluster_agreement_flags.size() && result.cluster_agreement_flags[i]) {
                        total_cluster_agreement += 1.0;
                    }
                }
            }

            double avg_exact = total_tests > 0 ? (total_exact_agreement / total_tests) * 100.0 : 0.0;
            double avg_cluster = total_tests > 0 ? (total_cluster_agreement / total_tests) * 100.0 : 0.0;

            file << std::setw(10) << iterations << " | "
                 << std::fixed << std::setprecision(1) << std::setw(6) << avg_exact << "% "
                 << "(" << static_cast<int>(total_exact_agreement) << "/" << total_tests << ") | "
                 << std::setw(6) << avg_cluster << "% "
                 << "(" << static_cast<int>(total_cluster_agreement) << "/" << total_tests << ")\n";
        }
    }

    file << "=========================================\n";
    file << "\n";

    // Write timing breakdown summary
    if (!results_.empty()) {
        file << "Average Timing Breakdown per Test:\n";
        file << "-------------------------------------------\
";

        double total_allgrid = 0.0, total_clustered = 0.0, total_analysis = 0.0, total_time = 0.0;
        double total_allgrid_rollout = 0.0, total_clustered_rollout = 0.0, total_clustering = 0.0;
        int total_allgrid_rollout_count = 0, total_clustered_rollout_count = 0;

        for (const auto& result : results_) {
            total_allgrid += result.allgrid_time_sec;
            total_clustered += result.clustered_total_time_sec;
            total_analysis += result.cluster_analysis_time_sec;
            total_time += result.total_test_time_sec;

            // Sum rollout times
            total_allgrid_rollout += result.allgrid_result.rollout_time_sec;
            total_allgrid_rollout_count += result.allgrid_result.rollout_count;

            for (const auto& clustered_res : result.clustered_results) {
                total_clustered_rollout += clustered_res.rollout_time_sec;
                total_clustered_rollout_count += clustered_res.rollout_count;
                total_clustering += clustered_res.clustering_time_sec;
            }
        }

        int num_tests = results_.size();
        double avg_allgrid = total_allgrid / num_tests;
        double avg_clustered = total_clustered / num_tests;
        double avg_analysis = total_analysis / num_tests;
        double avg_total = total_time / num_tests;
        double avg_other = avg_total - avg_allgrid - avg_clustered;

        double avg_allgrid_rollout = total_allgrid_rollout / num_tests;
        double avg_clustered_rollout = total_clustered_rollout / num_tests;
        double avg_clustering = total_clustering / num_tests;

        file << "  AllGrid MCTS:       " << std::setw(8) << std::fixed << std::setprecision(2)
             << avg_allgrid << "s (" << std::setw(5) << std::setprecision(1)
             << (avg_allgrid / avg_total * 100) << "%)\n";
        file << "    - Rollouts:       " << std::setw(8) << avg_allgrid_rollout << "s ("
             << std::setw(5) << (avg_allgrid_rollout / avg_allgrid * 100) << "% of AllGrid)\n";
        file << "  Clustered MCTS:     " << std::setw(8) << avg_clustered << "s ("
             << std::setw(5) << (avg_clustered / avg_total * 100) << "%)\n";
        file << "    - Clustering:     " << std::setw(8) << avg_clustering << "s ("
             << std::setw(5) << (avg_clustering / avg_clustered * 100) << "% of Clustered)\n";
        file << "    - Rollouts:       " << std::setw(8) << avg_clustered_rollout << "s ("
             << std::setw(5) << (avg_clustered_rollout / avg_clustered * 100) << "% of Clustered)\n";
        file << "  Cluster Analysis:   " << std::setw(8) << avg_analysis << "s ("
             << std::setw(5) << (avg_analysis / avg_total * 100) << "%)\n";
        file << "  Other:              " << std::setw(8) << avg_other << "s ("
             << std::setw(5) << (avg_other / avg_total * 100) << "%)\n";
        file << "  Total:              " << std::setw(8) << avg_total << "s\n";
        file << "\n";
        file << "Average Rollout Counts:\n";
        file << "  AllGrid:   " << (total_allgrid_rollout_count / num_tests) << " rollouts\n";
        file << "  Clustered: " << (total_clustered_rollout_count / num_tests) << " rollouts (total across all iterations)\n";
        file << "\n";
        file << "Total Cumulative Time: " << std::setprecision(2) << total_time << "s\n";
        file << "=========================================\n";
        file << "\n";
    }

    // Write detailed breakdown by test case
    file << "Detailed Breakdown by Test Case:\n";
    file << "-------------------------------------------\
";

    for (const auto& result : results_) {
        file << "\nTest " << result.test_id << ": " << result.test_description << "\n";
        file << "  AllGrid (Ground Truth): Grid " << result.allgrid_result.selected_grid_id
             << " (WinRate: " << std::fixed << std::setprecision(3) << result.allgrid_result.win_rate
             << ", Time: " << std::setprecision(2) << result.allgrid_result.elapsed_time_sec << "s)\n";

        for (size_t i = 0; i < result.clustered_results.size(); ++i) {
            const auto& clustered_res = result.clustered_results[i];
            bool exact_agrees = i < result.agreement_flags.size() ? result.agreement_flags[i] : false;
            bool cluster_agrees = i < result.cluster_agreement_flags.size() ? result.cluster_agreement_flags[i] : false;

            file << "  Clustered (Iter " << std::setw(6) << clustered_res.iterations << "): Grid "
                 << std::setw(2) << clustered_res.selected_grid_id
                 << " (WinRate: " << std::fixed << std::setprecision(3) << clustered_res.win_rate
                 << ", Time: " << std::setprecision(2) << clustered_res.elapsed_time_sec << ")"
                 << " | Exact: " << (exact_agrees ? "YES" : "NO")
                 << " | Cluster: " << (cluster_agrees ? "YES" : "NO") << "\n";
        }

        // Add timing breakdown for this test
        file << "\n  [Timing Breakdown for Test " << result.test_id << "]\n";
        file << "    AllGrid MCTS:       " << std::setw(8) << std::fixed << std::setprecision(2)
             << result.allgrid_time_sec << "s ("
             << std::setw(5) << std::setprecision(1) << (result.allgrid_time_sec / result.total_test_time_sec * 100) << "%)\n";
        file << "    Clustered MCTS:     " << std::setw(8) << result.clustered_total_time_sec << "s ("
             << std::setw(5) << (result.clustered_total_time_sec / result.total_test_time_sec * 100) << "%)\n";
        file << "    Cluster Analysis:   " << std::setw(8) << result.cluster_analysis_time_sec << "s ("
             << std::setw(5) << (result.cluster_analysis_time_sec / result.total_test_time_sec * 100) << "%)\n";
        double other_time = result.total_test_time_sec - result.allgrid_time_sec - result.clustered_total_time_sec;
        file << "    Other:              " << std::setw(8) << other_time << "s ("
             << std::setw(5) << (other_time / result.total_test_time_sec * 100) << "%)\n";
        file << "    Total:              " << std::setw(8) << result.total_test_time_sec << "s\n";
    }

    file.close();
    std::cout << "\nSummary exported to: " << filename << "\n";
}

// Generate filename with grid size, depth, cluster info, and test case count
std::string AgreementExperiment::generateFilename(const std::string& prefix, const std::string& extension, int depth) const {
    std::ostringstream filename;
    filename << prefix << "_Grid" << (grid_m_ * grid_n_)
             << "_Depth" << depth
             << "_Clusters" << cluster_num_
             << "_Tests" << results_.size()
             << extension;
    return filename.str();
}

// Export results to CSV
void AgreementExperiment::exportResultsToCSV(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << "\n";
        return;
    }

    // Write CSV header
    file << "TestID,Description,ShotNum,Method,Iterations,GridID,WinRate,ElapsedTime,ExactAgreement,ClusterAgreement,SilhouetteScore\n";

    for (const auto& result : results_) {
        // Write AllGrid result
        file << result.test_id << ","
             << "\"" << result.test_description << "\","
             << result.shot_number << ","
             << "AllGrid,"
             << result.allgrid_result.iterations << ","
             << result.allgrid_result.selected_grid_id << ","
             << std::fixed << std::setprecision(4) << result.allgrid_result.win_rate << ","
             << std::setprecision(2) << result.allgrid_result.elapsed_time_sec << ","
             << "N/A,N/A,-1\n";

        // Write Clustered results
        for (size_t i = 0; i < result.clustered_results.size(); ++i) {
            const auto& clustered = result.clustered_results[i];
            bool exact_agree = i < result.agreement_flags.size() ? result.agreement_flags[i] : false;
            bool cluster_agree = i < result.cluster_agreement_flags.size() ? result.cluster_agreement_flags[i] : false;

            file << result.test_id << ","
                 << "\"" << result.test_description << "\","
                 << result.shot_number << ","
                 << "Clustered,"
                 << clustered.iterations << ","
                 << clustered.selected_grid_id << ","
                 << std::fixed << std::setprecision(4) << clustered.win_rate << ","
                 << std::setprecision(2) << clustered.elapsed_time_sec << ","
                 << (exact_agree ? "YES" : "NO") << ","
                 << (cluster_agree ? "YES" : "NO") << ","
                 << std::setprecision(4) << clustered.silhouette_score << "\n";
        }
    }

    file.close();
    std::cout << "Results exported to: " << filename << "\n";
}

// Simulate a single shot multiple times and return statistics
ShotSimulationResult AgreementExperiment::simulateShotMultipleTimes(
    const dc::GameState& initial_state,
    const ShotInfo& shot,
    int shot_id,
    int num_simulations
) {
    ShotSimulationResult result;
    result.shot_id = shot_id;
    result.mean_score = 0.0f;
    result.std_score = 0.0f;

    // Run simulations
    for (int i = 0; i < num_simulations; ++i) {
        dc::GameState simulated_state = simulator_clustered_->run_single_simulation(initial_state, shot);
        float score = simulator_clustered_->evaluate(simulated_state);
        result.final_scores.push_back(score);
    }

    // Calculate mean
    float sum = 0.0f;
    for (float s : result.final_scores) {
        sum += s;
    }
    result.mean_score = result.final_scores.empty() ? 0.0f : sum / result.final_scores.size();

    // Calculate std
    float variance_sum = 0.0f;
    for (float s : result.final_scores) {
        variance_sum += (s - result.mean_score) * (s - result.mean_score);
    }
    result.std_score = result.final_scores.empty() ? 0.0f : std::sqrt(variance_sum / result.final_scores.size());

    return result;
}

// Analyze a single cluster
ClusterMemberAnalysis AgreementExperiment::analyzeSingleCluster(
    const dc::GameState& initial_state,
    int cluster_id,
    const std::vector<int>& member_ids,
    int allgrid_shot_id,
    int clustered_shot_id
) {
    ClusterMemberAnalysis analysis;
    analysis.cluster_id = cluster_id;
    analysis.allgrid_selected_shot_id = allgrid_shot_id;
    analysis.clustered_selected_shot_id = clustered_shot_id;
    analysis.member_shot_ids = member_ids;
    analysis.contains_allgrid_shot = false;
    analysis.contains_clustered_shot = false;

    float total_score = 0.0f;
    std::vector<float> all_mean_scores;

    for (int member_id : member_ids) {
        if (state_to_shot_table_.find(member_id) == state_to_shot_table_.end()) {
            continue;
        }
        const ShotInfo& shot = state_to_shot_table_.at(member_id);
        ShotSimulationResult sim_result = simulateShotMultipleTimes(initial_state, shot, member_id, simulations_per_shot_);
        analysis.member_results.push_back(sim_result);
        all_mean_scores.push_back(sim_result.mean_score);
        total_score += sim_result.mean_score;

        if (member_id == allgrid_shot_id) {
            analysis.contains_allgrid_shot = true;
        }
        if (member_id == clustered_shot_id) {
            analysis.contains_clustered_shot = true;
        }
    }

    // Calculate cluster mean and variance
    analysis.cluster_mean_score = member_ids.empty() ? 0.0f : total_score / member_ids.size();

    float variance_sum = 0.0f;
    for (float s : all_mean_scores) {
        variance_sum += (s - analysis.cluster_mean_score) * (s - analysis.cluster_mean_score);
    }
    analysis.cluster_score_variance = member_ids.empty() ? 0.0f : variance_sum / member_ids.size();

    return analysis;
}

// Analyze cluster members when cluster agreement is found
ClusterMemberAnalysis AgreementExperiment::analyzeClusterMembers(
    const dc::GameState& initial_state,
    int allgrid_shot_id,
    int clustered_shot_id,
    const std::vector<std::vector<int>>& cluster_table
) {
    // Find which cluster contains both shots (assuming cluster agreement)
    int target_cluster_id = -1;
    for (size_t i = 0; i < cluster_table.size(); ++i) {
        for (int member_id : cluster_table[i]) {
            if (member_id == allgrid_shot_id || member_id == clustered_shot_id) {
                target_cluster_id = static_cast<int>(i);
                break;
            }
        }
        if (target_cluster_id >= 0) break;
    }

    if (target_cluster_id >= 0 && target_cluster_id < static_cast<int>(cluster_table.size())) {
        return analyzeSingleCluster(initial_state, target_cluster_id, cluster_table[target_cluster_id], allgrid_shot_id, clustered_shot_id);
    }

    // Return empty analysis if cluster not found
    ClusterMemberAnalysis empty_analysis;
    empty_analysis.cluster_id = -1;
    empty_analysis.allgrid_selected_shot_id = allgrid_shot_id;
    empty_analysis.clustered_selected_shot_id = clustered_shot_id;
    return empty_analysis;
}

// Analyze all clusters
AllClusterAnalysis AgreementExperiment::analyzeAllClusters(
    const dc::GameState& initial_state,
    int allgrid_shot_id,
    int clustered_shot_id,
    const std::vector<std::vector<int>>& cluster_table
) {
    AllClusterAnalysis all_analysis;
    all_analysis.allgrid_cluster_id = -1;
    all_analysis.clustered_cluster_id = -1;
    all_analysis.best_cluster_id = -1;
    float best_mean_score = -1e9f;

    for (size_t i = 0; i < cluster_table.size(); ++i) {
        ClusterMemberAnalysis cluster_analysis = analyzeSingleCluster(
            initial_state, static_cast<int>(i), cluster_table[i], allgrid_shot_id, clustered_shot_id);
        all_analysis.all_clusters.push_back(cluster_analysis);

        if (cluster_analysis.contains_allgrid_shot) {
            all_analysis.allgrid_cluster_id = static_cast<int>(i);
        }
        if (cluster_analysis.contains_clustered_shot) {
            all_analysis.clustered_cluster_id = static_cast<int>(i);
        }
        if (cluster_analysis.cluster_mean_score > best_mean_score) {
            best_mean_score = cluster_analysis.cluster_mean_score;
            all_analysis.best_cluster_id = static_cast<int>(i);
        }
    }

    return all_analysis;
}

// Generate best shot comparison
BestShotComparison AgreementExperiment::generateBestShotComparison(
    const dc::GameState& initial_state,
    int allgrid_shot_id,
    int clustered_shot_id
) {
    BestShotComparison comparison;
    comparison.allgrid_shot_id = allgrid_shot_id;
    comparison.clustered_shot_id = clustered_shot_id;
    comparison.best_overall_shot_id = -1;
    comparison.best_overall_mean_score = -1e9f;
    comparison.allgrid_mean_score = 0.0f;
    comparison.clustered_mean_score = 0.0f;

    // Simulate all shots
    for (const auto& [shot_id, shot_info] : state_to_shot_table_) {
        ShotSimulationResult result = simulateShotMultipleTimes(initial_state, shot_info, shot_id, simulations_per_shot_);
        comparison.all_shot_results.push_back(result);

        if (result.mean_score > comparison.best_overall_mean_score) {
            comparison.best_overall_mean_score = result.mean_score;
            comparison.best_overall_shot_id = shot_id;
        }

        if (shot_id == allgrid_shot_id) {
            comparison.allgrid_mean_score = result.mean_score;
        }
        if (shot_id == clustered_shot_id) {
            comparison.clustered_mean_score = result.mean_score;
        }
    }

    return comparison;
}