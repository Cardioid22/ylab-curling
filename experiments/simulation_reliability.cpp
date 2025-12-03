#include "simulation_reliability.h"
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <iomanip>

namespace dc = digitalcurling3;

// ========================================
// SimulationReliabilityExperiment Implementation
// ========================================

SimulationReliabilityExperiment::SimulationReliabilityExperiment(
    dc::Team team,
    std::shared_ptr<SimulatorWrapper> sim_wrapper,
    int grid_m,
    int grid_n
)
    : team_(team),
      simulator_(sim_wrapper),
      grid_m_(grid_m),
      grid_n_(grid_n)
{
}

ReliabilityExperimentResult SimulationReliabilityExperiment::runExperiment(
    int num_patterns_per_type,
    int leaves_per_test_case,
    int simulations_per_leaf,
    int mcts_iterations
) {
    ReliabilityExperimentResult result;
    result.num_test_cases = 0;
    result.leaves_per_test_case = leaves_per_test_case;
    result.simulations_per_leaf = simulations_per_leaf;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Simulation Reliability Experiment" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Patterns per type: " << num_patterns_per_type << std::endl;
    std::cout << "Leaves per test case: " << leaves_per_test_case << std::endl;
    std::cout << "Simulations per leaf: " << simulations_per_leaf << std::endl;
    std::cout << "MCTS iterations: " << mcts_iterations << std::endl;
    std::cout << "========================================\n" << std::endl;

    // ========================================
    // PHASE 1: Generate test states and collect all leaves
    // ========================================
    std::cout << "\n=== PHASE 1: Collecting Leaves ===" << std::endl;

    ClusteringValidation validator(team_);
    std::vector<TestState> test_states = validator.generateTestStates(num_patterns_per_type);
    result.num_test_cases = static_cast<int>(test_states.size());

    std::cout << "Generated " << test_states.size() << " test states" << std::endl;

    int global_leaf_id = 0;
    std::vector<LeafStateInfo> all_leaf_infos;  // Collect all leaves here

    // For each test case, collect sampled leaves
    for (size_t test_idx = 0; test_idx < test_states.size(); ++test_idx) {
        const auto& test_state = test_states[test_idx];

        // Store test case description for later use in export
        result.test_case_descriptions[test_state.test_id] = test_state.description;

        std::cout << "\n--- Test Case " << test_idx + 1 << "/" << test_states.size()
                  << " (" << test_state.description << ") ---" << std::endl;

        // Step 1a: Build MCTS tree
        std::cout << "Building MCTS tree..." << std::endl;
        auto root = buildMCTSTree(test_state.state, mcts_iterations);

        if (!root) {
            std::cerr << "Failed to build MCTS tree for test case " << test_idx << std::endl;
            continue;
        }

        // Step 1b: Extract all leaf nodes
        std::vector<MCTS_Node*> all_leaves;
        extractLeafNodes(root.get(), all_leaves);

        std::cout << "Extracted " << all_leaves.size() << " leaf nodes" << std::endl;

        if (all_leaves.empty()) {
            std::cerr << "No leaf nodes found for test case " << test_idx << std::endl;
            continue;
        }

        // Step 1c: Randomly sample leaf nodes
        int num_samples = std::min(leaves_per_test_case, static_cast<int>(all_leaves.size()));
        std::vector<MCTS_Node*> sampled_leaves = sampleRandomLeaves(all_leaves, num_samples);

        std::cout << "Sampled " << sampled_leaves.size() << " random leaf nodes" << std::endl;

        // Step 1d: Create leaf info for all sampled leaves
        for (size_t leaf_idx = 0; leaf_idx < sampled_leaves.size(); ++leaf_idx) {
            MCTS_Node* leaf = sampled_leaves[leaf_idx];
            LeafStateInfo leaf_info = createLeafInfo(leaf, test_state.test_id, global_leaf_id);
            all_leaf_infos.push_back(leaf_info);
            global_leaf_id++;
        }
    }

    std::cout << "\n=== PHASE 1 Complete ===" << std::endl;
    std::cout << "Total leaves collected: " << all_leaf_infos.size() << std::endl;

    // ========================================
    // PHASE 2: Run simulations for all collected leaves (batch processing)
    // ========================================
    std::cout << "\n=== PHASE 2: Running Simulations (Batch) ===" << std::endl;

    result.sampled_leaves = all_leaf_infos;  // Store all leaf infos
    result.variance_data.reserve(all_leaf_infos.size());  // Pre-allocate

    for (size_t i = 0; i < all_leaf_infos.size(); ++i) {
        const auto& leaf_info = all_leaf_infos[i];

        std::cout << "Leaf " << i + 1 << "/" << all_leaf_infos.size()
                  << " (ID: " << leaf_info.leaf_id
                  << ", TestCase: " << leaf_info.test_case_id << ")..." << std::endl;

        // Run simulations for this leaf
        SimulationVarianceData variance_data = measureSimulationVariance(
            leaf_info,
            simulations_per_leaf
        );
        result.variance_data.push_back(variance_data);

        std::cout << "  Mean score: " << variance_data.mean_score
                  << ", Score variance: " << variance_data.score_variance
                  << ", Position variance: " << variance_data.position_variance << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Experiment completed!" << std::endl;
    std::cout << "Total leaves sampled: " << result.sampled_leaves.size() << std::endl;
    std::cout << "Total simulations run: " << (result.sampled_leaves.size() * simulations_per_leaf) << std::endl;
    std::cout << "========================================\n" << std::endl;

    return result;
}

std::unique_ptr<MCTS_Node> SimulationReliabilityExperiment::buildMCTSTree(
    const dc::GameState& test_state,
    int mcts_iterations
) {
    // Create a simple MCTS tree with the given state as root
    // For simplicity, we'll create a minimal tree structure

    auto root = std::make_unique<MCTS_Node>(
        nullptr,                        // parent
        test_state,                     // state
        NodeSource::Random,             // source (use Random for exploration)
        simulator_,                     // simulator
        grid_m_,                        // grid M
        grid_n_,                        // grid N
        4,                              // cluster_num (not used for Random, but required)
        std::nullopt,                   // shot candidates
        std::nullopt                    // selected shot
    );

    // Run a few iterations to build a small tree
    // We'll manually expand the tree to create some leaf nodes
    for (int iter = 0; iter < mcts_iterations; ++iter) {
        MCTS_Node* node = root.get();

        // Navigate to a node that can be expanded
        while (node->selected && node->is_fully_expanded() && !node->children.empty()) {
            // Randomly select a child
            if (node->children.empty()) break;
            node = node->children[0].get();
        }

        // Expand the node if possible
        if (!node->terminal && node->degree < 3) {  // Limit degree for this experiment
            // Use empty vectors for expand (won't use clustering)
            std::vector<dc::GameState> empty_states;
            std::unordered_map<int, ShotInfo> empty_table;
            node->expand(empty_states, empty_table);
        }
    }

    return root;
}

void SimulationReliabilityExperiment::extractLeafNodes(
    MCTS_Node* root,
    std::vector<MCTS_Node*>& leaves
) {
    if (!root) return;

    // A node is a leaf if it has no children
    if (root->children.empty()) {
        leaves.push_back(root);
        return;
    }

    // Recursively traverse children
    for (const auto& child : root->children) {
        extractLeafNodes(child.get(), leaves);
    }
}

std::vector<MCTS_Node*> SimulationReliabilityExperiment::sampleRandomLeaves(
    const std::vector<MCTS_Node*>& all_leaves,
    int num_samples
) {
    std::vector<MCTS_Node*> sampled;

    if (num_samples >= static_cast<int>(all_leaves.size())) {
        // Return all leaves
        return all_leaves;
    }

    // Random sampling without replacement
    std::vector<int> indices(all_leaves.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    for (int i = 0; i < num_samples; ++i) {
        sampled.push_back(all_leaves[indices[i]]);
    }

    return sampled;
}

LeafStateInfo SimulationReliabilityExperiment::createLeafInfo(
    MCTS_Node* node,
    int test_case_id,
    int leaf_id
) {
    LeafStateInfo info;
    info.state = node->state;
    info.shot_to_reach = node->selected_shot;
    info.test_case_id = test_case_id;
    info.leaf_id = leaf_id;

    // Count stones
    info.total_stones = countStones(node->state);
    info.my_stones = countStones(node->state, team_);
    info.opponent_stones = countStones(node->state, dc::GetOpponentTeam(team_));

    // Extract positions
    for (size_t idx = 0; idx < 8; ++idx) {
        const auto& my_stone = node->state.stones[static_cast<size_t>(team_)][idx];
        if (my_stone) {
            info.my_stone_positions.push_back(my_stone->position);
        }

        const auto& opp_stone = node->state.stones[static_cast<size_t>(dc::GetOpponentTeam(team_))][idx];
        if (opp_stone) {
            info.opponent_stone_positions.push_back(opp_stone->position);
        }
    }

    return info;
}

SimulationVarianceData SimulationReliabilityExperiment::measureSimulationVariance(
    const LeafStateInfo& leaf_info,
    int num_simulations
) {
    SimulationVarianceData data;
    data.leaf_id = leaf_info.leaf_id;
    data.num_simulations = num_simulations;

    // Run multiple simulations with the same shot from the same state
    for (int i = 0; i < num_simulations; ++i) {
        // Run simulation
        dc::GameState result_state = simulator_->run_single_simulation(
            leaf_info.state,
            leaf_info.shot_to_reach
        );

        // Store result
        data.result_states.push_back(result_state);

        // Extract stone positions
        std::vector<dc::Vector2> positions = extractStonePositions(result_state);
        data.result_stone_positions.push_back(positions);

        // Evaluate score
        float score = simulator_->evaluate(result_state);
        data.scores.push_back(score);
    }

    // Calculate statistics
    if (!data.scores.empty()) {
        // Mean score
        data.mean_score = std::accumulate(data.scores.begin(), data.scores.end(), 0.0f)
                          / data.scores.size();

        // Score variance
        float sq_sum = 0.0f;
        for (float score : data.scores) {
            sq_sum += (score - data.mean_score) * (score - data.mean_score);
        }
        data.score_variance = sq_sum / data.scores.size();

        // Position variance
        data.position_variance = calculatePositionVariance(data.result_states);
    } else {
        data.mean_score = 0.0f;
        data.score_variance = 0.0f;
        data.position_variance = 0.0f;
    }

    return data;
}

float SimulationReliabilityExperiment::calculatePositionVariance(
    const std::vector<dc::GameState>& result_states
) {
    if (result_states.empty()) return 0.0f;

    // For each stone index, calculate variance across all simulations
    float total_variance = 0.0f;
    int variance_count = 0;

    // Check each team and stone index
    for (size_t team = 0; team < 2; ++team) {
        for (size_t idx = 0; idx < 8; ++idx) {
            std::vector<dc::Vector2> positions;

            // Collect positions for this stone across all result states
            for (const auto& state : result_states) {
                if (state.stones[team][idx]) {
                    positions.push_back(state.stones[team][idx]->position);
                }
            }

            // If this stone exists in all simulations, calculate variance
            if (positions.size() == result_states.size()) {
                // Calculate mean position
                float mean_x = 0.0f, mean_y = 0.0f;
                for (const auto& pos : positions) {
                    mean_x += pos.x;
                    mean_y += pos.y;
                }
                mean_x /= positions.size();
                mean_y /= positions.size();

                // Calculate variance (sum of squared distances from mean)
                float variance = 0.0f;
                for (const auto& pos : positions) {
                    float dx = pos.x - mean_x;
                    float dy = pos.y - mean_y;
                    variance += (dx * dx + dy * dy);
                }
                variance /= positions.size();

                total_variance += variance;
                variance_count++;
            }
        }
    }

    return variance_count > 0 ? total_variance / variance_count : 0.0f;
}

std::vector<dc::Vector2> SimulationReliabilityExperiment::extractStonePositions(
    const dc::GameState& state
) {
    std::vector<dc::Vector2> positions;

    for (size_t team = 0; team < 2; ++team) {
        for (size_t idx = 0; idx < 8; ++idx) {
            if (state.stones[team][idx]) {
                positions.push_back(state.stones[team][idx]->position);
            }
        }
    }

    return positions;
}

int SimulationReliabilityExperiment::countStones(
    const dc::GameState& state,
    std::optional<dc::Team> team
) {
    int count = 0;

    if (team.has_value()) {
        // Count stones for specific team
        size_t team_idx = static_cast<size_t>(team.value());
        for (size_t idx = 0; idx < 8; ++idx) {
            if (state.stones[team_idx][idx]) {
                count++;
            }
        }
    } else {
        // Count all stones
        for (size_t t = 0; t < 2; ++t) {
            for (size_t idx = 0; idx < 8; ++idx) {
                if (state.stones[t][idx]) {
                    count++;
                }
            }
        }
    }

    return count;
}

void SimulationReliabilityExperiment::exportResults(
    const ReliabilityExperimentResult& result,
    const std::string& output_dir
) {
    // Create main output directory
    std::filesystem::create_directories(output_dir);

    std::cout << "\n[SimulationReliability] Exporting results to: " << output_dir << std::endl;

    // ========================================
    // Group leaves and variance data by test_case_id
    // ========================================
    std::map<int, std::vector<LeafStateInfo>> leaves_by_test_case;
    std::map<int, std::vector<SimulationVarianceData>> variance_by_test_case;

    // Build leaf ID to variance data mapping
    std::map<int, SimulationVarianceData> leaf_id_to_variance;
    for (const auto& variance : result.variance_data) {
        leaf_id_to_variance[variance.leaf_id] = variance;
    }

    // Group by test_case_id
    for (const auto& leaf : result.sampled_leaves) {
        leaves_by_test_case[leaf.test_case_id].push_back(leaf);

        // Find corresponding variance data
        if (leaf_id_to_variance.find(leaf.leaf_id) != leaf_id_to_variance.end()) {
            variance_by_test_case[leaf.test_case_id].push_back(leaf_id_to_variance[leaf.leaf_id]);
        }
    }

    // ========================================
    // Export data for each test case in separate folders
    // ========================================
    for (const auto& [test_case_id, leaves] : leaves_by_test_case) {
        // Get test case description
        std::string description = "unknown";
        if (result.test_case_descriptions.find(test_case_id) != result.test_case_descriptions.end()) {
            description = result.test_case_descriptions.at(test_case_id);
        }

        // Create test case directory
        std::string test_case_dir = output_dir + "/test_case_" + std::to_string(test_case_id)
                                    + "_" + description;
        std::filesystem::create_directories(test_case_dir);

        std::cout << "\n--- Exporting Test Case " << test_case_id << " (" << description << ") ---" << std::endl;

        // Get variance data for this test case
        const auto& variances = variance_by_test_case[test_case_id];

        // 1. Export leaf state information
        {
            std::string path = test_case_dir + "/leaf_states.csv";
            std::ofstream ofs(path);
            ofs << "leaf_id,test_case_id,total_stones,my_stones,opponent_stones,shot_vx,shot_vy,shot_rot\n";

            for (const auto& leaf : leaves) {
                ofs << leaf.leaf_id << ","
                    << leaf.test_case_id << ","
                    << leaf.total_stones << ","
                    << leaf.my_stones << ","
                    << leaf.opponent_stones << ","
                    << leaf.shot_to_reach.vx << ","
                    << leaf.shot_to_reach.vy << ","
                    << leaf.shot_to_reach.rot << "\n";
            }

            ofs.close();
            std::cout << "  Exported: " << path << std::endl;
        }

        // 2. Export simulation variance data
        {
            std::string path = test_case_dir + "/simulation_variance.csv";
            std::ofstream ofs(path);
            ofs << "leaf_id,num_simulations,mean_score,score_variance,position_variance\n";

            for (const auto& variance : variances) {
                ofs << variance.leaf_id << ","
                    << variance.num_simulations << ","
                    << variance.mean_score << ","
                    << variance.score_variance << ","
                    << variance.position_variance << "\n";
            }

            ofs.close();
            std::cout << "  Exported: " << path << std::endl;
        }

        // 3. Export detailed stone positions for each simulation
        {
            std::string path = test_case_dir + "/result_stone_positions.csv";
            std::ofstream ofs(path);
            ofs << "leaf_id,simulation_id,stone_idx,x,y\n";

            for (const auto& variance : variances) {
                for (size_t sim_idx = 0; sim_idx < variance.result_stone_positions.size(); ++sim_idx) {
                    const auto& positions = variance.result_stone_positions[sim_idx];
                    for (size_t stone_idx = 0; stone_idx < positions.size(); ++stone_idx) {
                        ofs << variance.leaf_id << ","
                            << sim_idx << ","
                            << stone_idx << ","
                            << positions[stone_idx].x << ","
                            << positions[stone_idx].y << "\n";
                    }
                }
            }

            ofs.close();
            std::cout << "  Exported: " << path << std::endl;
        }

        // 4. Export individual scores for each simulation
        {
            std::string path = test_case_dir + "/simulation_scores.csv";
            std::ofstream ofs(path);
            ofs << "leaf_id,simulation_id,score\n";

            for (const auto& variance : variances) {
                for (size_t sim_idx = 0; sim_idx < variance.scores.size(); ++sim_idx) {
                    ofs << variance.leaf_id << ","
                        << sim_idx << ","
                        << variance.scores[sim_idx] << "\n";
                }
            }

            ofs.close();
            std::cout << "  Exported: " << path << std::endl;
        }

        // 5. Export per-test-case summary
        {
            std::string path = test_case_dir + "/summary.csv";
            std::ofstream ofs(path);
            ofs << "metric,value\n";
            ofs << "test_case_id," << test_case_id << "\n";
            ofs << "description," << description << "\n";
            ofs << "num_leaves," << leaves.size() << "\n";
            ofs << "simulations_per_leaf," << result.simulations_per_leaf << "\n";
            ofs << "total_simulations," << (leaves.size() * result.simulations_per_leaf) << "\n";

            // Calculate average variances for this test case
            if (!variances.empty()) {
                float avg_score_var = 0.0f;
                float avg_pos_var = 0.0f;
                for (const auto& v : variances) {
                    avg_score_var += v.score_variance;
                    avg_pos_var += v.position_variance;
                }
                avg_score_var /= variances.size();
                avg_pos_var /= variances.size();

                ofs << "avg_score_variance," << avg_score_var << "\n";
                ofs << "avg_position_variance," << avg_pos_var << "\n";
            }

            ofs.close();
            std::cout << "  Exported: " << path << std::endl;
        }
    }

    // ========================================
    // Export global summary at top level
    // ========================================
    {
        std::string path = output_dir + "/global_summary.csv";
        std::ofstream ofs(path);
        ofs << "metric,value\n";
        ofs << "num_test_cases," << result.num_test_cases << "\n";
        ofs << "leaves_per_test_case," << result.leaves_per_test_case << "\n";
        ofs << "simulations_per_leaf," << result.simulations_per_leaf << "\n";
        ofs << "total_leaves_sampled," << result.sampled_leaves.size() << "\n";
        ofs << "total_simulations," << (result.sampled_leaves.size() * result.simulations_per_leaf) << "\n";

        // Calculate global average variances
        if (!result.variance_data.empty()) {
            float avg_score_var = 0.0f;
            float avg_pos_var = 0.0f;
            for (const auto& v : result.variance_data) {
                avg_score_var += v.score_variance;
                avg_pos_var += v.position_variance;
            }
            avg_score_var /= result.variance_data.size();
            avg_pos_var /= result.variance_data.size();

            ofs << "avg_score_variance," << avg_score_var << "\n";
            ofs << "avg_position_variance," << avg_pos_var << "\n";
        }

        ofs.close();
        std::cout << "\nExported global summary: " << path << std::endl;
    }

    std::cout << "\n[SimulationReliability] Export completed!\n" << std::endl;
    std::cout << "Total test case folders created: " << leaves_by_test_case.size() << std::endl;
}
