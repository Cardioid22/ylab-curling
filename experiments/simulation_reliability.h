#pragma once
#ifndef _SIMULATION_RELIABILITY_H_
#define _SIMULATION_RELIABILITY_H_

#include "../src/structure.h"
#include "../src/mcts.h"
#include "../src/simulator.h"
#include "clustering_validation.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>

namespace dc = digitalcurling3;

/**
 * @brief Information about a leaf state in MCTS tree
 */
struct LeafStateInfo {
    dc::GameState state;                // Leaf game state
    ShotInfo shot_to_reach;             // Shot that led to this leaf
    int test_case_id;                   // ID of the test case this leaf came from
    int leaf_id;                        // Unique ID for this leaf
    int total_stones;                   // Total number of stones on the sheet
    int my_stones;                      // Number of my team's stones
    int opponent_stones;                // Number of opponent's stones
    std::vector<dc::Vector2> my_stone_positions;       // Positions of my stones
    std::vector<dc::Vector2> opponent_stone_positions; // Positions of opponent's stones
};

/**
 * @brief Result of multiple simulations from a single leaf state
 */
struct SimulationVarianceData {
    int leaf_id;                                          // ID of the leaf state
    int num_simulations;                                  // Number of simulations performed
    std::vector<dc::GameState> result_states;             // All resulting states
    std::vector<std::vector<dc::Vector2>> result_stone_positions; // Stone positions for each result
    float position_variance;                              // Variance in stone positions
    float score_variance;                                 // Variance in evaluation scores
    float mean_score;                                     // Mean evaluation score
    std::vector<float> scores;                            // All scores from simulations
};

/**
 * @brief Complete experiment results
 */
struct ReliabilityExperimentResult {
    std::vector<LeafStateInfo> sampled_leaves;            // All sampled leaf states
    std::vector<SimulationVarianceData> variance_data;    // Variance data for each leaf
    int num_test_cases;                                   // Number of test cases used
    int leaves_per_test_case;                             // Number of leaves sampled per test case
    int simulations_per_leaf;                             // Number of simulations per leaf
    std::map<int, std::string> test_case_descriptions;    // Map: test_case_id -> description
};

/**
 * @brief Experiment class to measure simulation reliability
 *
 * This class tests the consistency of the physics simulator by:
 * 1. Taking test cases from ClusteringValidation
 * 2. Building MCTS trees from each test case
 * 3. Randomly sampling leaf nodes from each tree
 * 4. Running multiple simulations from each leaf with the same shot
 * 5. Measuring the variance in the resulting states
 */
class SimulationReliabilityExperiment {
public:
    /**
     * @brief Constructor
     *
     * @param team The team for which to run the experiment
     * @param sim_wrapper Simulator wrapper for running simulations
     * @param grid_m Grid size M
     * @param grid_n Grid size N
     */
    SimulationReliabilityExperiment(
        dc::Team team,
        std::shared_ptr<SimulatorWrapper> sim_wrapper,
        int grid_m,
        int grid_n
    );

    /**
     * @brief Run the complete experiment
     *
     * @param num_patterns_per_type Number of test patterns per pattern type
     * @param leaves_per_test_case Number of leaves to sample from each test case
     * @param simulations_per_leaf Number of simulations to run from each leaf
     * @param mcts_iterations Number of MCTS iterations to build the tree
     * @return Complete experiment results
     */
    ReliabilityExperimentResult runExperiment(
        int num_patterns_per_type = 1,
        int leaves_per_test_case = 5,
        int simulations_per_leaf = 100,
        int mcts_iterations = 50
    );

    /**
     * @brief Export experiment results to CSV files
     *
     * Creates the following files:
     * 1. leaf_states.csv - Information about sampled leaf states
     * 2. simulation_variance.csv - Variance data for each leaf
     * 3. result_stone_positions.csv - Detailed stone positions from all simulations
     * 4. summary.csv - Overall experiment summary
     *
     * @param result Experiment results
     * @param output_dir Output directory
     */
    void exportResults(
        const ReliabilityExperimentResult& result,
        const std::string& output_dir
    );

private:
    dc::Team team_;
    std::shared_ptr<SimulatorWrapper> simulator_;
    int grid_m_;
    int grid_n_;

    /**
     * @brief Build MCTS tree from a test state
     *
     * @param test_state The test state to start from
     * @param mcts_iterations Number of iterations
     * @return Root node of the MCTS tree
     */
    std::unique_ptr<MCTS_Node> buildMCTSTree(
        const dc::GameState& test_state,
        int mcts_iterations
    );

    /**
     * @brief Extract all leaf nodes from an MCTS tree
     *
     * @param root Root node of the tree
     * @param leaves Output vector of leaf nodes
     */
    void extractLeafNodes(
        MCTS_Node* root,
        std::vector<MCTS_Node*>& leaves
    );

    /**
     * @brief Randomly sample N leaf nodes
     *
     * @param all_leaves All available leaf nodes
     * @param num_samples Number of samples to take
     * @return Sampled leaf nodes
     */
    std::vector<MCTS_Node*> sampleRandomLeaves(
        const std::vector<MCTS_Node*>& all_leaves,
        int num_samples
    );

    /**
     * @brief Create LeafStateInfo from an MCTS_Node
     *
     * @param node MCTS node (leaf)
     * @param test_case_id Test case ID
     * @param leaf_id Unique leaf ID
     * @return LeafStateInfo
     */
    LeafStateInfo createLeafInfo(
        MCTS_Node* node,
        int test_case_id,
        int leaf_id
    );

    /**
     * @brief Run multiple simulations from a leaf state and measure variance
     *
     * @param leaf_info Information about the leaf state
     * @param num_simulations Number of simulations to run
     * @return Simulation variance data
     */
    SimulationVarianceData measureSimulationVariance(
        const LeafStateInfo& leaf_info,
        int num_simulations
    );

    /**
     * @brief Calculate variance in stone positions across multiple result states
     *
     * @param result_states Vector of resulting game states
     * @return Position variance
     */
    float calculatePositionVariance(
        const std::vector<dc::GameState>& result_states
    );

    /**
     * @brief Extract stone positions from a game state
     *
     * @param state Game state
     * @return Vector of stone positions
     */
    std::vector<dc::Vector2> extractStonePositions(
        const dc::GameState& state
    );

    /**
     * @brief Count stones in a game state
     *
     * @param state Game state
     * @param team Team to count (or both if nullopt)
     * @return Number of stones
     */
    int countStones(
        const dc::GameState& state,
        std::optional<dc::Team> team = std::nullopt
    );
};

#endif // _SIMULATION_RELIABILITY_H_
