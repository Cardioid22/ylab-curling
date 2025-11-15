#pragma once
#ifndef _CLUSTERING_VALIDATION_H_
#define _CLUSTERING_VALIDATION_H_

#include "../src/structure.h"
#include "../src/clustering.h"
#include "../src/clustering-v2.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include <vector>
#include <string>
#include <map>

namespace dc = digitalcurling3;

/**
 * @brief Types of stone placement patterns for testing
 */
enum class StonePattern {
    // Basic patterns
    CenterGuard,        // 1 guard stone in center
    DoubleGuard,        // 2 guard stones in center
    CornerGuards,       // Guard stones in left and right corners

    // House patterns
    SingleDraw,         // 1 stone in house center
    DoubleDraw,         // 2 stones in house
    TripleDraw,         // 3 stones in house
    HouseCorners,       // Stones placed at four corners of house

    // Mixed patterns
    GuardAndDraw,       // 1 guard + 1 stone in house
    Split,              // Stones split left and right
    Crowded,            // Densely packed stones
    Scattered,          // Scattered stones

    // Tactical patterns
    FreezeAttempt,      // After freeze shot (clustered near house center)
    Takeout,            // After takeout (few stones)
    Promotion,          // After promotion (guard moved into house)
    Corner,             // Corner guard tactics

    // Symmetric/Asymmetric patterns
    Symmetric,          // Symmetric left-right placement
    Asymmetric,         // Asymmetric placement

    // Random
    Random              // Random placement
};

/**
 * @brief Test state information
 */
struct TestState {
    StonePattern pattern;           // Pattern type
    std::string description;        // Pattern description
    dc::GameState state;            // GameState
    int test_id;                    // Test ID
};

/**
 * @brief Clustering validation results
 */
struct ValidationResult {
    // Test information
    int num_test_states;
    int num_clusters;

    // Clustering V1 results
    std::vector<std::vector<int>> v1_clusters;
    std::vector<int> v1_representatives;

    // Clustering V2 results
    std::vector<std::vector<int>> v2_clusters;
    std::vector<int> v2_representatives;

    // Quality evaluation
    float v2_quality_score;

    // Execution time
    double v1_time_ms;
    double v2_time_ms;
};

/**
 * @brief ClusteringV2 validation experiment class
 */
class ClusteringValidation {
public:
    ClusteringValidation(dc::Team team);

    /**
     * @brief Generate test stone placements
     *
     * Generate various patterns of stone placements to validate clustering quality
     *
     * @param num_patterns_per_type Number of patterns to generate per type
     * @return Vector of test states
     */
    std::vector<TestState> generateTestStates(int num_patterns_per_type = 3);

    /**
     * @brief Generate stone placement for specific pattern
     *
     * @param pattern Pattern type
     * @param variation Variation number (0-based)
     * @param test_id Test ID
     * @return TestState
     */
    TestState createPattern(StonePattern pattern, int variation, int test_id);

    /**
     * @brief Run and compare both clustering methods
     *
     * @param test_states Test states
     * @param target_clusters Target number of clusters
     * @return ValidationResult
     */
    ValidationResult runComparison(
        const std::vector<TestState>& test_states,
        int target_clusters
    );

    /**
     * @brief Export validation results to CSV
     *
     * Generates the following files:
     * 1. test_states.csv - Information about each test state
     * 2. v1_clusters.csv - V1 clustering results
     * 3. v2_clusters.csv - V2 clustering results
     * 4. stone_coordinates.csv - Stone coordinates (for visualization)
     * 5. comparison_summary.csv - Comparison summary
     *
     * @param result Validation results
     * @param test_states Test states
     * @param output_dir Output directory
     */
    void exportResults(
        const ValidationResult& result,
        const std::vector<TestState>& test_states,
        const std::string& output_dir
    );

    /**
     * @brief Export stone coordinates to CSV (for visualization)
     *
     * Format compatible with existing draw_remake_house.py
     *
     * @param states Vector of GameStates
     * @param output_path Output path
     */
    void exportStoneCoordinates(
        const std::vector<dc::GameState>& states,
        const std::string& output_path
    );

    /**
     * @brief Export cluster distribution in grid format (for visualization)
     *
     * Format compatible with existing draw_distribution.py
     *
     * @param cluster_assignments Cluster ID for each state
     * @param grid_m Grid row count
     * @param grid_n Grid column count
     * @param output_path Output path
     */
    void exportClusterDistribution(
        const std::vector<int>& cluster_assignments,
        int grid_m,
        int grid_n,
        const std::string& output_path
    );

private:
    dc::Team team_;
    dc::GameSetting game_setting_;

    // Constants
    static constexpr float HouseRadius_ = 1.829f;
    static constexpr float HouseCenterX_ = 0.0f;
    static constexpr float HouseCenterY_ = 38.405f;
    static constexpr float GuardLineY_ = 34.747f;  // Hog line + approx 4m

    /**
     * @brief Generate a basic empty GameState
     */
    dc::GameState createEmptyState();

    /**
     * @brief Place a stone at a specific position
     *
     * @param state GameState
     * @param team Team
     * @param index Stone index (0-7)
     * @param x X coordinate
     * @param y Y coordinate
     */
    void placeStone(
        dc::GameState& state,
        dc::Team team,
        int index,
        float x,
        float y
    );

    /**
     * @brief Place a stone at a random position
     */
    void placeRandomStone(
        dc::GameState& state,
        dc::Team team,
        int index
    );

    /**
     * @brief Convert pattern name to string
     */
    std::string patternToString(StonePattern pattern);
};

#endif // _CLUSTERING_VALIDATION_H_
