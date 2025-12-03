#include "clustering_validation.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <filesystem>
#include <cmath>

namespace dc = digitalcurling3;

// ========================================
// ClusteringValidation Implementation
// ========================================

ClusteringValidation::ClusteringValidation(dc::Team team)
    : team_(team)
{
    // Initialize game settings
    game_setting_.max_end = 1;
    game_setting_.five_rock_rule = true;
    game_setting_.thinking_time[0] = std::chrono::seconds(86400);
    game_setting_.thinking_time[1] = std::chrono::seconds(86400);
}

std::vector<TestState> ClusteringValidation::generateTestStates(int num_patterns_per_type) {
    std::vector<TestState> test_states;
    int test_id = 0;

    // Enumerate all pattern types (reduced to 10 representative patterns)
    std::vector<StonePattern> all_patterns = {
        StonePattern::CenterGuard,      // 1. Basic guard
        StonePattern::CornerGuards,     // 2. Corner tactics
        StonePattern::SingleDraw,       // 3. Single stone in house
        StonePattern::DoubleDraw,       // 4. Two stones in house
        StonePattern::HouseCorners,     // 5. Four corners placement
        StonePattern::GuardAndDraw,     // 6. Guard + house combination
        StonePattern::Crowded,          // 7. Dense clustering
        StonePattern::FreezeAttempt,    // 8. Freeze shot scenario
        StonePattern::Corner,           // 9. Complex corner tactics
        StonePattern::Random            // 10. Random placement
    };

    std::cout << "[ClusteringValidation] Generating test states..." << std::endl;

    for (auto pattern : all_patterns) {
        for (int variation = 0; variation < num_patterns_per_type; ++variation) {
            test_states.push_back(createPattern(pattern, variation, test_id));
            test_id++;
        }
    }

    std::cout << "[ClusteringValidation] Generated " << test_states.size()
              << " test states" << std::endl;

    return test_states;
}

TestState ClusteringValidation::createPattern(StonePattern pattern, int variation, int test_id) {
    TestState test_state;
    test_state.pattern = pattern;
    test_state.test_id = test_id;
    test_state.description = patternToString(pattern) + "_v" + std::to_string(variation);
    test_state.state = createEmptyState();

    auto& state = test_state.state;
    float offset = variation * 0.3f;  // Offset for variation

    switch (pattern) {
    case StonePattern::CenterGuard:
        // 1 guard stone in center
        placeStone(state, team_, 0, 0.0f, GuardLineY_ + offset);
        break;

    case StonePattern::DoubleGuard:
        // 2 guard stones in center
        placeStone(state, team_, 0, -0.3f, GuardLineY_ + offset);
        placeStone(state, team_, 1, 0.3f, GuardLineY_ + offset);
        break;

    case StonePattern::CornerGuards:
        // Guard stones in left and right corners
        placeStone(state, team_, 0, -1.5f, GuardLineY_ + offset);
        placeStone(state, team_, 1, 1.5f, GuardLineY_ + offset);
        break;

    case StonePattern::SingleDraw:
        // 1 stone in house center
        placeStone(state, team_, 0, 0.0f + offset * 0.3f, HouseCenterY_);
        break;

    case StonePattern::DoubleDraw:
        // 2 stones in house
        placeStone(state, team_, 0, -0.4f + offset * 0.2f, HouseCenterY_ - 0.3f);
        placeStone(state, team_, 1, 0.4f + offset * 0.2f, HouseCenterY_ + 0.3f);
        break;

    case StonePattern::TripleDraw:
        // 3 stones in house
        placeStone(state, team_, 0, 0.0f, HouseCenterY_);
        placeStone(state, team_, 1, -0.6f + offset * 0.2f, HouseCenterY_ - 0.5f);
        placeStone(state, team_, 2, 0.6f + offset * 0.2f, HouseCenterY_ + 0.5f);
        break;

    case StonePattern::HouseCorners:
        // Stones placed at four corners of house
        placeStone(state, team_, 0, -1.2f, HouseCenterY_ - 1.2f);
        placeStone(state, team_, 1, 1.2f, HouseCenterY_ - 1.2f);
        placeStone(state, dc::GetOpponentTeam(team_), 0, -1.2f, HouseCenterY_ + 1.2f);
        placeStone(state, dc::GetOpponentTeam(team_), 1, 1.2f, HouseCenterY_ + 1.2f);
        break;

    case StonePattern::GuardAndDraw:
        // 1 guard + 1 stone in house
        placeStone(state, team_, 0, 0.0f + offset * 0.5f, GuardLineY_);
        placeStone(state, team_, 1, 0.0f + offset * 0.3f, HouseCenterY_);
        break;

    case StonePattern::Split:
        // Stones split left and right
        placeStone(state, team_, 0, -1.5f, HouseCenterY_ - 0.5f);
        placeStone(state, team_, 1, -1.3f, GuardLineY_);
        placeStone(state, dc::GetOpponentTeam(team_), 0, 1.5f, HouseCenterY_ + 0.5f);
        placeStone(state, dc::GetOpponentTeam(team_), 1, 1.3f, GuardLineY_);
        break;

    case StonePattern::Crowded:
        // Densely packed stones (near house center)
        placeStone(state, team_, 0, -0.2f, HouseCenterY_ - 0.3f);
        placeStone(state, team_, 1, 0.2f, HouseCenterY_ + 0.2f);
        placeStone(state, team_, 2, 0.0f, HouseCenterY_ + 0.5f);
        placeStone(state, dc::GetOpponentTeam(team_), 0, -0.3f, HouseCenterY_ + 0.1f);
        placeStone(state, dc::GetOpponentTeam(team_), 1, 0.3f, HouseCenterY_ - 0.2f);
        break;

    case StonePattern::Scattered:
        // Scattered stones
        placeStone(state, team_, 0, -1.8f, GuardLineY_);
        placeStone(state, team_, 1, 0.5f, HouseCenterY_ - 1.5f);
        placeStone(state, dc::GetOpponentTeam(team_), 0, 1.5f, GuardLineY_ + 1.0f);
        placeStone(state, dc::GetOpponentTeam(team_), 1, -0.8f, HouseCenterY_ + 1.0f);
        break;

    case StonePattern::FreezeAttempt:
        // After freeze shot (clustered)
        placeStone(state, dc::GetOpponentTeam(team_), 0, 0.0f, HouseCenterY_);
        placeStone(state, team_, 0, 0.15f + offset * 0.1f, HouseCenterY_ + 0.15f);
        placeStone(state, team_, 1, -0.2f, HouseCenterY_ - 0.3f);
        break;

    case StonePattern::Takeout:
        // After takeout (few stones)
        placeStone(state, team_, 0, 0.0f + offset * 0.5f, HouseCenterY_ + 0.8f);
        break;

    case StonePattern::Promotion:
        // After promotion (guard moved into house)
        placeStone(state, team_, 0, -0.5f, GuardLineY_);
        placeStone(state, team_, 1, -0.3f + offset * 0.3f, HouseCenterY_ - 0.5f);
        placeStone(state, dc::GetOpponentTeam(team_), 0, 0.0f, HouseCenterY_);
        break;

    case StonePattern::Corner:
        // Corner guard tactics
        placeStone(state, team_, 0, -1.6f, GuardLineY_);
        placeStone(state, team_, 1, -1.4f, GuardLineY_ + 1.0f);
        placeStone(state, team_, 2, -0.8f + offset * 0.3f, HouseCenterY_ - 0.5f);
        placeStone(state, dc::GetOpponentTeam(team_), 0, 0.0f, HouseCenterY_);
        break;

    case StonePattern::Symmetric:
        // Symmetric left-right placement
        placeStone(state, team_, 0, -1.0f, GuardLineY_);
        placeStone(state, team_, 1, 1.0f, GuardLineY_);
        placeStone(state, team_, 2, -0.5f, HouseCenterY_ - 0.5f);
        placeStone(state, team_, 3, 0.5f, HouseCenterY_ - 0.5f);
        break;

    case StonePattern::Asymmetric:
        // Asymmetric placement
        placeStone(state, team_, 0, -1.5f, GuardLineY_);
        placeStone(state, team_, 1, 0.3f, GuardLineY_ + 0.5f);
        placeStone(state, team_, 2, -0.2f, HouseCenterY_);
        placeStone(state, dc::GetOpponentTeam(team_), 0, 1.2f, HouseCenterY_ + 0.8f);
        break;

    case StonePattern::Random:
        // Random placement
        {
            std::random_device rd;
            std::mt19937 gen(rd() + variation);  // Change seed by variation
            std::uniform_int_distribution<> num_stones(2, 6);
            int n = num_stones(gen);

            for (int i = 0; i < n; ++i) {
                dc::Team t = (i % 2 == 0) ? team_ : dc::GetOpponentTeam(team_);
                placeRandomStone(state, t, i / 2);
            }
        }
        break;
    }

    return test_state;
}

ValidationResult ClusteringValidation::runComparison(
    const std::vector<TestState>& test_states,
    int target_clusters
) {
    ValidationResult result;
    result.num_test_states = static_cast<int>(test_states.size());
    result.num_clusters = target_clusters;

    // Create vector of GameStates
    std::vector<dc::GameState> states;
    states.reserve(test_states.size());
    for (const auto& ts : test_states) {
        states.push_back(ts.state);
    }

    std::cout << "\n[ClusteringValidation] Running comparison..." << std::endl;
    std::cout << "Test states: " << states.size() << std::endl;
    std::cout << "Target clusters: " << target_clusters << std::endl;

    // === Clustering V1 ===
    std::cout << "\n--- Clustering V1 ---" << std::endl;
    auto start_v1 = std::chrono::high_resolution_clock::now();

    Clustering v1_algo(target_clusters, states, 1, states.size(), team_);
    auto v1_clusters_set = v1_algo.getClusters();
    result.v1_representatives = v1_algo.getRecommendedStates();

    auto end_v1 = std::chrono::high_resolution_clock::now();
    result.v1_time_ms = std::chrono::duration<double, std::milli>(end_v1 - start_v1).count();

    // Convert set to vector
    result.v1_clusters.clear();
    for (const auto& cluster_set : v1_clusters_set) {
        std::vector<int> cluster_vec(cluster_set.begin(), cluster_set.end());
        result.v1_clusters.push_back(cluster_vec);
    }

    std::cout << "V1 Clusters: " << result.v1_clusters.size() << std::endl;
    std::cout << "V1 Time: " << result.v1_time_ms << " ms" << std::endl;

    // === Clustering V2 ===
    std::cout << "\n--- Clustering V2 ---" << std::endl;
    auto start_v2 = std::chrono::high_resolution_clock::now();

    ClusteringV2 v2_algo(target_clusters, states, 1, states.size(), team_);
    auto v2_clusters_obj = v2_algo.getClusters();
    result.v2_representatives = v2_algo.getRecommendedStates();
    result.v2_quality_score = v2_algo.evaluateClusteringQuality();

    auto end_v2 = std::chrono::high_resolution_clock::now();
    result.v2_time_ms = std::chrono::duration<double, std::milli>(end_v2 - start_v2).count();

    // Convert Cluster object to vector
    result.v2_clusters.clear();
    for (const auto& cluster : v2_clusters_obj) {
        result.v2_clusters.push_back(cluster.state_ids);
    }

    std::cout << "V2 Clusters: " << result.v2_clusters.size() << std::endl;
    std::cout << "V2 Quality Score: " << result.v2_quality_score << std::endl;
    std::cout << "V2 Time: " << result.v2_time_ms << " ms" << std::endl;

    std::cout << "\n[ClusteringValidation] Comparison completed!" << std::endl;

    return result;
}

void ClusteringValidation::exportResults(
    const ValidationResult& result,
    const std::vector<TestState>& test_states,
    const std::string& output_dir
) {
    // Create directory
    std::filesystem::create_directories(output_dir);

    std::cout << "\n[ClusteringValidation] Exporting results to: " << output_dir << std::endl;

    // 1. Test state information
    {
        std::string path = output_dir + "/test_states.csv";
        std::ofstream ofs(path);
        ofs << "test_id,pattern,description,total_stones,my_stones,opponent_stones\n";

        for (const auto& ts : test_states) {
            int total = 0, my = 0, opp = 0;
            for (size_t team = 0; team < 2; ++team) {
                for (size_t idx = 0; idx < 8; ++idx) {
                    if (ts.state.stones[team][idx]) {
                        total++;
                        if (team == static_cast<size_t>(team_)) my++;
                        else opp++;
                    }
                }
            }
            ofs << ts.test_id << ","
                << patternToString(ts.pattern) << ","
                << ts.description << ","
                << total << "," << my << "," << opp << "\n";
        }
        ofs.close();
        std::cout << "Exported: " << path << std::endl;
    }

    // 2. V1 clustering results
    {
        std::string path = output_dir + "/v1_clusters.csv";
        std::ofstream ofs(path);
        ofs << "cluster_id,state_id,is_representative\n";

        for (size_t c = 0; c < result.v1_clusters.size(); ++c) {
            for (int state_id : result.v1_clusters[c]) {
                bool is_rep = std::find(result.v1_representatives.begin(),
                                       result.v1_representatives.end(),
                                       state_id) != result.v1_representatives.end();
                ofs << c << "," << state_id << "," << (is_rep ? 1 : 0) << "\n";
            }
        }
        ofs.close();
        std::cout << "Exported: " << path << std::endl;
    }

    // 3. V2 clustering results
    {
        std::string path = output_dir + "/v2_clusters.csv";
        std::ofstream ofs(path);
        ofs << "cluster_id,state_id,is_representative\n";

        for (size_t c = 0; c < result.v2_clusters.size(); ++c) {
            for (int state_id : result.v2_clusters[c]) {
                bool is_rep = std::find(result.v2_representatives.begin(),
                                       result.v2_representatives.end(),
                                       state_id) != result.v2_representatives.end();
                ofs << c << "," << state_id << "," << (is_rep ? 1 : 0) << "\n";
            }
        }
        ofs.close();
        std::cout << "Exported: " << path << std::endl;
    }

    // 4. Stone coordinates (for visualization)
    {
        std::vector<dc::GameState> states;
        for (const auto& ts : test_states) {
            states.push_back(ts.state);
        }
        exportStoneCoordinates(states, output_dir + "/stone_coordinates.csv");
    }

    // 5. Comparison summary
    {
        std::string path = output_dir + "/comparison_summary.csv";
        std::ofstream ofs(path);
        ofs << "metric,v1,v2\n";
        ofs << "num_clusters," << result.v1_clusters.size() << "," << result.v2_clusters.size() << "\n";
        ofs << "time_ms," << result.v1_time_ms << "," << result.v2_time_ms << "\n";
        ofs << "quality_score,N/A," << result.v2_quality_score << "\n";
        ofs.close();
        std::cout << "Exported: " << path << std::endl;
    }

    std::cout << "[ClusteringValidation] Export completed!\n" << std::endl;
}

void ClusteringValidation::exportStoneCoordinates(
    const std::vector<dc::GameState>& states,
    const std::string& output_path
) {
    std::ofstream ofs(output_path);

    // Header (8 stones per team × 2 coordinates)
    ofs << "state_id";
    for (int team = 0; team < 2; ++team) {
        for (int idx = 0; idx < 8; ++idx) {
            ofs << ",team" << team << "_stone" << idx << "_x"
                << ",team" << team << "_stone" << idx << "_y";
        }
    }
    ofs << "\n";

    // Stone coordinates for each state
    for (size_t state_id = 0; state_id < states.size(); ++state_id) {
        ofs << state_id;
        for (size_t team = 0; team < 2; ++team) {
            for (size_t idx = 0; idx < 8; ++idx) {
                const auto& stone = states[state_id].stones[team][idx];
                if (stone) {
                    ofs << "," << stone->position.x << "," << stone->position.y;
                } else {
                    ofs << ",,";  // Empty field
                }
            }
        }
        ofs << "\n";
    }

    ofs.close();
    std::cout << "Exported: " << output_path << std::endl;
}

void ClusteringValidation::exportClusterDistribution(
    const std::vector<int>& cluster_assignments,
    int grid_m,
    int grid_n,
    const std::string& output_path
) {
    std::ofstream ofs(output_path);

    // Output in grid format
    for (int i = 0; i < grid_m; ++i) {
        for (int j = 0; j < grid_n; ++j) {
            int idx = i * grid_n + j;
            if (idx < static_cast<int>(cluster_assignments.size())) {
                ofs << cluster_assignments[idx];
            } else {
                ofs << -1;
            }
            if (j < grid_n - 1) ofs << ",";
        }
        ofs << "\n";
    }

    ofs.close();
    std::cout << "Exported: " << output_path << std::endl;
}

// ========================================
// Private helper functions
// ========================================

dc::GameState ClusteringValidation::createEmptyState() {
    // Use proper constructor to initialize scores vector
    dc::GameState state(game_setting_);

    // GameState(GameSetting) already initializes shot=0, end=0, and all stones to nullopt
    // We just need to ensure hammer is set correctly if needed
    state.hammer = team_;

    return state;
}

void ClusteringValidation::placeStone(
    dc::GameState& state,
    dc::Team team,
    int index,
    float x,
    float y
) {
    if (index < 0 || index >= 8) return;

    dc::Transform transform;
    transform.position.x = x;
    transform.position.y = y;
    transform.angle = 0.0f;

    state.stones[static_cast<size_t>(team)][index] = transform;
}

void ClusteringValidation::placeRandomStone(
    dc::GameState& state,
    dc::Team team,
    int index
) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> y_dist(GuardLineY_, HouseCenterY_ + 1.5f);

    float x = x_dist(gen);
    float y = y_dist(gen);

    placeStone(state, team, index, x, y);
}

std::string ClusteringValidation::patternToString(StonePattern pattern) {
    switch (pattern) {
    case StonePattern::CenterGuard: return "CenterGuard";
    case StonePattern::DoubleGuard: return "DoubleGuard";
    case StonePattern::CornerGuards: return "CornerGuards";
    case StonePattern::SingleDraw: return "SingleDraw";
    case StonePattern::DoubleDraw: return "DoubleDraw";
    case StonePattern::TripleDraw: return "TripleDraw";
    case StonePattern::HouseCorners: return "HouseCorners";
    case StonePattern::GuardAndDraw: return "GuardAndDraw";
    case StonePattern::Split: return "Split";
    case StonePattern::Crowded: return "Crowded";
    case StonePattern::Scattered: return "Scattered";
    case StonePattern::FreezeAttempt: return "FreezeAttempt";
    case StonePattern::Takeout: return "Takeout";
    case StonePattern::Promotion: return "Promotion";
    case StonePattern::Corner: return "Corner";
    case StonePattern::Symmetric: return "Symmetric";
    case StonePattern::Asymmetric: return "Asymmetric";
    case StonePattern::Random: return "Random";
    default: return "Unknown";
    }
}
