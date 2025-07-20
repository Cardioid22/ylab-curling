#include "analysis.h"
#include "structure.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <filesystem>
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

Analysis::Analysis(int grid_size_m, int grid_size_n)
    : GridSize_M(grid_size_m), GridSize_N(grid_size_n) {
}

void Analysis::SaveSimilarityTableToCSV(const std::vector<std::vector<float>>& table, int shot_number) const {
    std::string folder = "table_outputs_" + std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    std::string filename = folder + "state_similarity_shot_" + std::to_string(shot_number) + ".csv";
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    for (const auto& row : table) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << std::fixed << std::setprecision(10) << row[i];
            if (i != row.size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Saved similarity table to: " << filename << "\n";
}

void Analysis::export_best_shot_comparison_to_csv(
    const ShotInfo& best,
    const ShotInfo& best_allgrid,
    int best_state,
    int best_allgrid_state,
    int shot_num,
    int iter,
    const std::string& filename
) {
    // Create the directory if it doesn't exist
    std::string folder = "../Iter_"+ std::to_string(iter) + "/MCTS_Output_BestShotComparison_" + std::to_string(GridSize_M * GridSize_N);
    std::filesystem::create_directories(folder);
    std::string new_filename = folder + "/" + filename + "_" + std::to_string(shot_num) + ".csv";
    std::ofstream file(new_filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << new_filename << "\n";
        return;
    }

    // Write header
    file << "Type,Vx,Vy,Rot,StateID\n";

    // Write MCTS best shot
    file << "MCTS," << best.vx << "," << best.vy << "," << best.rot << "," << best_state << "\n";

    // Write AllGrid best shot
    file << "AllGrid," << best_allgrid.vx << "," << best_allgrid.vy << "," << best_allgrid.rot << "," << best_allgrid_state << "\n";

    file.close();
}

void Analysis::cluster_id_to_state_csv(std::vector<std::vector<int>> cluster_id_to_state, int shot_num, int iter) const {
    std::string folder = "../Iter_" + std::to_string(iter) + "/MCTS_Output_ClusteringId_" + std::to_string(cluster_id_to_state.size()) + "_Clusters_/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    std::string new_filename = folder + "cluster_ids_" + std::to_string(shot_num) + ".csv";
    std::ofstream file(new_filename);

    file << "ClusterId,StateId\n";

    for (int cid = 0; cid < cluster_id_to_state.size(); ++cid) {
        for (int stateId : cluster_id_to_state[cid]) {
            file << cid << "," << stateId << "\n";
        }
    }
    file.close();
}

void Analysis::OutputClusterGridToCSV(const std::vector<int>& state_index_to_cluster,
    int rows, int cols,
    const std::string& filename, const int shot_num) {
    std::string folder = "hierarchical_clustering/cluster_distribution_" + std::to_string(rows) + "_" + std::to_string(cols) + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    std::string new_filename = folder + filename + "_" + std::to_string(shot_num) + ".csv";
    std::ofstream file(new_filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << new_filename << "\n";
        return;
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int index = r * cols + c;
            int cluster = state_index_to_cluster.at(index);  // Assumes all indices exist
            file << cluster;
            if (c < cols - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Cluster grid written to " << new_filename << "\n";
}

void Analysis::ExportStoneCoordinatesToCSV(const dc::GameState& game_state, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << "\n";
        return;
    }

    // Write header
    for (int team = 0; team < 2; ++team) {
        for (int i = 0; i < 8; ++i) {
            file << "team" << team << "_stone" << i << "_x,";
            file << "team" << team << "_stone" << i << "_y,";
        }
    }
    file << "\n";

    // Write data rows
    const auto& state = game_state;
    for (int team = 0; team < 2; ++team) {
        for (int i = 0; i < 8; ++i) {
            if (state.stones[team][i]) {
                file << std::fixed << std::setprecision(3) << state.stones[team][i]->position.x << ",";
                file << std::fixed << std::setprecision(3) << state.stones[team][i]->position.y << ",";
            }
            else {
                file << "NaN,NaN,";
            }
        }
    }
    file << "\n";

    file.close();
    std::cout << "Stone coordinates exported to " << filename << "\n";
}

void Analysis::ExportStonesByCluster(
    const std::vector<int>& state_index_to_cluster,
    const std::vector<dc::GameState>& all_game_states, const int shot_num)
{
    std::string base_folder = "hierarchical_clustering/Stone_Coordinates_" +
        std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::string shot_folder = base_folder + "shot" + std::to_string(shot_num) + "/";
    // Delete old shot folder if it exists
    if (std::filesystem::exists(shot_folder)) {
        std::filesystem::remove_all(shot_folder);
        std::cout << "Old folder removed: " << shot_folder << "\n";
    }
    std::filesystem::create_directories(shot_folder);

    for (int index = 0; index < state_index_to_cluster.size(); index++) {
        int state_index = index;
        int cluster_id = state_index_to_cluster[state_index];
        if (state_index >= all_game_states.size()) {
            std::cerr << "Invalid state index: " << state_index << "\n";
            continue;
        }

        const auto& game_state = all_game_states[state_index];

        // Construct folder: hierarchical_clustering/Stone_Coordinates_M_N/shotK/ClusterX/
        std::stringstream cluster_folder_ss;
        cluster_folder_ss << shot_folder << "Cluster" << cluster_id << "/";
        std::string cluster_folder = cluster_folder_ss.str();
        std::filesystem::create_directories(cluster_folder);

        // File: ClusterX/stateY.csv
        std::string state_filename = "state" + std::to_string(state_index);

        ExportStoneCoordinatesToCSV(game_state, cluster_folder + state_filename + ".csv");
    }

    std::cout << "Export complete: Stones sorted into cluster folders.\n";
}

void Analysis::IntraToCSV(const std::vector<float>& scores, const int shot_num) {
    std::string folder = "hierarchical_clustering/Intra_Cluster_Scores_" + std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    // Create filename with shot number
    std::string filename = "intra_cluster_scores_shot_" + std::to_string(shot_num) + ".csv";
    std::string new_filename = folder + filename;

    std::ofstream file(new_filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << new_filename << "\n";
        return;
    }

    file << "k,intra_score\n";
    for (size_t k = 2; k < scores.size(); ++k) {
        file << k << "," << std::fixed << std::setprecision(5) << scores[k] << "\n";
    }

    file.close();
    std::cout << "Intra-cluster scores saved to: " << new_filename << "\n";
}

void Analysis::SilhouetteToCSV(float score, int shot_num, int k_cluster) {
    std::string folder = "hierarchical_clustering/SilhouetteScores_" + std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::filesystem::create_directories(folder);
    std::string filename = folder + "silhouette_scores_cluster_" + std::to_string(k_cluster) + ".csv";

    // Check if file exists
    bool file_exists = std::filesystem::exists(filename);

    std::ofstream file(filename, std::ios::app); // Open in append mode
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return;
    }

    // Write header only if file didn't exist before
    if (!file_exists) {
        file << "shot,silhouette_score\n";
    }

    // Write current score
    file << shot_num << "," << std::fixed << std::setprecision(5) << score << "\n";
    file.close();

    std::cout << "Silhouette score for shot " << shot_num << " saved to: " << filename << "\n";
}

float Analysis::ComputeSilhouetteScore(
    const std::vector<std::vector<float>>& distance_matrix,
    const std::vector<int>& state_index_to_cluster
) {
    int N = distance_matrix.size();
    std::vector<float> silhouette_values(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        int cluster_i = state_index_to_cluster[i];
        std::vector<float> a_dists, b_dists;

        float a = 0.0f, b = std::numeric_limits<float>::max();

        // Distances to same-cluster members (a)
        int a_count = 0;
        for (int j = 0; j < N; ++j) {
            if (i != j && state_index_to_cluster[j] == cluster_i) {
                a += distance_matrix[i][j];
                ++a_count;
            }
        }
        if (a_count > 0) a /= a_count;

        // Distances to other clusters (b)
        std::map<int, std::pair<float, int>> cluster_sums;
        for (int j = 0; j < N; ++j) {
            int cluster_j = state_index_to_cluster[j];
            if (cluster_j != cluster_i) {
                cluster_sums[cluster_j].first += distance_matrix[i][j];
                cluster_sums[cluster_j].second++;
            }
        }

        for (const auto& [cluster, sum_pair] : cluster_sums) {
            float avg = sum_pair.first / sum_pair.second;
            if (avg < b) b = avg;
        }

        float s = 0.0f;
        if (a_count > 0 && std::max(a, b) > 0.0f) {
            s = (b - a) / std::max(a, b);
        }
        silhouette_values[i] = s;
    }

    // Average silhouette score
    float total = std::accumulate(silhouette_values.begin(), silhouette_values.end(), 0.0f);
    return total / N;
}
float Analysis::ComputeIntraClusterDistance(
    const std::vector<std::vector<float>>& dist,
    const std::vector<std::set<int>>& clsters)  {

    float total = 0.0f;
    int count = 0;
    for (const auto& cluster_set : clsters) {
        std::vector<int> cluster(cluster_set.begin(), cluster_set.end());
        float sum = 0.0f;
        count = 0;
        for (int i = 0; i < cluster.size(); ++i) {
            for (int j = i + 1; j < cluster.size(); ++j) {
                sum += dist[cluster[i]][cluster[j]];
                count++;
            }
        }
        total += sum;
    }
    if (count > 0) total /= count;
    return total;
}