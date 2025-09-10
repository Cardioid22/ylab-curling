#include <iostream>
#include <set>
#include <fstream>
#include "clustering.h"
#include "structure.h"
#include "analysis.h"
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;
Clustering::Clustering(int k_clusters, std::vector<dc::GameState> all_states, int gridM, int gridN) 
: n_desired_clusters(k_clusters), cluster_exists(false)
{
    GridSize_M_ = gridM;
    GridSize_N_ = gridN;
    n_desired_clusters = std::log2(gridM * gridN);
    states.resize(all_states.size());
    std::copy(all_states.begin(), all_states.end(), states.begin());
    clusters.resize(all_states.size());
    if (states.empty()) {
        std::cerr << "[Clustering Error] 'states' is empty in getClusters().\n";
        return;
    }
}

bool Clustering::IsInHouse(const std::optional<dc::Transform>& stone) const {
    float x = stone->position.x;
    float y = stone->position.y;
    const float HouseRadius = 1.829;
    const float HouseCenterY = 38.405;
    return std::pow(x, 2) + std::pow(y - HouseCenterY, 2) <= std::pow(2 * HouseRadius / 3, 2);
}

std::vector<std::pair<size_t, size_t>> Clustering::SortStones(const std::array<std::array<std::optional<dc::Transform>, 8>, 2>& all_stones) const {
    std::vector<std::tuple<float, size_t, size_t>> distances;

    for (size_t team = 0; team < 2; ++team) {
        for (size_t index = 0; index < 8; ++index) {
            if (all_stones[team][index]) {
                float x = all_stones[team][index]->position.x;
                float y = all_stones[team][index]->position.y;
                float dist = std::sqrt(std::pow(x, 2) + std::pow(y - HouseCenterY_, 2));
                distances.emplace_back(dist, team, index);
            }
        }
    }

    // Sort based on distance
    std::sort(distances.begin(), distances.end());

    // Extract sorted (team, index) pairs
    std::vector<std::pair<size_t, size_t>> sorted_stones;
    for (auto& [dist, team, index] : distances) {
        sorted_stones.emplace_back(team, index);
    }

    return sorted_stones;
}


float Clustering::dist(dc::GameState const& a, dc::GameState const& b) const {
    int v = 0;
    int p = 2;
    for (size_t team = 0; team < 2; team++) {
        auto const& stones_a = a.stones[team];
        auto const& stones_b = b.stones[team];
        for (size_t index = 0; index < 8; index++) {
            if (stones_a[index] && stones_b[index]) v++;
        }
    }
    if (v == 0) return 100.0f;
    //std::cout << "v: " << v << "\n";

    float distance = 0.0f;
    for (size_t team = 0; team < 2; team++) {
        auto const& stones_a = a.stones[team];
        auto const& stones_b = b.stones[team];
        for (size_t index = 0; index < 8; index++) {
            if (stones_a[index] && stones_b[index]) {
                float dist_x = stones_a[index]->position.x - stones_b[index]->position.x;
                float dist_y = stones_a[index]->position.y - stones_b[index]->position.y;
                distance += std::sqrt((std::pow(dist_x, 2) + std::pow(dist_y, 2)));
                if ((IsInHouse(stones_a[index]) && !IsInHouse(stones_b[index])) || (!IsInHouse(stones_a[index]) && IsInHouse(stones_b[index]))) {
                    distance += 8.0f;
                }
            }
            else if (stones_b[index]) { // ex: new shot
                float dist_x = stones_b[index]->position.x;
                float dist_y = stones_b[index]->position.y - HouseCenterY_;
                distance += std::sqrt((std::pow(dist_x, 2) + std::pow(dist_y, 2)));
                if (IsInHouse(stones_b[index])) {
                    distance += 8.0f;
                }
            }
            else if (stones_a[index]) { // stone has taken away
                float dist_x = stones_a[index]->position.x;
                float dist_y = stones_a[index]->position.y - HouseCenterY_;
                distance += std::sqrt((std::pow(dist_x, 2) + std::pow(dist_y, 2)));
                if (IsInHouse(stones_a[index])) {
                    distance += 8.0f;
                }
            }
            else {
                continue;
                //std::cout << "All Stones has taken away!\n";
            }
        }
    }
    auto const& all_stones_a = a.stones;
    auto const& all_stones_b = b.stones;
    std::vector<std::pair<size_t, size_t>> sorted_stones_a, sorted_stones_b;
    sorted_stones_a = SortStones(all_stones_a);
    sorted_stones_b = SortStones(all_stones_b);
    auto [team_a, index_a] = sorted_stones_a[0];
    auto [team_b, index_b] = sorted_stones_b[0];
    if (team_a != team_b) distance += 12.0f;

    return distance;
}

std::vector<std::vector<float>> Clustering::MakeDistanceTable(std::vector<dc::GameState> const& states) {
    std::cout << "Make DitanceTable Begin\n";
    std::vector<std::vector<float>> states_table;
    const int S = GridSize_M_ * GridSize_N_;
    for (int m = 0; m < S; m++) {
        std::vector<float> category(S);
        dc::GameState s_m = states[m];
        for (int n = 0; n < S; n++) {
            dc::GameState s_n = states[n];
            if (m == n) {
                category[n] = -1.0f;
            }
            else {
                category[n] = dist(s_m, s_n);
            }
        }
        states_table.push_back(category);
    }
    std::cout << "Make DitanceTable Done\n";
    int shot_num = static_cast<int>(states[0].shot);
    if (shot_num % 2 == 1) {
        Analysis an(GridSize_M_, GridSize_N_);       
        an.SaveSimilarityTableToCSV(states_table, shot_num);
    }
    return states_table;
}

std::tuple<int, int, float> Clustering::findClosestClusters(const std::vector<std::vector<float>>& dist, const std::vector<std::set<int>>& clusters) {
    float min_dist = std::numeric_limits<float>::max();
    int cluster_a = -1, cluster_b = -1;

    for (int i = 0; i < clusters.size(); i++) {
        for (int j = i + 1; j < clusters.size(); j++) {
            float total_dist = 0.0f;
            int pair_count = 0;
            for (int a : clusters[i]) {
                for (int b : clusters[j]) {
                    total_dist += dist[a][b];
                    pair_count++;
                }
            }
            float avg_dist = total_dist / pair_count;

            if (avg_dist < min_dist) {
                min_dist = avg_dist;
                cluster_a = i;
                cluster_b = j;
            }
        }
    }
    return { cluster_a, cluster_b, min_dist };
}

LinkageMatrix Clustering::hierarchicalClustering(const std::vector<std::vector<float>>& dist, std::vector<std::set<int>>& clusters, int n_desired_clusters) {
    int n_samples = dist.size();
    std::vector<int> cluster_ids(n_samples);

    for (int i = 0; i < n_samples; i++) {
        clusters[i].insert(i);
        cluster_ids[i] = i;
    }
    int next_cluster_id = n_samples;
    LinkageMatrix linkage;
    std::cout << "Clustering Algo working...\n";
    while (clusters.size() > n_desired_clusters) {
        auto [i, j, d] = findClosestClusters(dist, clusters);
        if (i == -1 || j == -1) {
            std::cout << "Failed to find the minimum clusters\n";
            break;
        }
        int id_i = cluster_ids[i];
        int id_j = cluster_ids[j];
        int new_size = clusters[i].size() + clusters[j].size();
        linkage.emplace_back(id_i, id_j, d, new_size);

        clusters[i].insert(clusters[j].begin(), clusters[j].end());
        //std::cout << "Merge cluster[" << j << "] into cluster[" << i << "]\n";
        clusters.erase(clusters.begin() + j);
        cluster_ids[i] = next_cluster_id++;
        cluster_ids.erase(cluster_ids.begin() + j);
    }
    cluster_exists = true;
    return linkage;
}

std::vector<std::vector<int>> Clustering::calculateMedioid(const std::vector<std::vector<float>>& dist, std::vector<std::set<int>>& clusters) {
    std::vector<std::vector<int>> medoids;
    for (const auto& cluster: clusters) {
        std::vector<int> cluster_medoids;
        if (cluster.empty()) {
            continue;
        }
        if (cluster.size() == 1) {
            cluster_medoids.push_back(*cluster.begin());
        }
        else {
            float min_total_distance = std::numeric_limits<float>::max();
            int best_medoid = -1;
            for (int candidate : cluster) {
                float total_distance = 0.0f;
                for (int other : cluster) {
                    if (candidate != other) {
                        total_distance += dist[candidate][other];
                    }
                }
                if (total_distance < min_total_distance) {
                    min_total_distance = total_distance;
                    best_medoid = candidate;
                }
            }
            cluster_medoids.push_back(best_medoid);
        }
        medoids.push_back(cluster_medoids);
    }
    return medoids;
}

std::vector<std::set<int>> Clustering::getClusters() {
    if (cluster_exists) {
        return clusters;
    }
    auto distance_table = MakeDistanceTable(states);
    linkage = hierarchicalClustering(distance_table, clusters, n_desired_clusters);
    recommend_states = calculateMedioid(distance_table, clusters);
    return clusters;
}

std::vector<int> Clustering::getRecommendedStates() {
    auto clusters = getClusters();
    std::vector<int> recommend;
    for (const auto& cluster : recommend_states) {
        if (!cluster.empty()) {
            recommend.push_back(*cluster.begin());
        }
    }
    return recommend;
}

std::vector<std::vector<int>> Clustering::get_clusters_id_table() {
    int cluster_id = 0;
    std::vector<std::vector<int>> cluster_id_to_state(clusters.size());
    for (auto cluster : clusters) {
        for (int state_id : cluster) {
            cluster_id_to_state[cluster_id].push_back(state_id);
        }
        cluster_id++;
    }
    return cluster_id_to_state;
}


