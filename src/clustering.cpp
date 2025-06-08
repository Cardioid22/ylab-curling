#include <iostream>
#include "clustering.h"
#include "structure.h"
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

Clustering::Clustering() : n_desired_clusters(4), states(states){}

float Clustering::dist(dc::GameState const& a, dc::GameState const& b) {
    int v = 0;
    int p = 2;
    for (size_t team = 0; team < 2; team++) {
        auto const stones_a = a.stones[team];
        auto const stones_b = b.stones[team];
        for (size_t index = 0; index < 8; index++) {
            if (stones_a[index] && stones_b[index]) v++;
        }
    }
    if (v == 0) return 100.0f;
    //std::cout << "v: " << v << "\n";

    float distance = 0.0f;
    for (size_t team = 0; team < 2; team++) {
        auto const stones_a = a.stones[team];
        auto const stones_b = b.stones[team];
        for (size_t index = 0; index < 8; index++) {
            if (stones_a[index] && stones_b[index]) {
                float dist_x = stones_a[index]->position.x - stones_b[index]->position.x;
                float dist_y = stones_a[index]->position.y - stones_b[index]->position.y;
                distance += std::sqrt((std::pow(dist_x, 2) + std::pow(dist_y, 2)));
            }
            else if (stones_b[index]) { // ex: new shot
                float dist_x = stones_b[index]->position.x;
                float dist_y = stones_b[index]->position.y - HouseCenterY;
                distance += std::sqrt((std::pow(dist_x, 2) + std::pow(dist_y, 2))); // 欠損値処理をしている
            }
            else if (stones_a[index]) { // stone has taken away
                float dist_x = stones_a[index]->position.x;
                float dist_y = stones_a[index]->position.y - HouseCenterY;
                distance += std::sqrt( (std::pow(dist_x, 2) + std::pow(dist_y, 2))); // 欠損値処理をしている
            }
            else {
                break;
                //std::cout << "All Stones has taken away!\n";
            }
        }
    }
    return distance;
}

std::vector<std::vector<float>> Clustering::MakeDistanceTable(std::vector<dc::GameState> const& states) {
    std::vector<std::vector<float>> states_table;
    int S = GridSize_M * GridSize_N;
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
    return states_table;
}

std::tuple<int, int, float> Clustering::findClosestClusters(const std::vector<std::vector<float>>& dist, const std::vector<std::set<int>>& clusters) {
    float min_dist = std::numeric_limits<float>::max();
    int cluster_a = -1, cluster_b = -1;

    for (int i = 0; i < clusters.size(); i++) {
        for (int j = i + 1; j < clusters.size(); j++) {
            for (int a : clusters[i]) {
                for (int b : clusters[j]) {
                    if (dist[a][b] < min_dist) {
                        min_dist = dist[a][b];
                        cluster_a = i;
                        cluster_b = j;
                    }
                }
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
    return linkage;
}

std::vector<std::set<int>> Clustering::getClusters() {
    auto distance_table = MakeDistanceTable(states);
    linkage = hierarchicalClustering(distance_table, clusters, n_desired_clusters);
    return clusters;
}

std::vector<int> Clustering::getRecommndedStates(std::vector<std::set<int>> clusters) {
    std::vector<int> recommend;
    for (const auto& cluster : clusters) {
        if (!cluster.empty()) {
            recommend.push_back(*cluster.begin());
        }
    }
    return recommend;
}