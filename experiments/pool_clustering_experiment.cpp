#include "pool_clustering_experiment.h"
#include "pool_experiment.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <map>

namespace dc = digitalcurling3;

// --- ClusterAnalysis ---

float ClusterAnalysis::purity() const {
    int total = static_cast<int>(member_indices.size());
    if (total == 0) return 0.0f;
    int max_count = std::max({count_draw, count_hit, count_freeze, count_guard, count_other});
    return static_cast<float>(max_count) / total;
}

std::string ClusterAnalysis::dominantType() const {
    int max_count = std::max({count_draw, count_hit, count_freeze, count_guard, count_other});
    if (max_count == count_draw) return "DRAW";
    if (max_count == count_hit) return "HIT";
    if (max_count == count_freeze) return "FREEZE";
    if (max_count == count_guard) return "GUARD";
    return "OTHER";
}

// --- PoolClusteringResult ---

float PoolClusteringResult::weightedPurity() const {
    float total_purity = 0.0f;
    int total_members = 0;
    for (auto& c : clusters) {
        int size = static_cast<int>(c.member_indices.size());
        total_purity += c.purity() * size;
        total_members += size;
    }
    return total_members > 0 ? total_purity / total_members : 0.0f;
}

float PoolClusteringResult::typeCoverage() const {
    std::set<std::string> dominant_types;
    for (auto& c : clusters) {
        dominant_types.insert(c.dominantType());
    }
    return static_cast<float>(dominant_types.size()) / n_clusters;
}

// --- PoolClusteringExperiment ---

PoolClusteringExperiment::PoolClusteringExperiment(dc::GameSetting const& game_setting)
    : game_setting_(game_setting)
{
}

std::vector<dc::GameState> PoolClusteringExperiment::createTestStates() {
    std::vector<dc::GameState> states;

    // テスト盤面1: 空場
    {
        dc::GameState s(game_setting_);
        s.shot = 0;
        states.push_back(s);
    }

    // テスト盤面2: 相手石がティー近くに1個
    {
        dc::GameState s(game_setting_);
        s.shot = 2;
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY), 0.f));
        states.push_back(s);
    }

    // テスト盤面3: 相手石2個 + 自分石1個
    {
        dc::GameState s(game_setting_);
        s.shot = 5;
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(-0.5f, kHouseCenterY + 0.3f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY - 0.2f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY + 0.8f), 0.f));
        states.push_back(s);
    }

    // テスト盤面4: 混雑した盤面
    {
        dc::GameState s(game_setting_);
        s.shot = 10;
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(-1.0f, kHouseCenterY - 0.5f), 0.f));
        s.stones[0][2].emplace(dc::Transform(dc::Vector2(0.5f, kHouseCenterY + 1.0f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY + 0.1f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.3f, kHouseCenterY + 1.5f), 0.f));
        s.stones[1][2].emplace(dc::Transform(dc::Vector2(0.8f, kHouseCenterY - 0.8f), 0.f));
        states.push_back(s);
    }

    return states;
}

bool PoolClusteringExperiment::isInHouse(const std::optional<dc::Transform>& stone) const {
    float x = stone->position.x;
    float y = stone->position.y;
    return std::pow(x, 2) + std::pow(y - kHouseCenterY, 2) <= std::pow(2 * kHouseRadius / 3, 2);
}

float PoolClusteringExperiment::dist(const dc::GameState& a, const dc::GameState& b) const {
    // Clusteringクラスと同じ距離関数
    int v = 0;
    for (size_t team = 0; team < 2; team++) {
        for (size_t index = 0; index < 8; index++) {
            if (a.stones[team][index] && b.stones[team][index]) v++;
        }
    }
    if (v == 0) return 100.0f;

    float distance = 0.0f;
    for (size_t team = 0; team < 2; team++) {
        for (size_t index = 0; index < 8; index++) {
            if (a.stones[team][index] && b.stones[team][index]) {
                float dx = a.stones[team][index]->position.x - b.stones[team][index]->position.x;
                float dy = a.stones[team][index]->position.y - b.stones[team][index]->position.y;
                distance += std::sqrt(dx * dx + dy * dy);
                if (isInHouse(a.stones[team][index]) != isInHouse(b.stones[team][index])) {
                    distance += 8.0f;
                }
            }
            else if (b.stones[team][index]) {
                float dx = b.stones[team][index]->position.x;
                float dy = b.stones[team][index]->position.y - kHouseCenterY;
                distance += std::sqrt(dx * dx + dy * dy);
                if (isInHouse(b.stones[team][index])) {
                    distance += 8.0f;
                }
            }
            else if (a.stones[team][index]) {
                float dx = a.stones[team][index]->position.x;
                float dy = a.stones[team][index]->position.y - kHouseCenterY;
                distance += std::sqrt(dx * dx + dy * dy);
                if (isInHouse(a.stones[team][index])) {
                    distance += 8.0f;
                }
            }
        }
    }

    // No.1ストーンのチーム一致判定
    float closest_a = std::numeric_limits<float>::max();
    float closest_b = std::numeric_limits<float>::max();
    int team_a = -1, team_b = -1;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (a.stones[t][i]) {
                float d = std::sqrt(std::pow(a.stones[t][i]->position.x, 2) +
                          std::pow(a.stones[t][i]->position.y - kHouseCenterY, 2));
                if (d < closest_a) { closest_a = d; team_a = t; }
            }
            if (b.stones[t][i]) {
                float d = std::sqrt(std::pow(b.stones[t][i]->position.x, 2) +
                          std::pow(b.stones[t][i]->position.y - kHouseCenterY, 2));
                if (d < closest_b) { closest_b = d; team_b = t; }
            }
        }
    }
    if (team_a >= 0 && team_b >= 0 && team_a != team_b) {
        distance += 12.0f;
    }

    return distance;
}

std::vector<std::vector<float>> PoolClusteringExperiment::makeDistanceTable(
    const std::vector<dc::GameState>& states
) {
    int n = static_cast<int>(states.size());
    std::vector<std::vector<float>> table(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float d = dist(states[i], states[j]);
            table[i][j] = d;
            table[j][i] = d;
        }
        table[i][i] = -1.0f;
    }
    return table;
}

std::vector<std::set<int>> PoolClusteringExperiment::runClustering(
    const std::vector<std::vector<float>>& dist_table,
    int n_desired_clusters
) {
    int n = static_cast<int>(dist_table.size());
    std::vector<std::set<int>> clusters(n);
    for (int i = 0; i < n; i++) {
        clusters[i].insert(i);
    }

    while (static_cast<int>(clusters.size()) > n_desired_clusters) {
        // 最も近いクラスタペアを探す（平均連結法）
        float min_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;

        for (int i = 0; i < static_cast<int>(clusters.size()); i++) {
            for (int j = i + 1; j < static_cast<int>(clusters.size()); j++) {
                float total = 0.0f;
                int count = 0;
                for (int a : clusters[i]) {
                    for (int b : clusters[j]) {
                        total += dist_table[a][b];
                        count++;
                    }
                }
                float avg = total / count;
                if (avg < min_dist) {
                    min_dist = avg;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i == -1) break;

        // マージ
        clusters[best_i].insert(clusters[best_j].begin(), clusters[best_j].end());
        clusters.erase(clusters.begin() + best_j);
    }

    return clusters;
}

std::vector<int> PoolClusteringExperiment::calculateMedoids(
    const std::vector<std::vector<float>>& dist_table,
    const std::vector<std::set<int>>& clusters
) {
    std::vector<int> medoids;
    for (auto& cluster : clusters) {
        if (cluster.empty()) {
            medoids.push_back(-1);
            continue;
        }
        if (cluster.size() == 1) {
            medoids.push_back(*cluster.begin());
            continue;
        }
        float min_total = std::numeric_limits<float>::max();
        int best = -1;
        for (int c : cluster) {
            float total = 0.0f;
            for (int o : cluster) {
                if (c != o) total += dist_table[c][o];
            }
            if (total < min_total) {
                min_total = total;
                best = c;
            }
        }
        medoids.push_back(best);
    }
    return medoids;
}

PoolClusteringResult PoolClusteringExperiment::analyzeClusterComposition(
    const std::string& state_name,
    const CandidatePool& pool,
    const std::vector<std::set<int>>& clusters,
    const std::vector<int>& medoids
) {
    PoolClusteringResult result;
    result.state_name = state_name;
    result.n_candidates = static_cast<int>(pool.candidates.size());
    result.n_clusters = static_cast<int>(clusters.size());

    for (int c = 0; c < static_cast<int>(clusters.size()); c++) {
        ClusterAnalysis ca;
        ca.cluster_id = c;
        ca.medoid_index = medoids[c];
        if (ca.medoid_index >= 0) {
            ca.medoid_type = pool.candidates[ca.medoid_index].type;
            ca.medoid_label = pool.candidates[ca.medoid_index].label;
        }

        for (int idx : clusters[c]) {
            ca.member_indices.push_back(idx);
            ShotType type = pool.candidates[idx].type;
            ca.member_types.push_back(type);
            ca.member_labels.push_back(pool.candidates[idx].label);

            switch (type) {
                case ShotType::DRAW: ca.count_draw++; break;
                case ShotType::HIT: ca.count_hit++; break;
                case ShotType::FREEZE: ca.count_freeze++; break;
                case ShotType::PREGUARD: case ShotType::POSTGUARD: ca.count_guard++; break;
                default: ca.count_other++; break;
            }
        }
        result.clusters.push_back(ca);
    }

    return result;
}

void PoolClusteringExperiment::printResult(const PoolClusteringResult& result) {
    std::cout << "\n=== " << result.state_name
              << " (candidates=" << result.n_candidates
              << ", clusters=" << result.n_clusters << ") ===" << std::endl;
    std::cout << "  Weighted Purity: " << std::fixed << std::setprecision(3)
              << result.weightedPurity() << std::endl;
    std::cout << "  Type Coverage: " << result.typeCoverage()
              << " (" << static_cast<int>(result.typeCoverage() * result.n_clusters)
              << "/" << result.n_clusters << " distinct dominant types)" << std::endl;

    for (auto& ca : result.clusters) {
        std::cout << "\n  Cluster " << ca.cluster_id
                  << " (size=" << ca.member_indices.size()
                  << ", purity=" << std::setprecision(2) << ca.purity()
                  << ", dominant=" << ca.dominantType() << ")" << std::endl;
        std::cout << "    Type breakdown: Draw=" << ca.count_draw
                  << " Hit=" << ca.count_hit
                  << " Freeze=" << ca.count_freeze
                  << " Guard=" << ca.count_guard
                  << " Other=" << ca.count_other << std::endl;
        std::cout << "    Medoid: [" << ca.medoid_index << "] " << ca.medoid_label << std::endl;

        // メンバー一覧（ラベル表示）
        std::cout << "    Members: ";
        for (size_t i = 0; i < ca.member_labels.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << ca.member_labels[i];
        }
        std::cout << std::endl;
    }
}

void PoolClusteringExperiment::exportResultCSV(
    const PoolClusteringResult& result,
    const std::string& output_dir
) {
    std::string filename = output_dir + "/cluster_analysis_" + result.state_name
                         + "_k" + std::to_string(result.n_clusters) + ".csv";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return;
    }

    ofs << "candidate_index,cluster_id,shot_type,label,is_medoid,spin,target_index,param,vx,vy,rot" << std::endl;

    for (auto& ca : result.clusters) {
        for (size_t i = 0; i < ca.member_indices.size(); i++) {
            int idx = ca.member_indices[i];
            auto& cand = result.state_name.empty() ? throw std::runtime_error("") :
                         // dummy - we need pool reference
                         ca.member_labels[i]; // just for label
            (void)cand;

            ofs << idx << ","
                << ca.cluster_id << ","
                << static_cast<int>(ca.member_types[i]) << ","
                << "\"" << ca.member_labels[i] << "\","
                << (idx == ca.medoid_index ? 1 : 0)
                << std::endl;
        }
    }

    std::cout << "  Exported to " << filename << std::endl;
}

void PoolClusteringExperiment::run() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Pool Clustering Experiment" << std::endl;
    std::cout << "  - Extended candidate pool + hierarchical clustering" << std::endl;
    std::cout << "  - Verifying if clustering reproduces shot type grouping" << std::endl;
    std::cout << "================================================" << std::endl;

    auto test_states = createTestStates();
    std::string state_names[] = { "empty", "opp1_tee", "opp2_my1", "crowded" };

    std::string output_dir = "experiments/pool_clustering_results";
    std::filesystem::create_directories(output_dir);

    ShotGenerator generator(game_setting_);
    auto grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    // 異なるクラスタ数でテスト
    std::vector<int> k_values = {4, 6, 8};

    for (size_t s = 0; s < test_states.size(); s++) {
        auto& state = test_states[s];
        dc::Team my_team = dc::Team::k0;

        std::cout << "\n---------- State: " << state_names[s]
                  << " (shot=" << state.shot << ") ----------" << std::endl;

        // プール生成
        auto t0 = std::chrono::high_resolution_clock::now();
        auto pool = generator.generatePool(state, my_team, grid);
        auto t1 = std::chrono::high_resolution_clock::now();
        double pool_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  Pool generated: " << pool.candidates.size()
                  << " candidates (" << pool_ms << " ms)" << std::endl;

        // 距離テーブル計算
        auto t2 = std::chrono::high_resolution_clock::now();
        auto dist_table = makeDistanceTable(pool.result_states);
        auto t3 = std::chrono::high_resolution_clock::now();
        double dist_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "  Distance table: " << dist_ms << " ms" << std::endl;

        for (int k : k_values) {
            if (k >= static_cast<int>(pool.candidates.size())) continue;

            auto clusters = runClustering(dist_table, k);
            auto medoids = calculateMedoids(dist_table, clusters);
            auto result = analyzeClusterComposition(state_names[s], pool, clusters, medoids);

            printResult(result);

            // CSVエクスポート（候補手情報付き）
            {
                std::string csv = output_dir + "/cluster_" + state_names[s]
                                + "_k" + std::to_string(k) + ".csv";
                std::ofstream ofs(csv);
                ofs << "candidate_index,cluster_id,shot_type,shot_type_name,label,is_medoid,"
                    << "spin,target_index,param,vx,vy,rot";
                // 結果盤面の石の位置
                for (int t = 0; t < 2; t++) {
                    for (int i = 0; i < 8; i++) {
                        ofs << ",team" << t << "_stone" << i << "_x"
                            << ",team" << t << "_stone" << i << "_y"
                            << ",team" << t << "_stone" << i << "_active";
                    }
                }
                ofs << "\n";

                auto typeToName = [](ShotType t) -> std::string {
                    switch (t) {
                        case ShotType::PASS: return "PASS";
                        case ShotType::DRAW: return "DRAW";
                        case ShotType::PREGUARD: return "PREGUARD";
                        case ShotType::HIT: return "HIT";
                        case ShotType::FREEZE: return "FREEZE";
                        case ShotType::PEEL: return "PEEL";
                        case ShotType::COMEAROUND: return "COMEAROUND";
                        case ShotType::POSTGUARD: return "POSTGUARD";
                        case ShotType::DRAWRAISE: return "DRAWRAISE";
                        default: return "OTHER";
                    }
                };

                for (size_t ci = 0; ci < clusters.size(); ci++) {
                    for (int idx : clusters[ci]) {
                        auto& cand = pool.candidates[idx];
                        auto& rs = pool.result_states[idx];
                        ofs << idx << ","
                            << ci << ","
                            << static_cast<int>(cand.type) << ","
                            << typeToName(cand.type) << ","
                            << "\"" << cand.label << "\","
                            << (idx == medoids[ci] ? 1 : 0) << ","
                            << cand.spin << ","
                            << cand.target_index << ","
                            << cand.param << ","
                            << std::fixed << std::setprecision(6)
                            << cand.shot.vx << ","
                            << cand.shot.vy << ","
                            << cand.shot.rot;

                        for (int t = 0; t < 2; t++) {
                            for (int i = 0; i < 8; i++) {
                                if (rs.stones[t][i].has_value()) {
                                    ofs << "," << rs.stones[t][i]->position.x
                                        << "," << rs.stones[t][i]->position.y
                                        << ",1";
                                } else {
                                    ofs << ",0,0,0";
                                }
                            }
                        }
                        ofs << "\n";
                    }
                }
                std::cout << "  Exported: " << csv << std::endl;
            }
        }
    }

    // サマリー表
    std::cout << "\n================================================" << std::endl;
    std::cout << "  Summary" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << std::setw(12) << "State"
              << std::setw(8) << "K"
              << std::setw(12) << "Purity"
              << std::setw(12) << "Coverage" << std::endl;
    std::cout << std::string(44, '-') << std::endl;

    // 再度計算してサマリー出力
    for (size_t s = 0; s < test_states.size(); s++) {
        auto pool = generator.generatePool(test_states[s], dc::Team::k0, grid);
        auto dist_table = makeDistanceTable(pool.result_states);

        for (int k : k_values) {
            if (k >= static_cast<int>(pool.candidates.size())) continue;
            auto clusters = runClustering(dist_table, k);
            auto medoids = calculateMedoids(dist_table, clusters);
            auto result = analyzeClusterComposition(state_names[s], pool, clusters, medoids);

            std::cout << std::setw(12) << state_names[s]
                      << std::setw(8) << k
                      << std::setw(12) << std::fixed << std::setprecision(3) << result.weightedPurity()
                      << std::setw(12) << result.typeCoverage() << std::endl;
        }
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Pool Clustering Experiment Complete" << std::endl;
    std::cout << "================================================" << std::endl;
}
