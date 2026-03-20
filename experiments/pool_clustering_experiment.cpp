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
        s.shot = 4;  // team0の手番 (偶数 = team0)
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

// ゾーン判定: 石の位置がどの戦略的ゾーンに属するか
// Zone 0: ハウス内（得点圏）
// Zone 1: ガードゾーン（ハウス手前、y < HouseCenterY - HouseRadius）
// Zone 2: 遠方（シート外に近い、または横方向に大きく外れた位置）
int PoolClusteringExperiment::getZone(const std::optional<dc::Transform>& stone) const {
    if (!stone) return -1;
    float x = stone->position.x;
    float y = stone->position.y;
    float dist_to_tee = std::sqrt(x * x + (y - kHouseCenterY) * (y - kHouseCenterY));

    if (dist_to_tee <= kHouseRadius) return 0;  // ハウス内
    if (y < kHouseCenterY - kHouseRadius && y > kHouseCenterY - 3.0f * kHouseRadius) return 1;  // ガードゾーン
    return 2;  // 遠方
}

// 盤面の得点評価（カーリングの公式ルール準拠）
// team0の視点: 正=team0有利, 負=team1有利
float PoolClusteringExperiment::evaluateBoard(const dc::GameState& state) const {
    struct StoneInfo { float dist; int team; };
    std::vector<StoneInfo> in_house;

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (!state.stones[t][i]) continue;
            float dx = state.stones[t][i]->position.x;
            float dy = state.stones[t][i]->position.y - kHouseCenterY;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= kHouseRadius + 0.145f) {  // ハウス半径 + 石の半径
                in_house.push_back({d, t});
            }
        }
    }
    if (in_house.empty()) return 0.0f;

    std::sort(in_house.begin(), in_house.end(),
              [](auto& a, auto& b) { return a.dist < b.dist; });

    int scoring_team = in_house[0].team;
    int score = 0;
    for (auto& s : in_house) {
        if (s.team == scoring_team) score++;
        else break;
    }
    return scoring_team == 0 ? static_cast<float>(score) : -static_cast<float>(score);
}

float PoolClusteringExperiment::distDelta(
    const dc::GameState& input,
    const dc::GameState& a,
    const dc::GameState& b
) const {
    // 改良版デルタ距離関数 v2
    // 基本方針: 入力盤面からの「変化」を多次元で比較
    //
    // 改良点:
    // 1. 盤面スコア差ペナルティ: 戦略的結果の違いを直接反映
    // 2. 石インタラクションペナルティ: 既存石を動かしたか否かの違い
    //    (Draw=非接触 vs TOUCH/Freeze=接触 の区別)
    // 3. ゾーンペナルティ増大: Guard vs House Draw の分離を強化
    // 4. 新石の近接度: Freeze(密着) vs Draw(非密着) の区別

    constexpr float MOVE_THRESHOLD = 0.01f;        // 1cm以下は「不変」
    constexpr float PENALTY_EXISTENCE = 30.0f;      // 石の有無が異なるペナルティ
    constexpr float PENALTY_ZONE = 12.0f;           // ゾーン差ペナルティ (6→12)
    constexpr float NEW_STONE_WEIGHT = 4.0f;        // 新石位置差の重み (3→4)
    constexpr float MOVED_STONE_WEIGHT = 2.0f;      // 押された石の移動差
    constexpr float PENALTY_INTERACTION = 15.0f;    // 石接触有無の差ペナルティ (NEW)
    constexpr float INTERACTION_THRESHOLD = 0.03f;  // 3cm以上動いたら「接触」
    constexpr float SCORE_WEIGHT = 8.0f;            // 盤面スコア差の重み (NEW)
    constexpr float PROXIMITY_WEIGHT = 5.0f;        // 新石近接度差の重み (NEW)

    float distance = 0.0f;
    float max_displacement_a = 0.0f;
    float max_displacement_b = 0.0f;
    int new_stone_team = -1, new_stone_idx = -1;

    for (int team = 0; team < 2; team++) {
        for (int idx = 0; idx < 8; idx++) {
            bool in_input = input.stones[team][idx].has_value();
            bool in_a = a.stones[team][idx].has_value();
            bool in_b = b.stones[team][idx].has_value();

            if (in_input) {
                // === 入力盤面に存在した石 ===
                if (in_a && in_b) {
                    float dx_a = a.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dy_a = a.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float dx_b = b.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dy_b = b.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float move_a = std::sqrt(dx_a * dx_a + dy_a * dy_a);
                    float move_b = std::sqrt(dx_b * dx_b + dy_b * dy_b);

                    // 最大変位を追跡（インタラクション判定用）
                    max_displacement_a = std::max(max_displacement_a, move_a);
                    max_displacement_b = std::max(max_displacement_b, move_b);

                    if (move_a < MOVE_THRESHOLD && move_b < MOVE_THRESHOLD) {
                        continue;  // 両方不変 → 距離0
                    }

                    float ddx = dx_a - dx_b;
                    float ddy = dy_a - dy_b;
                    distance += MOVED_STONE_WEIGHT * std::sqrt(ddx * ddx + ddy * ddy);

                    int zone_a = getZone(a.stones[team][idx]);
                    int zone_b = getZone(b.stones[team][idx]);
                    if (zone_a != zone_b) {
                        distance += PENALTY_ZONE;
                    }
                }
                else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            }
            else {
                // === 新規配置石 ===
                if (in_a && in_b) {
                    new_stone_team = team;
                    new_stone_idx = idx;

                    float dx = a.stones[team][idx]->position.x - b.stones[team][idx]->position.x;
                    float dy = a.stones[team][idx]->position.y - b.stones[team][idx]->position.y;
                    distance += NEW_STONE_WEIGHT * std::sqrt(dx * dx + dy * dy);

                    int zone_a = getZone(a.stones[team][idx]);
                    int zone_b = getZone(b.stones[team][idx]);
                    if (zone_a != zone_b) {
                        distance += PENALTY_ZONE;
                    }
                }
                else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            }
        }
    }

    // (1) 石インタラクションペナルティ: 既存石を動かしたか否か
    // Draw/Guard=非接触, TOUCH/Freeze=接触 を区別
    bool interacted_a = max_displacement_a > INTERACTION_THRESHOLD;
    bool interacted_b = max_displacement_b > INTERACTION_THRESHOLD;
    if (interacted_a != interacted_b) {
        distance += PENALTY_INTERACTION;
    }

    // (2) 新石の近接度: 結果盤面で新石が既存石にどれだけ近いか
    // Freeze(密着≈0.29m) vs Draw(非密着≈1m+) の区別に有効
    if (new_stone_team >= 0) {
        auto computeMinProximity = [&](const dc::GameState& state) -> float {
            float min_dist = std::numeric_limits<float>::max();
            float nx = state.stones[new_stone_team][new_stone_idx]->position.x;
            float ny = state.stones[new_stone_team][new_stone_idx]->position.y;
            for (int t = 0; t < 2; t++) {
                for (int i = 0; i < 8; i++) {
                    if (t == new_stone_team && i == new_stone_idx) continue;
                    if (!state.stones[t][i].has_value()) continue;
                    float dx = nx - state.stones[t][i]->position.x;
                    float dy = ny - state.stones[t][i]->position.y;
                    min_dist = std::min(min_dist, std::sqrt(dx * dx + dy * dy));
                }
            }
            return min_dist;
        };

        float prox_a = computeMinProximity(a);
        float prox_b = computeMinProximity(b);
        // 両方に既存石がある場合のみ比較（空場ではスキップ）
        if (prox_a < 100.0f || prox_b < 100.0f) {
            distance += PROXIMITY_WEIGHT * std::abs(prox_a - prox_b);
        }
    }

    // (3) 盤面スコア差ペナルティ: 戦略的結果の違いを直接反映
    float score_a = evaluateBoard(a);
    float score_b = evaluateBoard(b);
    distance += SCORE_WEIGHT * std::abs(score_a - score_b);

    // (4) No.1ストーンのチーム一致判定
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
        distance += 10.0f;
    }

    return distance;
}

std::vector<std::vector<float>> PoolClusteringExperiment::makeDistanceTableDelta(
    const dc::GameState& input_state,
    const std::vector<dc::GameState>& result_states
) {
    int n = static_cast<int>(result_states.size());
    std::vector<std::vector<float>> table(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float d = distDelta(input_state, result_states[i], result_states[j]);
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
    std::cout << "  - Comparing Original vs Delta distance functions" << std::endl;
    std::cout << "================================================" << std::endl;

    auto test_states = createTestStates();
    std::string state_names[] = { "empty", "opp1_tee", "opp2_my1", "crowded" };

    std::string output_dir = "experiments/pool_clustering_results";
    std::filesystem::create_directories(output_dir);

    ShotGenerator generator(game_setting_);
    auto grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    std::vector<int> k_values = {4, 6, 8, 12, 16};

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

    // サマリー用の蓄積
    struct SummaryRow {
        std::string state;
        int k;
        float purity_orig;
        float coverage_orig;
        float purity_delta;
        float coverage_delta;
        int max_cluster_orig;   // 最大クラスタのサイズ
        int max_cluster_delta;
    };
    std::vector<SummaryRow> summary;

    for (size_t s = 0; s < test_states.size(); s++) {
        auto& state = test_states[s];
        dc::Team my_team = dc::Team::k0;

        std::cout << "\n========== State: " << state_names[s]
                  << " (shot=" << state.shot << ") ==========" << std::endl;

        // プール生成（1回だけ）
        auto t0 = std::chrono::high_resolution_clock::now();
        auto pool = generator.generatePool(state, my_team, grid);
        auto t1 = std::chrono::high_resolution_clock::now();
        double pool_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  Pool: " << pool.candidates.size()
                  << " candidates (" << std::fixed << std::setprecision(1) << pool_ms << " ms)" << std::endl;

        // 距離テーブル計算（両方）
        auto dist_orig = makeDistanceTable(pool.result_states);
        auto dist_delta = makeDistanceTableDelta(state, pool.result_states);

        // デバッグ: 代表的な距離値の比較（最初のDraw, 最初のHit, 最初のGuard）
        {
            int draw_idx = -1, hit_idx = -1, guard_idx = -1, freeze_idx = -1;
            for (int i = 0; i < static_cast<int>(pool.candidates.size()); i++) {
                if (draw_idx < 0 && pool.candidates[i].type == ShotType::DRAW) draw_idx = i;
                if (hit_idx < 0 && pool.candidates[i].type == ShotType::HIT) hit_idx = i;
                if (guard_idx < 0 && (pool.candidates[i].type == ShotType::PREGUARD || pool.candidates[i].type == ShotType::POSTGUARD)) guard_idx = i;
                if (freeze_idx < 0 && pool.candidates[i].type == ShotType::FREEZE) freeze_idx = i;
            }
            std::cout << "  Distance samples (orig/delta):" << std::endl;
            if (draw_idx >= 0 && hit_idx >= 0)
                std::cout << "    Draw-Hit:    " << std::setprecision(2) << dist_orig[draw_idx][hit_idx]
                          << " / " << dist_delta[draw_idx][hit_idx]
                          << "  [" << pool.candidates[draw_idx].label << " vs " << pool.candidates[hit_idx].label << "]" << std::endl;
            if (draw_idx >= 0 && guard_idx >= 0)
                std::cout << "    Draw-Guard:  " << dist_orig[draw_idx][guard_idx]
                          << " / " << dist_delta[draw_idx][guard_idx]
                          << "  [" << pool.candidates[draw_idx].label << " vs " << pool.candidates[guard_idx].label << "]" << std::endl;
            if (draw_idx >= 0 && freeze_idx >= 0)
                std::cout << "    Draw-Freeze: " << dist_orig[draw_idx][freeze_idx]
                          << " / " << dist_delta[draw_idx][freeze_idx]
                          << "  [" << pool.candidates[draw_idx].label << " vs " << pool.candidates[freeze_idx].label << "]" << std::endl;
            // Draw同士の距離
            if (draw_idx >= 0) {
                for (int i = draw_idx + 1; i < static_cast<int>(pool.candidates.size()); i++) {
                    if (pool.candidates[i].type == ShotType::DRAW && pool.candidates[i].label != pool.candidates[draw_idx].label) {
                        std::cout << "    Draw-Draw:   " << dist_orig[draw_idx][i]
                                  << " / " << dist_delta[draw_idx][i]
                                  << "  [" << pool.candidates[draw_idx].label << " vs " << pool.candidates[i].label << "]" << std::endl;
                        break;
                    }
                }
            }
            // Hit(STRONG)がある場合
            int hit_strong_idx = -1;
            for (int i = 0; i < static_cast<int>(pool.candidates.size()); i++) {
                if (pool.candidates[i].type == ShotType::HIT && pool.candidates[i].param == 3) { hit_strong_idx = i; break; }
            }
            if (draw_idx >= 0 && hit_strong_idx >= 0)
                std::cout << "    Draw-HitSTR: " << dist_orig[draw_idx][hit_strong_idx]
                          << " / " << dist_delta[draw_idx][hit_strong_idx]
                          << "  [" << pool.candidates[draw_idx].label << " vs " << pool.candidates[hit_strong_idx].label << "]" << std::endl;
        }

        for (int k : k_values) {
            if (k >= static_cast<int>(pool.candidates.size())) continue;

            // --- Original distance ---
            auto clusters_orig = runClustering(dist_orig, k);
            auto medoids_orig = calculateMedoids(dist_orig, clusters_orig);
            auto result_orig = analyzeClusterComposition(state_names[s], pool, clusters_orig, medoids_orig);

            // --- Delta distance ---
            auto clusters_delta = runClustering(dist_delta, k);
            auto medoids_delta = calculateMedoids(dist_delta, clusters_delta);
            auto result_delta = analyzeClusterComposition(state_names[s], pool, clusters_delta, medoids_delta);

            // 最大クラスタサイズ
            int max_orig = 0, max_delta = 0;
            for (auto& c : result_orig.clusters) max_orig = std::max(max_orig, static_cast<int>(c.member_indices.size()));
            for (auto& c : result_delta.clusters) max_delta = std::max(max_delta, static_cast<int>(c.member_indices.size()));

            // 出力
            std::cout << "\n  --- K=" << k << " ---" << std::endl;
            std::cout << "  [Original] Purity=" << std::setprecision(3) << result_orig.weightedPurity()
                      << "  Coverage=" << result_orig.typeCoverage()
                      << "  MaxCluster=" << max_orig << "/" << pool.candidates.size() << std::endl;
            std::cout << "  [Delta]    Purity=" << std::setprecision(3) << result_delta.weightedPurity()
                      << "  Coverage=" << result_delta.typeCoverage()
                      << "  MaxCluster=" << max_delta << "/" << pool.candidates.size() << std::endl;

            // Delta距離のクラスタ詳細を表示
            printResult(result_delta);

            // サマリー蓄積
            summary.push_back({
                state_names[s], k,
                result_orig.weightedPurity(), result_orig.typeCoverage(),
                result_delta.weightedPurity(), result_delta.typeCoverage(),
                max_orig, max_delta
            });

            // Delta距離のCSVエクスポート
            {
                std::string csv = output_dir + "/delta_" + state_names[s]
                                + "_k" + std::to_string(k) + ".csv";
                std::ofstream ofs(csv);
                ofs << "candidate_index,cluster_id,shot_type,shot_type_name,label,is_medoid,"
                    << "spin,target_index,param,vx,vy,rot";
                for (int t = 0; t < 2; t++) {
                    for (int i = 0; i < 8; i++) {
                        ofs << ",team" << t << "_stone" << i << "_x"
                            << ",team" << t << "_stone" << i << "_y"
                            << ",team" << t << "_stone" << i << "_active";
                    }
                }
                ofs << "\n";

                for (size_t ci = 0; ci < clusters_delta.size(); ci++) {
                    for (int idx : clusters_delta[ci]) {
                        auto& cand = pool.candidates[idx];
                        auto& rs = pool.result_states[idx];
                        ofs << idx << ","
                            << ci << ","
                            << static_cast<int>(cand.type) << ","
                            << typeToName(cand.type) << ","
                            << "\"" << cand.label << "\","
                            << (idx == medoids_delta[ci] ? 1 : 0) << ","
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

    // 比較サマリー表
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Comparison Summary: Original vs Delta Distance" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::setw(12) << "State"
              << std::setw(4) << "K"
              << "  | " << std::setw(8) << "Purity" << std::setw(8) << "Cover" << std::setw(8) << "MaxCl"
              << "  | " << std::setw(8) << "Purity" << std::setw(8) << "Cover" << std::setw(8) << "MaxCl"
              << "  | " << std::setw(8) << "Improv" << std::endl;
    std::cout << std::setw(16) << ""
              << "  | " << std::setw(24) << "--- Original ---"
              << "  | " << std::setw(24) << "--- Delta ---"
              << "  |" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    for (auto& row : summary) {
        float improvement = row.purity_delta - row.purity_orig;
        std::cout << std::setw(12) << row.state
                  << std::setw(4) << row.k
                  << "  | " << std::setw(8) << std::fixed << std::setprecision(3) << row.purity_orig
                  << std::setw(8) << row.coverage_orig
                  << std::setw(8) << row.max_cluster_orig
                  << "  | " << std::setw(8) << row.purity_delta
                  << std::setw(8) << row.coverage_delta
                  << std::setw(8) << row.max_cluster_delta
                  << "  | " << std::setw(7) << std::showpos << improvement << std::noshowpos
                  << std::endl;
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << "  Pool Clustering Experiment Complete" << std::endl;
    std::cout << "================================================" << std::endl;
}
