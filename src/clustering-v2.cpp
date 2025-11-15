#include "clustering-v2.h"
#include "analysis.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <numeric>

namespace dc = digitalcurling3;

// ========================================
// StateFeature 実装
// ========================================

StateFeature::StateFeature()
    : total_stones(0)
    , my_stones(0)
    , opponent_stones(0)
    , my_stones_in_house(0)
    , opponent_stones_in_house(0)
    , has_no1_stone(false)
    , no1_team(-1)
    , original_state_id(-1)
{
    my_team_distribution.fill(0);
    opponent_team_distribution.fill(0);
}

// ========================================
// Cluster 実装
// ========================================

Cluster::Cluster()
    : representative_id(-1)
    , internal_variance(0.0f)
{
}

// ========================================
// ClusteringV2 実装
// ========================================

ClusteringV2::ClusteringV2(
    int k_clusters,
    std::vector<dc::GameState> all_states,
    int gridM,
    int gridN,
    dc::Team team
)
    : GridSize_M_(gridM)
    , GridSize_N_(gridN)
    , n_desired_clusters_(k_clusters)
    , g_team_(team)
    , clustering_done_(false)
{
    if (all_states.empty()) {
        std::cerr << "[ClusteringV2 Error] Input states are empty!" << std::endl;
        return;
    }

    states_ = std::move(all_states);
    features_.reserve(states_.size());

    std::cout << "[ClusteringV2] Initialized with " << states_.size()
              << " states, target clusters: " << n_desired_clusters_ << std::endl;
}

// ========================================
// メイン関数
// ========================================

std::vector<Cluster> ClusteringV2::getClusters() {
    if (clustering_done_) {
        return clusters_;
    }

    std::cout << "[ClusteringV2] Starting clustering process..." << std::endl;

    // ステップ1: 特徴抽出
    extractFeatures();

    // ステップ2: 粗分類 (総石数でグループ化)
    auto coarse_groups = coarseGrouping();
    std::cout << "[ClusteringV2] Coarse grouping created "
              << coarse_groups.size() << " groups" << std::endl;

    // ステップ3: 各グループ内で細分類
    clusters_.clear();

    // クラスタ数の配分計算
    int total_states = static_cast<int>(states_.size());
    for (const auto& [stone_count, group_ids] : coarse_groups) {
        // このグループに割り当てるクラスタ数を計算
        // グループサイズに比例して配分
        int group_k = std::max(1, static_cast<int>(
            std::round(n_desired_clusters_ * group_ids.size() / static_cast<float>(total_states))
        ));

        // グループサイズより多いクラスタ数は無意味
        group_k = std::min(group_k, static_cast<int>(group_ids.size()));

        std::cout << "[ClusteringV2] Group (stones=" << stone_count
                  << ", size=" << group_ids.size()
                  << ") -> " << group_k << " clusters" << std::endl;

        // 細分類実行
        auto group_clusters = fineGrainedClustering(group_ids, group_k);

        // 結果をマージ
        clusters_.insert(clusters_.end(), group_clusters.begin(), group_clusters.end());
    }

    // クラスタ数が目標と異なる場合は調整
    adjustClusterCount();

    // 各クラスタの代表を選出
    for (auto& cluster : clusters_) {
        selectRepresentative(cluster);
        cluster.internal_variance = computeVariance(cluster);
    }

    clustering_done_ = true;

    std::cout << "[ClusteringV2] Clustering completed! Final clusters: "
              << clusters_.size() << std::endl;

    // デバッグ情報
    printDebugInfo();

    return clusters_;
}

std::vector<int> ClusteringV2::getRecommendedStates() {
    if (!clustering_done_) {
        getClusters();
    }

    std::vector<int> recommended;
    recommended.reserve(clusters_.size());

    for (const auto& cluster : clusters_) {
        if (cluster.representative_id >= 0) {
            recommended.push_back(cluster.representative_id);
        }
    }

    std::cout << "[ClusteringV2] Recommended " << recommended.size()
              << " representative states" << std::endl;

    return recommended;
}

std::vector<std::vector<int>> ClusteringV2::getClusterIdTable() {
    if (!clustering_done_) {
        getClusters();
    }

    std::vector<std::vector<int>> table;
    table.reserve(clusters_.size());

    for (const auto& cluster : clusters_) {
        table.push_back(cluster.state_ids);
    }

    return table;
}

float ClusteringV2::evaluateClusteringQuality() const {
    if (!clustering_done_ || clusters_.empty()) {
        return 0.0f;
    }

    // シルエット係数の簡易版
    // クラスタ内距離 vs クラスタ間距離
    float total_score = 0.0f;
    int count = 0;

    for (size_t i = 0; i < clusters_.size(); ++i) {
        for (int state_id : clusters_[i].state_ids) {
            // クラスタ内平均距離
            float intra_dist = 0.0f;
            for (int other_id : clusters_[i].state_ids) {
                if (state_id != other_id) {
                    intra_dist += computeDistance(features_[state_id], features_[other_id]);
                }
            }
            if (clusters_[i].state_ids.size() > 1) {
                intra_dist /= (clusters_[i].state_ids.size() - 1);
            }

            // 最近傍クラスタへの平均距離
            float min_inter_dist = std::numeric_limits<float>::max();
            for (size_t j = 0; j < clusters_.size(); ++j) {
                if (i == j) continue;

                float inter_dist = 0.0f;
                for (int other_id : clusters_[j].state_ids) {
                    inter_dist += computeDistance(features_[state_id], features_[other_id]);
                }
                inter_dist /= clusters_[j].state_ids.size();
                min_inter_dist = std::min(min_inter_dist, inter_dist);
            }

            // シルエット値
            float s = 0.0f;
            if (intra_dist < min_inter_dist) {
                s = 1.0f - intra_dist / min_inter_dist;
            } else if (intra_dist > min_inter_dist) {
                s = min_inter_dist / intra_dist - 1.0f;
            }

            total_score += s;
            count++;
        }
    }

    return (count > 0) ? (total_score / count) : 0.0f;
}

// ========================================
// ステップ1: 特徴抽出
// ========================================

void ClusteringV2::extractFeatures() {
    std::cout << "[ClusteringV2] Extracting features from " << states_.size()
              << " states..." << std::endl;

    features_.clear();
    features_.reserve(states_.size());

    for (size_t i = 0; i < states_.size(); ++i) {
        features_.push_back(computeFeature(states_[i], static_cast<int>(i)));
    }

    std::cout << "[ClusteringV2] Feature extraction completed." << std::endl;
}

StateFeature ClusteringV2::computeFeature(const dc::GameState& state, int state_id) {
    StateFeature feature;
    feature.original_state_id = state_id;

    // 初期化
    feature.my_team_distribution.fill(0);
    feature.opponent_team_distribution.fill(0);

    // No.1ストーン情報
    float min_distance_to_center = std::numeric_limits<float>::max();
    int no1_team = -1;

    // 各チーム、各石を調査
    for (size_t team = 0; team < 2; ++team) {
        for (size_t index = 0; index < 8; ++index) {
            const auto& stone = state.stones[team][index];
            if (!stone) continue;

            float x = stone->position.x;
            float y = stone->position.y;

            // 総石数カウント
            feature.total_stones++;
            if (team == static_cast<size_t>(g_team_)) {
                feature.my_stones++;
            } else {
                feature.opponent_stones++;
            }

            // ハウス判定
            if (isInHouse(stone)) {
                if (team == static_cast<size_t>(g_team_)) {
                    feature.my_stones_in_house++;
                } else {
                    feature.opponent_stones_in_house++;
                }
            }

            // 領域判定
            GridRegion region = getRegion(x, y);
            if (region != GridRegion::OutOfBounds) {
                int region_idx = static_cast<int>(region);
                if (team == static_cast<size_t>(g_team_)) {
                    feature.my_team_distribution[region_idx]++;
                } else {
                    feature.opponent_team_distribution[region_idx]++;
                }
            }

            // No.1ストーン判定
            float dist_to_center = std::sqrt(
                std::pow(x - HouseCenterX_, 2) + std::pow(y - HouseCenterY_, 2)
            );
            if (dist_to_center < min_distance_to_center) {
                min_distance_to_center = dist_to_center;
                no1_team = static_cast<int>(team);
            }
        }
    }

    // No.1ストーン情報を設定
    if (no1_team >= 0) {
        feature.has_no1_stone = true;
        feature.no1_team = no1_team;
    }

    return feature;
}

GridRegion ClusteringV2::getRegion(float x, float y) const {
    // Y方向の判定
    bool is_upper = (y >= HouseCenterY_);

    // X方向の判定
    int x_zone;
    if (x < -1.0f) {
        x_zone = 0; // 左
    } else if (x <= 1.0f) {
        x_zone = 1; // 中央
    } else {
        x_zone = 2; // 右
    }

    // 領域インデックス計算
    int region_idx;
    if (is_upper) {
        region_idx = x_zone; // 0, 1, 2
    } else {
        region_idx = 3 + x_zone; // 3, 4, 5
    }

    return static_cast<GridRegion>(region_idx);
}

bool ClusteringV2::isInHouse(const std::optional<dc::Transform>& stone) const {
    if (!stone) return false;

    float x = stone->position.x;
    float y = stone->position.y;
    float dist_sq = std::pow(x - HouseCenterX_, 2) + std::pow(y - HouseCenterY_, 2);

    return dist_sq <= std::pow(HouseRadius_, 2);
}

// ========================================
// ステップ2: 類似度計算
// ========================================

float ClusteringV2::computeDistance(const StateFeature& f1, const StateFeature& f2) const {
    float distance = 0.0f;

    // 1. 総石数の差 (重み: 5.0)
    float stone_diff = std::abs(f1.total_stones - f2.total_stones);
    distance += stone_diff * 5.0f;

    // 2. 6領域分布の差 (重み: 3.0)
    for (int i = 0; i < 6; ++i) {
        float my_diff = std::abs(f1.my_team_distribution[i] - f2.my_team_distribution[i]);
        float opp_diff = std::abs(f1.opponent_team_distribution[i] - f2.opponent_team_distribution[i]);
        distance += (my_diff + opp_diff) * 3.0f;
    }

    // 3. ハウス内石数の差 (重み: 8.0)
    float house_my_diff = std::abs(f1.my_stones_in_house - f2.my_stones_in_house);
    float house_opp_diff = std::abs(f1.opponent_stones_in_house - f2.opponent_stones_in_house);
    distance += (house_my_diff + house_opp_diff) * 8.0f;

    // 4. No.1ストーンチームの一致/不一致 (重み: 12.0)
    if (f1.has_no1_stone && f2.has_no1_stone) {
        if (f1.no1_team != f2.no1_team) {
            distance += 12.0f;
        }
    } else if (f1.has_no1_stone != f2.has_no1_stone) {
        // 片方にしかNo.1ストーンがない場合
        distance += 6.0f;
    }

    return distance;
}

// ========================================
// ステップ3: クラスタリング
// ========================================

std::map<int, std::vector<int>> ClusteringV2::coarseGrouping() {
    std::map<int, std::vector<int>> groups;

    for (size_t i = 0; i < features_.size(); ++i) {
        int total_stones = features_[i].total_stones;
        groups[total_stones].push_back(static_cast<int>(i));
    }

    return groups;
}

std::vector<Cluster> ClusteringV2::fineGrainedClustering(
    const std::vector<int>& group_ids,
    int k
) {
    // グループサイズが小さい場合は各状態を個別クラスタに
    if (group_ids.size() <= static_cast<size_t>(k)) {
        std::vector<Cluster> clusters;
        for (int id : group_ids) {
            Cluster cluster;
            cluster.state_ids.push_back(id);
            cluster.representative_id = id;
            clusters.push_back(cluster);
        }
        return clusters;
    }

    // k-meansクラスタリング実行
    return kMeansClustering(group_ids, k);
}

std::vector<Cluster> ClusteringV2::kMeansClustering(
    const std::vector<int>& feature_indices,
    int k,
    int max_iterations
) {
    std::vector<Cluster> clusters(k);
    std::random_device rd;
    std::mt19937 gen(rd());

    // 初期重心をランダムに選択 (k-means++)
    std::vector<int> centroid_indices;
    {
        std::uniform_int_distribution<> dis(0, feature_indices.size() - 1);
        centroid_indices.push_back(feature_indices[dis(gen)]);

        for (int i = 1; i < k; ++i) {
            std::vector<float> min_dists(feature_indices.size(), std::numeric_limits<float>::max());

            // 各点から最も近い既存重心への距離を計算
            for (size_t j = 0; j < feature_indices.size(); ++j) {
                for (int c_idx : centroid_indices) {
                    float d = computeDistance(features_[feature_indices[j]], features_[c_idx]);
                    min_dists[j] = std::min(min_dists[j], d);
                }
            }

            // 距離に比例した確率で次の重心を選択
            std::discrete_distribution<> dist_dist(min_dists.begin(), min_dists.end());
            int next_idx = dist_dist(gen);
            centroid_indices.push_back(feature_indices[next_idx]);
        }
    }

    // 重心を設定
    for (int i = 0; i < k; ++i) {
        clusters[i].centroid = features_[centroid_indices[i]];
    }

    // k-means反復
    bool converged = false;
    for (int iter = 0; iter < max_iterations && !converged; ++iter) {
        // クラスタをクリア
        for (auto& cluster : clusters) {
            cluster.state_ids.clear();
        }

        // 各点を最も近い重心に割り当て
        for (int idx : feature_indices) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;

            for (int c = 0; c < k; ++c) {
                float dist = computeDistance(features_[idx], clusters[c].centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }

            clusters[best_cluster].state_ids.push_back(idx);
        }

        // 重心を更新
        converged = true;
        for (auto& cluster : clusters) {
            if (cluster.state_ids.empty()) continue;

            StateFeature new_centroid = computeCentroid(cluster);

            // 変化があったかチェック
            if (computeDistance(cluster.centroid, new_centroid) > 0.01f) {
                converged = false;
            }

            cluster.centroid = new_centroid;
        }
    }

    // 空のクラスタを除去
    clusters.erase(
        std::remove_if(clusters.begin(), clusters.end(),
            [](const Cluster& c) { return c.state_ids.empty(); }),
        clusters.end()
    );

    return clusters;
}

StateFeature ClusteringV2::computeCentroid(const Cluster& cluster) const {
    if (cluster.state_ids.empty()) {
        return StateFeature();
    }

    StateFeature centroid;

    // 各特徴の平均を計算
    for (int id : cluster.state_ids) {
        const auto& f = features_[id];
        centroid.total_stones += f.total_stones;
        centroid.my_stones += f.my_stones;
        centroid.opponent_stones += f.opponent_stones;
        centroid.my_stones_in_house += f.my_stones_in_house;
        centroid.opponent_stones_in_house += f.opponent_stones_in_house;

        for (int i = 0; i < 6; ++i) {
            centroid.my_team_distribution[i] += f.my_team_distribution[i];
            centroid.opponent_team_distribution[i] += f.opponent_team_distribution[i];
        }
    }

    int n = static_cast<int>(cluster.state_ids.size());
    centroid.total_stones /= n;
    centroid.my_stones /= n;
    centroid.opponent_stones /= n;
    centroid.my_stones_in_house /= n;
    centroid.opponent_stones_in_house /= n;

    for (int i = 0; i < 6; ++i) {
        centroid.my_team_distribution[i] /= n;
        centroid.opponent_team_distribution[i] /= n;
    }

    // No.1ストーンチームは多数決
    std::map<int, int> team_count;
    for (int id : cluster.state_ids) {
        if (features_[id].has_no1_stone) {
            team_count[features_[id].no1_team]++;
        }
    }
    if (!team_count.empty()) {
        auto max_it = std::max_element(team_count.begin(), team_count.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        centroid.has_no1_stone = true;
        centroid.no1_team = max_it->first;
    }

    return centroid;
}

void ClusteringV2::selectRepresentative(Cluster& cluster) {
    if (cluster.state_ids.empty()) {
        return;
    }

    // 重心に最も近い状態を選ぶ
    float min_dist = std::numeric_limits<float>::max();
    int best_id = cluster.state_ids[0];

    for (int id : cluster.state_ids) {
        float dist = computeDistance(features_[id], cluster.centroid);
        if (dist < min_dist) {
            min_dist = dist;
            best_id = id;
        }
    }

    cluster.representative_id = best_id;
}

float ClusteringV2::computeVariance(const Cluster& cluster) const {
    if (cluster.state_ids.size() <= 1) {
        return 0.0f;
    }

    float variance = 0.0f;
    for (int id : cluster.state_ids) {
        float dist = computeDistance(features_[id], cluster.centroid);
        variance += dist * dist;
    }

    return variance / cluster.state_ids.size();
}

// ========================================
// ユーティリティ
// ========================================

void ClusteringV2::adjustClusterCount() {
    // クラスタ数が多すぎる場合: 小さいクラスタ同士をマージ
    while (static_cast<int>(clusters_.size()) > n_desired_clusters_) {
        // 最も近い2つのクラスタを見つける
        float min_dist = std::numeric_limits<float>::max();
        size_t merge_i = 0, merge_j = 1;

        for (size_t i = 0; i < clusters_.size(); ++i) {
            for (size_t j = i + 1; j < clusters_.size(); ++j) {
                float dist = computeDistance(clusters_[i].centroid, clusters_[j].centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    merge_i = i;
                    merge_j = j;
                }
            }
        }

        // マージ
        clusters_[merge_i].state_ids.insert(
            clusters_[merge_i].state_ids.end(),
            clusters_[merge_j].state_ids.begin(),
            clusters_[merge_j].state_ids.end()
        );
        clusters_[merge_i].centroid = computeCentroid(clusters_[merge_i]);

        clusters_.erase(clusters_.begin() + merge_j);
    }

    // クラスタ数が少なすぎる場合: 大きいクラスタを分割
    while (static_cast<int>(clusters_.size()) < n_desired_clusters_) {
        // 最も分散が大きいクラスタを見つける
        size_t split_idx = 0;
        float max_var = -1.0f;

        for (size_t i = 0; i < clusters_.size(); ++i) {
            if (clusters_[i].state_ids.size() < 2) continue;

            float var = computeVariance(clusters_[i]);
            if (var > max_var) {
                max_var = var;
                split_idx = i;
            }
        }

        if (clusters_[split_idx].state_ids.size() < 2) {
            // これ以上分割できない
            break;
        }

        // 2つに分割
        auto split_clusters = kMeansClustering(clusters_[split_idx].state_ids, 2);

        // 元のクラスタを置き換え
        clusters_[split_idx] = split_clusters[0];
        if (split_clusters.size() > 1) {
            clusters_.push_back(split_clusters[1]);
        }
    }

    std::cout << "[ClusteringV2] Adjusted to " << clusters_.size() << " clusters" << std::endl;
}

void ClusteringV2::printDebugInfo() const {
    std::cout << "\n=== ClusteringV2 Debug Info ===" << std::endl;
    std::cout << "Total states: " << states_.size() << std::endl;
    std::cout << "Total clusters: " << clusters_.size() << std::endl;

    for (size_t i = 0; i < clusters_.size(); ++i) {
        std::cout << "Cluster " << i << ": "
                  << clusters_[i].state_ids.size() << " states, "
                  << "representative: " << clusters_[i].representative_id << ", "
                  << "variance: " << clusters_[i].internal_variance << std::endl;
    }

    float quality = evaluateClusteringQuality();
    std::cout << "Clustering quality (silhouette): " << quality << std::endl;
    std::cout << "================================\n" << std::endl;
}
