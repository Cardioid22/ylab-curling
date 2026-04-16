#include "clustering_effectiveness_experiment.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>
#include <random>
#include <set>
#include <cassert>

namespace dc = digitalcurling3;

ClusteringEffectivenessExperiment::ClusteringEffectivenessExperiment(
    dc::GameSetting const& game_setting)
    : game_setting_(game_setting)
{
    policy_ = std::make_unique<RolloutPolicy>();
    shot_gen_ = std::make_unique<ShotGenerator>(game_setting);
}

// ============================================================
// Phase 1: テスト盤面の生成 (gPolicyで自己対戦)
// ============================================================
std::vector<ClusteringEffectivenessExperiment::GameRecord>
ClusteringEffectivenessExperiment::generateTestPositions() {
    std::vector<GameRecord> records;

    std::cout << "\n=== Phase 1: Generating test positions via self-play ===" << std::endl;
    std::cout << "  Games: " << test_games_ << std::endl;

    SimulatorWrapper sim(dc::Team::k0, game_setting_);

    for (int game = 0; game < test_games_; ++game) {
        std::cout << "  Game " << game << "..." << std::flush;

        dc::GameState state(game_setting_);
        // 明示的に初期化
        state.end = 0;
        state.shot = 0;
        int positions_collected = 0;

        // 1ゲーム分 (8エンド) を自己対戦
        // IsGameOver は end >= max_end で判定
        int max_shots = game_setting_.max_end * 16;  // 8エンド × 16ショット
        for (int total_shot = 0; total_shot < max_shots && !state.IsGameOver(); ++total_shot) {
            int end = static_cast<int>(state.end);
            int shot_num = static_cast<int>(state.shot);

            // 現在のチーム (偶数ショット=Team0)
            dc::Team current_team = (shot_num % 2 == 0) ? dc::Team::k0 : dc::Team::k1;

            // この盤面をテスト用に記録
            GameRecord rec;
            rec.game_id = game;
            rec.end = end;
            rec.shot_num = shot_num;
            rec.state = state;
            rec.current_team = current_team;
            records.push_back(rec);
            positions_collected++;

            // gPolicy で手を選んでシミュレーション → 次の状態へ
            auto candidates = shot_gen_->generateCandidates(state, current_team);

            // Pass除外
            std::vector<CandidateShot> filtered;
            for (auto& c : candidates) {
                if (c.type != ShotType::PASS) filtered.push_back(c);
            }
            if (filtered.empty()) filtered = candidates;

            int sel = policy_->selectShot(state, filtered, shot_num, current_team, end, 0);
            ShotInfo shot = filtered[sel].shot;
            state = sim.run_single_simulation(state, shot);

        }
        std::cout << " " << positions_collected << " positions" << std::endl;
    }

    std::cout << "  Total: " << records.size() << " test positions" << std::endl;
    return records;
}

// ============================================================
// 候補手のロールアウト評価
// ============================================================
double ClusteringEffectivenessExperiment::evaluateCandidate(
    const dc::GameState& state,
    const CandidateShot& candidate,
    SimulatorWrapper& sim)
{
    // エンド終了までに制限 (16ショット/エンド)
    int remaining = 16 - static_cast<int>(state.shot);
    sim.max_rollout_shots = remaining;

    double total = 0.0;
    for (int r = 0; r < rollout_count_; ++r) {
        total += sim.run_policy_rollout(
            state, candidate.shot, *policy_, *shot_gen_, 1);
    }
    return total / rollout_count_;
}

// ============================================================
// Phase 2: 各盤面で AllGrid vs Clustered を比較
// ============================================================
TestCaseResult ClusteringEffectivenessExperiment::evaluatePosition(
    const GameRecord& record,
    int retention_pct,
    SimulatorWrapper& sim)
{
    TestCaseResult result;
    result.game_id = record.game_id;
    result.end = record.end;
    result.shot_num = record.shot_num;

    // 候補手生成 + シミュレーション
    auto pool = shot_gen_->generatePool(record.state, record.current_team);
    auto& candidates = pool.candidates;
    auto& result_states = pool.result_states;
    int n = static_cast<int>(candidates.size());
    result.n_candidates = n;

    if (n == 0) {
        result.exact_match = true;
        result.same_cluster = true;
        result.same_type = true;
        result.score_diff = 0;
        return result;
    }

    // --- AllGrid: 全候補をロールアウト評価 ---
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<double> allgrid_scores(n);
    for (int i = 0; i < n; ++i) {
        allgrid_scores[i] = evaluateCandidate(record.state, candidates[i], sim);
    }
    int allgrid_best = static_cast<int>(
        std::max_element(allgrid_scores.begin(), allgrid_scores.end()) - allgrid_scores.begin());

    auto t1 = std::chrono::high_resolution_clock::now();
    result.allgrid_time_sec = std::chrono::duration<double>(t1 - t0).count();
    result.allgrid_best_idx = allgrid_best;
    result.allgrid_best_label = candidates[allgrid_best].label;
    result.allgrid_best_type = candidates[allgrid_best].type;
    result.allgrid_best_score = allgrid_scores[allgrid_best];

    // --- クラスタリング ---
    auto t2 = std::chrono::high_resolution_clock::now();

    int n_clusters = std::max(1, n * retention_pct / 100);
    result.n_clustered = n_clusters;

    auto dist_table = makeDistanceTableDelta(record.state, result_states);
    auto clusters = runClustering(dist_table, n_clusters);
    auto medoids = calculateMedoids(dist_table, clusters);

    // シルエットスコア
    result.silhouette_score = calcSilhouetteScore(dist_table, clusters);

    // AllGrid最良手のクラスタを特定
    result.allgrid_cluster_id = -1;
    for (int c = 0; c < static_cast<int>(clusters.size()); ++c) {
        if (clusters[c].count(allgrid_best)) {
            result.allgrid_cluster_id = c;
            break;
        }
    }

    // クラスタ構成を記録 (proposed)
    for (int c = 0; c < static_cast<int>(clusters.size()); ++c) {
        ClusterInfo ci;
        ci.game_id = record.game_id;
        ci.end = record.end;
        ci.shot_num = record.shot_num;
        ci.n_candidates = n;
        ci.method = "proposed";
        ci.cluster_id = c;
        ci.cluster_size = static_cast<int>(clusters[c].size());
        ci.medoid_idx = medoids[c];
        ci.medoid_label = (medoids[c] >= 0 && medoids[c] < n) ? candidates[medoids[c]].label : "N/A";
        cluster_details_.push_back(ci);
    }

    // --- Clustered: メドイドのみ評価 ---
    double clustered_best_score = -1e9;
    int clustered_best_medoid_idx = -1;
    int clustered_best_cluster = -1;

    for (int c = 0; c < static_cast<int>(medoids.size()); ++c) {
        int m = medoids[c];
        if (m < 0 || m >= n) continue;
        double score = evaluateCandidate(record.state, candidates[m], sim);
        if (score > clustered_best_score) {
            clustered_best_score = score;
            clustered_best_medoid_idx = m;
            clustered_best_cluster = c;
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    result.clustered_time_sec = std::chrono::duration<double>(t3 - t2).count();
    result.clustered_best_idx = clustered_best_medoid_idx;
    result.clustered_best_label = (clustered_best_medoid_idx >= 0) ?
        candidates[clustered_best_medoid_idx].label : "N/A";
    result.clustered_best_type = (clustered_best_medoid_idx >= 0) ?
        candidates[clustered_best_medoid_idx].type : ShotType::PASS;
    result.clustered_best_score = clustered_best_score;
    result.clustered_cluster_id = clustered_best_cluster;

    // --- 一致指標 ---
    result.exact_match = (allgrid_best == clustered_best_medoid_idx);
    result.same_cluster = (result.allgrid_cluster_id == clustered_best_cluster);
    result.same_type = (result.allgrid_best_type == result.clustered_best_type);
    result.score_diff = result.allgrid_best_score - result.clustered_best_score;

    // --- Spatial Clustering (ベースライン): 結果盤面の石座標距離のみ ---
    auto t_sp0 = std::chrono::high_resolution_clock::now();

    auto spatial_dist_table = makeDistanceTableSpatial(result_states);
    auto spatial_clusters = runClustering(spatial_dist_table, n_clusters);
    auto spatial_medoids = calculateMedoids(spatial_dist_table, spatial_clusters);

    result.spatial_silhouette_score = calcSilhouetteScore(spatial_dist_table, spatial_clusters);

    // クラスタ構成を記録 (spatial)
    for (int c = 0; c < static_cast<int>(spatial_clusters.size()); ++c) {
        ClusterInfo ci;
        ci.game_id = record.game_id;
        ci.end = record.end;
        ci.shot_num = record.shot_num;
        ci.n_candidates = n;
        ci.method = "spatial";
        ci.cluster_id = c;
        ci.cluster_size = static_cast<int>(spatial_clusters[c].size());
        ci.medoid_idx = spatial_medoids[c];
        ci.medoid_label = (spatial_medoids[c] >= 0 && spatial_medoids[c] < n) ? candidates[spatial_medoids[c]].label : "N/A";
        cluster_details_.push_back(ci);
    }

    double spatial_best_score = -1e9;
    int spatial_best_medoid_idx = -1;
    int spatial_best_cluster = -1;

    for (int c = 0; c < static_cast<int>(spatial_medoids.size()); ++c) {
        int m = spatial_medoids[c];
        if (m < 0 || m >= n) continue;
        double score = evaluateCandidate(record.state, candidates[m], sim);
        if (score > spatial_best_score) {
            spatial_best_score = score;
            spatial_best_medoid_idx = m;
            spatial_best_cluster = c;
        }
    }

    auto t_sp1 = std::chrono::high_resolution_clock::now();
    result.spatial_time_sec = std::chrono::duration<double>(t_sp1 - t_sp0).count();
    result.spatial_best_idx = spatial_best_medoid_idx;
    result.spatial_best_label = (spatial_best_medoid_idx >= 0) ?
        candidates[spatial_best_medoid_idx].label : "N/A";
    result.spatial_best_type = (spatial_best_medoid_idx >= 0) ?
        candidates[spatial_best_medoid_idx].type : ShotType::PASS;
    result.spatial_best_score = spatial_best_score;

    result.spatial_exact_match = (allgrid_best == spatial_best_medoid_idx);
    int allgrid_spatial_cluster_id = -1;
    for (int c = 0; c < static_cast<int>(spatial_clusters.size()); ++c) {
        if (spatial_clusters[c].count(allgrid_best)) { allgrid_spatial_cluster_id = c; break; }
    }
    result.spatial_same_cluster = (allgrid_spatial_cluster_id == spatial_best_cluster);
    result.spatial_same_type = (result.allgrid_best_type == result.spatial_best_type);
    result.spatial_score_diff = result.allgrid_best_score - spatial_best_score;

    // --- Random Clustering (ベースライン): ランダムにクラスタ割り当て ---
    auto t4 = std::chrono::high_resolution_clock::now();

    // シードは盤面ごとに変える（再現性のため game_id, end, shot_num を混ぜる）
    std::mt19937 rng(42 + record.game_id * 10000 + record.end * 100 + record.shot_num);
    auto random_clusters = runRandomClustering(n, n_clusters, rng);
    auto random_medoids = calculateMedoids(dist_table, random_clusters);

    result.random_silhouette_score = calcSilhouetteScore(dist_table, random_clusters);

    // クラスタ構成を記録 (random)
    for (int c = 0; c < static_cast<int>(random_clusters.size()); ++c) {
        ClusterInfo ci;
        ci.game_id = record.game_id;
        ci.end = record.end;
        ci.shot_num = record.shot_num;
        ci.n_candidates = n;
        ci.method = "random";
        ci.cluster_id = c;
        ci.cluster_size = static_cast<int>(random_clusters[c].size());
        ci.medoid_idx = random_medoids[c];
        ci.medoid_label = (random_medoids[c] >= 0 && random_medoids[c] < n) ? candidates[random_medoids[c]].label : "N/A";
        cluster_details_.push_back(ci);
    }

    double random_best_score = -1e9;
    int random_best_medoid_idx = -1;

    for (int c = 0; c < static_cast<int>(random_medoids.size()); ++c) {
        int m = random_medoids[c];
        if (m < 0 || m >= n) continue;
        double score = evaluateCandidate(record.state, candidates[m], sim);
        if (score > random_best_score) {
            random_best_score = score;
            random_best_medoid_idx = m;
        }
    }

    auto t5 = std::chrono::high_resolution_clock::now();
    result.random_time_sec = std::chrono::duration<double>(t5 - t4).count();
    result.random_best_idx = random_best_medoid_idx;
    result.random_best_label = (random_best_medoid_idx >= 0) ?
        candidates[random_best_medoid_idx].label : "N/A";
    result.random_best_type = (random_best_medoid_idx >= 0) ?
        candidates[random_best_medoid_idx].type : ShotType::PASS;
    result.random_best_score = random_best_score;

    result.random_exact_match = (allgrid_best == random_best_medoid_idx);
    // AllGrid最良手が属するランダムクラスタを特定
    int allgrid_random_cluster_id = -1;
    int random_best_cluster_id = -1;
    for (int c = 0; c < static_cast<int>(random_clusters.size()); ++c) {
        if (random_clusters[c].count(allgrid_best)) allgrid_random_cluster_id = c;
        if (random_clusters[c].count(random_best_medoid_idx)) random_best_cluster_id = c;
    }
    result.random_same_cluster = (allgrid_random_cluster_id == random_best_cluster_id);
    result.random_same_type = (result.allgrid_best_type == result.random_best_type);
    result.random_score_diff = result.allgrid_best_score - random_best_score;

    return result;
}

// ============================================================
// 距離関数・クラスタリング (pool_clustering_experiment から移植)
// ============================================================

int ClusteringEffectivenessExperiment::getZone(const std::optional<dc::Transform>& stone) const {
    if (!stone) return -1;
    float x = stone->position.x;
    float y = stone->position.y;
    float dist_to_tee = std::sqrt(x * x + (y - kHouseCenterY) * (y - kHouseCenterY));
    if (dist_to_tee <= kHouseRadius) return 0;
    if (y < kHouseCenterY - kHouseRadius && y > kHouseCenterY - 3.0f * kHouseRadius) return 1;
    return 2;
}

float ClusteringEffectivenessExperiment::evaluateBoard(const dc::GameState& state) const {
    struct SI { float dist; int team; };
    std::vector<SI> in_house;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (!state.stones[t][i]) continue;
            float dx = state.stones[t][i]->position.x;
            float dy = state.stones[t][i]->position.y - kHouseCenterY;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= kHouseRadius + 0.145f) in_house.push_back({d, t});
        }
    }
    if (in_house.empty()) return 0.0f;
    std::sort(in_house.begin(), in_house.end(), [](auto& a, auto& b) { return a.dist < b.dist; });
    int scoring = in_house[0].team;
    int score = 0;
    for (auto& s : in_house) {
        if (s.team == scoring) score++;
        else break;
    }
    return scoring == 0 ? (float)score : -(float)score;
}

float ClusteringEffectivenessExperiment::distDelta(
    const dc::GameState& input, const dc::GameState& a, const dc::GameState& b) const
{
    constexpr float MOVE_THRESHOLD = 0.01f;
    constexpr float PENALTY_EXISTENCE = 30.0f;
    constexpr float PENALTY_ZONE = 12.0f;
    constexpr float NEW_STONE_WEIGHT = 4.0f;
    constexpr float MOVED_STONE_WEIGHT = 2.0f;
    constexpr float PENALTY_INTERACTION = 15.0f;
    constexpr float INTERACTION_THRESHOLD = 0.03f;
    constexpr float SCORE_WEIGHT = 20.0f;            // 8→20: スコア差をより重視
    constexpr float PROXIMITY_WEIGHT = 5.0f;

    float distance = 0.0f;
    float max_disp_a = 0.0f, max_disp_b = 0.0f;
    int new_team = -1, new_idx = -1;

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            bool in_inp = input.stones[t][i].has_value();
            bool in_a = a.stones[t][i].has_value();
            bool in_b = b.stones[t][i].has_value();
            if (in_inp) {
                if (in_a && in_b) {
                    float dxa = a.stones[t][i]->position.x - input.stones[t][i]->position.x;
                    float dya = a.stones[t][i]->position.y - input.stones[t][i]->position.y;
                    float dxb = b.stones[t][i]->position.x - input.stones[t][i]->position.x;
                    float dyb = b.stones[t][i]->position.y - input.stones[t][i]->position.y;
                    float ma = std::sqrt(dxa*dxa+dya*dya), mb = std::sqrt(dxb*dxb+dyb*dyb);
                    max_disp_a = std::max(max_disp_a, ma);
                    max_disp_b = std::max(max_disp_b, mb);
                    if (ma < MOVE_THRESHOLD && mb < MOVE_THRESHOLD) continue;
                    float ddx = dxa-dxb, ddy = dya-dyb;
                    distance += MOVED_STONE_WEIGHT * std::sqrt(ddx*ddx+ddy*ddy);
                    if (getZone(a.stones[t][i]) != getZone(b.stones[t][i])) distance += PENALTY_ZONE;
                } else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            } else {
                if (in_a && in_b) {
                    new_team = t; new_idx = i;
                    float dx = a.stones[t][i]->position.x - b.stones[t][i]->position.x;
                    float dy = a.stones[t][i]->position.y - b.stones[t][i]->position.y;
                    distance += NEW_STONE_WEIGHT * std::sqrt(dx*dx+dy*dy);
                    if (getZone(a.stones[t][i]) != getZone(b.stones[t][i])) distance += PENALTY_ZONE;
                } else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            }
        }
    }
    if ((max_disp_a > INTERACTION_THRESHOLD) != (max_disp_b > INTERACTION_THRESHOLD))
        distance += PENALTY_INTERACTION;

    if (new_team >= 0) {
        auto minProx = [&](const dc::GameState& s) -> float {
            float mn = 1e9f;
            float nx = s.stones[new_team][new_idx]->position.x;
            float ny = s.stones[new_team][new_idx]->position.y;
            for (int t=0;t<2;t++) for (int i=0;i<8;i++) {
                if (t==new_team && i==new_idx) continue;
                if (!s.stones[t][i]) continue;
                float dx=nx-s.stones[t][i]->position.x, dy=ny-s.stones[t][i]->position.y;
                mn = std::min(mn, std::sqrt(dx*dx+dy*dy));
            }
            return mn;
        };
        float pa = minProx(a), pb = minProx(b);
        if (pa < 100.0f || pb < 100.0f) distance += PROXIMITY_WEIGHT * std::abs(pa-pb);
    }
    distance += SCORE_WEIGHT * std::abs(evaluateBoard(a) - evaluateBoard(b));

    float ca=1e9f, cb=1e9f; int ta=-1, tb=-1;
    for (int t=0;t<2;t++) for (int i=0;i<8;i++) {
        if (a.stones[t][i]) { float d=std::sqrt(std::pow(a.stones[t][i]->position.x,2)+std::pow(a.stones[t][i]->position.y-kHouseCenterY,2)); if(d<ca){ca=d;ta=t;} }
        if (b.stones[t][i]) { float d=std::sqrt(std::pow(b.stones[t][i]->position.x,2)+std::pow(b.stones[t][i]->position.y-kHouseCenterY,2)); if(d<cb){cb=d;tb=t;} }
    }
    if (ta>=0 && tb>=0 && ta!=tb) distance += 25.0f;  // 10→25: No.1石チーム差を重視
    return distance;
}

std::vector<std::vector<float>> ClusteringEffectivenessExperiment::makeDistanceTableDelta(
    const dc::GameState& input_state, const std::vector<dc::GameState>& result_states)
{
    int n = static_cast<int>(result_states.size());
    std::vector<std::vector<float>> table(n, std::vector<float>(n, 0.0f));
    for (int i=0; i<n; i++) {
        for (int j=i+1; j<n; j++) {
            float d = distDelta(input_state, result_states[i], result_states[j]);
            table[i][j] = d; table[j][i] = d;
        }
        table[i][i] = -1.0f;
    }
    return table;
}

std::vector<std::vector<float>> ClusteringEffectivenessExperiment::makeDistanceTableSpatial(
    const std::vector<dc::GameState>& result_states)
{
    int n = static_cast<int>(result_states.size());
    std::vector<std::vector<float>> table(n, std::vector<float>(n, 0.0f));

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dist = 0.0f;
            // 全石の座標差のユークリッド距離
            for (int t = 0; t < 2; t++) {
                for (int s = 0; s < 8; s++) {
                    bool in_a = result_states[i].stones[t][s].has_value();
                    bool in_b = result_states[j].stones[t][s].has_value();
                    if (in_a && in_b) {
                        float dx = result_states[i].stones[t][s]->position.x
                                 - result_states[j].stones[t][s]->position.x;
                        float dy = result_states[i].stones[t][s]->position.y
                                 - result_states[j].stones[t][s]->position.y;
                        dist += dx * dx + dy * dy;
                    } else if (in_a != in_b) {
                        // 一方にだけ石がある → 大きなペナルティ
                        dist += 100.0f;
                    }
                }
            }
            dist = std::sqrt(dist);
            table[i][j] = dist;
            table[j][i] = dist;
        }
        table[i][i] = -1.0f;
    }
    return table;
}

std::vector<std::set<int>> ClusteringEffectivenessExperiment::runClustering(
    const std::vector<std::vector<float>>& dist_table, int n_desired)
{
    int n = static_cast<int>(dist_table.size());
    std::vector<std::set<int>> clusters(n);
    for (int i=0; i<n; i++) clusters[i].insert(i);
    while (static_cast<int>(clusters.size()) > n_desired) {
        float mn = 1e18f; int bi=-1, bj=-1;
        for (int i=0; i<(int)clusters.size(); i++)
            for (int j=i+1; j<(int)clusters.size(); j++) {
                float total=0; int cnt=0;
                for (int a:clusters[i]) for (int b:clusters[j]) { total+=dist_table[a][b]; cnt++; }
                float avg=total/cnt;
                if (avg<mn) { mn=avg; bi=i; bj=j; }
            }
        if (bi==-1) break;
        clusters[bi].insert(clusters[bj].begin(), clusters[bj].end());
        clusters.erase(clusters.begin()+bj);
    }
    return clusters;
}

std::vector<std::set<int>> ClusteringEffectivenessExperiment::runRandomClustering(
    int n_items, int n_desired_clusters, std::mt19937& rng)
{
    int k = std::min(n_desired_clusters, n_items);
    std::vector<std::set<int>> clusters(k);

    // 各アイテムをランダムなクラスタに割り当て
    // まず各クラスタに最低1つは割り当てる (空クラスタを防ぐ)
    std::vector<int> indices(n_items);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < k; ++i) {
        clusters[i].insert(indices[i]);
    }
    // 残りをランダムに割り当て
    std::uniform_int_distribution<int> dist(0, k - 1);
    for (int i = k; i < n_items; ++i) {
        clusters[dist(rng)].insert(indices[i]);
    }

    return clusters;
}

std::vector<int> ClusteringEffectivenessExperiment::calculateMedoids(
    const std::vector<std::vector<float>>& dist_table, const std::vector<std::set<int>>& clusters)
{
    std::vector<int> medoids;
    for (auto& cluster : clusters) {
        if (cluster.empty()) { medoids.push_back(-1); continue; }
        if (cluster.size() == 1) { medoids.push_back(*cluster.begin()); continue; }
        int best = -1; float best_sum = 1e18f;
        for (int c : cluster) {
            float sum = 0;
            for (int o : cluster) if (c != o) sum += dist_table[c][o];
            if (sum < best_sum) { best_sum = sum; best = c; }
        }
        medoids.push_back(best);
    }
    return medoids;
}

double ClusteringEffectivenessExperiment::calcSilhouetteScore(
    const std::vector<std::vector<float>>& dist_table, const std::vector<std::set<int>>& clusters)
{
    int n = static_cast<int>(dist_table.size());
    if (clusters.size() <= 1) return 0.0;

    // 各点のクラスタID
    std::vector<int> label(n, -1);
    for (int c = 0; c < (int)clusters.size(); c++)
        for (int i : clusters[c]) label[i] = c;

    double total = 0.0;
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (label[i] < 0) continue;
        // a(i): 同クラスタ内平均距離
        double a_sum = 0; int a_cnt = 0;
        for (int j : clusters[label[i]]) {
            if (i == j) continue;
            a_sum += dist_table[i][j]; a_cnt++;
        }
        double a_i = (a_cnt > 0) ? a_sum / a_cnt : 0;
        // b(i): 最近隣クラスタの平均距離
        double b_i = 1e18;
        for (int c = 0; c < (int)clusters.size(); c++) {
            if (c == label[i]) continue;
            double sum = 0; int cnt = 0;
            for (int j : clusters[c]) { sum += dist_table[i][j]; cnt++; }
            if (cnt > 0) b_i = std::min(b_i, sum / cnt);
        }
        double s_i = (std::max(a_i, b_i) > 0) ? (b_i - a_i) / std::max(a_i, b_i) : 0;
        total += s_i; count++;
    }
    return (count > 0) ? total / count : 0;
}

// ============================================================
// 結果出力
// ============================================================
void ClusteringEffectivenessExperiment::printSummary(
    const std::vector<TestCaseResult>& results, int retention_pct)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Retention: " << retention_pct << "%  |  B=" << rollout_count_
              << "  |  N=" << results.size() << " positions" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 全体集計
    int exact = 0, same_c = 0, same_t = 0;
    double total_sdiff = 0, total_ag_time = 0, total_cl_time = 0;
    double total_sil = 0;
    int rnd_exact = 0, rnd_same_c = 0, rnd_same_t = 0;
    double rnd_total_sdiff = 0, rnd_total_time = 0, rnd_total_sil = 0;
    int sp_exact = 0, sp_same_c = 0, sp_same_t = 0;
    double sp_total_sdiff = 0, sp_total_time = 0, sp_total_sil = 0;
    for (auto& r : results) {
        if (r.exact_match) exact++;
        if (r.same_cluster) same_c++;
        if (r.same_type) same_t++;
        total_sdiff += std::abs(r.score_diff);
        total_ag_time += r.allgrid_time_sec;
        total_cl_time += r.clustered_time_sec;
        total_sil += r.silhouette_score;
        // Spatial
        if (r.spatial_exact_match) sp_exact++;
        if (r.spatial_same_cluster) sp_same_c++;
        if (r.spatial_same_type) sp_same_t++;
        sp_total_sdiff += std::abs(r.spatial_score_diff);
        sp_total_time += r.spatial_time_sec;
        sp_total_sil += r.spatial_silhouette_score;
        // Random
        if (r.random_exact_match) rnd_exact++;
        if (r.random_same_cluster) rnd_same_c++;
        if (r.random_same_type) rnd_same_t++;
        rnd_total_sdiff += std::abs(r.random_score_diff);
        rnd_total_time += r.random_time_sec;
        rnd_total_sil += r.random_silhouette_score;
    }
    int N = static_cast<int>(results.size());

    std::cout << "  ---- Proposed Clustering ----" << std::endl;
    std::cout << "    Exact Match:   " << exact << "/" << N << " (" << 100.0*exact/N << "%)" << std::endl;
    std::cout << "    Same Cluster:  " << same_c << "/" << N << " (" << 100.0*same_c/N << "%)" << std::endl;
    std::cout << "    Same Type:     " << same_t << "/" << N << " (" << 100.0*same_t/N << "%)" << std::endl;
    std::cout << "    Avg |ScoreDiff|: " << total_sdiff/N << std::endl;
    std::cout << "    Avg Silhouette:  " << total_sil/N << std::endl;
    std::cout << "    Time:          " << total_cl_time << "s total" << std::endl;

    std::cout << "\n  ---- Spatial Clustering (Baseline) ----" << std::endl;
    std::cout << "    Exact Match:   " << sp_exact << "/" << N << " (" << 100.0*sp_exact/N << "%)" << std::endl;
    std::cout << "    Same Cluster:  " << sp_same_c << "/" << N << " (" << 100.0*sp_same_c/N << "%)" << std::endl;
    std::cout << "    Same Type:     " << sp_same_t << "/" << N << " (" << 100.0*sp_same_t/N << "%)" << std::endl;
    std::cout << "    Avg |ScoreDiff|: " << sp_total_sdiff/N << std::endl;
    std::cout << "    Avg Silhouette:  " << sp_total_sil/N << std::endl;
    std::cout << "    Time:          " << sp_total_time << "s total" << std::endl;

    std::cout << "\n  ---- Random Clustering (Baseline) ----" << std::endl;
    std::cout << "    Exact Match:   " << rnd_exact << "/" << N << " (" << 100.0*rnd_exact/N << "%)" << std::endl;
    std::cout << "    Same Cluster:  " << rnd_same_c << "/" << N << " (" << 100.0*rnd_same_c/N << "%)" << std::endl;
    std::cout << "    Same Type:     " << rnd_same_t << "/" << N << " (" << 100.0*rnd_same_t/N << "%)" << std::endl;
    std::cout << "    Avg |ScoreDiff|: " << rnd_total_sdiff/N << std::endl;
    std::cout << "    Avg Silhouette:  " << rnd_total_sil/N << std::endl;
    std::cout << "    Time:          " << rnd_total_time << "s total" << std::endl;

    std::cout << "\n  ---- AllGrid (Ground Truth) ----" << std::endl;
    std::cout << "    Time:          " << total_ag_time << "s total" << std::endl;
    std::cout << "    Speedup (Proposed): " << total_ag_time/std::max(total_cl_time, 0.001) << "x" << std::endl;

    // ショット番号ごとの集計
    std::cout << "\n  By Shot Number (Proposed / Random):" << std::endl;
    std::cout << "    Shot | N  | Exact%P | Exact%R | SameClus% | SameType%P | SameType%R | |SDiff|P  | |SDiff|R" << std::endl;
    std::cout << "    -----|----|---------|---------|-----------|-----------|-----------|-----------|---------" << std::endl;
    for (int s = 0; s < 16; s++) {
        int cnt=0, ex=0, sc=0, st=0; double sd=0, cands=0;
        int rex=0, rst=0; double rsd=0;
        for (auto& r : results) {
            if (r.shot_num != s) continue;
            cnt++;
            if(r.exact_match) ex++; if(r.same_cluster) sc++;
            if(r.same_type) st++; sd += std::abs(r.score_diff);
            cands += r.n_candidates;
            if(r.random_exact_match) rex++;
            if(r.random_same_type) rst++;
            rsd += std::abs(r.random_score_diff);
        }
        if (cnt == 0) continue;
        std::cout << "    " << std::setw(4) << s << " | "
                  << std::setw(2) << cnt << " | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << 100.0*ex/cnt << "% | "
                  << std::setw(6) << 100.0*rex/cnt << "% | "
                  << std::setw(8) << 100.0*sc/cnt << "% | "
                  << std::setw(9) << 100.0*st/cnt << "% | "
                  << std::setw(9) << 100.0*rst/cnt << "% | "
                  << std::setw(9) << std::setprecision(3) << sd/cnt << " | "
                  << std::setw(7) << rsd/cnt << std::endl;
    }

    // エンド番号ごとの集計
    std::cout << "\n  By End Number:" << std::endl;
    std::cout << "    End  | N  | Exact% | SameClus% | SameType%" << std::endl;
    std::cout << "    -----|----|---------|-----------|---------" << std::endl;
    for (int e = 0; e < 10; e++) {
        int cnt=0, ex=0, sc=0, st=0;
        for (auto& r : results) {
            if (r.end != e) continue;
            cnt++; if(r.exact_match) ex++; if(r.same_cluster) sc++; if(r.same_type) st++;
        }
        if (cnt == 0) continue;
        std::cout << "    " << std::setw(4) << e << " | "
                  << std::setw(2) << cnt << " | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << 100.0*ex/cnt << "% | "
                  << std::setw(8) << 100.0*sc/cnt << "% | "
                  << std::setw(8) << 100.0*st/cnt << "%" << std::endl;
    }
}

void ClusteringEffectivenessExperiment::exportCSV(
    const std::vector<TestCaseResult>& results, int retention_pct)
{
    std::string dir = "experiment_results";
    std::filesystem::create_directories(dir);

    // タイムスタンプ付きファイル名（上書き防止）
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[20];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);

    std::string path = dir + "/clustering_effectiveness_ret" + std::to_string(retention_pct)
                     + "_B" + std::to_string(rollout_count_)
                     + "_" + std::string(ts) + ".csv";

    std::ofstream ofs(path);
    ofs << "game_id,end,shot_num,n_candidates,n_clustered,"
        << "allgrid_best_label,allgrid_best_type,allgrid_best_score,allgrid_time,"
        << "clustered_best_label,clustered_best_type,clustered_best_score,clustered_time,"
        << "exact_match,same_cluster,same_type,score_diff,silhouette,"
        << "spatial_best_label,spatial_best_type,spatial_best_score,spatial_time,"
        << "spatial_exact_match,spatial_same_cluster,spatial_same_type,spatial_score_diff,spatial_silhouette,"
        << "random_best_label,random_best_type,random_best_score,random_time,"
        << "random_exact_match,random_same_cluster,random_same_type,random_score_diff,random_silhouette"
        << std::endl;

    for (auto& r : results) {
        ofs << r.game_id << "," << r.end << "," << r.shot_num << ","
            << r.n_candidates << "," << r.n_clustered << ","
            << "\"" << r.allgrid_best_label << "\"," << static_cast<int>(r.allgrid_best_type) << ","
            << r.allgrid_best_score << "," << r.allgrid_time_sec << ","
            << "\"" << r.clustered_best_label << "\"," << static_cast<int>(r.clustered_best_type) << ","
            << r.clustered_best_score << "," << r.clustered_time_sec << ","
            << r.exact_match << "," << r.same_cluster << "," << r.same_type << ","
            << r.score_diff << "," << r.silhouette_score << ","
            << "\"" << r.spatial_best_label << "\"," << static_cast<int>(r.spatial_best_type) << ","
            << r.spatial_best_score << "," << r.spatial_time_sec << ","
            << r.spatial_exact_match << "," << r.spatial_same_cluster << "," << r.spatial_same_type << ","
            << r.spatial_score_diff << "," << r.spatial_silhouette_score << ","
            << "\"" << r.random_best_label << "\"," << static_cast<int>(r.random_best_type) << ","
            << r.random_best_score << "," << r.random_time_sec << ","
            << r.random_exact_match << "," << r.random_same_cluster << "," << r.random_same_type << ","
            << r.random_score_diff << "," << r.random_silhouette_score
            << std::endl;
    }
    std::cout << "  Exported: " << path << std::endl;
}

void ClusteringEffectivenessExperiment::exportClusterDetailsCSV(int retention_pct) {
    std::string dir = "experiment_results";
    std::filesystem::create_directories(dir);

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[20];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);

    std::string path = dir + "/cluster_details_ret" + std::to_string(retention_pct)
                     + "_B" + std::to_string(rollout_count_)
                     + "_" + std::string(ts) + ".csv";

    std::ofstream ofs(path);
    ofs << "game_id,end,shot_num,n_candidates,method,cluster_id,cluster_size,medoid_idx,medoid_label"
        << std::endl;

    for (auto& ci : cluster_details_) {
        ofs << ci.game_id << "," << ci.end << "," << ci.shot_num << ","
            << ci.n_candidates << "," << ci.method << "," << ci.cluster_id << ","
            << ci.cluster_size << "," << ci.medoid_idx << ","
            << "\"" << ci.medoid_label << "\"" << std::endl;
    }
    std::cout << "  Exported cluster details: " << path << std::endl;
}

// ============================================================
// メイン実行
// ============================================================
void ClusteringEffectivenessExperiment::run() {
    std::cout << "=== Clustering Effectiveness Experiment ===" << std::endl;

    // gPolicy 読み込み
    if (!policy_->load("data/policy_param.dat")) {
        std::cerr << "Failed to load gPolicy. Aborting." << std::endl;
        return;
    }

    // テスト盤面生成
    auto records = generateTestPositions();

    // SimulatorWrapper (ロールアウト評価用)
    SimulatorWrapper sim(dc::Team::k0, game_setting_);

    // 各保持率で実験
    for (int ret : retention_rates_) {
        std::cout << "\n=== Running with retention=" << ret << "% ===" << std::endl;

        std::vector<TestCaseResult> results;
        auto exp_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < records.size(); ++i) {
            auto& rec = records[i];
            std::cout << "  [" << i+1 << "/" << records.size() << "] "
                      << "Game" << rec.game_id << " End" << rec.end
                      << " Shot" << rec.shot_num << " ... " << std::flush;

            auto result = evaluatePosition(rec, ret, sim);
            results.push_back(result);

            std::cout << "P:" << (result.exact_match ? "EXACT" : (result.same_cluster ? "SameClus" : "DIFF"))
                      << " R:" << (result.random_exact_match ? "EXACT" : "DIFF")
                      << " | AG=" << result.allgrid_best_label
                      << " CL=" << result.clustered_best_label
                      << " RD=" << result.random_best_label
                      << std::endl;
        }

        auto exp_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(exp_end - exp_start).count();

        printSummary(results, ret);
        exportCSV(results, ret);
        exportClusterDetailsCSV(ret);
        cluster_details_.clear();  // 次の保持率用にリセット

        std::cout << "\n  Total experiment time: " << total_time << "s" << std::endl;
    }

    std::cout << "\n=== Experiment Complete ===" << std::endl;
}
