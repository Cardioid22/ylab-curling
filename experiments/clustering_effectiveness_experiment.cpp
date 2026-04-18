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
// CSVからテスト盤面を読み込み (generate_test_positions のバッチ形式)
// ============================================================
std::vector<ClusteringEffectivenessExperiment::GameRecord>
ClusteringEffectivenessExperiment::loadTestPositionsFromCSV(
    const std::string& dir, int max_n)
{
    std::vector<GameRecord> records;

    std::cout << "\n=== Phase 1 (Load): Reading test positions from " << dir << " ===" << std::endl;

    if (!std::filesystem::exists(dir)) {
        std::cerr << "Error: directory does not exist: " << dir << std::endl;
        return records;
    }

    // ディレクトリ内のbatch_*.csvをソートして順に読み込み
    std::vector<std::filesystem::path> batch_files;
    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".csv" &&
            entry.path().filename().string().find("batch_") == 0) {
            batch_files.push_back(entry.path());
        }
    }
    std::sort(batch_files.begin(), batch_files.end());

    std::cout << "  Found " << batch_files.size() << " batch files" << std::endl;

    for (auto& bf : batch_files) {
        std::ifstream ifs(bf);
        if (!ifs) continue;

        std::string header;
        std::getline(ifs, header);  // ヘッダ行スキップ

        std::string line;
        while (std::getline(ifs, line)) {
            if (max_n > 0 && static_cast<int>(records.size()) >= max_n) break;

            // カラム分割
            std::vector<std::string> cols;
            std::stringstream ss(line);
            std::string col;
            while (std::getline(ss, col, ',')) cols.push_back(col);

            if (cols.size() < 4 + 16 * 3) continue;  // 最小カラム数

            GameRecord rec;
            rec.game_id = std::stoi(cols[0]);
            rec.end = std::stoi(cols[1]);
            rec.shot_num = std::stoi(cols[2]);
            int team_int = std::stoi(cols[3]);
            rec.current_team = (team_int == 0) ? dc::Team::k0 : dc::Team::k1;

            // GameState を復元
            dc::GameState state(game_setting_);
            state.end = static_cast<std::uint8_t>(rec.end);
            state.shot = static_cast<std::uint8_t>(rec.shot_num);

            // 全石をクリア
            for (int t = 0; t < 2; ++t)
                for (int s = 0; s < 8; ++s)
                    state.stones[t][s].reset();

            // 石情報を読み込み (4列目以降: team × 8石 × (inplay, x, y))
            int col_idx = 4;
            for (int t = 0; t < 2; ++t) {
                for (int s = 0; s < 8; ++s) {
                    int inplay = std::stoi(cols[col_idx++]);
                    float x = std::stof(cols[col_idx++]);
                    float y = std::stof(cols[col_idx++]);
                    if (inplay == 1) {
                        state.stones[t][s].emplace(dc::Vector2(x, y), 0.0f);
                    }
                }
            }

            rec.state = state;
            records.push_back(rec);
        }

        if (max_n > 0 && static_cast<int>(records.size()) >= max_n) break;
    }

    std::cout << "  Loaded: " << records.size() << " test positions" << std::endl;
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

    // B回ロールアウトし、最大値を採用 (AllGridが最大値で候補を選ぶのと同じポリシー)
    // 「この候補の上振れポテンシャル」で評価する
    double max_score = -1e9;
    for (int r = 0; r < rollout_count_; ++r) {
        double s = sim.run_policy_rollout(
            state, candidate.shot, *policy_, *shot_gen_, 1);
        if (s > max_score) max_score = s;
    }
    return max_score;
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

    // --- 予算計算 ---
    // AllGrid: 全N候補 × B_ag回 = 総予算
    // Proposed/Simple/Random: K候補 × B_sub回 = 同じ総予算
    int B_allgrid = rollout_count_;
    int n_clusters = std::max(1, n * retention_pct / 100);
    int B_sub = std::max(1, (n * B_allgrid) / n_clusters);  // 予算公平: B_sub = N/K * B_ag

    // --- AllGrid: 全候補をロールアウト評価 ---
    auto t0 = std::chrono::high_resolution_clock::now();

    int saved_rollout = rollout_count_;
    rollout_count_ = B_allgrid;
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

    // n_clusters は上で計算済み
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

    // --- Clustered: メドイドのみ評価 (B_sub回、予算公平) ---
    rollout_count_ = B_sub;
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

    // --- Simple Clustering (ベースライン): 速度ベクトルのグリッド分割 ---
    // 「打つ瞬間の情報で似ているものを削る」ポリシー
    auto t_si0 = std::chrono::high_resolution_clock::now();

    auto simple_selected = selectByVelocityGrid(candidates, n_clusters);

    // Simple用のクラスタ構成を記録
    for (int c = 0; c < static_cast<int>(simple_selected.size()); ++c) {
        ClusterInfo ci;
        ci.game_id = record.game_id;
        ci.end = record.end;
        ci.shot_num = record.shot_num;
        ci.n_candidates = n;
        ci.method = "simple";
        ci.cluster_id = c;
        ci.cluster_size = 1;  // グリッド分割なので各セル1手
        ci.medoid_idx = simple_selected[c];
        ci.medoid_label = candidates[simple_selected[c]].label;
        cluster_details_.push_back(ci);
    }

    // Simple も B_sub 回（予算公平）
    // ただし simple_selected の数は n_clusters と異なる場合がある
    int B_simple = std::max(1, (n * B_allgrid) / std::max(1, (int)simple_selected.size()));
    rollout_count_ = B_simple;
    double simple_best_score = -1e9;
    int simple_best_idx = -1;

    for (int idx : simple_selected) {
        double score = evaluateCandidate(record.state, candidates[idx], sim);
        if (score > simple_best_score) {
            simple_best_score = score;
            simple_best_idx = idx;
        }
    }

    auto t_si1 = std::chrono::high_resolution_clock::now();
    result.simple_time_sec = std::chrono::duration<double>(t_si1 - t_si0).count();
    result.simple_best_idx = simple_best_idx;
    result.simple_best_label = (simple_best_idx >= 0) ?
        candidates[simple_best_idx].label : "N/A";
    result.simple_best_type = (simple_best_idx >= 0) ?
        candidates[simple_best_idx].type : ShotType::PASS;
    result.simple_best_score = simple_best_score;

    result.simple_exact_match = (allgrid_best == simple_best_idx);
    // Simple のクラスタ一致: AllGrid最良手がSimple選出手に含まれているか
    result.simple_same_cluster = false;
    for (int idx : simple_selected) {
        if (idx == allgrid_best) { result.simple_same_cluster = true; break; }
    }
    result.simple_same_type = (result.allgrid_best_type == result.simple_best_type);
    result.simple_score_diff = result.allgrid_best_score - simple_best_score;

    // --- Random (ベースライン): ランダムにK手選ぶ ---
    auto t4 = std::chrono::high_resolution_clock::now();

    std::mt19937 rng(42 + record.game_id * 10000 + record.end * 100 + record.shot_num);
    auto random_selected = selectRandom(n, n_clusters, rng);

    // same_cluster指標用にランダムクラスタも割り当て
    auto random_clusters = assignRandomClusters(n, n_clusters, rng);

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
        ci.medoid_idx = -1;
        ci.medoid_label = "N/A";
        cluster_details_.push_back(ci);
    }

    // Random も B_sub 回（予算公平）
    int B_random = std::max(1, (n * B_allgrid) / std::max(1, (int)random_selected.size()));
    rollout_count_ = B_random;
    double random_best_score = -1e9;
    int random_best_medoid_idx = -1;

    for (int idx : random_selected) {
        double score = evaluateCandidate(record.state, candidates[idx], sim);
        if (score > random_best_score) {
            random_best_score = score;
            random_best_medoid_idx = idx;
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

    rollout_count_ = saved_rollout;  // 元に戻す
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

// ランダムにK手を選択
std::vector<int> ClusteringEffectivenessExperiment::selectRandom(
    int n_items, int k, std::mt19937& rng)
{
    k = std::min(k, n_items);
    std::vector<int> indices(n_items);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(k);
    return indices;
}

// same_cluster指標用のランダムクラスタ割り当て
std::vector<std::set<int>> ClusteringEffectivenessExperiment::assignRandomClusters(
    int n_items, int n_clusters, std::mt19937& rng)
{
    int k = std::min(n_clusters, n_items);
    std::vector<std::set<int>> clusters(k);
    std::vector<int> indices(n_items);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < k; ++i) clusters[i].insert(indices[i]);
    std::uniform_int_distribution<int> dist(0, k - 1);
    for (int i = k; i < n_items; ++i) clusters[dist(rng)].insert(indices[i]);
    return clusters;
}

// Simple: 速度ベクトル(vx, vy, rot)のグリッド分割で代表手を選出
// 「打つ瞬間の情報で似ているものを削る」ポリシー
// Simple: 速度ベクトル(vx, vy, rot)の固定グリッド分割で代表手を選出
// カーリングの物理に基づいた固定境界:
//   vy: ガード / ドロー / ヒット弱 / ヒット強
//   vx: 左 / 中央 / 右
//   rot: CW / CCW
// 合計: 4 × 3 × 2 = 24 セル（非空セルのみ代表を選出）
std::vector<int> ClusteringEffectivenessExperiment::selectByVelocityGrid(
    const std::vector<CandidateShot>& candidates, int k)
{
    int n = static_cast<int>(candidates.size());
    if (n <= k) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    // 固定グリッド境界（カーリングの物理に基づく）
    // vy: 投擲方向の速度（大きいほど強い）
    constexpr float vy_bounds[] = { 2.0f, 2.5f, 3.2f };  // 4区間
    constexpr int n_vy = 4;  // [0,2.0), [2.0,2.5), [2.5,3.2), [3.2,∞)

    // vx: 横方向の速度
    constexpr float vx_bounds[] = { -0.05f, 0.05f };  // 3区間
    constexpr int n_vx = 3;  // (-∞,-0.05), [-0.05,0.05], (0.05,∞)

    // rot: 2区間 (CW=1, CCW=0)
    constexpr int n_rot = 2;

    auto getVyBin = [](float vy) -> int {
        if (vy < 2.0f) return 0;       // ガード・弱ドロー
        if (vy < 2.5f) return 1;       // ドロー
        if (vy < 3.2f) return 2;       // ヒット弱〜中
        return 3;                       // ヒット強
    };

    auto getVxBin = [](float vx) -> int {
        if (vx < -0.05f) return 0;     // 左
        if (vx <= 0.05f) return 1;     // 中央
        return 2;                       // 右
    };

    // 各候補をグリッドセルに割り当て
    std::map<int, std::vector<int>> grid_cells;
    for (int i = 0; i < n; ++i) {
        int rot_bin = candidates[i].shot.rot;  // 0 or 1
        int vx_bin = getVxBin(candidates[i].shot.vx);
        int vy_bin = getVyBin(candidates[i].shot.vy);
        int cell = rot_bin * (n_vx * n_vy) + vy_bin * n_vx + vx_bin;
        grid_cells[cell].push_back(i);
    }

    // 各非空セルから代表を1つ選出（セル内の中央に最も近い候補）
    std::vector<int> selected;
    for (auto& [cell, members] : grid_cells) {
        if (members.empty()) continue;
        float avg_vx = 0, avg_vy = 0;
        for (int idx : members) {
            avg_vx += candidates[idx].shot.vx;
            avg_vy += candidates[idx].shot.vy;
        }
        avg_vx /= members.size();
        avg_vy /= members.size();
        int best = members[0];
        float best_dist = 1e9f;
        for (int idx : members) {
            float dx = candidates[idx].shot.vx - avg_vx;
            float dy = candidates[idx].shot.vy - avg_vy;
            float d = dx * dx + dy * dy;
            if (d < best_dist) { best_dist = d; best = idx; }
        }
        selected.push_back(best);
    }

    return selected;
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
    int si_exact = 0, si_same_c = 0, si_same_t = 0;
    double si_total_sdiff = 0, si_total_time = 0;
    for (auto& r : results) {
        if (r.exact_match) exact++;
        if (r.same_cluster) same_c++;
        if (r.same_type) same_t++;
        total_sdiff += std::abs(r.score_diff);
        total_ag_time += r.allgrid_time_sec;
        total_cl_time += r.clustered_time_sec;
        total_sil += r.silhouette_score;
        // Spatial
        if (r.simple_exact_match) si_exact++;
        if (r.simple_same_cluster) si_same_c++;
        if (r.simple_same_type) si_same_t++;
        si_total_sdiff += std::abs(r.simple_score_diff);
        si_total_time += r.simple_time_sec;
        
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

    std::cout << "\n  ---- Simple Clustering (Baseline) ----" << std::endl;
    std::cout << "    Exact Match:   " << si_exact << "/" << N << " (" << 100.0*si_exact/N << "%)" << std::endl;
    std::cout << "    Same Cluster:  " << si_same_c << "/" << N << " (" << 100.0*si_same_c/N << "%)" << std::endl;
    std::cout << "    Same Type:     " << si_same_t << "/" << N << " (" << 100.0*si_same_t/N << "%)" << std::endl;
    std::cout << "    Avg |ScoreDiff|: " << si_total_sdiff/N << std::endl;
    std::cout << "    Time:          " << si_total_time << "s total" << std::endl;

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
        << "simple_best_label,simple_best_type,simple_best_score,simple_time,"
        << "simple_exact_match,simple_same_cluster,simple_same_type,simple_score_diff,"
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
            << "\"" << r.simple_best_label << "\"," << static_cast<int>(r.simple_best_type) << ","
            << r.simple_best_score << "," << r.simple_time_sec << ","
            << r.simple_exact_match << "," << r.simple_same_cluster << "," << r.simple_same_type << ","
            << r.simple_score_diff << ","
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

    // テスト盤面生成 or CSV読み込み
    std::vector<GameRecord> records;
    if (!load_positions_dir_.empty()) {
        records = loadTestPositionsFromCSV(load_positions_dir_, max_positions_);
    } else {
        records = generateTestPositions();
        if (max_positions_ > 0 && static_cast<int>(records.size()) > max_positions_) {
            records.resize(max_positions_);
        }
    }

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
                      << " S:" << (result.simple_exact_match ? "EXACT" : "DIFF")
                      << " R:" << (result.random_exact_match ? "EXACT" : "DIFF")
                      << " | AG=" << result.allgrid_best_label
                      << "(" << std::fixed << std::setprecision(2) << result.allgrid_best_score << ")"
                      << " CL=" << result.clustered_best_label
                      << "(" << result.clustered_best_score << ")"
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
