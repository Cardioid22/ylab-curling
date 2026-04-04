#include "depth1_mcts_experiment.h"
#include "pool_experiment.h"
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

namespace dc = digitalcurling3;

Depth1MctsExperiment::Depth1MctsExperiment(dc::GameSetting const& game_setting)
    : game_setting_(game_setting)
{
}

// ========== 盤面評価 ==========

float Depth1MctsExperiment::evaluateEndScore(const dc::GameState& state, dc::Team my_team) const {
    // カーリングの得点ルール: ハウス内でティーに最も近い石のチームが得点
    // 連続値で返す（歩に近い設計）
    struct StoneInfo { float dist; int team; };
    std::vector<StoneInfo> in_house;

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (!state.stones[t][i]) continue;
            float dx = state.stones[t][i]->position.x - kHouseCenterX;
            float dy = state.stones[t][i]->position.y - kHouseCenterY;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= kHouseRadius + 0.145f) {
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

    int my_team_int = static_cast<int>(my_team);
    return scoring_team == my_team_int ? static_cast<float>(score) : -static_cast<float>(score);
}

// ========== ロールアウト ==========

double Depth1MctsExperiment::rollout(
    SimulatorWrapper& sim,
    const dc::GameState& state,
    const ShotInfo& shot,
    int remaining_shots
) {
    // 1. 候補手を適用
    dc::GameState sim_state = sim.run_single_simulation(state, shot);
    int shots_played = 1;

    // 2. 残りショットをShotGeneratorポリシーで消化
    //    各手番でShotGeneratorで候補を生成し、ランダムに1つ選ぶ
    std::random_device rd;
    std::mt19937 gen(rd());

    while (shots_played < remaining_shots && !sim_state.IsGameOver()) {
        // 現在の手番のチーム
        dc::Team current_team = (sim_state.shot % 2 == 0) ? dc::Team::k0 : dc::Team::k1;

        // ShotGeneratorで盤面に応じた候補手を生成（シミュレーションなし）
        auto candidates = rollout_generator_->generateCandidates(sim_state, current_team, rollout_grid_);

        ShotInfo chosen_shot;
        if (candidates.empty()) {
            // 候補がない場合はグリッドからランダム
            std::uniform_int_distribution<int> grid_dist(0, static_cast<int>(sim.initialShotData.size()) - 1);
            chosen_shot = sim.initialShotData[grid_dist(gen)];
        } else {
            std::uniform_int_distribution<int> cand_dist(0, static_cast<int>(candidates.size()) - 1);
            chosen_shot = candidates[cand_dist(gen)].shot;
        }

        sim_state = sim.run_single_simulation(sim_state, chosen_shot);
        shots_played++;
    }

    // 3. エンドスコアで評価（連続値）
    return static_cast<double>(evaluateEndScore(sim_state, dc::Team::k0));
}

// ========== UCB1 ==========

double Depth1MctsExperiment::ucb1Score(const Arm& arm, int total_visits, double c) const {
    if (arm.visits == 0) return 1e9;  // 未訪問は最優先
    double exploit = arm.mean();
    double explore = std::sqrt(std::log(static_cast<double>(total_visits)) / arm.visits);
    return exploit + c * explore;
}

// ========== フラットMC実行 ==========

int Depth1MctsExperiment::runFlatMC(
    std::vector<Arm>& arms,
    SimulatorWrapper& sim,
    const dc::GameState& state,
    int budget,
    int remaining_shots
) {
    int n_arms = static_cast<int>(arms.size());
    if (n_arms == 0) return -1;

    int total_visits = 0;

    for (int iter = 0; iter < budget; iter++) {
        // UCB1で腕を選択
        int best_arm = 0;
        double best_ucb = -1e9;
        for (int a = 0; a < n_arms; a++) {
            double ucb = ucb1Score(arms[a], total_visits);
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_arm = a;
            }
        }

        // ロールアウト実行
        double reward = rollout(sim, state, arms[best_arm].shot, remaining_shots);
        arms[best_arm].total_reward += reward;
        arms[best_arm].total_reward_sq += reward * reward;
        arms[best_arm].visits++;
        total_visits++;
    }

    // 最多訪問数の腕ではなく、平均報酬最大の腕を選択（歩に近い設計）
    int best = 0;
    double best_mean = -1e9;
    for (int a = 0; a < n_arms; a++) {
        if (arms[a].visits > 0 && arms[a].mean() > best_mean) {
            best_mean = arms[a].mean();
            best = a;
        }
    }
    return best;
}

// ========== 距離関数・クラスタリング（pool_clustering_experimentから移植）==========

int Depth1MctsExperiment::getZone(const std::optional<dc::Transform>& stone) const {
    if (!stone) return -1;
    float x = stone->position.x;
    float y = stone->position.y;
    float dist_to_tee = std::sqrt(x * x + (y - kHouseCenterY) * (y - kHouseCenterY));
    if (dist_to_tee <= kHouseRadius) return 0;
    if (y < kHouseCenterY - kHouseRadius && y > kHouseCenterY - 3.0f * kHouseRadius) return 1;
    return 2;
}

float Depth1MctsExperiment::evaluateBoard(const dc::GameState& state) const {
    struct StoneInfo { float dist; int team; };
    std::vector<StoneInfo> in_house;
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
    int scoring_team = in_house[0].team;
    int score = 0;
    for (auto& s : in_house) { if (s.team == scoring_team) score++; else break; }
    return scoring_team == 0 ? static_cast<float>(score) : -static_cast<float>(score);
}

float Depth1MctsExperiment::distDelta(
    const dc::GameState& input, const dc::GameState& a, const dc::GameState& b
) const {
    constexpr float MOVE_THRESHOLD = 0.01f;
    constexpr float PENALTY_EXISTENCE = 30.0f;
    constexpr float PENALTY_ZONE = 12.0f;
    constexpr float NEW_STONE_WEIGHT = 4.0f;
    constexpr float MOVED_STONE_WEIGHT = 2.0f;
    constexpr float PENALTY_INTERACTION = 15.0f;
    constexpr float INTERACTION_THRESHOLD = 0.03f;
    constexpr float SCORE_WEIGHT = 8.0f;
    constexpr float PROXIMITY_WEIGHT = 5.0f;

    float distance = 0.0f;
    float max_displacement_a = 0.0f, max_displacement_b = 0.0f;
    int new_stone_team = -1, new_stone_idx = -1;

    for (int team = 0; team < 2; team++) {
        for (int idx = 0; idx < 8; idx++) {
            bool in_input = input.stones[team][idx].has_value();
            bool in_a = a.stones[team][idx].has_value();
            bool in_b = b.stones[team][idx].has_value();

            if (in_input) {
                if (in_a && in_b) {
                    float dx_a = a.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dy_a = a.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float dx_b = b.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dy_b = b.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float move_a = std::sqrt(dx_a*dx_a + dy_a*dy_a);
                    float move_b = std::sqrt(dx_b*dx_b + dy_b*dy_b);
                    max_displacement_a = std::max(max_displacement_a, move_a);
                    max_displacement_b = std::max(max_displacement_b, move_b);
                    if (move_a < MOVE_THRESHOLD && move_b < MOVE_THRESHOLD) continue;
                    float ddx = dx_a - dx_b, ddy = dy_a - dy_b;
                    distance += MOVED_STONE_WEIGHT * std::sqrt(ddx*ddx + ddy*ddy);
                    if (getZone(a.stones[team][idx]) != getZone(b.stones[team][idx])) distance += PENALTY_ZONE;
                } else if (in_a != in_b) { distance += PENALTY_EXISTENCE; }
            } else {
                if (in_a && in_b) {
                    new_stone_team = team; new_stone_idx = idx;
                    float dx = a.stones[team][idx]->position.x - b.stones[team][idx]->position.x;
                    float dy = a.stones[team][idx]->position.y - b.stones[team][idx]->position.y;
                    distance += NEW_STONE_WEIGHT * std::sqrt(dx*dx + dy*dy);
                    if (getZone(a.stones[team][idx]) != getZone(b.stones[team][idx])) distance += PENALTY_ZONE;
                } else if (in_a != in_b) { distance += PENALTY_EXISTENCE; }
            }
        }
    }

    bool interacted_a = max_displacement_a > INTERACTION_THRESHOLD;
    bool interacted_b = max_displacement_b > INTERACTION_THRESHOLD;
    if (interacted_a != interacted_b) distance += PENALTY_INTERACTION;

    if (new_stone_team >= 0) {
        auto computeMinProx = [&](const dc::GameState& st) -> float {
            float min_d = std::numeric_limits<float>::max();
            float nx = st.stones[new_stone_team][new_stone_idx]->position.x;
            float ny = st.stones[new_stone_team][new_stone_idx]->position.y;
            for (int t = 0; t < 2; t++) for (int i = 0; i < 8; i++) {
                if (t == new_stone_team && i == new_stone_idx) continue;
                if (!st.stones[t][i]) continue;
                float dx = nx - st.stones[t][i]->position.x;
                float dy = ny - st.stones[t][i]->position.y;
                min_d = std::min(min_d, std::sqrt(dx*dx + dy*dy));
            }
            return min_d;
        };
        float pa = computeMinProx(a), pb = computeMinProx(b);
        if (pa < 100.0f || pb < 100.0f) distance += PROXIMITY_WEIGHT * std::abs(pa - pb);
    }

    distance += SCORE_WEIGHT * std::abs(evaluateBoard(a) - evaluateBoard(b));

    float closest_a = 1e9f, closest_b = 1e9f;
    int team_a = -1, team_b = -1;
    for (int t = 0; t < 2; t++) for (int i = 0; i < 8; i++) {
        if (a.stones[t][i]) { float d = std::sqrt(std::pow(a.stones[t][i]->position.x,2)+std::pow(a.stones[t][i]->position.y-kHouseCenterY,2)); if (d < closest_a) { closest_a = d; team_a = t; } }
        if (b.stones[t][i]) { float d = std::sqrt(std::pow(b.stones[t][i]->position.x,2)+std::pow(b.stones[t][i]->position.y-kHouseCenterY,2)); if (d < closest_b) { closest_b = d; team_b = t; } }
    }
    if (team_a >= 0 && team_b >= 0 && team_a != team_b) distance += 10.0f;
    return distance;
}

std::vector<std::vector<float>> Depth1MctsExperiment::makeDistanceTableDelta(
    const dc::GameState& input_state, const std::vector<dc::GameState>& result_states
) {
    int n = static_cast<int>(result_states.size());
    std::vector<std::vector<float>> table(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) { for (int j = i+1; j < n; j++) { float d = distDelta(input_state, result_states[i], result_states[j]); table[i][j] = d; table[j][i] = d; } table[i][i] = -1.0f; }
    return table;
}

std::vector<std::set<int>> Depth1MctsExperiment::runClustering(
    const std::vector<std::vector<float>>& dist_table, int n_desired_clusters
) {
    int n = static_cast<int>(dist_table.size());
    std::vector<std::set<int>> clusters(n);
    for (int i = 0; i < n; i++) clusters[i].insert(i);
    while (static_cast<int>(clusters.size()) > n_desired_clusters) {
        float min_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;
        for (int i = 0; i < static_cast<int>(clusters.size()); i++) {
            for (int j = i+1; j < static_cast<int>(clusters.size()); j++) {
                float total = 0.0f; int count = 0;
                for (int a : clusters[i]) for (int b : clusters[j]) { total += dist_table[a][b]; count++; }
                float avg = total / count;
                if (avg < min_dist) { min_dist = avg; best_i = i; best_j = j; }
            }
        }
        if (best_i == -1) break;
        clusters[best_i].insert(clusters[best_j].begin(), clusters[best_j].end());
        clusters.erase(clusters.begin() + best_j);
    }
    return clusters;
}

std::vector<int> Depth1MctsExperiment::calculateMedoids(
    const std::vector<std::vector<float>>& dist_table, const std::vector<std::set<int>>& clusters
) {
    std::vector<int> medoids;
    for (auto& cluster : clusters) {
        if (cluster.empty()) { medoids.push_back(-1); continue; }
        if (cluster.size() == 1) { medoids.push_back(*cluster.begin()); continue; }
        float min_total = std::numeric_limits<float>::max(); int best = -1;
        for (int c : cluster) { float total = 0; for (int o : cluster) if (c != o) total += dist_table[c][o]; if (total < min_total) { min_total = total; best = c; } }
        medoids.push_back(best);
    }
    return medoids;
}

// ========== テスト盤面生成 ==========

std::vector<dc::GameState> Depth1MctsExperiment::createTestStates() {
    std::vector<dc::GameState> states;
    test_state_names_.clear();

    auto calcShot = [](const dc::GameState& s) -> int {
        int total = 0;
        for (int t = 0; t < 2; t++) for (int i = 0; i < 8; i++) if (s.stones[t][i].has_value()) total++;
        return (total % 2 == 0) ? total : total + 1;
    };

    auto pseudoRand = [](int seed) -> float {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return (seed % 2001 - 1000) / 1000.0f;
    };

    // 石がある3盤面のみ
    // 1. 自分1+相手2
    { dc::GameState s(game_setting_); s.stones[0][0].emplace(dc::Transform(dc::Vector2(-0.5f, kHouseCenterY+0.3f), 0.f)); s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY-0.2f), 0.f)); s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY+0.8f), 0.f)); s.shot = calcShot(s); states.push_back(s); test_state_names_.push_back("opp2_my1"); }

    // 2. 3v3密集
    { dc::GameState s(game_setting_); s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY), 0.f)); s.stones[0][1].emplace(dc::Transform(dc::Vector2(-1.0f, kHouseCenterY-0.5f), 0.f)); s.stones[0][2].emplace(dc::Transform(dc::Vector2(0.5f, kHouseCenterY+1.0f), 0.f)); s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY+0.1f), 0.f)); s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.3f, kHouseCenterY+1.5f), 0.f)); s.stones[1][2].emplace(dc::Transform(dc::Vector2(0.8f, kHouseCenterY-0.8f), 0.f)); s.shot = calcShot(s); states.push_back(s); test_state_names_.push_back("crowded_3v3"); }

    // 3. ボタン争い
    { dc::GameState s(game_setting_); s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.15f, kHouseCenterY+0.1f), 0.f)); s.stones[1][0].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY-0.05f), 0.f)); s.shot = calcShot(s); states.push_back(s); test_state_names_.push_back("button_fight"); }

    // プログラム生成は省略（時間短縮のため）
    for (int i = 0; i < 0; i++) {
        dc::GameState s(game_setting_);
        int n_my = 1 + (i % 3);
        int n_opp = 1 + ((i+1) % 3);
        for (int j = 0; j < n_my && j < 4; j++) {
            s.stones[0][j].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i*17+j*3+200) * 1.2f,
                kHouseCenterY + pseudoRand(i*17+j*3+201) * 1.5f), 0.f));
        }
        for (int j = 0; j < n_opp && j < 4; j++) {
            s.stones[1][j].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i*17+j*3+210) * 1.2f,
                kHouseCenterY + pseudoRand(i*17+j*3+211) * 1.5f), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("gen" + std::to_string(i));
    }

    return states;
}

// ========== メイン実行 ==========

void Depth1MctsExperiment::run() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  Depth-1 Flat MC Experiment (Ayumu-style)" << std::endl;
    std::cout << "  UCB1 adaptive sampling, continuous evaluation" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto test_states = createTestStates();
    std::cout << "Test states: " << test_states.size() << std::endl;

    std::string output_dir = "experiments/depth1_mcts_results";
    std::filesystem::create_directories(output_dir);

    ShotGenerator generator(game_setting_);
    auto grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    // ロールアウト用ShotGenerator + グリッド
    rollout_generator_ = std::make_unique<ShotGenerator>(game_setting_);
    rollout_grid_ = grid;

    // SimulatorWrapper（ロールアウト用）
    SimulatorWrapper sim(dc::Team::k0, game_setting_);
    // グリッドショットをinitialShotDataに設定（フォールバック用）
    for (auto& pos : grid) {
        ShotInfo shot = sim.FindShot(pos);
        sim.initialShotData.push_back(shot);
    }

    auto classifyType = [](ShotType t) -> std::string {
        switch (t) {
            case ShotType::DRAW: return "Draw";
            case ShotType::HIT: return "Hit";
            case ShotType::FREEZE: return "Freeze";
            case ShotType::PREGUARD: case ShotType::POSTGUARD: return "Guard";
            case ShotType::PASS: return "Pass";
            default: return "Other";
        }
    };

    // 実験パラメータ
    std::vector<int> budgets = {500, 1000};
    std::vector<float> retention_ratios = {0.2f, 0.3f, 0.5f};

    struct ResultRow {
        std::string state;
        int n_candidates;
        int budget;
        std::string method;  // "FullPool" or "Clustered_XX%"
        int n_arms;
        std::string best_label;
        std::string best_type;
        double best_mean_reward;
        double best_variance;
        int best_visits;
        double elapsed_ms;
    };
    std::vector<ResultRow> results;

    for (size_t s = 0; s < test_states.size(); s++) {
        auto& state = test_states[s];
        dc::Team my_team = dc::Team::k0;

        auto pool = generator.generatePool(state, my_team, grid);
        int n = static_cast<int>(pool.candidates.size());
        if (n <= 2) continue;

        int remaining_shots = 16 - state.shot;

        // 距離テーブル + クラスタリング（全保持率で共有）
        auto dist_delta = makeDistanceTableDelta(state, pool.result_states);

        std::cout << "\n[" << (s+1) << "/" << test_states.size() << "] "
                  << test_state_names_[s] << " (N=" << n
                  << ", remaining=" << remaining_shots << ")" << std::endl;

        for (int budget : budgets) {
            // === FullPool: 全候補にロールアウト ===
            {
                std::vector<Arm> arms(n);
                for (int i = 0; i < n; i++) {
                    arms[i].candidate_idx = i;
                    arms[i].label = pool.candidates[i].label;
                    arms[i].type = classifyType(pool.candidates[i].type);
                    arms[i].shot = pool.candidates[i].shot;
                }

                auto t0 = std::chrono::high_resolution_clock::now();
                int best = runFlatMC(arms, sim, state, budget, remaining_shots);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                results.push_back({
                    test_state_names_[s], n, budget, "FullPool", n,
                    arms[best].label, arms[best].type,
                    arms[best].mean(), arms[best].variance(), arms[best].visits, ms
                });

                std::cout << "  B=" << budget << " Full(" << n << "arms): "
                          << arms[best].label << " mean=" << std::fixed << std::setprecision(2)
                          << arms[best].mean() << " visits=" << arms[best].visits
                          << " [" << std::setprecision(0) << ms << "ms]" << std::endl;
            }

            // === Clustered: クラスタリングで絞った候補にロールアウト ===
            for (float ratio : retention_ratios) {
                int k = std::max(2, static_cast<int>(std::round(n * ratio)));
                if (k >= n) continue;

                auto clusters = runClustering(dist_delta, k);
                auto medoids = calculateMedoids(dist_delta, clusters);

                std::vector<Arm> arms;
                for (int m : medoids) {
                    if (m < 0) continue;
                    Arm arm;
                    arm.candidate_idx = m;
                    arm.label = pool.candidates[m].label;
                    arm.type = classifyType(pool.candidates[m].type);
                    arm.shot = pool.candidates[m].shot;
                    arms.push_back(arm);
                }

                auto t0 = std::chrono::high_resolution_clock::now();
                int best = runFlatMC(arms, sim, state, budget, remaining_shots);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                int ratio_pct = static_cast<int>(std::round(ratio * 100));
                std::string method = "Clustered_" + std::to_string(ratio_pct) + "%";

                results.push_back({
                    test_state_names_[s], n, budget, method,
                    static_cast<int>(arms.size()),
                    arms[best].label, arms[best].type,
                    arms[best].mean(), arms[best].variance(), arms[best].visits, ms
                });

                std::cout << "  B=" << budget << " C" << ratio_pct << "%("
                          << arms.size() << "arms): "
                          << arms[best].label << " mean=" << std::fixed << std::setprecision(2)
                          << arms[best].mean() << " visits=" << arms[best].visits
                          << " [" << std::setprecision(0) << ms << "ms]" << std::endl;
            }
        }
    }

    // ========== サマリー ==========
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Summary: Flat MC Best Shot by Method & Budget" << std::endl;
    std::cout << "================================================================" << std::endl;

    // method × budget 別の平均報酬
    struct AggKey {
        std::string method;
        int budget;
        bool operator<(const AggKey& o) const {
            if (budget != o.budget) return budget < o.budget;
            return method < o.method;
        }
    };
    std::map<AggKey, std::vector<double>> agg_rewards;
    std::map<AggKey, std::vector<double>> agg_times;
    std::map<AggKey, std::vector<int>> agg_visits;

    for (auto& row : results) {
        AggKey key{row.method, row.budget};
        agg_rewards[key].push_back(row.best_mean_reward);
        agg_times[key].push_back(row.elapsed_ms);
        agg_visits[key].push_back(row.best_visits);
    }

    std::cout << std::setw(18) << "Method" << std::setw(8) << "Budget"
              << std::setw(12) << "AvgReward" << std::setw(12) << "AvgVisits"
              << std::setw(12) << "AvgTime(ms)" << std::setw(6) << "N" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (auto& [key, rewards] : agg_rewards) {
        double avg_r = 0; for (double r : rewards) avg_r += r; avg_r /= rewards.size();
        double avg_t = 0; for (double t : agg_times[key]) avg_t += t; avg_t /= agg_times[key].size();
        double avg_v = 0; for (int v : agg_visits[key]) avg_v += v; avg_v /= agg_visits[key].size();
        std::cout << std::setw(18) << key.method << std::setw(8) << key.budget
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg_r
                  << std::setw(12) << std::setprecision(0) << avg_v
                  << std::setw(12) << std::setprecision(0) << avg_t
                  << std::setw(6) << rewards.size() << std::endl;
    }

    // CSVエクスポート
    {
        std::string csv_path = output_dir + "/depth1_mcts_results.csv";
        std::ofstream ofs(csv_path);
        ofs << "state,n_candidates,budget,method,n_arms,best_label,best_type,"
            << "best_mean_reward,best_variance,best_visits,elapsed_ms" << std::endl;
        for (auto& row : results) {
            ofs << row.state << "," << row.n_candidates << "," << row.budget << ","
                << row.method << "," << row.n_arms << ","
                << "\"" << row.best_label << "\"," << row.best_type << ","
                << std::fixed << std::setprecision(4) << row.best_mean_reward << ","
                << row.best_variance << "," << row.best_visits << ","
                << std::setprecision(1) << row.elapsed_ms << std::endl;
        }
        std::cout << "\nCSV exported to: " << csv_path << std::endl;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Depth-1 Flat MC Experiment Complete" << std::endl;
    std::cout << "================================================================" << std::endl;
}
