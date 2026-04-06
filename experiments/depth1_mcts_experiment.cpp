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
    const uint8_t starting_end = state.end;

    // 1. 候補手を適用
    dc::GameState sim_state = sim.run_single_simulation(state, shot);
    int shots_played = 1;

    // 2. 残りショットをε-greedyグリッドポリシーで消化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> grid_dist(0, static_cast<int>(sim.initialShotData.size()) - 1);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    constexpr double EPSILON = 0.3;

    // エンド終了（石リセット）でループを抜ける
    while (shots_played < remaining_shots && !sim_state.IsGameOver()
           && sim_state.end == starting_end) {
        // 現在の手番チーム
        bool is_team1_turn = (sim_state.shot % 2 == 1);

        ShotInfo chosen;
        if (prob(gen) < EPSILON) {
            chosen = sim.initialShotData[grid_dist(gen)];
        } else {
            double best_score = -1e9;
            chosen = sim.initialShotData[0];
            for (auto& grid_shot : sim.initialShotData) {
                dc::GameState next = sim.run_single_simulation(sim_state, grid_shot);
                double score;
                if (next.end > starting_end || next.IsGameOver()) {
                    // エンド終了: scoresから読む
                    int t0 = next.scores[0][starting_end].value_or(0);
                    int t1 = next.scores[1][starting_end].value_or(0);
                    score = static_cast<double>(t0 - t1);
                } else {
                    score = static_cast<double>(evaluateEndScore(next, dc::Team::k0));
                }
                // 相手ターンでは符号反転（相手は自分に不利な手を選ぶ）
                if (is_team1_turn) score = -score;
                if (score > best_score) {
                    best_score = score;
                    chosen = grid_shot;
                }
            }
        }
        sim_state = sim.run_single_simulation(sim_state, chosen);
        shots_played++;
    }

    // 3. エンドスコアで評価
    // エンド終了後は石がリセットされているのでscores配列から読む
    if (sim_state.end > starting_end || sim_state.IsGameOver()) {
        if (starting_end < static_cast<uint8_t>(sim_state.scores[0].size())) {
            int t0 = sim_state.scores[0][starting_end].value_or(0);
            int t1 = sim_state.scores[1][starting_end].value_or(0);
            return static_cast<double>(t0 - t1);
        }
    }
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

    // 最終ショット付近のテスト盤面のみ（速度重視）

    // 1. 最終ショット（残り1手）: 自分1+相手2 → 最後の1手
    { dc::GameState s(game_setting_); s.stones[0][0].emplace(dc::Transform(dc::Vector2(-0.3f, kHouseCenterY+0.2f), 0.f)); s.stones[0][1].emplace(dc::Transform(dc::Vector2(0.5f, kHouseCenterY-0.3f), 0.f)); s.stones[0][2].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY+1.0f), 0.f)); s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY-0.05f), 0.f)); s.stones[1][1].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY+0.5f), 0.f)); s.stones[1][2].emplace(dc::Transform(dc::Vector2(-0.5f, kHouseCenterY-0.8f), 0.f)); s.stones[1][3].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY+1.2f), 0.f)); s.shot = 15; states.push_back(s); test_state_names_.push_back("last_shot"); }

    // 5. 残り2手: 終盤接戦
    { dc::GameState s(game_setting_); s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.1f, kHouseCenterY+0.1f), 0.f)); s.stones[0][1].emplace(dc::Transform(dc::Vector2(-0.4f, kHouseCenterY+0.7f), 0.f)); s.stones[0][2].emplace(dc::Transform(dc::Vector2(0.6f, kHouseCenterY-0.4f), 0.f)); s.stones[1][0].emplace(dc::Transform(dc::Vector2(-0.05f, kHouseCenterY-0.02f), 0.f)); s.stones[1][1].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY+0.3f), 0.f)); s.stones[1][2].emplace(dc::Transform(dc::Vector2(-0.2f, kHouseCenterY-0.6f), 0.f)); s.shot = 14; states.push_back(s); test_state_names_.push_back("last_2shots"); }

    // 6. 残り3手
    { dc::GameState s(game_setting_); s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY), 0.f)); s.stones[0][1].emplace(dc::Transform(dc::Vector2(-0.8f, kHouseCenterY+0.5f), 0.f)); s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY+0.15f), 0.f)); s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.3f, kHouseCenterY+1.0f), 0.f)); s.stones[1][2].emplace(dc::Transform(dc::Vector2(0.5f, kHouseCenterY-0.5f), 0.f)); s.shot = 13; states.push_back(s); test_state_names_.push_back("last_3shots"); }

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
    std::cout << "  1-shot-ahead vs End-of-end Evaluation Comparison" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto test_states = createTestStates();
    std::cout << "Test states: " << test_states.size() << std::endl;

    std::string output_dir = "experiments/depth1_mcts_results";
    std::filesystem::create_directories(output_dir);

    ShotGenerator generator(game_setting_);
    auto grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    SimulatorWrapper sim(dc::Team::k0, game_setting_);
    for (auto& pos : grid) {
        sim.initialShotData.push_back(sim.FindShot(pos));
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
    int n_rollouts = 3;  // 候補あたりのロールアウト回数
    std::vector<float> retention_ratios = {0.1f, 0.2f, 0.3f, 0.5f, 0.7f};

    struct ResultRow {
        std::string state;
        int n_candidates;
        float ratio;
        int k;
        // 1手先評価（静的）
        std::string static_best_label;
        std::string static_best_type;
        float static_best_score;
        // エンド終了評価（ロールアウト）
        std::string rollout_best_label;
        std::string rollout_best_type;
        double rollout_best_mean;
        // DCクラスタリング後の最良手
        std::string dc_static_best_label;
        std::string dc_rollout_best_label;
        // 一致度
        bool static_vs_rollout_exact;  // 1手先とエンド終了で同じ手を選んだか
        bool dc_static_same_cluster;   // DC(静的)の手が歩(静的)と同じクラスタか
        bool dc_rollout_same_cluster;  // DC(ロールアウト)の手が歩(ロールアウト)と同じクラスタか
        float static_score_diff;
        double rollout_score_diff;
    };
    std::vector<ResultRow> results;

    for (size_t s = 0; s < test_states.size(); s++) {
        auto& state = test_states[s];
        dc::Team my_team = dc::Team::k0;

        auto pool = generator.generatePool(state, my_team, grid);
        int n = static_cast<int>(pool.candidates.size());
        if (n <= 2) continue;

        int remaining_shots = 16 - state.shot;

        std::cout << "\n[" << (s+1) << "/" << test_states.size() << "] "
                  << test_state_names_[s] << " (N=" << n
                  << ", remaining=" << remaining_shots << ")" << std::endl;

        // === 全候補の評価 ===
        // (A) 1手先の静的評価（前回の実験と同じ）
        std::vector<float> static_scores(n);
        for (int i = 0; i < n; i++) {
            static_scores[i] = evaluateBoard(pool.result_states[i]);
        }
        int static_best_idx = std::max_element(static_scores.begin(), static_scores.end()) - static_scores.begin();

        // (B) エンド終了までロールアウトした評価
        std::vector<double> rollout_scores(n, 0.0);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++) {
            for (int r = 0; r < n_rollouts; r++) {
                rollout_scores[i] += rollout(sim, state, pool.candidates[i].shot, remaining_shots);
            }
            rollout_scores[i] /= n_rollouts;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int rollout_best_idx = std::max_element(rollout_scores.begin(), rollout_scores.end()) - rollout_scores.begin();

        bool static_vs_rollout = (static_best_idx == rollout_best_idx);
        std::cout << "  Static best: " << pool.candidates[static_best_idx].label
                  << " (score=" << static_scores[static_best_idx] << ")" << std::endl;
        std::cout << "  Rollout best: " << pool.candidates[rollout_best_idx].label
                  << " (mean=" << std::fixed << std::setprecision(2) << rollout_scores[rollout_best_idx] << ")"
                  << " [" << std::setprecision(0) << ms << "ms]"
                  << (static_vs_rollout ? " SAME" : " DIFFERENT") << std::endl;

        // === クラスタリング + 各保持率 ===
        auto dist_delta = makeDistanceTableDelta(state, pool.result_states);

        for (float ratio : retention_ratios) {
            int k = std::max(2, static_cast<int>(std::round(n * ratio)));
            if (k >= n) continue;

            auto clusters = runClustering(dist_delta, k);
            auto medoids = calculateMedoids(dist_delta, clusters);

            // DC: 静的評価での最良メドイド
            int dc_static_best = -1;
            float dc_static_best_score = -1e9f;
            for (int m : medoids) {
                if (m < 0) continue;
                if (static_scores[m] > dc_static_best_score) {
                    dc_static_best_score = static_scores[m];
                    dc_static_best = m;
                }
            }

            // DC: ロールアウト評価での最良メドイド（既に計算済みのrollout_scoresを使う）
            int dc_rollout_best = -1;
            double dc_rollout_best_score = -1e9;
            for (int m : medoids) {
                if (m < 0) continue;
                if (rollout_scores[m] > dc_rollout_best_score) {
                    dc_rollout_best_score = rollout_scores[m];
                    dc_rollout_best = m;
                }
            }

            // Same Cluster判定
            auto findCluster = [&](int idx) -> int {
                for (int ci = 0; ci < static_cast<int>(clusters.size()); ci++) {
                    if (clusters[ci].count(idx)) return ci;
                }
                return -1;
            };

            int static_best_cl = findCluster(static_best_idx);
            int rollout_best_cl = findCluster(rollout_best_idx);
            int dc_static_cl = findCluster(dc_static_best);
            int dc_rollout_cl = findCluster(dc_rollout_best);

            int ratio_pct = static_cast<int>(std::round(ratio * 100));
            std::cout << "  " << ratio_pct << "%(K=" << k << ")"
                      << "  DC_static:" << pool.candidates[dc_static_best].label
                      << (dc_static_best == static_best_idx ? " EXACT" : (dc_static_cl == static_best_cl ? " SameCluster" : " MISS"))
                      << "  DC_rollout:" << pool.candidates[dc_rollout_best].label
                      << (dc_rollout_best == rollout_best_idx ? " EXACT" : (dc_rollout_cl == rollout_best_cl ? " SameCluster" : " MISS"))
                      << std::endl;

            ResultRow row;
            row.state = test_state_names_[s];
            row.n_candidates = n; row.ratio = ratio; row.k = k;
            row.static_best_label = pool.candidates[static_best_idx].label;
            row.static_best_type = classifyType(pool.candidates[static_best_idx].type);
            row.static_best_score = static_scores[static_best_idx];
            row.rollout_best_label = pool.candidates[rollout_best_idx].label;
            row.rollout_best_type = classifyType(pool.candidates[rollout_best_idx].type);
            row.rollout_best_mean = rollout_scores[rollout_best_idx];
            row.dc_static_best_label = pool.candidates[dc_static_best].label;
            row.dc_rollout_best_label = pool.candidates[dc_rollout_best].label;
            row.static_vs_rollout_exact = static_vs_rollout;
            row.dc_static_same_cluster = (dc_static_cl == static_best_cl);
            row.dc_rollout_same_cluster = (dc_rollout_cl == rollout_best_cl);
            row.static_score_diff = dc_static_best_score - static_scores[static_best_idx];
            row.rollout_score_diff = dc_rollout_best_score - rollout_scores[rollout_best_idx];
            results.push_back(row);
        }
    }

    // ========== サマリー ==========
    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Summary" << std::endl;
    std::cout << "================================================================" << std::endl;

    // 1. 静的評価 vs ロールアウト評価で最良手は変わるか
    int n_same = 0, n_total = 0;
    for (auto& row : results) {
        if (row.ratio == retention_ratios[0]) {  // 重複カウント防止
            n_total++;
            if (row.static_vs_rollout_exact) n_same++;
        }
    }
    std::cout << "\n  [Static vs Rollout] Same best shot: " << n_same << "/" << n_total
              << " (" << std::fixed << std::setprecision(0)
              << (n_total > 0 ? 100.0f * n_same / n_total : 0) << "%)" << std::endl;

    // 2. 保持率別のSame Cluster率比較
    std::cout << "\n  [DC Same Cluster by Retention Ratio]" << std::endl;
    std::cout << std::setw(8) << "Ratio"
              << std::setw(22) << "DC_Static SameClust"
              << std::setw(22) << "DC_Rollout SameClust" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (float ratio : retention_ratios) {
        int rp = static_cast<int>(std::round(ratio * 100));
        int count = 0, sc_static = 0, sc_rollout = 0;
        for (auto& row : results) {
            if (static_cast<int>(std::round(row.ratio * 100)) == rp) {
                count++;
                if (row.dc_static_same_cluster) sc_static++;
                if (row.dc_rollout_same_cluster) sc_rollout++;
            }
        }
        if (count == 0) continue;
        std::cout << std::setw(6) << rp << "%"
                  << std::setw(10) << sc_static << "/" << count
                  << " (" << std::setw(3) << std::setprecision(0) << (100.0f * sc_static / count) << "%)"
                  << std::setw(10) << sc_rollout << "/" << count
                  << " (" << std::setw(3) << (100.0f * sc_rollout / count) << "%)"
                  << std::endl;
    }

    // CSVエクスポート
    {
        std::string csv_path = output_dir + "/static_vs_rollout.csv";
        std::ofstream ofs(csv_path);
        ofs << "state,n_candidates,ratio,k,"
            << "static_best_label,static_best_type,static_best_score,"
            << "rollout_best_label,rollout_best_type,rollout_best_mean,"
            << "dc_static_best_label,dc_rollout_best_label,"
            << "static_vs_rollout_exact,dc_static_same_cluster,dc_rollout_same_cluster,"
            << "static_score_diff,rollout_score_diff" << std::endl;
        for (auto& row : results) {
            ofs << row.state << "," << row.n_candidates << ","
                << std::fixed << std::setprecision(2) << row.ratio << "," << row.k << ","
                << "\"" << row.static_best_label << "\"," << row.static_best_type << ","
                << std::setprecision(1) << row.static_best_score << ","
                << "\"" << row.rollout_best_label << "\"," << row.rollout_best_type << ","
                << std::setprecision(4) << row.rollout_best_mean << ","
                << "\"" << row.dc_static_best_label << "\","
                << "\"" << row.dc_rollout_best_label << "\","
                << (row.static_vs_rollout_exact ? 1 : 0) << ","
                << (row.dc_static_same_cluster ? 1 : 0) << ","
                << (row.dc_rollout_same_cluster ? 1 : 0) << ","
                << std::setprecision(2) << row.static_score_diff << ","
                << row.rollout_score_diff << std::endl;
        }
        std::cout << "\nCSV exported to: " << csv_path << std::endl;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  Experiment Complete (n_rollouts=" << n_rollouts << " per candidate)" << std::endl;
    std::cout << "================================================================" << std::endl;
}
