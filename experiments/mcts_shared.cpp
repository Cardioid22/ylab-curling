#include "mcts_shared.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

namespace mcts_shared {

// ========== 盤面評価 ==========

float evaluateEndScore(const dc::GameState& state, dc::Team my_team) {
    struct StoneInfo { float dist; int team; };
    std::vector<StoneInfo> in_house;

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (!state.stones[t][i]) continue;
            float dx = state.stones[t][i]->position.x - kHouseCenterX;
            float dy = state.stones[t][i]->position.y - kHouseCenterY;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= kHouseRadius + kStoneRadius) {
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
    return scoring_team == my_team_int
        ? static_cast<float>(score)
        : -static_cast<float>(score);
}

float evaluateBoard(const dc::GameState& state) {
    struct StoneInfo { float dist; int team; };
    std::vector<StoneInfo> in_house;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (!state.stones[t][i]) continue;
            float dx = state.stones[t][i]->position.x;
            float dy = state.stones[t][i]->position.y - kHouseCenterY;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= kHouseRadius + kStoneRadius) in_house.push_back({d, t});
        }
    }
    if (in_house.empty()) return 0.0f;
    std::sort(in_house.begin(), in_house.end(),
              [](auto& a, auto& b) { return a.dist < b.dist; });
    int scoring_team = in_house[0].team;
    int score = 0;
    for (auto& s : in_house) { if (s.team == scoring_team) score++; else break; }
    return scoring_team == 0 ? static_cast<float>(score) : -static_cast<float>(score);
}

int getZone(const std::optional<dc::Transform>& stone) {
    if (!stone) return -1;
    float x = stone->position.x;
    float y = stone->position.y;
    float dist_to_tee = std::sqrt(x * x + (y - kHouseCenterY) * (y - kHouseCenterY));
    if (dist_to_tee <= kHouseRadius) return 0;
    if (y < kHouseCenterY - kHouseRadius && y > kHouseCenterY - 3.0f * kHouseRadius) return 1;
    return 2;
}

// ========== Delta距離関数 ==========

float distDelta(const dc::GameState& input,
                const dc::GameState& a,
                const dc::GameState& b) {
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
                } else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            } else {
                if (in_a && in_b) {
                    new_stone_team = team; new_stone_idx = idx;
                    float dx = a.stones[team][idx]->position.x - b.stones[team][idx]->position.x;
                    float dy = a.stones[team][idx]->position.y - b.stones[team][idx]->position.y;
                    distance += NEW_STONE_WEIGHT * std::sqrt(dx*dx + dy*dy);
                    if (getZone(a.stones[team][idx]) != getZone(b.stones[team][idx])) distance += PENALTY_ZONE;
                } else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
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
        if (a.stones[t][i]) {
            float d = std::sqrt(std::pow(a.stones[t][i]->position.x, 2)
                              + std::pow(a.stones[t][i]->position.y - kHouseCenterY, 2));
            if (d < closest_a) { closest_a = d; team_a = t; }
        }
        if (b.stones[t][i]) {
            float d = std::sqrt(std::pow(b.stones[t][i]->position.x, 2)
                              + std::pow(b.stones[t][i]->position.y - kHouseCenterY, 2));
            if (d < closest_b) { closest_b = d; team_b = t; }
        }
    }
    if (team_a >= 0 && team_b >= 0 && team_a != team_b) distance += 10.0f;
    return distance;
}

std::vector<std::vector<float>> makeDistanceTableDelta(
    const dc::GameState& input_state,
    const std::vector<dc::GameState>& result_states)
{
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

// ========== 階層的クラスタリング ==========

std::vector<std::set<int>> runClustering(
    const std::vector<std::vector<float>>& dist_table,
    int n_desired_clusters)
{
    int n = static_cast<int>(dist_table.size());
    std::vector<std::set<int>> clusters(n);
    for (int i = 0; i < n; i++) clusters[i].insert(i);

    while (static_cast<int>(clusters.size()) > n_desired_clusters) {
        float min_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;
        int cs = static_cast<int>(clusters.size());
        for (int i = 0; i < cs; i++) {
            for (int j = i + 1; j < cs; j++) {
                float total = 0.0f;
                int count = 0;
                for (int a : clusters[i]) for (int b : clusters[j]) {
                    total += dist_table[a][b];
                    count++;
                }
                if (count == 0) continue;
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

std::vector<int> calculateMedoids(
    const std::vector<std::vector<float>>& dist_table,
    const std::vector<std::set<int>>& clusters)
{
    std::vector<int> medoids;
    for (auto& cluster : clusters) {
        if (cluster.empty()) { medoids.push_back(-1); continue; }
        if (cluster.size() == 1) { medoids.push_back(*cluster.begin()); continue; }
        float min_total = std::numeric_limits<float>::max();
        int best = -1;
        for (int c : cluster) {
            float total = 0;
            for (int o : cluster) if (c != o) total += dist_table[c][o];
            if (total < min_total) { min_total = total; best = c; }
        }
        medoids.push_back(best);
    }
    return medoids;
}

// ========== ロールアウト ==========

double rolloutFromState(
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    const dc::GameState& state,
    int remaining_shots,
    dc::Team root_team,
    std::mt19937& rng,
    double epsilon)
{
    const uint8_t starting_end = state.end;
    dc::GameState sim_state = state;
    int shots_played = 0;

    std::uniform_real_distribution<double> prob(0.0, 1.0);

    while (shots_played < remaining_shots
           && !sim_state.IsGameOver()
           && sim_state.end == starting_end) {

        bool is_team1_turn = (sim_state.shot % 2 == 1);
        dc::Team cur = is_team1_turn ? dc::Team::k1 : dc::Team::k0;

        // 手番チーム視点の「賢い」ロールアウト候補 (ドロー群 + 相手No.1へのHit/Freeze + 疎ならガード)
        // 物理シミュ不要の速度計算のみで生成 (毎手呼んでも安い)
        auto cands = gen.generateRolloutCandidates(sim_state, cur);
        if (cands.empty()) break;

        ShotInfo chosen;
        if (prob(rng) < epsilon) {
            std::uniform_int_distribution<int> cd(0, static_cast<int>(cands.size()) - 1);
            chosen = cands[cd(rng)].shot;
        } else {
            // 候補全探索で1手先読みの最高評価を選択
            double best_score = -1e9;
            chosen = cands[0].shot;
            for (auto& cand : cands) {
                dc::GameState next = sim.run_single_simulation(sim_state, cand.shot);
                double score;
                if (next.end > starting_end || next.IsGameOver()) {
                    int t0 = next.scores[0][starting_end].value_or(0);
                    int t1 = next.scores[1][starting_end].value_or(0);
                    score = static_cast<double>(t0 - t1);
                } else {
                    score = static_cast<double>(evaluateEndScore(next, dc::Team::k0));
                }
                if (is_team1_turn) score = -score;
                if (score > best_score) { best_score = score; chosen = cand.shot; }
            }
        }
        sim_state = sim.run_single_simulation(sim_state, chosen);
        shots_played++;
    }

    // エンド末端スコア
    if (sim_state.end > starting_end || sim_state.IsGameOver()) {
        if (starting_end < static_cast<uint8_t>(sim_state.scores[0].size())) {
            int t0 = sim_state.scores[0][starting_end].value_or(0);
            int t1 = sim_state.scores[1][starting_end].value_or(0);
            double diff = static_cast<double>(t0 - t1);
            // root_team 視点に変換
            return (root_team == dc::Team::k0) ? diff : -diff;
        }
    }
    // エンド未終了の場合は evaluateEndScore を使う
    return static_cast<double>(evaluateEndScore(sim_state, root_team));
}

double rollout(
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    const dc::GameState& state,
    const ShotInfo& initial_shot,
    int remaining_shots,
    dc::Team root_team,
    std::mt19937& rng,
    double epsilon)
{
    // initial_shot を適用
    dc::GameState sim_state = sim.run_single_simulation(state, initial_shot);
    int remaining_after = std::max(0, remaining_shots - 1);
    return rolloutFromState(sim, gen, sim_state, remaining_after, root_team, rng, epsilon);
}

// ========== UCB1 ==========

double ucb1Score(double mean, int visits, int total_visits, double c) {
    if (visits <= 0) return 1e9;
    if (total_visits <= 0) return mean;
    double explore = std::sqrt(std::log(static_cast<double>(total_visits))
                               / static_cast<double>(visits));
    return mean + c * explore;
}

// ========== テスト盤面ローダ ==========

std::vector<TestPositionRecord> loadTestPositionsFromCSV(
    const std::string& dir,
    const dc::GameSetting& game_setting,
    int max_n)
{
    std::vector<TestPositionRecord> records;

    if (!std::filesystem::exists(dir)) {
        std::cerr << "Error: directory does not exist: " << dir << std::endl;
        return records;
    }

    // batch_*.csv を名前順に読む
    std::vector<std::filesystem::path> batch_files;
    for (auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".csv" &&
            entry.path().filename().string().find("batch_") == 0) {
            batch_files.push_back(entry.path());
        }
    }
    std::sort(batch_files.begin(), batch_files.end());

    std::cout << "  [loader] " << batch_files.size() << " batch files in " << dir << std::endl;

    for (auto& bf : batch_files) {
        std::ifstream ifs(bf);
        if (!ifs) continue;

        std::string header;
        std::getline(ifs, header);  // ヘッダ行

        std::string line;
        while (std::getline(ifs, line)) {
            if (max_n > 0 && static_cast<int>(records.size()) >= max_n) break;

            std::vector<std::string> cols;
            std::stringstream ss(line);
            std::string col;
            while (std::getline(ss, col, ',')) cols.push_back(col);

            if (cols.size() < 4 + 16 * 3) continue;

            TestPositionRecord rec;
            try {
                rec.game_id = std::stoi(cols[0]);
                rec.end = std::stoi(cols[1]);
                rec.shot_num = std::stoi(cols[2]);
                int team_int = std::stoi(cols[3]);
                rec.current_team = (team_int == 0) ? dc::Team::k0 : dc::Team::k1;
            } catch (...) {
                continue;
            }

            dc::GameState state(game_setting);
            state.end = static_cast<std::uint8_t>(rec.end);
            state.shot = static_cast<std::uint8_t>(rec.shot_num);
            for (int t = 0; t < 2; ++t)
                for (int s = 0; s < 8; ++s)
                    state.stones[t][s].reset();

            int col_idx = 4;
            bool parse_ok = true;
            for (int t = 0; t < 2 && parse_ok; ++t) {
                for (int s = 0; s < 8 && parse_ok; ++s) {
                    try {
                        int inplay = std::stoi(cols[col_idx++]);
                        float x = std::stof(cols[col_idx++]);
                        float y = std::stof(cols[col_idx++]);
                        if (inplay == 1) {
                            state.stones[t][s].emplace(dc::Vector2(x, y), 0.0f);
                        }
                    } catch (...) {
                        parse_ok = false;
                    }
                }
            }
            if (!parse_ok) continue;

            rec.state = state;
            records.push_back(std::move(rec));
        }
        if (max_n > 0 && static_cast<int>(records.size()) >= max_n) break;
    }

    std::cout << "  [loader] loaded " << records.size() << " positions" << std::endl;
    return records;
}

std::vector<TestPositionRecord> sampleTestPositions(
    const std::vector<TestPositionRecord>& all,
    int n,
    uint64_t seed)
{
    if (static_cast<int>(all.size()) <= n) return all;

    std::vector<int> indices(all.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937_64 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<TestPositionRecord> sampled;
    sampled.reserve(n);
    for (int i = 0; i < n; i++) sampled.push_back(all[indices[i]]);
    return sampled;
}

}  // namespace mcts_shared
