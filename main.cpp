#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "digitalcurling3/digitalcurling3.hpp"

#include <fstream>
#include <iomanip>
#include <set>

namespace dc = digitalcurling3;

namespace {

    struct Position {
        float x;
        float y;
    };
    struct ShotInfo {
        float vx;
        float vy;
        int rot;
    };

    using LinkageRow = std::tuple<int, int, float, int>;
    using LinkageMatrix = std::vector<LinkageRow>;

dc::Team g_team;  // 自身のチームID
dc::GameSetting g_game_setting; // ゲーム設定
std::unique_ptr<dc::ISimulator> g_simulator; // シミュレーターのインターフェイス
std::unique_ptr<dc::ISimulatorStorage> g_simulator_storage; // シミュレーションの状態を保存するストレージ
std::array<std::unique_ptr<dc::IPlayer>, 4> g_players; // ゲームプレイヤー

const auto HouseRadius = 1.829;
const auto AreaMaxX = 2.375;
const auto AreaMaxY = 40.234;
const auto HouseCenterX = 0;
const auto HouseCenterY = 38.405;
const int GridSize_M = 4; // rows
const int GridSize_N = 4; // columns

std::vector<std::vector<Position>> grid(GridSize_M, std::vector<Position>(GridSize_N));
std::vector<std::vector<ShotInfo>> shotData(GridSize_M, std::vector<ShotInfo>(GridSize_N));
std::vector<dc::GameState> grid_states(GridSize_M * GridSize_N);

std::vector<std::vector<Position>> MakeGrid(const int m, const int n) {
    float x_grid = 2 * (2 * HouseRadius / 3) / (m - 1);
    float y_grid = 2 * (2 * HouseRadius / 3) / (n - 1);
    Position pos;
    std::vector<std::vector<Position>> result(GridSize_M, std::vector<Position>(GridSize_N));
    for (float i = 0; i < m; i++) {
        float y = HouseCenterY + (2 * HouseRadius / 3) - i * y_grid;
        for (int j = 0; j < n; j++) {
            float x = -(2 * HouseRadius / 3) + j * x_grid;
            pos.x = x;
            pos.y = y;
            result[i][j] = pos;
        }
    }
    return result;
}

dc::GameState run(dc::GameState const& game_state, ShotInfo shotinfo) {
    g_simulator->Load(*g_simulator_storage);
    dc::GameState state = game_state;
    auto& current_player = *g_players[game_state.shot / 4];
    dc::Vector2 shot_velocity(shotinfo.vx, shotinfo.vy);
    dc::moves::Shot::Rotation rotation = shotinfo.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    dc::moves::Shot shot{ shot_velocity, rotation };
    dc::Move move{ shot };
    dc::ApplyMove(g_game_setting, *g_simulator, current_player, state, move, std::chrono::milliseconds(0));
    g_simulator->Save(*g_simulator_storage);
    return state;
}

dc::Vector2 EstimateShotVelocityFCV1(dc::Vector2 const& target_position, float target_speed, dc::moves::Shot::Rotation rotation)
{
    assert(target_speed >= 0.f);
    assert(target_speed <= 4.f);

    // 初速度の大きさを逆算する
    // 逆算には専用の関数を用いる．

    float const v0_speed = [&target_position, target_speed]
        {
            auto const target_r = target_position.Length();
            assert(target_r > 0.f);

            if (target_speed <= 0.05f)
            {
                float constexpr kC0[] = { 0.0005048122574925176, 0.2756242531609261 };
                float constexpr kC1[] = { 0.00046669575066030805, -29.898958358378636, -0.0014030973174948508 };
                float constexpr kC2[] = { 0.13968687866736632, 0.41120940058777616 };

                float const c0 = kC0[0] * target_r + kC0[1];
                float const c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
                float const c2 = kC2[0] * target_r + kC2[1];

                return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
            }
            else if (target_speed <= 1.f)
            {
                float constexpr kC0[] = { -0.0014309170115803444, 0.9858457898438147 };
                float constexpr kC1[] = { -0.0008339331735471273, -29.86751291726946, -0.19811799977982522 };
                float constexpr kC2[] = { 0.13967323742978, 0.42816312110477517 };

                float const c0 = kC0[0] * target_r + kC0[1];
                float const c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
                float const c2 = kC2[0] * target_r + kC2[1];

                return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
            }
            else
            {
                float constexpr kC0[] = { 1.0833113118071224e-06, -0.00012132851917870833, 0.004578093297561233, 0.9767006869364527 };
                float constexpr kC1[] = { 0.07950648211492622, -8.228225657195706, -0.05601306077702578 };
                float constexpr kC2[] = { 0.14140440186382008, 0.3875782508767419 };

                float const c0 = kC0[0] * target_r * target_r * target_r + kC0[1] * target_r * target_r + kC0[2] * target_r + kC0[3];
                float const c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
                float const c2 = kC2[0] * target_r + kC2[1];

                return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
            }
        }();

    // assert(target_speed < v0_speed);

    // 一度シミュレーションを行い，発射方向を決定する

    dc::Vector2 const delta = [rotation, v0_speed, target_speed]
        {
            float const rotation_factor = rotation == dc::moves::Shot::Rotation::kCCW ? 1.f : -1.f;

            // シミュレータは FCV1 シミュレータを使用する．
            thread_local std::unique_ptr<dc::ISimulator> s_simulator;
            if (s_simulator == nullptr)
            {
                s_simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
            }

            dc::ISimulator::AllStones init_stones;
            init_stones[0].emplace(dc::Vector2(), 0.f, dc::Vector2(0.f, v0_speed), 1.57f * rotation_factor);
            s_simulator->SetStones(init_stones);

            while (!s_simulator->AreAllStonesStopped())
            {
                auto const& stones = s_simulator->GetStones();
                auto const speed = stones[0]->linear_velocity.Length();
                if (speed <= target_speed)
                {
                    return stones[0]->position;
                }
                s_simulator->Step();
            }

            return s_simulator->GetStones()[0]->position;
        }();

    float const delta_angle = std::atan2(delta.x, delta.y); // 注: delta.x, delta.y の順番で良い
    float const target_angle = std::atan2(target_position.y, target_position.x);
    float const v0_angle = target_angle + delta_angle; // 発射方向

    return dc::Vector2(v0_speed * std::cos(v0_angle), v0_speed * std::sin(v0_angle));
}

ShotInfo FindShot(Position const& pos) {
    dc::Vector2 target_position = { pos.x, pos.y };
    dc::Vector2 final_speed(0, 0);
    dc::moves::Shot::Rotation rotation = (pos.x > 0 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW);
    final_speed = EstimateShotVelocityFCV1(target_position, 0, rotation);
    ShotInfo shot;
    shot.vx = final_speed.x;
    shot.vy = final_speed.y;
    shot.rot = rotation == dc::moves::Shot::Rotation::kCW ? 1 : 0;
    return shot;
}

float dist(dc::GameState const& a, dc::GameState const& b) {
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

std::vector<std::vector<float>> MakeDistanceTable(std::vector<dc::GameState> const& states) {
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

void SaveSimilarityTableToCSV(const std::vector<std::vector<float>>& table, int shot_number) {
    std::string folder = "table_outputs_" + std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    std::string filename = folder + "state_similarity_shot_" + std::to_string(shot_number) + ".csv";
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    for (const auto& row : table) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << std::fixed << std::setprecision(10) << row[i];
            if (i != row.size() - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Saved similarity table to: " << filename << "\n";
}

std::tuple<int, int, float> findClosestClusters(const std::vector<std::vector<float>>& dist, const std::vector<std::set<int>>& clusters) {
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

LinkageMatrix hierarchicalClustering(const std::vector<std::vector<float>>& dist, std::vector<std::set<int>>& clusters, int n_desired_clusters = 1) {
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

void printLinkage(LinkageMatrix linkage) {
    std::cout << "Linkage Matrix:\n";
    std::cout << "Cluster1 Cluster2 Distance Size\n";
    for (auto const& [a, b, dist, size] : linkage) {
        std::cout << std::setw(8) << a
            << std::setw(9) << b
            << std::setw(9) << std::fixed << std::setprecision(2) << dist
            << std::setw(6) << size << "\n";
    }
}

void OutputClusterGridToCSV(const std::vector<int>& state_index_to_cluster,
    int rows, int cols,
    const std::string& filename, const int shot_num) {
    std::string folder = "hierarchical_clustering/cluster_distribution_" + std::to_string(rows) + "_" + std::to_string(cols) + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    std::string new_filename = folder + filename + "_" + std::to_string(shot_num) + ".csv";
    std::ofstream file(new_filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << new_filename << "\n";
        return;
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int index = r * cols + c;
            int cluster = state_index_to_cluster.at(index);  // Assumes all indices exist
            file << cluster;
            if (c < cols - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    std::cout << "Cluster grid written to " << new_filename << "\n";
}

float ComputeSilhouetteScore(
    const std::vector<std::vector<float>>& distance_matrix,
    const std::vector<int>& state_index_to_cluster
) {
    int N = distance_matrix.size();
    std::vector<float> silhouette_values(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        int cluster_i = state_index_to_cluster[i];
        std::vector<float> a_dists, b_dists;

        float a = 0.0f, b = std::numeric_limits<float>::max();

        // Distances to same-cluster members (a)
        int a_count = 0;
        for (int j = 0; j < N; ++j) {
            if (i != j && state_index_to_cluster[j] == cluster_i) {
                a += distance_matrix[i][j];
                ++a_count;
            }
        }
        if (a_count > 0) a /= a_count;

        // Distances to other clusters (b)
        std::map<int, std::pair<float, int>> cluster_sums;
        for (int j = 0; j < N; ++j) {
            int cluster_j = state_index_to_cluster[j];
            if (cluster_j != cluster_i) {
                cluster_sums[cluster_j].first += distance_matrix[i][j];
                cluster_sums[cluster_j].second++;
            }
        }

        for (const auto& [cluster, sum_pair] : cluster_sums) {
            float avg = sum_pair.first / sum_pair.second;
            if (avg < b) b = avg;
        }

        float s = 0.0f;
        if (a_count > 0 && std::max(a, b) > 0.0f) {
            s = (b - a) / std::max(a, b);
        }
        silhouette_values[i] = s;
    }

    // Average silhouette score
    float total = std::accumulate(silhouette_values.begin(), silhouette_values.end(), 0.0f);
    return total / N;
}

void ExportStoneCoordinatesToCSV(const dc::GameState& game_state, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << "\n";
        return;
    }

    // Write header
    for (int team = 0; team < 2; ++team) {
        for (int i = 0; i < 8; ++i) {
            file << "team" << team << "_stone" << i << "_x,";
            file << "team" << team << "_stone" << i << "_y,";
        }
    }
    file << "\n";

    // Write data rows
    const auto& state = game_state;
    for (int team = 0; team < 2; ++team) {
        for (int i = 0; i < 8; ++i) {
            if (state.stones[team][i]) {
                file << std::fixed << std::setprecision(3) << state.stones[team][i]->position.x << ",";
                file << std::fixed << std::setprecision(3) << state.stones[team][i]->position.y << ",";
            }
            else {
                file << "NaN,NaN,";
            }
        }
    }
    file << "\n";

    file.close();
    std::cout << "Stone coordinates exported to " << filename << "\n";
}

void ExportStonesByCluster(
    const std::vector<int>& state_index_to_cluster,
    const std::vector<dc::GameState>& all_game_states, const int shot_num)
{
    std::string base_folder = "hierarchical_clustering/Stone_Coordinates_" +
        std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::string shot_folder = base_folder + "shot" + std::to_string(shot_num) + "/";
    // Delete old shot folder if it exists
    if (std::filesystem::exists(shot_folder)) {
        std::filesystem::remove_all(shot_folder);
        std::cout << "Old folder removed: " << shot_folder << "\n";
    }
    std::filesystem::create_directories(shot_folder);

    for (int index = 0; index < state_index_to_cluster.size(); index++) {
        int state_index = index;
        int cluster_id = state_index_to_cluster[state_index];
        if (state_index >= all_game_states.size()) {
            std::cerr << "Invalid state index: " << state_index << "\n";
            continue;
        }

        const auto& game_state = all_game_states[state_index];

        // Construct folder: hierarchical_clustering/Stone_Coordinates_M_N/shotK/ClusterX/
        std::stringstream cluster_folder_ss;
        cluster_folder_ss << shot_folder << "Cluster" << cluster_id << "/";
        std::string cluster_folder = cluster_folder_ss.str();
        std::filesystem::create_directories(cluster_folder);

        // File: ClusterX/stateY.csv
        std::string state_filename = "state" + std::to_string(state_index);

        ExportStoneCoordinatesToCSV(game_state, cluster_folder + state_filename + ".csv");
    }

    std::cout << "Export complete: Stones sorted into cluster folders.\n";
}

float ComputeIntraClusterDistance(
    const std::vector<std::vector<float>>& dist,
    const std::vector<std::set<int>>& clsters) {

    float total = 0.0f;
    int count = 0;
    for (const auto& cluster_set : clsters) {
        std::vector<int> cluster(cluster_set.begin(), cluster_set.end());
        float sum = 0.0f;
        count = 0;
        for (int i = 0; i < cluster.size(); ++i) {
            for (int j = i + 1; j < cluster.size(); ++j) {
                sum += dist[cluster[i]][cluster[j]];
                count++;
            }
        }
        total += sum;
    }
    if (count > 0) total /= count;
    return total;
}

void IntraToCSV(const std::vector<float>& scores, const int shot_num) {
    std::string folder = "hierarchical_clustering/Intra_Cluster_Scores_" + std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    // Create filename with shot number
    std::string filename = "intra_cluster_scores_shot_" + std::to_string(shot_num) + ".csv";
    std::string new_filename = folder + filename;

    std::ofstream file(new_filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << new_filename << "\n";
        return;
    }

    file << "k,intra_score\n";
    for (size_t k = 2; k < scores.size(); ++k) {
        file << k << "," << std::fixed << std::setprecision(5) << scores[k] << "\n";
    }

    file.close();
    std::cout << "Intra-cluster scores saved to: " << new_filename << "\n";
}

void SilhouetteToCSV(float score, int shot_num, int k_cluster) {
    std::string folder = "hierarchical_clustering/SilhouetteScores_" + std::to_string(GridSize_M) + "_" + std::to_string(GridSize_N) + "/";
    std::filesystem::create_directories(folder);
    std::string filename = folder + "silhouette_scores_cluster_" + std::to_string(k_cluster) + ".csv";

    // Check if file exists
    bool file_exists = std::filesystem::exists(filename);

    std::ofstream file(filename, std::ios::app); // Open in append mode
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << "\n";
        return;
    }

    // Write header only if file didn't exist before
    if (!file_exists) {
        file << "shot,silhouette_score\n";
    }

    // Write current score
    file << shot_num << "," << std::fixed << std::setprecision(5) << score << "\n";
    file.close();

    std::cout << "Silhouette score for shot " << shot_num << " saved to: " << filename << "\n";
}


void OnInit(
    dc::Team team,
    dc::GameSetting const& game_setting,
    std::unique_ptr<dc::ISimulatorFactory> simulator_factory,
    std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories,
    std::array<size_t, 4>& player_order)
{
    // TODO AIを作る際はここを編集してください
    
    g_team = team;
    g_game_setting = game_setting;
    if (simulator_factory) {
        g_simulator = simulator_factory->CreateSimulator(); // simulator 生成 
    }
    else {
        g_simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
    }
    g_simulator_storage = g_simulator->CreateStorage();
    
    // プレイヤーを生成する
    // 非対応の場合は NormalDistプレイヤーを使用する．csv
    assert(g_players.size() == player_factories.size());
    for (size_t i = 0; i < g_players.size(); ++i) {
        auto const& player_factory = player_factories[player_order[i]];
        if (player_factory) {
            g_players[i] = player_factory->CreatePlayer();
        }
        else {
            g_players[i] = dc::players::PlayerNormalDistFactory().CreatePlayer();
        }

    }
    
    grid = MakeGrid(GridSize_M, GridSize_N);
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[i].size(); j++) {
            shotData[i][j] = FindShot(grid[i][j]); // 初回参考速度生成
        }
    }
    std::cout << "This is the end of the OnInit\n";
}
dc::Move OnMyTurn(dc::GameState const& game_state)
{
    int k = 0;
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[i].size(); j++) {
            grid_states[k++] = run(game_state, shotData[i][j]); // 類似度計算で使うサンプル生成
        }
    }
    auto distance_table = MakeDistanceTable(grid_states);
    SaveSimilarityTableToCSV(distance_table, game_state.shot);

    std::vector<std::set<int>> clusters(distance_table.size()); // 最初はすべて一つずつの集合として捉える
    int n_desired_clusters = GridSize_M == 4 ? 4 : 8;
    std::vector<int> state_index_to_cluster(grid_states.size());
    //ExportStoneCoordinatesToCSV(game_state, "shot_", game_state.shot);  
    if (game_state.shot % 2 == 0) { // 偶数投げのとき、elbowをチェック(クラスタ数が変動する)
        std::vector<float> intra_score(17);
        for (int k = 2; k <= 16; k++) {
            std::vector<std::set<int>> clusters(distance_table.size());
            std::cout << "k =" << k << "\n";
            LinkageMatrix linkage = hierarchicalClustering(distance_table, clusters, k);
            intra_score[k] = ComputeIntraClusterDistance(distance_table, clusters);
            IntraToCSV(intra_score, game_state.shot);
            std::cout << "k = " << k << "score: " << intra_score[k] << "\n";
        }
    }
    else { // 奇数投げのとき、シルエットをチェック(クラスタ数不変)
        float silh_score = 0.f;
        LinkageMatrix linkage = hierarchicalClustering(distance_table, clusters, n_desired_clusters);
        //printLinkage(linkage);
        for (int i = 0; i < clusters.size(); i++) {
            auto const& cluster = clusters[i];
            if (cluster.size() > 0) {
                std::cout << "Cluster[" << i << "]=(";
                for (auto const label : cluster) {
                    state_index_to_cluster[label] = i;
                    std::cout << label << ", ";
                }
                std::cout << ")\n";
            }
            ExportStonesByCluster(state_index_to_cluster, grid_states, game_state.shot);
        }
        silh_score = ComputeSilhouetteScore(distance_table, state_index_to_cluster);
        std::cout << "Silhouette Score: " << silh_score << "\n";
        SilhouetteToCSV(silh_score, game_state.shot, n_desired_clusters);
    }
    OutputClusterGridToCSV(state_index_to_cluster, GridSize_M, GridSize_N, "cluster_distribution_test", game_state.shot);

    dc::moves::Shot shot;
    int row = game_state.shot / GridSize_M;
    int col = game_state.shot % GridSize_N;
    shot.velocity.x = shotData[row][col].vx;
    shot.velocity.y = shotData[row][col].vy;
    shot.rotation = shotData[row][col].rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    std::cout << "Shot: " << shot.velocity.x << ", " << shot.velocity.y << "\n";
    return shot;
}



/// \brief 相手チームのターンに呼ばれます．AIを作る際にこの関数の中身を記述する必要は無いかもしれません．
///
/// ひとつ前の手番で自分が行った行動の結果を見ることができます．
///
/// \param game_state 現在の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
void OnOpponentTurn(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
}



/// \brief ゲームが正常に終了した際にはこの関数が呼ばれます．
///
/// \param game_state 試合終了後の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
void OnGameOver(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください

    if (game_state.game_result->winner == g_team) {
        std::cout << "won the game" << std::endl;
    } else {
        std::cout << "lost the game" << std::endl;
    }
}



} // unnamed namespace



int main(int argc, char const * argv[])
{
    using boost::asio::ip::tcp;
    using nlohmann::json;

    // TODO AIの名前を変更する場合はここを変更してください．
    constexpr auto kName = "ylab-project";

    constexpr int kSupportedProtocolVersionMajor = 1;

    try {
        if (argc != 3) {
            std::cerr << "Usage: command <host> <port>" << std::endl;
            return 1;
        }

        boost::asio::io_context io_context;

        tcp::socket socket(io_context);
        tcp::resolver resolver(io_context);
        boost::asio::connect(socket, resolver.resolve(argv[1], argv[2]));  // 引数のホスト，ポートに接続します．

        // ソケットから1行読む関数です．バッファが空の場合，新しい行が来るまでスレッドをブロックします．
        auto read_next_line = [&socket, input_buffer = std::string()] () mutable {
            // read_untilの結果，input_bufferに複数行入ることがあるため，1行ずつ取り出す処理を行っている
            if (input_buffer.empty()) {
                boost::asio::read_until(socket, boost::asio::dynamic_buffer(input_buffer), '\n');
            }
            auto new_line_pos = input_buffer.find_first_of('\n');
            auto line = input_buffer.substr(0, new_line_pos + 1);
            input_buffer.erase(0, new_line_pos + 1);
            return line;
        };

        // コマンドが予期したものかチェックする関数です．
        auto check_command = [] (nlohmann::json const& jin, std::string_view expected_cmd) {
            auto const actual_cmd = jin.at("cmd").get<std::string>();
            if (actual_cmd != expected_cmd) {
                std::ostringstream buf;
                buf << "Unexpected cmd (expected: \"" << expected_cmd << "\", actual: \"" << actual_cmd << "\")";
                throw std::runtime_error(buf.str());
            }
        };

        dc::Team team = dc::Team::kInvalid;

        // [in] dc
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "dc");

            auto const& jin_version = jin.at("version");
            if (jin_version.at("major").get<int>() != kSupportedProtocolVersionMajor) {
                throw std::runtime_error("Unexpected protocol version");
            }

            std::cout << "[in] dc" << std::endl;
            std::cout << "  game_id  : " << jin.at("game_id").get<std::string>() << std::endl;
            std::cout << "  date_time: " << jin.at("date_time").get<std::string>() << std::endl;
        }

        // [out] dc_ok
        {
            json const jout = {
                { "cmd", "dc_ok" },
                { "name", kName }
            };
            auto const output_message = jout.dump() + '\n';
            boost::asio::write(socket, boost::asio::buffer(output_message));

            std::cout << "[out] dc_ok" << std::endl;
            std::cout << "  name: " << kName << std::endl;
        }


        // [in] is_ready
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "is_ready");

            if (jin.at("game").at("rule").get<std::string>() != "normal") {
                throw std::runtime_error("Unexpected rule");
            }

            team = jin.at("team").get<dc::Team>();

            auto const game_setting = jin.at("game").at("setting").get<dc::GameSetting>();

            auto const& jin_simulator = jin.at("game").at("simulator");
            std::unique_ptr<dc::ISimulatorFactory> simulator_factory;
            try {
                simulator_factory = jin_simulator.get<std::unique_ptr<dc::ISimulatorFactory>>();
            } catch (std::exception & e) {
                std::cout << "Exception: " << e.what() << std::endl;
            }

            auto const& jin_player_factories = jin.at("game").at("players").at(dc::ToString(team));
            std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories;
            for (size_t i = 0; i < 4; ++i) {
                std::unique_ptr<dc::IPlayerFactory> player_factory;
                try {
                    player_factory = jin_player_factories[i].get<std::unique_ptr<dc::IPlayerFactory>>();
                } catch (std::exception & e) {
                    std::cout << "Exception: " << e.what() << std::endl;
                }
                player_factories[i] = std::move(player_factory);
            }

            std::cout << "[in] is_ready" << std::endl;
        
        // [out] ready_ok

            std::array<size_t, 4> player_order{ 0, 1, 2, 3 };
            OnInit(team, game_setting, std::move(simulator_factory), std::move(player_factories), player_order);

            json const jout = {
                { "cmd", "ready_ok" },
                { "player_order", player_order }
            };
            auto const output_message = jout.dump() + '\n';
            boost::asio::write(socket, boost::asio::buffer(output_message));

            std::cout << "[out] ready_ok" << std::endl;
            std::cout << "  player order: " << jout.at("player_order").dump() << std::endl;
        }

        // [in] new_game
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "new_game");

            std::cout << "[in] new_game" << std::endl;
            std::cout << "  team 0: " << jin.at("name").at("team0") << std::endl;
            std::cout << "  team 1: " << jin.at("name").at("team1") << std::endl;
        }

        dc::GameState game_state;

        while (true) {
            // [in] update
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "update");

            game_state = jin.at("state").get<dc::GameState>();

            std::cout << "[in] update (end: " << int(game_state.end) << ", shot: " << int(game_state.shot) << ")" << std::endl;

            // if game was over
            if (game_state.game_result) {
                break;
            }

            if (game_state.GetNextTeam() == team) { // my turn
                // [out] move
                auto move = OnMyTurn(game_state);
                json jout = {
                    { "cmd", "move" },
                    { "move", move }
                };
                auto const output_message = jout.dump() + '\n';
                boost::asio::write(socket, boost::asio::buffer(output_message));
                
                std::cout << "[out] move" << std::endl;
                if (std::holds_alternative<dc::moves::Shot>(move)) {
                    dc::moves::Shot const& shot = std::get<dc::moves::Shot>(move);
                    std::cout << "  type    : shot" << std::endl;
                    std::cout << "  velocity: [" << shot.velocity.x << ", " << shot.velocity.y << "]" << std::endl;
                    std::cout << "  rotation: " << (shot.rotation == dc::moves::Shot::Rotation::kCCW ? "ccw" : "cw") << std::endl;
                } else if (std::holds_alternative<dc::moves::Concede>(move)) {
                    std::cout << "  type: concede" << std::endl;
                }

            } else { // opponent turn
                OnOpponentTurn(game_state);
            }
        }

        // [in] game_over
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "game_over");

            std::cout << "[in] game_over" << std::endl;
        }

        // 終了．
        OnGameOver(game_state);

    } catch (std::exception & e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}