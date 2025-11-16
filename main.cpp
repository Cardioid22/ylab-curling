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
#include <filesystem>
#include "src/mcts.h"
#include "src/structure.h"
#include "src/clustering.h"
#include "src/simulator.h"
#include "src/analysis.h"
#include "experiments/simple_experiment.h"
#include "experiments/clustering_validation.h"
#include "experiments/agreement_experiment.h"
#define DBL_EPSILON 2.2204460492503131e-016

namespace dc = digitalcurling3;

namespace {


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

std::vector<Position> grid;
std::vector<ShotInfo> shotData;
std::unordered_map<int, ShotInfo> state_to_shot_table;
std::vector<dc::GameState> grid_states;
std::shared_ptr<SimulatorWrapper> simWrapper;
std::shared_ptr<SimulatorWrapper> simWrapper_allgrid;

std::vector<Position> MakeGrid(const int m, const int n) {
    const float x_min = -HouseRadius;
    const float x_max = HouseRadius;
    const float y_min = 36.0;
    const float y_max = 40.0;

    float x_grid = (x_max - x_min) / (m - 1);
    float y_grid = (y_max - y_min) / (n - 1);

    float x_start = -((m - 1) / 2.0) * x_grid; // center based
    Position pos;
    std::vector<Position> result;
    for (int j = 0; j < n; ++j) {
        float y = y_max - j * y_grid;
        for (int i = 0; i < m; ++i) {
            float x = x_start + i * x_grid;
            pos.x = x;
            pos.y = y;
            result.push_back(pos);
        }
    }
    return result;
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
    //std::cout << "Finding Shot...\n";
    dc::Vector2 target_position = { pos.x, pos.y };
    dc::Vector2 final_speed(0, 0);
    dc::moves::Shot::Rotation rotation = (pos.x > 0 ? dc::moves::Shot::Rotation::kCCW : dc::moves::Shot::Rotation::kCW);
    final_speed = EstimateShotVelocityFCV1(target_position, 0, rotation);
    ShotInfo shot;
    shot.vx = final_speed.x;
    shot.vy = final_speed.y;
    shot.rot = rotation == dc::moves::Shot::Rotation::kCW ? 1 : 0;
    return shot;
}

dc::GameState run_single_simulation(dc::GameState const& state, const ShotInfo& shot) {
    dc::GameState new_state = state;
    if (!g_simulator || !g_simulator_storage) throw std::runtime_error("Simulator or storage not initialized");
    g_simulator->Load(*g_simulator_storage);
    auto& current_player = *g_players[new_state.shot / 4];
    if (!&current_player) {
        std::cout << "Player is null.\n";
    }
    dc::Vector2 velocity(shot.vx, shot.vy);
    auto rot = shot.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    dc::moves::Shot shot_move{ velocity, rot };
    dc::Move move{ shot_move };
    dc::ApplyMove(g_game_setting, *g_simulator, current_player, new_state, move, std::chrono::milliseconds(0));
    g_simulator->Save(*g_simulator_storage);
    return new_state;
}

dc::moves::Shot test(dc::GameState const& game_state) {
    int shot_num = static_cast<int>(game_state.shot);
    dc::moves::Shot shot;
    dc::Vector2 vel;
    switch (shot_num)
    {
    case 0: 
        vel =  dc::Vector2(0.0, 38.8);
        vel = EstimateShotVelocityFCV1(vel, 0, dc::moves::Shot::Rotation::kCCW);
        shot.velocity.x = vel.x;
        shot.velocity.y = vel.y;
        shot.rotation = dc::moves::Shot::Rotation::kCCW;
        break;
    case 1:
        vel = dc::Vector2(0.2, 38.405);
        vel = EstimateShotVelocityFCV1(vel, 0, dc::moves::Shot::Rotation::kCCW);
        shot.velocity.x = vel.x;
        shot.velocity.y = vel.y;
        shot.rotation = dc::moves::Shot::Rotation::kCCW;
        break;
    case 2:
        vel = dc::Vector2(-0.5, 38.2);
        vel = EstimateShotVelocityFCV1(vel, 0, dc::moves::Shot::Rotation::kCW);
        shot.velocity.x = vel.x;
        shot.velocity.y = vel.y;
        shot.rotation = dc::moves::Shot::Rotation::kCW;
        break;
    case 3:
        vel = dc::Vector2(-0.3, 38.0);
        vel = EstimateShotVelocityFCV1(vel, 0, dc::moves::Shot::Rotation::kCW);
        shot.velocity.x = vel.x;
        shot.velocity.y = vel.y;
        shot.rotation = dc::moves::Shot::Rotation::kCW;
        break;
    case 4:
        vel = dc::Vector2(1.2, 38.6);
        vel = EstimateShotVelocityFCV1(vel, 0, dc::moves::Shot::Rotation::kCW);
        shot.velocity.x = vel.x;
        shot.velocity.y = vel.y;
        shot.rotation = dc::moves::Shot::Rotation::kCW;
        break;
    case 5:
        vel = dc::Vector2(0.8, 37.8);
        vel = EstimateShotVelocityFCV1(vel, 0, dc::moves::Shot::Rotation::kCW);
        shot.velocity.x = vel.x;
        shot.velocity.y = vel.y;
        shot.rotation = dc::moves::Shot::Rotation::kCW;
        break;
    default:
        break;
    }
    return shot;
}

int findStateFromShot(ShotInfo target) {
    for (int i = 0; i < state_to_shot_table.size(); i++) {
        ShotInfo shotinfo = state_to_shot_table[i];
        if (fabs(shotinfo.vx - target.vx) <= DBL_EPSILON) {
            return i;
        }
    }
    return -1;
}

long long int calcIteration(long long int c, long long int d) {
    long long int res = 0;
    for (long long int i = 0; i <= d; i++) {
        res += pow(c, i);
    }
    return res;
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
    simWrapper = std::make_unique<SimulatorWrapper>(g_team, g_game_setting);
    simWrapper_allgrid = std::make_unique<SimulatorWrapper>(g_team, g_game_setting);
    std::cout << "CurlingAI Initialize Begin.\n";
    int S = GridSize_M * GridSize_N;
    grid.resize(S);
    shotData.resize(S);
    grid_states.resize(S);
    grid = MakeGrid(GridSize_M, GridSize_N);
    std::ofstream outFile("shotInfo_" + std::to_string(GridSize_M * GridSize_N));
    for (int i = 0; i < S; ++i) {
        ShotInfo shotinfo = FindShot(grid[i]);
        outFile << "  ShotInfo for grid[" << i << "]: "
            << "vx = " << shotinfo.vx << ", "
            << "vy = " << shotinfo.vy << ", "
            << "rot = " << shotinfo.rot << "\n";
        shotData[i] = shotinfo;
        simWrapper->initialShotData.push_back(shotinfo);
        simWrapper_allgrid->initialShotData.push_back(shotinfo);
        state_to_shot_table[i] = shotinfo;
    }
    outFile.close();
    std::cout << "CurlingAI Initialize Done.\n";
}
dc::Move OnMyTurn(dc::GameState const& game_state)
{
    //dc::moves::Shot shot;
    //shot.velocity.x = shotData[static_cast<int>(game_state.shot)].vx;
    //shot.velocity.y = shotData[static_cast<int>(game_state.shot)].vy;
    //shot.rotation = shotData[static_cast<int>(game_state.shot)].rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    //return shot;
    //if (game_state.shot < 6) {
    //    dc::moves::Shot test_shot = test(game_state);
    //    return test_shot;
    //}
    //if (game_state.hammer == g_team) {
    //    dc::moves::Shot shot;
    //    shot.velocity.x = 0.1;
    //    shot.velocity.y = 2.5;
    //    shot.rotation = dc::moves::Shot::Rotation::kCCW;
    //    return shot;
    //}
    for (int i = 0; i < GridSize_M * GridSize_N; ++i) {
        ShotInfo shot = shotData[i];
        //std::cout << "shotData in My Turn. shot.vx: " << shot.vx << ", shot.vy: " << shot.vy << "\n";
        dc::GameState result_state = run_single_simulation(game_state, shot); // simulate one outcome
        grid_states[i] = result_state;
    }
    dc::GameState const& current_state = game_state;
    int shot_num = static_cast<int>(game_state.shot);
    std::cout << "CurlingAI grid_states Calculation Done.\n";
    long long int S = GridSize_M * GridSize_N * 1LL;
    Clustering algo(static_cast<int>(std::log2(S)), grid_states, GridSize_M, GridSize_N, g_team);
    Analysis an(GridSize_M, GridSize_N);
    auto clusters = algo.getRecommendedStates(); // for debugging
    auto cluster_id_to_state = algo.get_clusters_id_table(); // for debugging

    // --- MCTS Search ---

    std::cout << "------Clustered Tree------" << '\n';
    long long int mcts_child_num = static_cast<long long int>(std::log2(S));
    long long int search_depth = 2;
    long long int mcts_iter = calcIteration(mcts_child_num, search_depth);
    std::cout << search_depth << "\n";
    std::cout << mcts_iter << "\n";
    long long int cluster_child_num = S;
    long long int cluster_iter = calcIteration(cluster_child_num, search_depth);
    std::cout << cluster_iter << "\n";
    an.cluster_id_to_state_csv(cluster_id_to_state, shot_num, mcts_iter); // for debugging
    MCTS mcts_clustered(current_state, NodeSource::Clustered, grid_states, state_to_shot_table, simWrapper, GridSize_M, GridSize_N);
    mcts_clustered.grow_tree(mcts_iter, 86400.0);
    mcts_clustered.export_rollout_result_to_csv("root_children_score_clustered", shot_num, GridSize_M, GridSize_N, shotData);

    std::cout << "------AllGrid Tree------" << '\n';

    MCTS mcts_allgrid(current_state, NodeSource::AllGrid, grid_states, state_to_shot_table, simWrapper_allgrid, GridSize_M, GridSize_N);
    mcts_allgrid.grow_tree(cluster_iter, 86400.0);
    mcts_allgrid.export_rollout_result_to_csv("root_children_score_allgrid", shot_num, GridSize_M, GridSize_N, shotData);
    ShotInfo best = mcts_clustered.get_best_shot();
    ShotInfo best_allgrid = mcts_allgrid.get_best_shot();
    ////mcts_clustered.report_rollout_result(); // output on terminal
    ////mcts_allgrid.report_rollout_result(); // output on terminal
    dc::moves::Shot final_shot;
    final_shot.velocity.x = best.vx;
    final_shot.velocity.y = best.vy;
    final_shot.rotation = best.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    int best_state = findStateFromShot(best);
    int best_allgrid_state = findStateFromShot(best_allgrid);
    an.export_best_shot_comparison_to_csv(best, best_allgrid, best_state, best_allgrid_state, shot_num, mcts_iter, "best_shot_comparison");
    
    //std::cout << "MCTS Selected Shot: " << best.vx << ", " << best.vy << "," <<  best.rot << "\n";
    //std::cout << "AllGrid Shot: " << best_allgrid.vx << ", " << best_allgrid.vy << best_allgrid.rot << "\n";
    //std::cout << "Clustered Best Shot Place: " << best_state << "\n";
    //std::cout << "AllGrid Best Shot Place: " << best_allgrid_state << "\n";
    //dc::moves::Shot shot;
    //shot.velocity.x = shotData[static_cast<int>(game_state.shot)].vx;
    //shot.velocity.y = shotData[static_cast<int>(game_state.shot)].vy;
    //shot.rotation = shotData[static_cast<int>(game_state.shot)].rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    //return shot;
    return final_shot;
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



// 実験実行用の関数
void runSimpleExperiment() {
    std::cout << "\n=== Starting Simple Clustering vs AllGrid Experiment ===" << std::endl;

    // シンプル実験の初期化
    SimpleExperiment experiment(grid_states, state_to_shot_table, GridSize_M, GridSize_N);

    // 実験パラメータ
    int num_test_states = 3;      // テストする盤面の数
    int max_iterations = 1000000;    // 各手法の最大探索数

    std::cout << "Test states: " << num_test_states << std::endl;
    std::cout << "Max iterations per test: " << max_iterations << std::endl;

    // バッチ実験の実行
    auto results = experiment.runBatchExperiment(num_test_states, max_iterations);

    // グリッド数とパラメータに基づいたフォルダ名を生成
    int grid_size = GridSize_M * GridSize_N;
    std::string folder_name = "experiments/Grid" + std::to_string(grid_size) +
                              "_States" + std::to_string(num_test_states) +
                              "_MaxIter" + std::to_string(max_iterations);

    // フォルダが存在しない場合は作成 (analysis.cppと同じ方法)
    std::filesystem::create_directories(folder_name);

    // ファイル名にもパラメータ情報を含める
    std::string output_file = folder_name + "/experiment_results_grid" +
                              std::to_string(grid_size) + ".csv";

    // 結果をCSVに保存
    experiment.saveResults(results, output_file);

    std::cout << "\nExperiment completed!" << std::endl;
    std::cout << "Results saved to: " << output_file << std::endl;
    std::cout << "\nTo analyze results, run:" << std::endl;
    std::cout << "  python analyze_simple_results.py " << output_file << std::endl;
}

// クラスタリング検証実験を実行する関数
void runClusteringValidation() {
    std::cout << "\n=== Starting Clustering Validation Experiment ===" << std::endl;

    // ClusteringValidation インスタンス作成
    ClusteringValidation validation(dc::Team::k0);

    // テスト用石配置を生成（各パターン3バリエーション）
    auto test_states = validation.generateTestStates(3);

    std::cout << "\nGenerated " << test_states.size() << " test stone distributions" << std::endl;
    std::cout << "Pattern types:" << std::endl;
    std::cout << "  - Basic: CenterGuard, DoubleGuard, CornerGuards" << std::endl;
    std::cout << "  - House: SingleDraw, DoubleDraw, TripleDraw, HouseCorners" << std::endl;
    std::cout << "  - Mixed: GuardAndDraw, Split, Crowded, Scattered" << std::endl;
    std::cout << "  - Tactical: FreezeAttempt, Takeout, Promotion, Corner" << std::endl;
    std::cout << "  - Other: Symmetric, Asymmetric, Random" << std::endl;

    // 目標クラスタ数を計算（log2(状態数)）
    int target_clusters = static_cast<int>(std::log2(test_states.size()));
    if (target_clusters < 3) target_clusters = 3;  // 最低3クラスタ

    std::cout << "\nTarget clusters: " << target_clusters << std::endl;

    // 両方のクラスタリング手法で実行
    auto result = validation.runComparison(test_states, target_clusters);

    // 結果を出力
    std::string output_dir = "experiments/clustering_validation_results";
    validation.exportResults(result, test_states, output_dir);

    // サマリー表示
    std::cout << "\n=== Validation Summary ===" << std::endl;
    std::cout << "Test states: " << test_states.size() << std::endl;
    std::cout << "\nClustering V1 (Hierarchical):" << std::endl;
    std::cout << "  - Clusters: " << result.v1_clusters.size() << std::endl;
    std::cout << "  - Time: " << result.v1_time_ms << " ms" << std::endl;
    std::cout << "\nClustering V2 (Feature-based):" << std::endl;
    std::cout << "  - Clusters: " << result.v2_clusters.size() << std::endl;
    std::cout << "  - Quality Score: " << result.v2_quality_score << std::endl;
    std::cout << "  - Time: " << result.v2_time_ms << " ms" << std::endl;
    std::cout << "\nSpeedup: " << (result.v1_time_ms / result.v2_time_ms) << "x" << std::endl;

    std::cout << "\n=== Results saved to: " << output_dir << " ===" << std::endl;
    std::cout << "\nTo visualize the stone distributions, run:" << std::endl;
    std::cout << "  python python/visualize_clustering_validation.py" << std::endl;
    std::cout << "\n=== Clustering Validation Completed! ===" << std::endl;
}

int main(int argc, char const * argv[])
{
    using boost::asio::ip::tcp;
    using nlohmann::json;

    // TODO AIの名前を変更する場合はここを変更してください．
    constexpr auto kName = "ylab-project";

    constexpr int kSupportedProtocolVersionMajor = 1;

    try {
        // 実験モードのチェック（引数に--experimentが含まれているかチェック）
        bool experiment_mode = false;
        bool validate_clustering_mode = false;
        bool agreement_experiment_mode = false;

        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--experiment") {
                experiment_mode = true;
            }
            if (std::string(argv[i]) == "--validate-clustering") {
                validate_clustering_mode = true;
            }
            if (std::string(argv[i]) == "--agreement-experiment") {
                agreement_experiment_mode = true;
            }
        }

        if (experiment_mode) {
            std::cout << "Running in experiment mode..." << std::endl;
            // 初期化を簡略化して実験実行
            g_team = dc::Team::k0;
            g_game_setting.max_end = 1;
            g_game_setting.five_rock_rule = true;
            g_game_setting.thinking_time[0] = std::chrono::seconds(86400);
            g_game_setting.thinking_time[1] = std::chrono::seconds(86400);

            // 基本的な初期化（OnInit関数の代わり）
            g_simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
            g_simulator_storage = g_simulator->CreateStorage();

            for (size_t i = 0; i < g_players.size(); ++i) {
                g_players[i] = dc::players::PlayerNormalDistFactory().CreatePlayer();
            }

            simWrapper = std::make_unique<SimulatorWrapper>(g_team, g_game_setting);
            simWrapper_allgrid = std::make_unique<SimulatorWrapper>(g_team, g_game_setting);

            // グリッドとショット情報の初期化
            int S = GridSize_M * GridSize_N;
            grid.resize(S);
            shotData.resize(S);
            grid_states.resize(S);
            grid = MakeGrid(GridSize_M, GridSize_N);

            for (int i = 0; i < S; ++i) {
                ShotInfo shotinfo = FindShot(grid[i]);
                shotData[i] = shotinfo;
                simWrapper->initialShotData.push_back(shotinfo);
                simWrapper_allgrid->initialShotData.push_back(shotinfo);
                state_to_shot_table[i] = shotinfo;
            }

            // 実験実行
            runSimpleExperiment();
            return 0;
        }

        // クラスタリング検証実験モード
        if (validate_clustering_mode) {
            std::cout << "Running clustering validation mode..." << std::endl;
            runClusteringValidation();
            return 0;
        }

        // Agreement Experiment モード (Clustered vs AllGrid MCTS)
        if (agreement_experiment_mode) {
            std::cout << "Running Agreement Experiment mode..." << std::endl;

            // 初期化
            g_team = dc::Team::k0;
            g_game_setting.max_end = 1;
            g_game_setting.five_rock_rule = true;
            g_game_setting.thinking_time[0] = std::chrono::seconds(86400);
            g_game_setting.thinking_time[1] = std::chrono::seconds(86400);

            g_simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
            g_simulator_storage = g_simulator->CreateStorage();

            for (size_t i = 0; i < g_players.size(); ++i) {
                g_players[i] = dc::players::PlayerNormalDistFactory().CreatePlayer();
            }

            simWrapper = std::make_shared<SimulatorWrapper>(g_team, g_game_setting);
            simWrapper_allgrid = std::make_shared<SimulatorWrapper>(g_team, g_game_setting);

            // グリッドとショット情報の初期化
            int S = GridSize_M * GridSize_N;
            grid.resize(S);
            shotData.resize(S);
            grid_states.resize(S);
            grid = MakeGrid(GridSize_M, GridSize_N);

            std::cout << "Initializing " << S << " grid shots..." << std::endl;
            for (int i = 0; i < S; ++i) {
                ShotInfo shotinfo = FindShot(grid[i]);
                shotData[i] = shotinfo;
                simWrapper->initialShotData.push_back(shotinfo);
                simWrapper_allgrid->initialShotData.push_back(shotinfo);
                state_to_shot_table[i] = shotinfo;
            }
            std::cout << "Grid initialization complete.\n" << std::endl;

            // 初期状態からグリッド状態を生成 (use proper constructor to initialize scores vector)
            dc::GameState initial_state(g_game_setting);

            std::cout << "Simulating " << S << " grid states from initial state..." << std::endl;
            std::cout << "  Initial state - shot: " << static_cast<int>(initial_state.shot) << ", end: " << static_cast<int>(initial_state.end) << std::endl;
            for (int i = 0; i < S; ++i) {
                std::cout << "  Simulating grid #" << i << "..." << std::flush;
                try {
                    grid_states[i] = simWrapper->run_single_simulation(initial_state, shotData[i]);
                    std::cout << " done" << std::endl;
                } catch (const std::exception& e) {
                    std::cout << " ERROR: " << e.what() << std::endl;
                    throw;
                }
            }
            std::cout << "Grid states simulation complete.\n" << std::endl;

            // Agreement Experiment作成
            AgreementExperiment experiment(
                g_team,
                g_game_setting,
                GridSize_M,
                GridSize_N,
                grid_states,
                state_to_shot_table,
                simWrapper,
                simWrapper_allgrid
            );

            // 実験実行（各パターンタイプにつき1つのバリエーション）
            experiment.runExperiment(1);

            // 結果保存用ディレクトリ作成
            std::filesystem::create_directories("experiments/agreement_results");

            // タイムスタンプ付きファイル名を生成
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::tm tm_now;
            #ifdef _WIN32
            localtime_s(&tm_now, &time_t_now);
            #else
            localtime_r(&time_t_now, &tm_now);
            #endif
            std::ostringstream timestamp_stream;
            timestamp_stream << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
            std::string timestamp = timestamp_stream.str();

            std::string filename = "experiments/agreement_results/clustered_vs_allgrid_" + timestamp + ".csv";
            experiment.exportResultsToCSV(filename);

            std::cout << "\n========================================\n";
            std::cout << "Agreement Experiment Complete!\n";
            std::cout << "Results saved to: " << filename << "\n";
            std::cout << "========================================\n";

            return 0;
        }

        if (argc != 3) {
            std::cerr << "Usage: command <host> <port>" << std::endl;
            std::cerr << "       command --experiment              (for efficiency experiment)" << std::endl;
            std::cerr << "       command --validate-clustering     (for clustering validation)" << std::endl;
            std::cerr << "       command --agreement-experiment    (for Clustered vs AllGrid comparison)" << std::endl;
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