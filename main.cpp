#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

namespace {

    struct Position {
        float x;
        float y;
    };
    struct ShotInfo {
        float vx;
        float vy;
        uint8_t rotation;
    };

dc::Team g_team;  // 自身のチームID

const auto HouseRadius = 1.829;
const auto AreaMaxX = 2.375;
const auto AreaMaxY = 40.234;
const auto HouseCenterX = 0;
const auto HouseCenterY = 38.405;
const int GridSize_N = 3; // columns
const int GridSize_M = 3; // rows

std::vector<std::vector<Position>> grid(GridSize_M + 1, std::vector<Position>(GridSize_N + 1));
std::vector<std::vector<ShotInfo>> shotData(GridSize_M + 1, std::vector<ShotInfo>(GridSize_N + 1));

std::vector<std::vector<Position>> MakeGrid(const int m, const int n) {
    float x_grid = 2 * AreaMaxX / m;
    float y_grid = 2 * HouseRadius / n;
    Position pos;
    std::vector<std::vector<Position>> result(GridSize_M + 1, std::vector<Position>(GridSize_N + 1));
    for (int i = 0; i <= m; i++) {
        float y = AreaMaxY - i * y_grid;
        for (int j = 0; j <= n; j++) {
            float x = -AreaMaxX + j * x_grid;
            pos.x = x;
            pos.y = y;
            result[i][j] = pos;
        }
    }
    return result;
}

dc::Vector2 SimulateShot(float vx, float vy, dc::moves::Shot::Rotation rotation) {
    dc::ISimulator::AllStones init_stones;
    init_stones[0].emplace(dc::Vector2(), 0.f, dc::Vector2(vx, vy), rotation == dc::moves::Shot::Rotation::kCCW ? 1.57f : -1.57f);
    auto simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
    simulator->SetStones(init_stones);

    while (!simulator->AreAllStonesStopped()) {
        simulator->Step();
    }

    return simulator->GetStones()[0]->position;
}

ShotInfo FindOptimalShot(
    float target_x, float target_y)
{
    auto target = std::make_pair(target_x, target_y);


    float best_error = std::numeric_limits<float>::max();
    std::tuple<float, float, dc::moves::Shot::Rotation> best_shot;
    // ランダムジェネレータの準備
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rotation_dist(0, 1);
    constexpr int Iteration = 80;
    float base_adjustment = 0.1f;  // 基本の補正量
    float min_adjustment = 0.03f;  // 最小の補正量
    float max_adjustment = 0.15f;  // 最大の補正量

    //float adjustment_factor = 0.09f; // 調整係数
    float target_r = std::sqrt(target_x * target_x + target_y * target_y);
    float target_speed = 0.0f;  // ゴール時の目標速度（微調整可能）
    float v0_speed = 1.122 * 2.1f;
    dc::moves::Shot::Rotation rotation = dc::moves::Shot::Rotation::kCW;
    float vx = v0_speed * (target_x / target_r);
    float vy = v0_speed * (target_y / target_r);

    for (int iter = 0; iter < Iteration; ++iter) {
        // 回転方向をランダムに選択
        dc::moves::Shot::Rotation rotation = rotation_dist(gen) == 0
            ? dc::moves::Shot::Rotation::kCW
            : dc::moves::Shot::Rotation::kCCW;
        auto final_position = SimulateShot(vx, vy, rotation);
        float error_x = target_x - final_position.x;
        float error_y = target_y - final_position.y;

        float error = error_x * error_x + error_y * error_y;

        if (error < best_error) {
            best_error = error;
            best_shot = std::make_tuple(vx, vy, rotation);
            // 誤差が許容範囲内なら終了
            if (error < 0.001f) {
                //std::cout << "Found Optimal Shot and error is " << error << ", iteration=" << iter << "\n";
                break;
            }
        }

        // 動的な補正量を計算（誤差が大きいときは大きく、小さいときは細かく）
        float adjustment_factor = std::clamp(base_adjustment * std::sqrt(error), min_adjustment, max_adjustment);

        // 誤差方向の単位ベクトルを計算
        float error_norm = std::sqrt(error);
        float unit_error_x = error_x / error_norm;
        float unit_error_y = error_y / error_norm;

        // 単位ベクトルに基づいて速度を補正
        vx += adjustment_factor * unit_error_x;
        vy += adjustment_factor * unit_error_y;
    }
    ShotInfo shotinfo;
    std::tie(vx, vy, rotation) = best_shot;
    shotinfo.vx = vx;
    shotinfo.vy = vy;
    shotinfo.rotation = static_cast<uint8_t>(rotation);
    return shotinfo;
}

void OnInit(
    dc::Team team,
    dc::GameSetting const& game_setting,
    std::unique_ptr<dc::ISimulatorFactory> simulator_factory,
    std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories,
    std::array<size_t, 4> & player_order)
{
    // TODO AIを作る際はここを編集してください
    g_team = team;
    grid = MakeGrid(GridSize_M, GridSize_N);
}

dc::Move OnMyTurn(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
    for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[i].size(); ++j) {
            //std::cout << "grid[" << i << "][" << j << "] = ("
            //    << grid[i][j].x << ", " << grid[i][j].y << ")\n";
            shotData[i][j] = FindOptimalShot(grid[i][j].x, grid[i][j].y);
        }
    }
    for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[i].size(); ++j) {
            std::cout << "shotData[" << i << "][" << j << "] = ("
                << shotData[i][j].vx << ", " << shotData[i][j].vy << ", " << shotData[i][j].rotation << ")\n";
        }
    }

    dc::moves::Shot shot;

    // ショットの初速
    shot.velocity.x = 0.132f;
    shot.velocity.y = 2.3995f;

    // ショットの回転
    shot.rotation = dc::moves::Shot::Rotation::kCCW; // 反時計回り
    // shot.rotation = dc::moves::Shot::Rotation::kCW; // 時計回り

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