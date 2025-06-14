#include "simulator.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include "game_context.h"
#include <iostream>

namespace dc = digitalcurling3;

SimulatorWrapper::SimulatorWrapper(
    dc::Team team,
    dc::GameSetting const& game_setting
) : g_team(team), g_game_setting(game_setting) 
{
    initialize();
}

void SimulatorWrapper::initialize(
) 
{
    g_simulator_ = dc::simulators::SimulatorFCV1Factory().CreateSimulator();

    g_simulator_storage_ = g_simulator_->CreateStorage();

    // プレイヤーを生成する
    // 非対応の場合は NormalDistプレイヤーを使用する．
    for (size_t i = 0; i < g_players.size(); ++i) {
        g_players[i] = dc::players::PlayerNormalDistFactory().CreatePlayer();
    }
}

float SimulatorWrapper::evaluate(dc::GameState& state) const {
    dc::Team o_team = dc::GetOpponentTeam(g_team);
    if (state.IsGameOver()) {
        int my_team_score = state.GetTotalScore(g_team);
        int op_team_score = state.GameState::GetTotalScore(o_team);
        return my_team_score - op_team_score;
    }
    else return 0;
}

void SimulatorWrapper::run_single_simulation(dc::GameState const& state, const ShotInfo& shot) {
    std::cout << "Single Run Simulation Begin.\n";
    if (!g_simulator_ || !g_simulator_storage_) throw std::runtime_error("Simulator or storage not initialized");
    g_simulator_->Load(*g_simulator_storage_);
    dc::GameState sim_state = state;
    auto& current_player = *g_players[sim_state.shot / 4];
    if (!&current_player) {
        std::cout << "Player is null.\n";
    }
    dc::Vector2 velocity(shot.vx, shot.vy);
    auto rot = shot.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    dc::moves::Shot shot_move{ velocity, rot };
    dc::Move move{ shot_move };
    std::cout << "debug check1\n";
    dc::ApplyMove(g_game_setting, *g_simulator_, current_player, sim_state, move, std::chrono::milliseconds(0));
    std::cout << "debug check2\n";
    g_simulator_->Save(*g_simulator_storage_);
    std::cout << "Single Run Simulation Done.\n";
}

double SimulatorWrapper::run_simulation(dc::GameState const& state, const ShotInfo& shot) {
    std::cout << "Multi Run Simulation Begin.\n";
    dc::GameState sim_state = state;  // Copy state
    for (int i = sim_state.shot; i < 15; ++i) {
        g_simulator_->Load(*g_simulator_storage_);
        auto& current_player = *g_players[sim_state.shot / 4];
        if (!&current_player) {
            std::cout << "Player is null.\n";
        }
        dc::Vector2 velocity(shot.vx, shot.vy);
        auto rot = shot.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
        dc::moves::Shot shot_move{ velocity, rot };
        dc::Move move{ shot_move };
        dc::ApplyMove(g_game_setting, *g_simulator_, current_player, sim_state, move, std::chrono::milliseconds(0));
        g_simulator_->Save(*g_simulator_storage_);

        if (sim_state.IsGameOver()) break;
    }
    std::cout << "Multi Run Simulation Done.\n";
    return evaluate(sim_state);  // You define this: e.g., 1.0 for win, 0.0 for loss
}

dc::Vector2 SimulatorWrapper::EstimateShotVelocityFCV1(dc::Vector2 const& target_position, float target_speed, dc::moves::Shot::Rotation rotation)
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

ShotInfo SimulatorWrapper::FindShot(Position const& pos) {
    std::cout << "Finding Shot...\n";
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