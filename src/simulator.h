#pragma once
#ifndef _SIMULATORWRAPPER_H_
#define _SIMULATORWRAPPER_H_
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"

namespace dc = digitalcurling3;

class SimulatorWrapper {
public:
    SimulatorWrapper(
        dc::Team team,
        dc::GameSetting const& game_setting
    );

    dc::Team g_team;
    dc::GameSetting g_game_setting;
    std::unique_ptr<dc::ISimulatorStorage> g_simulator_storage_;
    std::array<std::unique_ptr<dc::IPlayer>, 4> g_players; // ゲームプレイヤー
    std::vector<ShotInfo> initialShotData;

    dc::GameState run_single_simulation(dc::GameState const& game_state, const ShotInfo& shot);
    double run_simulations(dc::GameState const& game_state, const ShotInfo& shot);  // return win/loss
    float evaluate(dc::GameState& game_state) const;
    dc::Vector2 EstimateShotVelocityFCV1(dc::Vector2 const& target_position, float target_speed, dc::moves::Shot::Rotation rotation);
    ShotInfo FindShot(Position const& pos);
private:
    std::unique_ptr<dc::ISimulator> g_simulator_;
    void initialize();
};

#endif