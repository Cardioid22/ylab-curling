#pragma once
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"

namespace dc = digitalcurling3;

class SimulatorWrapper {
public:
    SimulatorWrapper();
    dc::Team g_team;
    dc::GameSetting g_game_setting;
    void run_single_simulation(dc::GameState& state, const ShotInfo& shot);
    double run_simulation(dc::GameState& state, const ShotInfo& shot);  // return win/loss
    float evaluate(dc::GameState& state);
    dc::Vector2 SimulatorWrapper::EstimateShotVelocityFCV1(dc::Vector2 const& target_position, float target_speed, dc::moves::Shot::Rotation rotation);
    ShotInfo FindShot(Position const& pos);
private:
    std::unique_ptr<dc::ISimulator> simulator;
    std::array<std::unique_ptr<dc::IPlayer>, 4> players;

    void initialize();
};
