// SimulatorWrapper.cpp
#include "simulator.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"

SimulatorWrapper::SimulatorWrapper() {
    initialize();
}

void SimulatorWrapper::initialize() {
    simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
    for (auto& player : players) {
        player = dc::players::PlayerNormalDistFactory().CreatePlayer();
    }
}

double SimulatorWrapper::run_simulation(const GameState& state, const ShotInfo& shot) {
    GameState sim_state = state;  // Copy state
    for (int i = sim_state.shot; i < 15; ++i) {
        auto& current_player = *players[sim_state.shot / 4];
        dc::Vector2 velocity(shot.vx, shot.vy);
        auto rot = shot.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
        dc::moves::Shot shot_move{ velocity, rot };
        dc::Move move{ shot_move };
        dc::ApplyMove(g_game_setting, *simulator, current_player, sim_state, move, std::chrono::milliseconds(0));

        if (sim_state.IsGameOver()) break;
    }

    return evaluate(sim_state);  // You define this: e.g., 1.0 for win, 0.0 for loss
}
