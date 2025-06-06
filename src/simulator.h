#pragma once
#include "digitalcurling3/digitalcurling3.hpp"

class SimulatorWrapper {
public:
    SimulatorWrapper();
    double run_simulation(const dc::GameState& state, const ShotInfo& shot);  // return win/loss

private:
    std::unique_ptr<dc::ISimulator> simulator;
    std::array<std::unique_ptr<dc::IPlayer>, 4> players;

    void initialize();
};
