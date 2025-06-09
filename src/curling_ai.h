#pragma once
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include "simulator.h"
#include "clustering.h"
#include "mcts.h"

class CurlingAI {
public:
    CurlingAI(dc::Team team,
        const dc::GameSetting& game_setting,
        std::unique_ptr<dc::ISimulator> simulator,
        std::array<std::unique_ptr<dc::IPlayer>, 4> players);

    void Initialize();
    dc::Move DecideMove(const dc::GameState& game_state);

private:
    dc::Team team_;
    dc::GameSetting game_setting_;
    std::unique_ptr<dc::ISimulator> simulator_;
    std::array<std::unique_ptr<dc::IPlayer>, 4> players_;

    std::vector<std::vector<Position>> grid_;
    std::vector<std::vector<ShotInfo>> shotData_;
    std::vector<dc::GameState> grid_states_;
};
