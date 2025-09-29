#pragma once
#ifndef _GROUND_TRUTH_FINDER_H_
#define _GROUND_TRUTH_FINDER_H_

#include "../src/structure.h"
#include "../src/mcts.h"
#include "../src/simulator.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include <vector>
#include <memory>

namespace dc = digitalcurling3;

class GroundTruthFinder {
public:
    GroundTruthFinder(
        std::vector<dc::GameState> grid_states,
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        int gridM,
        int gridN
    );

    ShotInfo findGroundTruthByExtensiveSearch(
        const dc::GameState& state,
        int max_iterations = 100000
    );

    ShotInfo setGroundTruthManually(
        const dc::GameState& state,
        const std::vector<ShotInfo>& expert_shots
    );

    ShotInfo findGroundTruthByConsensus(
        const dc::GameState& state
    );

private:
    std::vector<dc::GameState> grid_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;
    int GridSize_M_;
    int GridSize_N_;

    ShotInfo runExhaustiveSearch(const dc::GameState& state);
    std::vector<ShotInfo> runMultipleMCTS(const dc::GameState& state, int num_runs = 10);
    bool validateGroundTruth(const dc::GameState& state, const ShotInfo& ground_truth);
};

#endif // _GROUND_TRUTH_FINDER_H_