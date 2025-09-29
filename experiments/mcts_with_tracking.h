#pragma once
#ifndef _MCTS_WITH_TRACKING_H_
#define _MCTS_WITH_TRACKING_H_

#include "../src/mcts.h"
#include "../src/structure.h"
#include "../src/simulator.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include <vector>
#include <memory>
#include <chrono>

namespace dc = digitalcurling3;

class MCTS_WithTracking {
private:
    ShotInfo ground_truth_shot_;
    std::vector<int> discovery_iterations_;
    std::vector<double> evaluation_scores_;

    std::unique_ptr<MCTS> mcts_;
    std::vector<dc::GameState> all_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;

public:
    struct SearchResult {
        int first_discovery_iteration;
        int best_evaluation_iteration;
        int convergence_iteration;
        std::vector<double> score_history;
        bool success;
        double final_score;
    };

    MCTS_WithTracking(
        dc::GameState const& root_state,
        NodeSource node_source,
        std::vector<dc::GameState> states,
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        std::shared_ptr<SimulatorWrapper> simWrapper,
        int gridM,
        int gridN
    );

    void setGroundTruth(const ShotInfo& ground_truth);
    SearchResult trackGroundTruthDiscovery(int max_iter, double max_time);

private:
    bool isGroundTruthShot(const ShotInfo& shot, double tolerance = 1e-6) const;
    void recordGroundTruthEvaluation(int iteration, double score);
    double getScoreForShot(const ShotInfo& shot) const;
    bool isShotInCandidates(const ShotInfo& shot) const;
    bool isBestShot(const ShotInfo& shot) const;
    void performOneMCTSIteration();
    double convergence_threshold_;
};

#endif // _MCTS_WITH_TRACKING_H_