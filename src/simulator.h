#pragma once
#ifndef _SIMULATORWRAPPER_H_
#define _SIMULATORWRAPPER_H_
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"

// Forward declarations
class RolloutPolicy;
class ShotGenerator;

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
    std::array<std::unique_ptr<dc::IPlayer>, 4> g_players;
    std::vector<ShotInfo> initialShotData;

    // Limit rollout depth: -1 = full game, N = max N shots per rollout sim.
    // Set to a small value (e.g. 16) for fast tournament play.
    int max_rollout_shots = -1;

    // Deterministic mode (PlayerIdentical, no noise). Must be set before initialize
    // or via setDeterministic() which reinitializes players.
    bool deterministic_ = false;
    void setDeterministic(bool d);

    dc::GameState run_single_simulation(dc::GameState const& game_state, const ShotInfo& shot);
    double run_simulations(dc::GameState const& game_state, const ShotInfo& shot);  // return win/loss (deprecated - use rollout methods)
    dc::GameState run_full_simulations(dc::GameState const& state, const ShotInfo& shot);

    // Rollout policies for ground truth oracle
    double run_grid_rollout(dc::GameState const& state);  // Grid random rollout
    double run_greedy_rollout(dc::GameState const& state, double epsilon = 0.3);  // ε-greedy rollout

    // Multiple simulations with random grid policy for MCTS rollout
    double run_multiple_simulations_with_random_policy(dc::GameState const& state, const ShotInfo& first_shot, int num_simulations);

    // gPolicy を使ったロールアウト (ShotGenerator + gPolicy でショット選択)
    // エンド終了まで実行し、評価値を返す
    double run_policy_rollout(
        dc::GameState const& state,
        const ShotInfo& first_shot,
        RolloutPolicy& policy,
        ShotGenerator& shot_gen,
        int num_simulations
    );

    float evaluate(dc::GameState& game_state) const;
    dc::Vector2 EstimateShotVelocityFCV1(dc::Vector2 const& target_position, float target_speed, dc::moves::Shot::Rotation rotation);
    ShotInfo FindShot(Position const& pos);
private:
    std::unique_ptr<dc::ISimulator> g_simulator_;
    void initialize();

    // Helper function for greedy rollout
    ShotInfo select_best_shot_by_evaluation(const dc::GameState& state, const std::vector<ShotInfo>& candidates);
};

#endif