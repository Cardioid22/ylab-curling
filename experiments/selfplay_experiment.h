#pragma once
#ifndef _SELFPLAY_EXPERIMENT_H_
#define _SELFPLAY_EXPERIMENT_H_

// 自己対戦ハーネス (A7 の"真の強さ"のバイアスフリー検証)
//   team0 = method A, team1 = method B で単一エンドを打ち切り、最終エンドスコアで勝敗を測る。
//   ハンマー(最終手=team1)の有利を打ち消すため、ゲームごとに team0/team1 の担当を入れ替え、
//   常に「A 視点の純得点」を集計する。regret のような指標バイアスと無関係な勝率評価。

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include "reinvest_experiment.h"

#include <cstdint>
#include <string>

namespace dc = digitalcurling3;

struct SelfPlayConfig {
    ReinvestConfig arm_a;         // 手法A (先に置く方)
    ReinvestConfig arm_b;         // 手法B
    int n_games = 100;            // 対戦エンド数 (ハンマー入替のため偶数推奨)
    int num_threads = 8;
    uint64_t seed = 42;
    std::string label_a = "A";
    std::string label_b = "B";
    std::string output_dir;
};

class SelfPlayExperiment {
public:
    SelfPlayExperiment(const dc::GameSetting& game_setting, const SelfPlayConfig& config);
    void run();

private:
    dc::GameSetting game_setting_;
    SelfPlayConfig config_;

    // 単一エンドを e0(team0)/e1(team1) で打ち切り、team0 の純得点(= s0 - s1)を返す
    double playOneEnd(ReinvestExperiment& e0, ReinvestExperiment& e1,
                      SimulatorWrapper& sim, ShotGenerator& gen,
                      std::mt19937& rng, uint64_t game_seed);
};

#endif  // _SELFPLAY_EXPERIMENT_H_
