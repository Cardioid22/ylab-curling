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
#include "../src/policy.h"
#include "reinvest_experiment.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dc = digitalcurling3;

struct SelfPlayConfig {
    ReinvestConfig arm_a;         // 手法A (team0 側; ゲーム毎に team を入替える)
    ReinvestConfig arm_b;         // 手法B
    bool a_is_ayumu = false;      // A が歩(gPolicyポリシープレイヤー)か
    bool b_is_ayumu = false;      // B が歩か
    int n_games = 100;            // 対戦ゲーム数 (先攻後攻入替のため偶数推奨)
    int n_ends = 1;               // 1ゲームのエンド数 (2 で2エンド戦)
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

    // n_ends エンドを e0(team0)/e1(team1) で打ち切り、team0 視点の各エンド純得点を返す
    // (長さ n_ends)。手番はハンマー依存の GetNextTeam() で決める。
    // t0_ayumu/t1_ayumu が真なら その手番は歩(policy) が打つ。
    std::vector<double> playGame(ReinvestExperiment& e0, ReinvestExperiment& e1,
                                 RolloutPolicy* policy, bool t0_ayumu, bool t1_ayumu,
                                 SimulatorWrapper& sim, ShotGenerator& gen,
                                 std::mt19937& rng, uint64_t game_seed, int n_ends);
};

#endif  // _SELFPLAY_EXPERIMENT_H_
