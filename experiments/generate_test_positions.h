#pragma once
#ifndef _GENERATE_TEST_POSITIONS_H_
#define _GENERATE_TEST_POSITIONS_H_

// テスト盤面生成実験
// gPolicyで自己対戦し、各対戦からランダムに1局面を抽出してCSV出力
// 序盤数手をランダム化することで盤面の多様性を確保

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include "../src/policy.h"
#include <vector>
#include <string>
#include <random>

namespace dc = digitalcurling3;

class GenerateTestPositionsExperiment {
public:
    GenerateTestPositionsExperiment(dc::GameSetting const& game_setting);

    // 実験パラメータ
    void setTotalGames(int n) { total_games_ = n; }
    void setBatchSize(int b) { batch_size_ = b; }
    void setOpeningRandom(int k) { opening_random_ = k; }

    void run();

private:
    dc::GameSetting game_setting_;
    int total_games_ = 10000;
    int batch_size_ = 1000;
    int opening_random_ = 3;  // 最初のN手をランダム化

    std::unique_ptr<RolloutPolicy> policy_;
    std::unique_ptr<ShotGenerator> shot_gen_;
    std::mt19937 rng_;

    // 1対戦から1ランダム局面を抽出
    struct PositionRecord {
        int match_id;
        int end;
        int shot_num;
        int current_team;
        dc::GameState state;
    };

    // 1ゲーム分を自己対戦して、全局面を返す
    std::vector<PositionRecord> playOneGame(int match_id, SimulatorWrapper& sim);

    // バッチをCSVに書き出す
    void exportBatch(const std::vector<PositionRecord>& positions,
                     const std::string& dir, int batch_num);

    // タイムスタンプ付きディレクトリ名を生成
    std::string makeOutputDir();
};

#endif
