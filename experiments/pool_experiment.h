#pragma once
#ifndef _POOL_EXPERIMENT_H_
#define _POOL_EXPERIMENT_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include <vector>
#include <string>

namespace dc = digitalcurling3;

class PoolExperiment {
public:
    PoolExperiment(dc::GameSetting const& game_setting);

    // テスト盤面を生成して拡張プールを構築・出力
    void runPoolGeneration();

    // プール生成結果をCSVに出力
    void exportPoolToCSV(
        const CandidatePool& pool,
        const std::string& filename
    );

    // グリッド位置を生成
    std::vector<Position> makeGrid(int m, int n);

private:
    dc::GameSetting game_setting_;

    // テスト盤面を生成
    std::vector<dc::GameState> createTestStates();
};

#endif
