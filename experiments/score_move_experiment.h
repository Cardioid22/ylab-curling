#pragma once
#ifndef _SCORE_MOVE_EXPERIMENT_H_
#define _SCORE_MOVE_EXPERIMENT_H_

// 審判 (referee): iso-budget 比較で「どちらが強い手を見つけたか」を判定するための
// 共通評価器。各候補手を「その手を着手 → エンド終了まで K 回ロールアウト」して
// 平均スコアで採点する。どの探索手法が選んだ手でも同じ物差しで質を測れる。
//
// depth_n_mcts_experiment と同じ rolloutFromState / 4x4 グリッドポリシーを使うので、
// 探索の葉評価と内部的に一貫した「手の真の価値」の低分散推定になる。
// K を大きくすれば AllGrid@少playouts のような自己不一致を避けられる。
//
// 出力 CSV (全候補の Q テーブル) を後段 Python で各手法の選択 index と突き合わせ、
// regret / head-to-head 勝率 / 順位 / Ω*(ε) を計算する。

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include "mcts_shared.h"

#include <cstdint>
#include <string>
#include <vector>

namespace dc = digitalcurling3;

struct ScoreMoveConfig {
    int n_states = 10;            // テスト局面数
    int score_rollouts = 500;     // 候補1手あたりのロールアウト回数 K
    bool resample_first_shot = true;  // true: 初手の実行ノイズも毎回振り直す (実行不確実性込みの価値)
                                      // false: プール生成時の1回の着地で固定 (探索木と同じ規約)
    double epsilon = 0.3;         // ロールアウトの ε-greedy 確率 (実験と一致させる)
    int num_threads = 8;
    uint64_t seed = 42;           // ロールアウト RNG のベースシード
    int start_index = 0;          // 並列実行用スライス (サンプリング後リストの開始 index)
    int max_positions = -1;       // 担当局面数 (-1 = 全部)
    std::string load_positions_dir;
    std::string output_dir;
};

// 候補1手分の採点結果
struct ScoredCandidate {
    int game_id = -1;
    int end = -1;
    int shot_num = -1;
    int candidate_idx = -1;       // generatePool() の候補 index (実験 CSV の proposed_idx/allgrid_idx と整合)
    std::string label;
    int shot_type = -1;           // ShotType を int 化したもの
    double q_ref_mean = 0.0;      // K 回ロールアウト平均 (root_team 視点)
    double q_ref_sd = 0.0;        // K 回のばらつき
    int n_rollouts = 0;
    int resampled = 1;            // 1: 初手ノイズ再サンプルあり / 0: 固定着地
};

class ScoreMoveExperiment {
public:
    ScoreMoveExperiment(const dc::GameSetting& game_setting, const ScoreMoveConfig& config);
    void run();

private:
    dc::GameSetting game_setting_;
    ScoreMoveConfig config_;

    void writeCSV(const std::vector<ScoredCandidate>& rows, const std::string& path) const;
};

#endif  // _SCORE_MOVE_EXPERIMENT_H_
