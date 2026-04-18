#pragma once
#ifndef _POLICY_H_
#define _POLICY_H_

// 大渡さんの歩(Ayumu) gPolicy の移植
// 線形ソフトマックスポリシー関数
// MCTSロールアウト中の手選択に使用
//
// 参考: 大渡勝己, 田中哲朗. "モンテカルロ木探索によるデジタルカーリングAI" GPW 2016
// ソース: https://github.com/u-tokyo-gps-tanaka-lab/gpw2016

#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include "shot_generator.h"
#include <vector>
#include <string>
#include <random>
#include <cmath>

namespace dc = digitalcurling3;

// Ayumu StoneInfo の軽量版 (DC3座標系用)
struct StoneInfoDC3 {
    float x = 0, y = 0;        // Ayumu座標 (ティー原点)
    float r = 0, th = 0;       // ティーからの極座標
    float tr = 0, tth = 0;     // 投擲位置からの極座標
    float hitLimit_[2][2] = {}; // [spin:RIGHT=0,LEFT=1][lr:0=left,1=right]

    // DC3座標から初期化
    void setFromDC3(float dc3_x, float dc3_y);

    float hitLimit(int spin, int lr) const { return hitLimit_[spin][lr]; }
};

// 2石間の相対位置情報
struct RelStoneInfoDC3 {
    float r = 0;   // 石間距離
    float th = 0;  // 相対角度
    int rp = -1;   // 相対位置ラベル (0-15, -1=無効)

    void set(const StoneInfoDC3& base, const StoneInfoDC3& obj);
};

// 相対位置ラベル (Ayumu logic.hpp labelRelPos 移植)
int labelRelPos(const StoneInfoDC3& base, const StoneInfoDC3& obj, const RelStoneInfoDC3& rel);

class RolloutPolicy {
public:
    // パラ���ータ定数 (Ayumu policy.hpp より)
    static constexpr int N_STANDARD_MOVES = 13;
    static constexpr int N_TURN_INDICES = 6;
    static constexpr int N_ZONES = 16;

    // 特徴グループごとのパラメータ数
    static constexpr int NUM_POL_1STONE = N_TURN_INDICES * N_STANDARD_MOVES * N_ZONES;  // 1248
    static constexpr int NUM_POL_2STONES = N_TURN_INDICES * N_STANDARD_MOVES * 2 * N_ZONES * N_ZONES;  // 39936
    static constexpr int NUM_POL_2SMOVE = N_TURN_INDICES * 2 * 2 * 17;  // 408
    static constexpr int NUM_POL_END_SCORE = N_TURN_INDICES * 2 * 5 * 4;  // 240
    static constexpr int NUM_POL_2ND_SCORE = N_TURN_INDICES * 2 * 5 * 4;  // 240
    static constexpr int N_PARAMS_ALL = 42072;

    // 特徴グループの開始インデックス
    static constexpr int IDX_1STONE = 0;
    static constexpr int IDX_2STONES = IDX_1STONE + NUM_POL_1STONE;       // 1248
    static constexpr int IDX_2SMOVE = IDX_2STONES + NUM_POL_2STONES;      // 41184
    static constexpr int IDX_END_SCORE = IDX_2SMOVE + NUM_POL_2SMOVE;     // 41592
    static constexpr int IDX_2ND_SCORE = IDX_END_SCORE + NUM_POL_END_SCORE; // 41832

    // Ayumu の Standard ショットタイプ (ayumu_dc.hpp)
    // ylab-curling の ShotType と対応
    static constexpr int AYUMU_PASS = 0;
    static constexpr int AYUMU_DRAW = 1;
    static constexpr int AYUMU_PREGUARD = 2;
    static constexpr int AYUMU_L1DRAW = 3;
    static constexpr int AYUMU_FREEZE = 4;
    static constexpr int AYUMU_COMEAROUND = 5;
    static constexpr int AYUMU_POSTGUARD = 6;
    static constexpr int AYUMU_HIT = 7;
    static constexpr int AYUMU_PEEL = 8;
    static constexpr int AYUMU_DRAWRAISE = 9;
    static constexpr int AYUMU_TAKEOUT = 10;
    static constexpr int AYUMU_DOUBLE = 11;
    static constexpr int AYUMU_RAISETAKEOUT = 12;

    // ショットタイプのカテゴリ境界 (IDX_TYPE_MOVE)
    static constexpr int TYPE_BOUNDARY_WHOLE = 1;  // DRAW, PREGUARD, L1DRAW
    static constexpr int TYPE_BOUNDARY_ONE = 4;     // FREEZE...TAKEOUT
    static constexpr int TYPE_BOUNDARY_TWO = 11;    // DOUBLE, RAISETAKEOUT

    // 物理定数 (Ayumu の dc.hpp と同じ値)
    static constexpr float FR_HOUSE_RAD = 1.83f;
    static constexpr float FR_STONE_RAD = 0.145f;
    static constexpr float FR_ACTIVE_LINE = FR_HOUSE_RAD + 3 * FR_STONE_RAD;  // 2.265
    static constexpr float HOUSE_PLUS_STONE = FR_HOUSE_RAD + FR_STONE_RAD;    // 1.975

    // DC3 座標系の定数
    static constexpr float DC3_TEE_Y = 38.405f;
    static constexpr float DC3_THROW_Y = 2.005f;  // 38.405 - 36.4

    RolloutPolicy();

    // パラメータファイルを読み込む
    bool load(const std::string& param_path);

    // 候補手の中から1手を確率的に選択
    // shot_num: 0-15 (0-indexed), my_team: 現在のチーム
    int selectShot(
        const dc::GameState& state,
        const std::vector<CandidateShot>& candidates,
        int shot_num,
        dc::Team my_team,
        int end_index,      // 現在のエンド番号
        int rel_score        // 自チーム - 相手チームのスコア差
    );

    // 候補手のスコアを計算 (デバッグ用)
    std::vector<double> scoreCandidates(
        const dc::GameState& state,
        const std::vector<CandidateShot>& candidates,
        int shot_num,
        dc::Team my_team,
        int end_index,
        int rel_score
    );

    bool isLoaded() const { return loaded_; }

    void setTemperature(double t) { temperature_ = t; }
    double temperature() const { return temperature_; }

    // deterministic=true: argmax 選択 (再現可能)
    // deterministic=false: softmax サンプリング (本番用)
    void setDeterministic(bool d) { deterministic_ = d; }
    bool isDeterministic() const { return deterministic_; }

private:
    double params_[N_PARAMS_ALL];
    double temperature_ = 0.8;
    bool loaded_ = false;
    bool deterministic_ = false;
    std::mt19937 rng_;

    // ShotType → Ayumu Standard type index
    int toAyumuType(ShotType type) const;

    // Ayumu type → カテゴリ (0=NONE, 1=WHOLE, 2=ONE, 3=TWO)
    int toTypeCategory(int ayumu_type) const;

    // DC3 shot_num (0-15) → Ayumu turn index (0-5)
    int toTurnIndex(int shot_num) const;

    // DC3 座標 → Ayumu 16ゾーン離散化
    int getPositionIndex(float dc3_x, float dc3_y) const;

    // 盤面のスコアを計算 (No.1石のチームの得点数)
    int countScore(const dc::GameState& state, dc::Team perspective_team) const;

    // clean (ブランクエンドが有利か) の判定
    bool isCleanBetter(bool is_second, int rel_score) const;

    // ドロー位置情報 (Ayumu drawPositionInfo 相当, type 0-12)
    StoneInfoDC3 drawPosInfo_[N_STANDARD_MOVES];
    bool drawPosInitialized_ = false;
    void initDrawPositionInfo();

    // 盤面上の石情報をStoneInfoDC3として構築
    struct BoardStoneInfo {
        StoneInfoDC3 info;
        int team;     // 0 or 1
        int index;    // team*8 + stone_idx (0-15)
    };
    std::vector<BoardStoneInfo> buildBoardStones(const dc::GameState& state) const;
};

#endif
