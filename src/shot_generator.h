#pragma once
#ifndef _SHOT_GENERATOR_H_
#define _SHOT_GENERATOR_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include <vector>
#include <string>
#include <memory>

namespace dc = digitalcurling3;

// ショットタイプ: 歩(Ayumu)の Standard enum に対応
enum class ShotType {
    PASS = 0,
    DRAW,       // 指定位置へのドロー
    PREGUARD,   // 事前ガード (ハウス手前)
    HIT,        // 対象石へのヒット
    FREEZE,     // 対象石に密着停止
    PEEL,       // 対象石と投球石を同時排除
    COMEAROUND, // 対象石の裏に回り込む
    POSTGUARD,  // 対象石の手前にガード
    DRAWRAISE,  // 自分石をティーに押し込む
    TAKEOUT,    // ガードを避けてテイクアウト (将来拡張)
    DOUBLE,     // ダブルテイクアウト (将来拡張)
};

// ヒットの強さ: 歩の HitWeight に対応
enum class HitWeight {
    TOUCH = 0,   // 0.3m先まで (ほぼ停止)
    WEAK = 1,    // 1.7m先まで
    MIDDLE = 2,  // 5.0m先まで (確実に弾き出す)
    STRONG = 3,  // 最高速
};

// カムアラウンドの距離
enum class ComeAroundLength {
    SHORT = 0,
    MIDDLE = 1,
    LONG = 2,
};

// ポストガードの距離
enum class PostGuardLength {
    SHORT = 0,
    MIDDLE = 1,
    LONG = 2,
};

// DrawPos: 歩の名前付きドロー位置 (16箇所)
enum class DrawPos {
    TEE = 0,
    S0, S2, S4, S6,
    L0, L1, L2, L3, L4, L5, L6, L7,
    G3, G4, G5,
    MAX = 16,
};

// 拡張候補手: 抽象手ラベル + 物理速度
struct CandidateShot {
    ShotType type;
    int spin;            // 1: CW, 0: CCW
    int target_index;    // 対象石のindex (-1 = なし)
    int param;           // HitWeight, ComeAroundLength等 (-1 = なし)
    ShotInfo shot;       // 物理速度 (vx, vy, rot)
    std::string label;   // デバッグ用ラベル ("Hit(R,3,MIDDLE)" 等)
};

// 候補手プール生成結果
struct CandidatePool {
    std::vector<CandidateShot> candidates;
    std::vector<dc::GameState> result_states;  // 各候補手のシミュレーション結果盤面
};

// 拡張候補手プール生成器
class ShotGenerator {
public:
    ShotGenerator(dc::GameSetting const& game_setting);

    // 候補手プールを生成 (シミュレーション結果付き)
    CandidatePool generatePool(
        const dc::GameState& state,
        dc::Team my_team,
        const std::vector<Position>& grid_positions
    );

    // 候補手のみ生成 (シミュレーションなし)
    std::vector<CandidateShot> generateCandidates(
        const dc::GameState& state,
        dc::Team my_team,
        const std::vector<Position>& grid_positions
    );

    // ノイズなしシミュレーション (歩の makeMoveNoRand 相当)
    dc::GameState simulateNoRand(
        const dc::GameState& state,
        const ShotInfo& shot
    );

private:
    dc::GameSetting game_setting_;
    std::unique_ptr<dc::ISimulator> simulator_;
    std::unique_ptr<dc::ISimulatorStorage> simulator_storage_;
    std::unique_ptr<dc::IPlayer> player_no_rand_;  // PlayerIdentical (ノイズなし)

    // DrawPos座標を取得
    Position getDrawPosition(DrawPos pos) const;

    // ショット速度計算
    ShotInfo calcDrawShot(const Position& target, int spin) const;
    ShotInfo calcHitShot(const Position& stone_pos, HitWeight weight, int spin) const;
    ShotInfo calcFreezeShot(const Position& stone_pos, int spin) const;
    ShotInfo calcPeelShot(const Position& stone_pos, int spin) const;
    ShotInfo calcComeAroundShot(const Position& stone_pos, ComeAroundLength length, int spin) const;
    ShotInfo calcPostGuardShot(const Position& stone_pos, PostGuardLength length, int spin) const;

    // EstimateShotVelocityFCV1 (逆運動学)
    dc::Vector2 estimateVelocity(dc::Vector2 const& target_position, float target_speed, dc::moves::Shot::Rotation rotation) const;

    // 指定位置を通過してさらにremaining_distance進む速度を計算
    // 歩の rotateToPassPointGoF に相当
    dc::Vector2 estimatePassThroughVelocity(dc::Vector2 const& pass_point, float remaining_distance, dc::moves::Shot::Rotation rotation) const;

    // 石の位置を取得するヘルパー
    Position getStonePosition(const dc::GameState& state, int stone_index) const;

    // 文字列変換
    static std::string shotTypeToString(ShotType type);
    static std::string hitWeightToString(HitWeight w);
};

#endif
