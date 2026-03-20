#include "shot_generator.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <cassert>

namespace dc = digitalcurling3;

// カーリング定数
static constexpr float kHouseCenterX = 0.0f;
static constexpr float kHouseCenterY = 38.405f;
static constexpr float kHouseRadius = 1.829f;
static constexpr float kStoneRadius = 0.145f;  // DC3の石の半径
static constexpr float kThrowY = 0.0f;         // 投擲点のY座標 (DC3基準)
static constexpr float kThrowX = 0.0f;         // 投擲点のX座標

// カムアラウンドの距離テーブル (歩の standardComeAroundLength に対応)
static constexpr float kComeAroundLength[] = { 1.0f, 2.2f, 3.5f };

// ポストガードの距離テーブル (歩の standardPostGuardLength に対応、手前方向なので負)
static constexpr float kPostGuardLength[] = { 1.0f, 2.0f, 3.0f };

// ヒットの残り距離テーブル (歩の rotateToPassPointGoF のパラメータ)
static constexpr float kHitRemainingDist[] = {
    0.3f,   // TOUCH
    1.7f,   // WEAK
    5.0f,   // MIDDLE
    -1.0f,  // STRONG (最高速: 別計算)
};

// ピールの速度 (歩: FV_MAX - 0.3 ≈ 高速)
static constexpr float kPeelSpeed = 3.5f;

ShotGenerator::ShotGenerator(dc::GameSetting const& game_setting)
    : game_setting_(game_setting)
{
    simulator_ = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
    simulator_storage_ = simulator_->CreateStorage();
    // ノイズなしプレイヤー (歩の makeMoveNoRand に相当)
    player_no_rand_ = dc::players::PlayerIdenticalFactory().CreatePlayer();
}

// ========== DrawPos座標 ==========

Position ShotGenerator::getDrawPosition(DrawPos pos) const {
    const float r_s = 0.9f;
    const float r_l = kHouseRadius - 2.0f * kStoneRadius;
    const float sqrt2_inv = 1.0f / std::sqrt(2.0f);

    switch (pos) {
        case DrawPos::TEE: return { kHouseCenterX, kHouseCenterY };
        case DrawPos::S0:  return { kHouseCenterX, kHouseCenterY + 4.0f * kStoneRadius };
        case DrawPos::S2:  return { kHouseCenterX + r_s, kHouseCenterY };
        case DrawPos::S4:  return { kHouseCenterX, kHouseCenterY + 2.0f * kStoneRadius };
        case DrawPos::S6:  return { kHouseCenterX - r_s, kHouseCenterY };
        case DrawPos::L0:  return { kHouseCenterX, kHouseCenterY + r_l };
        case DrawPos::L1:  return { kHouseCenterX + r_l * sqrt2_inv, kHouseCenterY + r_l * sqrt2_inv };
        case DrawPos::L2:  return { kHouseCenterX + r_l, kHouseCenterY };
        case DrawPos::L3:  return { kHouseCenterX + r_l * sqrt2_inv, kHouseCenterY - r_l * sqrt2_inv };
        case DrawPos::L4:  return { kHouseCenterX, kHouseCenterY - kHouseRadius };
        case DrawPos::L5:  return { kHouseCenterX - r_l * sqrt2_inv, kHouseCenterY - r_l * sqrt2_inv };
        case DrawPos::L6:  return { kHouseCenterX - r_l, kHouseCenterY };
        case DrawPos::L7:  return { kHouseCenterX - r_l * sqrt2_inv, kHouseCenterY + r_l * sqrt2_inv };
        case DrawPos::G3:  return { kHouseCenterX + 0.6f, kHouseCenterY - 2.17f };
        case DrawPos::G4:  return { kHouseCenterX, kHouseCenterY - 3.0f };
        case DrawPos::G5:  return { kHouseCenterX - 0.6f, kHouseCenterY - 2.17f };
        default: return { kHouseCenterX, kHouseCenterY };
    }
}

// ========== 逆運動学 ==========

dc::Vector2 ShotGenerator::estimateVelocity(
    dc::Vector2 const& target_position, float target_speed,
    dc::moves::Shot::Rotation rotation
) const {
    assert(target_speed >= 0.f);
    assert(target_speed <= 4.f);

    float const v0_speed = [&target_position, target_speed] {
        auto const target_r = target_position.Length();
        assert(target_r > 0.f);

        if (target_speed <= 0.05f) {
            float constexpr kC0[] = { 0.0005048122574925176f, 0.2756242531609261f };
            float constexpr kC1[] = { 0.00046669575066030805f, -29.898958358378636f, -0.0014030973174948508f };
            float constexpr kC2[] = { 0.13968687866736632f, 0.41120940058777616f };
            float const c0 = kC0[0] * target_r + kC0[1];
            float const c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
            float const c2 = kC2[0] * target_r + kC2[1];
            return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
        } else if (target_speed <= 1.f) {
            float constexpr kC0[] = { -0.0014309170115803444f, 0.9858457898438147f };
            float constexpr kC1[] = { -0.0008339331735471273f, -29.86751291726946f, -0.19811799977982522f };
            float constexpr kC2[] = { 0.13967323742978f, 0.42816312110477517f };
            float const c0 = kC0[0] * target_r + kC0[1];
            float const c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
            float const c2 = kC2[0] * target_r + kC2[1];
            return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
        } else {
            float constexpr kC0[] = { 1.0833113118071224e-06f, -0.00012132851917870833f, 0.004578093297561233f, 0.9767006869364527f };
            float constexpr kC1[] = { 0.07950648211492622f, -8.228225657195706f, -0.05601306077702578f };
            float constexpr kC2[] = { 0.14140440186382008f, 0.3875782508767419f };
            float const c0 = kC0[0] * target_r * target_r * target_r + kC0[1] * target_r * target_r + kC0[2] * target_r + kC0[3];
            float const c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
            float const c2 = kC2[0] * target_r + kC2[1];
            return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
        }
    }();

    dc::Vector2 const delta = [rotation, v0_speed, target_speed] {
        float const rotation_factor = rotation == dc::moves::Shot::Rotation::kCCW ? 1.f : -1.f;
        thread_local std::unique_ptr<dc::ISimulator> s_simulator;
        if (s_simulator == nullptr) {
            s_simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
        }
        dc::ISimulator::AllStones init_stones;
        init_stones[0].emplace(dc::Vector2(), 0.f, dc::Vector2(0.f, v0_speed), 1.57f * rotation_factor);
        s_simulator->SetStones(init_stones);
        while (!s_simulator->AreAllStonesStopped()) {
            auto const& stones = s_simulator->GetStones();
            auto const speed = stones[0]->linear_velocity.Length();
            if (speed <= target_speed) {
                return stones[0]->position;
            }
            s_simulator->Step();
        }
        return s_simulator->GetStones()[0]->position;
    }();

    float const delta_angle = std::atan2(delta.x, delta.y);
    float const target_angle = std::atan2(target_position.y, target_position.x);
    float const v0_angle = target_angle + delta_angle;

    return dc::Vector2(v0_speed * std::cos(v0_angle), v0_speed * std::sin(v0_angle));
}

dc::Vector2 ShotGenerator::estimatePassThroughVelocity(
    dc::Vector2 const& pass_point, float remaining_distance,
    dc::moves::Shot::Rotation rotation
) const {
    // 歩の rotateToPassPointGoF に相当:
    // pass_pointを通過した後、投擲点から見てさらにremaining_distance[m]進む速度を計算
    //
    // 手順:
    // 1. 投擲点からpass_pointまでの距離 pr を計算
    // 2. pr + remaining_distance の位置で停止する速度を逆算
    // 3. その速度でpass_pointを通過する角度を計算
    float pr = pass_point.Length();
    float target_r = pr + remaining_distance;

    // target_rの位置で停止する初速を逆算 (target_speed = 0)
    // ただし通過点での残り速度ではなく、より遠くで停止する初速を使う
    // → estimateVelocityの target_position を (0, target_r) 方向に設定して初速の大きさだけ得る
    dc::Vector2 dummy_target(0.f, target_r);
    dc::Vector2 v0_straight = estimateVelocity(dummy_target, 0.f, rotation);
    float v0_speed = v0_straight.Length();

    // この初速でpass_pointを通過する角度を計算 (estimateVelocity の方向補正ロジック)
    dc::Vector2 v0_at_pass = estimateVelocity(pass_point, 0.f, rotation);
    // v0_at_passは「pass_pointで停止する速度」なので、方向だけ借りて速さを差し替える
    float angle = std::atan2(v0_at_pass.x, v0_at_pass.y);

    // ただし上記は近似。より正確にはestimateVelocityにtarget_speedを渡すべき。
    // pass_pointでの残り速度 = target_r - pr の距離分に相当する速度
    // 簡易近似: target_speed ≈ 残り距離から逆算
    // DC3の EstimateShotVelocityFCV1 は target_speed をサポートしている (0〜4)
    // remaining_distance → target_speed の変換は近似が必要

    // シンプルな近似: remaining_distance を直接 target_speed として使う
    // ただし target_speed の上限は 4.0
    float target_speed = std::min(remaining_distance, 4.0f);

    return estimateVelocity(pass_point, target_speed, rotation);
}

// ========== ショット速度計算 ==========

ShotInfo ShotGenerator::calcDrawShot(const Position& target, int spin) const {
    dc::Vector2 target_pos(target.x, target.y);
    auto rotation = spin == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    dc::Vector2 vel = estimateVelocity(target_pos, 0.f, rotation);
    return { vel.x, vel.y, spin };
}

ShotInfo ShotGenerator::calcHitShot(const Position& stone_pos, HitWeight weight, int spin) const {
    // 歩: genStandardHit → 石の位置から0.015m手前を狙い、指定距離分先まで通過する速度
    dc::Vector2 pass_point(stone_pos.x, stone_pos.y - 0.015f);
    auto rotation = spin == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;

    if (weight == HitWeight::STRONG) {
        // STRONG: 最高速で通過 (target_speed = 4.0)
        return { estimateVelocity(pass_point, 4.0f, rotation).x,
                 estimateVelocity(pass_point, 4.0f, rotation).y, spin };
    }

    float remaining = kHitRemainingDist[static_cast<int>(weight)];
    dc::Vector2 vel = estimatePassThroughVelocity(pass_point, remaining, rotation);
    return { vel.x, vel.y, spin };
}

ShotInfo ShotGenerator::calcFreezeShot(const Position& stone_pos, int spin) const {
    // 対象石の手前 (石の直径 + 0.02m) にドロー
    float freeze_y = stone_pos.y - 2.0f * kStoneRadius - 0.02f;
    Position target = { stone_pos.x, freeze_y };
    return calcDrawShot(target, spin);
}

ShotInfo ShotGenerator::calcPeelShot(const Position& stone_pos, int spin) const {
    // 歩: 対象石に斜めに当てて両方排除
    // 高速で対象石の位置を通過
    dc::Vector2 pass_point(stone_pos.x, stone_pos.y - 0.015f);
    auto rotation = spin == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    dc::Vector2 vel = estimateVelocity(pass_point, kPeelSpeed, rotation);
    return { vel.x, vel.y, spin };
}

ShotInfo ShotGenerator::calcComeAroundShot(const Position& stone_pos, ComeAroundLength length, int spin) const {
    // 歩: 投擲点と対象石を結ぶ線上の延長、指定距離奥にドロー
    float dx = stone_pos.x - kThrowX;
    float dy = stone_pos.y - kThrowY;
    float dist = std::sqrt(dx * dx + dy * dy);
    float l = kComeAroundLength[static_cast<int>(length)];
    float rate = (dist + l) / dist;
    Position target = {
        kThrowX + dx * rate,
        kThrowY + dy * rate
    };
    return calcDrawShot(target, spin);
}

ShotInfo ShotGenerator::calcPostGuardShot(const Position& stone_pos, PostGuardLength length, int spin) const {
    // 歩: 投擲点と対象石を結ぶ線上の手前、指定距離にドロー
    float dx = stone_pos.x - kThrowX;
    float dy = stone_pos.y - kThrowY;
    float dist = std::sqrt(dx * dx + dy * dy);
    float l = kPostGuardLength[static_cast<int>(length)];
    float rate = (dist - l) / dist;
    Position target = {
        kThrowX + dx * rate,
        kThrowY + dy * rate
    };
    return calcDrawShot(target, spin);
}

// ========== 石の位置取得 ==========

Position ShotGenerator::getStonePosition(const dc::GameState& state, int stone_index) const {
    // DC3: stones[team][index]
    // stone_index: 0-15, 偶数=team0, 奇数=team1 (歩の方式)
    // DC3: team0は0〜7, team1は0〜7
    dc::Team team = (stone_index % 2 == 0) ? dc::Team::k0 : dc::Team::k1;
    int idx = stone_index / 2;
    auto& stone = state.stones[static_cast<int>(team)][idx];
    if (!stone.has_value()) return { 0.f, 0.f };
    return { stone->position.x, stone->position.y };
}

// ========== ノイズなしシミュレーション ==========

dc::GameState ShotGenerator::simulateNoRand(
    const dc::GameState& state,
    const ShotInfo& shot
) {
    dc::GameState new_state = state;
    simulator_->Load(*simulator_storage_);

    dc::Vector2 velocity(shot.vx, shot.vy);
    auto rot = shot.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    dc::moves::Shot shot_move{ velocity, rot };
    dc::Move move{ shot_move };

    // PlayerIdentical を使用 → ノイズなし (歩の makeMoveNoRand 相当)
    dc::ApplyMove(game_setting_, *simulator_, *player_no_rand_, new_state, move, std::chrono::milliseconds(0));
    simulator_->Save(*simulator_storage_);

    return new_state;
}

// ========== 候補手生成 ==========

std::vector<CandidateShot> ShotGenerator::generateCandidates(
    const dc::GameState& state,
    dc::Team my_team,
    const std::vector<Position>& grid_positions
) {
    std::vector<CandidateShot> candidates;
    dc::Team opp_team = dc::GetOpponentTeam(my_team);

    // --- 1. グリッドドロー (既存方式) ---
    for (size_t i = 0; i < grid_positions.size(); ++i) {
        for (int spin : {1, 0}) {
            CandidateShot c;
            c.type = ShotType::DRAW;
            c.spin = spin;
            c.target_index = -1;
            c.param = static_cast<int>(i);
            c.shot = calcDrawShot(grid_positions[i], spin);
            c.label = "Draw(" + std::string(spin == 1 ? "CW" : "CCW") + ",grid" + std::to_string(i) + ")";
            candidates.push_back(c);
        }
    }

    // --- 2. DrawPos名前付き位置 (歩の16位置のうちグリッドと重複しにくいもの) ---
    // ガード位置は固定グリッドに含まれないため明示的に追加
    DrawPos guard_positions[] = { DrawPos::G3, DrawPos::G4, DrawPos::G5 };
    for (auto dp : guard_positions) {
        for (int spin : {1, 0}) {
            Position pos = getDrawPosition(dp);
            CandidateShot c;
            c.type = ShotType::PREGUARD;
            c.spin = spin;
            c.target_index = -1;
            c.param = static_cast<int>(dp);
            c.shot = calcDrawShot(pos, spin);
            std::string name = (dp == DrawPos::G3) ? "G3" : (dp == DrawPos::G4) ? "G4" : "G5";
            c.label = "PreGuard(" + std::string(spin == 1 ? "CW" : "CCW") + "," + name + ")";
            candidates.push_back(c);
        }
    }

    // --- 3. 相手石へのヒット ---
    for (int idx = 0; idx < 8; ++idx) {
        int stone_index = idx * 2 + static_cast<int>(opp_team);
        auto& stone = state.stones[static_cast<int>(opp_team)][idx];
        if (!stone.has_value()) continue;

        Position stone_pos = { stone->position.x, stone->position.y };
        bool in_house = (std::pow(stone_pos.x - kHouseCenterX, 2) +
                         std::pow(stone_pos.y - kHouseCenterY, 2)) <=
                         std::pow(kHouseRadius + kStoneRadius, 2);

        // ヒット (MIDDLE と STRONG は常に生成、TOUCH/WEAK はハウス内のみ)
        std::vector<HitWeight> weights;
        if (in_house) {
            weights = { HitWeight::TOUCH, HitWeight::WEAK, HitWeight::MIDDLE, HitWeight::STRONG };
        } else {
            weights = { HitWeight::MIDDLE, HitWeight::STRONG };
        }

        for (auto w : weights) {
            for (int spin : {1, 0}) {
                CandidateShot c;
                c.type = ShotType::HIT;
                c.spin = spin;
                c.target_index = stone_index;
                c.param = static_cast<int>(w);
                c.shot = calcHitShot(stone_pos, w, spin);
                c.label = "Hit(" + std::string(spin == 1 ? "CW" : "CCW") + "," +
                          std::to_string(stone_index) + "," + hitWeightToString(w) + ")";
                candidates.push_back(c);
            }
        }

        // フリーズ (ハウス内の相手石のみ)
        if (in_house) {
            for (int spin : {1, 0}) {
                CandidateShot c;
                c.type = ShotType::FREEZE;
                c.spin = spin;
                c.target_index = stone_index;
                c.param = 0;
                c.shot = calcFreezeShot(stone_pos, spin);
                c.label = "Freeze(" + std::string(spin == 1 ? "CW" : "CCW") + "," +
                          std::to_string(stone_index) + ")";
                candidates.push_back(c);
            }
        }
    }

    // --- 4. 自分石へのヒット (Push), PostGuard ---
    for (int idx = 0; idx < 8; ++idx) {
        int stone_index = idx * 2 + static_cast<int>(my_team);
        auto& stone = state.stones[static_cast<int>(my_team)][idx];
        if (!stone.has_value()) continue;

        Position stone_pos = { stone->position.x, stone->position.y };
        bool in_house = (std::pow(stone_pos.x - kHouseCenterX, 2) +
                         std::pow(stone_pos.y - kHouseCenterY, 2)) <=
                         std::pow(kHouseRadius + kStoneRadius, 2);

        // ポストガード (ハウス内の自分石の手前にガード配置)
        if (in_house) {
            for (auto pl : { PostGuardLength::SHORT, PostGuardLength::MIDDLE, PostGuardLength::LONG }) {
                for (int spin : {1, 0}) {
                    CandidateShot c;
                    c.type = ShotType::POSTGUARD;
                    c.spin = spin;
                    c.target_index = stone_index;
                    c.param = static_cast<int>(pl);
                    c.shot = calcPostGuardShot(stone_pos, pl, spin);
                    c.label = "PostGuard(" + std::string(spin == 1 ? "CW" : "CCW") + "," +
                              std::to_string(stone_index) + "," + std::to_string(static_cast<int>(pl)) + ")";
                    candidates.push_back(c);
                }
            }
        }
    }

    // --- 5. PASS ---
    {
        CandidateShot c;
        c.type = ShotType::PASS;
        c.spin = 1;
        c.target_index = -1;
        c.param = -1;
        // PASS: 最低速度で外に投げる (盤面に影響しない)
        c.shot = { 0.0f, 0.01f, 1 };
        c.label = "Pass";
        candidates.push_back(c);
    }

    return candidates;
}

// ========== 候補手プール生成 (シミュレーション付き) ==========

CandidatePool ShotGenerator::generatePool(
    const dc::GameState& state,
    dc::Team my_team,
    const std::vector<Position>& grid_positions
) {
    CandidatePool pool;
    pool.candidates = generateCandidates(state, my_team, grid_positions);

    // 全候補手をノイズなしシミュレーション
    pool.result_states.reserve(pool.candidates.size());
    for (auto& candidate : pool.candidates) {
        dc::GameState result = simulateNoRand(state, candidate.shot);
        pool.result_states.push_back(result);
    }

    return pool;
}

// ========== 文字列変換 ==========

std::string ShotGenerator::shotTypeToString(ShotType type) {
    switch (type) {
        case ShotType::PASS: return "Pass";
        case ShotType::DRAW: return "Draw";
        case ShotType::PREGUARD: return "PreGuard";
        case ShotType::HIT: return "Hit";
        case ShotType::FREEZE: return "Freeze";
        case ShotType::PEEL: return "Peel";
        case ShotType::COMEAROUND: return "ComeAround";
        case ShotType::POSTGUARD: return "PostGuard";
        case ShotType::DRAWRAISE: return "DrawRaise";
        case ShotType::TAKEOUT: return "TakeOut";
        case ShotType::DOUBLE: return "Double";
        default: return "Unknown";
    }
}

std::string ShotGenerator::hitWeightToString(HitWeight w) {
    switch (w) {
        case HitWeight::TOUCH: return "TOUCH";
        case HitWeight::WEAK: return "WEAK";
        case HitWeight::MIDDLE: return "MIDDLE";
        case HitWeight::STRONG: return "STRONG";
        default: return "?";
    }
}
