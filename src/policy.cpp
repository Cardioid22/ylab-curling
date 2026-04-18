#define _USE_MATH_DEFINES
#include <cmath>
#include "policy.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

// ============================================================
// StoneInfoDC3: DC3座標からAyumu座標系の値を計算
// ============================================================
void StoneInfoDC3::setFromDC3(float dc3_x, float dc3_y) {
    constexpr float TEE_Y = RolloutPolicy::DC3_TEE_Y;    // 38.405
    constexpr float THROW_Y = RolloutPolicy::DC3_THROW_Y; // 2.005
    constexpr float STONE_RAD = RolloutPolicy::FR_STONE_RAD;
    constexpr float CURL_OFFSET = 0.05f;  // FCV1のカール量推定（約3度）

    // Ayumu座標 (ティー原点)
    x = dc3_x;
    y = dc3_y - TEE_Y;

    // ティーからの極座標
    r = std::hypot(x, y);
    th = std::atan2(x, y);  // Ayumu慣例: atan2(dx, dy)

    // 投擲位置からの極座標
    float dx_throw = dc3_x - 0.0f;
    float dy_throw = dc3_y - THROW_Y;
    tr = std::hypot(dx_throw, dy_throw);
    tth = std::atan2(dx_throw, dy_throw);  // atan2(dx, dy)

    // hitLimit 近似 (方法A)
    float stone_angle = std::atan2(2.0f * STONE_RAD, tr);
    float h0 = -stone_angle - CURL_OFFSET;
    float h1 = +stone_angle + CURL_OFFSET;
    hitLimit_[0][0] = tth + h0;  // RIGHT(CW), left edge
    hitLimit_[0][1] = tth + h1;  // RIGHT(CW), right edge
    hitLimit_[1][0] = tth - h1;  // LEFT(CCW), left edge
    hitLimit_[1][1] = tth - h0;  // LEFT(CCW), right edge
}

// ============================================================
// RelStoneInfoDC3: 2石間の相対情報を計算
// ============================================================
void RelStoneInfoDC3::set(const StoneInfoDC3& base, const StoneInfoDC3& obj) {
    float dx = obj.x - base.x;
    float dy = obj.y - base.y;
    r = std::hypot(dx, dy);
    th = std::atan2(dx, dy);  // atan2(dx, dy) Ayumu慣例
    rp = labelRelPos(base, obj, *this);
}

// ============================================================
// labelRelPos: Ayumu logic.hpp の移植
// 2石間の相対位置を16カテゴリに分類 (0-15, -1=無効)
// ============================================================
int labelRelPos(const StoneInfoDC3& base, const StoneInfoDC3& obj, const RelStoneInfoDC3& rel) {
    constexpr float STONE_RAD = RolloutPolicy::FR_STONE_RAD;

    if (base.tr >= obj.tr) {
        // obj が手前 (投擲位置に近い側)
        float objRHL0 = obj.hitLimit(0, 0), objRHL1 = obj.hitLimit(0, 1);
        float baseRHL0 = base.hitLimit(0, 0), baseRHL1 = base.hitLimit(0, 1);
        float objLHL0 = obj.hitLimit(1, 0), objLHL1 = obj.hitLimit(1, 1);
        float baseLHL0 = base.hitLimit(1, 0), baseLHL1 = base.hitLimit(1, 1);

        // 右スピンとの関係
        int rs;
        if (objRHL1 < (baseRHL0 + baseRHL1) / 2.0f) {
            if (objRHL1 < baseRHL0) return -1;
            rs = 0;
        } else {
            if (objRHL0 > baseRHL1) {
                rs = 2;
            } else {
                rs = 1;
            }
        }
        // 左スピンとの関係
        int ls;
        if (objLHL0 < (baseLHL0 + baseLHL1) / 2.0f) {
            if (objLHL1 < baseLHL0) {
                ls = 2;
            } else {
                ls = 1;
            }
        } else {
            if (objLHL0 > baseLHL1) return -1;
            ls = 0;
        }

        if (rs == 2 && ls == 2) return -1;
        int bs = (int)(!(rs & ls))
               + (unsigned int)(rel.r > 4.0f * STONE_RAD) * 2
               + (unsigned int)(rel.th > 0.0f) * 4;
        return bs;  // 0-7
    } else {
        // obj が後方
        float ath = std::fabs(rel.th);
        if (rel.r <= 3.0f * STONE_RAD) {
            // 近距離
            return 8 + (unsigned int)(ath > M_PI / 6.0)
                     + (unsigned int)(rel.th > 0.0f) * 4;
        } else {
            // 遠距離
            if (ath > M_PI / 4.0) return -1;
            return 10 + (unsigned int)(ath > M_PI / 6.0)
                      + (unsigned int)(rel.th > 0.0f) * 4;
        }
    }
}

// ============================================================
// RolloutPolicy コンストラクタ
// ============================================================
RolloutPolicy::RolloutPolicy()
    : temperature_(0.8), loaded_(false), drawPosInitialized_(false) {
    std::memset(params_, 0, sizeof(params_));
    rng_.seed(std::random_device{}());
}

bool RolloutPolicy::load(const std::string& param_path) {
    std::ifstream ifs(param_path);
    if (!ifs) {
        std::cerr << "RolloutPolicy::load() : failed to open " << param_path << std::endl;
        return false;
    }
    for (int i = 0; i < N_PARAMS_ALL; ++i) {
        if (!(ifs >> params_[i])) {
            std::cerr << "RolloutPolicy::load() : failed to read param " << i << std::endl;
            return false;
        }
    }
    loaded_ = true;
    initDrawPositionInfo();
    std::cout << "RolloutPolicy: loaded " << N_PARAMS_ALL << " params, T=" << temperature_ << std::endl;
    return true;
}

// ============================================================
// ドロー位置情報の初期化 (Ayumu realizeDrawPosition 相当)
// Ayumu座標系で定義し、DC3座標に変換してsetFromDC3
// ============================================================
void RolloutPolicy::initDrawPositionInfo() {
    constexpr float r_s = 0.9f;                                  // Small ring radius
    constexpr float r_l = FR_HOUSE_RAD - 2.0f * FR_STONE_RAD;   // ≈1.54

    // Ayumu座標系での位置 (ティー原点)
    struct AyumuPos { float x, y; };
    AyumuPos positions[N_STANDARD_MOVES] = {
        {0, 0},                                        // 0: PASS → TEE
        {0, 4.0f * FR_STONE_RAD},                     // 1: DRAW → S0
        {r_s, 0},                                      // 2: PREGUARD → S2
        {0, 2.0f * FR_STONE_RAD},                     // 3: L1DRAW → S4
        {-r_s, 0},                                     // 4: FREEZE → S6
        {0, r_l},                                      // 5: COMEAROUND → L0
        {r_l / std::sqrt(2.0f), r_l / std::sqrt(2.0f)},  // 6: POSTGUARD → L1
        {r_l, 0},                                      // 7: HIT → L2
        {r_l / std::sqrt(2.0f), -r_l / std::sqrt(2.0f)}, // 8: PEEL → L3
        {0, -FR_HOUSE_RAD},                            // 9: DRAWRAISE → L4
        {-r_l / std::sqrt(2.0f), -r_l / std::sqrt(2.0f)}, // 10: TAKEOUT → L5
        {-r_l, 0},                                     // 11: DOUBLE → L6
        {-r_l / std::sqrt(2.0f), r_l / std::sqrt(2.0f)},  // 12: RAISETAKEOUT → L7
    };

    for (int i = 0; i < N_STANDARD_MOVES; ++i) {
        // Ayumu→DC3座標変換: dc3_y = ayumu_y + TEE_Y
        drawPosInfo_[i].setFromDC3(positions[i].x, positions[i].y + DC3_TEE_Y);
    }
    drawPosInitialized_ = true;
}

// ============================================================
// 盤面上の全石をStoneInfoDC3として構築
// ============================================================
std::vector<RolloutPolicy::BoardStoneInfo> RolloutPolicy::buildBoardStones(
    const dc::GameState& state) const
{
    std::vector<BoardStoneInfo> stones;
    for (int t = 0; t < 2; ++t) {
        for (int s = 0; s < 8; ++s) {
            auto& stone = state.stones[t][s];
            if (!stone) continue;
            BoardStoneInfo bsi;
            bsi.info.setFromDC3(stone->position.x, stone->position.y);
            bsi.team = t;
            bsi.index = s * 2 + t;  // ShotGenerator と同じインタリーブ方式
            stones.push_back(bsi);
        }
    }
    return stones;
}

// ============================================================
// ShotType → Ayumu Standard type index
// ============================================================
int RolloutPolicy::toAyumuType(ShotType type) const {
    switch (type) {
        case ShotType::PASS:       return AYUMU_PASS;       // 0
        case ShotType::DRAW:       return AYUMU_DRAW;       // 1
        case ShotType::PREGUARD:   return AYUMU_PREGUARD;   // 2
        case ShotType::FREEZE:     return AYUMU_FREEZE;     // 4
        case ShotType::COMEAROUND: return AYUMU_COMEAROUND; // 5
        case ShotType::POSTGUARD:  return AYUMU_POSTGUARD;  // 6
        case ShotType::HIT:        return AYUMU_HIT;        // 7
        case ShotType::PEEL:       return AYUMU_PEEL;       // 8
        case ShotType::DRAWRAISE:  return AYUMU_DRAWRAISE;  // 9
        case ShotType::TAKEOUT:    return AYUMU_TAKEOUT;    // 10
        case ShotType::DOUBLE:     return AYUMU_DOUBLE;     // 11
        default:                   return AYUMU_DRAW;       // fallback
    }
}

// ============================================================
// Ayumu type → カテゴリ (POL_END_SCORE, POL_2ND_SCORE 用)
//   0: NONE (Pass)
//   1: WHOLE (Draw, PreGuard, L1Draw)
//   2: ONE (Freeze...TakeOut)
//   3: TWO (Double, RaiseTakeOut)
// ============================================================
int RolloutPolicy::toTypeCategory(int ayumu_type) const {
    if (ayumu_type < TYPE_BOUNDARY_WHOLE) return 0;
    if (ayumu_type < TYPE_BOUNDARY_ONE)   return 1;
    if (ayumu_type < TYPE_BOUNDARY_TWO)   return 2;
    return 3;
}

// ============================================================
// DC3 shot_num (0-15) → Ayumu turn index (0-5)
//
// Ayumu のターン番号は逆順: turn=0 が最終投, turn=15 が先頭投
// TtoTI:
//   turn==0 (TURN_LAST) → 0
//   turn==1 (TURN_BEFORE_LAST) → 1
//   WHITE (後攻) + not free guard → 2
//   BLACK (先攻) + not free guard → 3
//   WHITE (後攻) + free guard → 4
//   BLACK (先攻) + free guard → 5
// ============================================================
int RolloutPolicy::toTurnIndex(int shot_num) const {
    int ayumu_turn = 15 - shot_num;  // DC3→Ayumu変換

    if (ayumu_turn == 0) return 0;   // 最終投
    if (ayumu_turn == 1) return 1;   // 最終投の1つ前

    bool is_free_guard = (ayumu_turn >= 12);  // turns 12-15 = free guard
    bool is_black = (ayumu_turn & 1);         // odd turns = BLACK (先攻)

    if (!is_black) {
        return is_free_guard ? 4 : 2;  // WHITE (後攻)
    } else {
        return is_free_guard ? 5 : 3;  // BLACK (先攻)
    }
}

// ============================================================
// DC3 座標 → Ayumu 16ゾーン離散化 (logic.hpp getPositionIndex)
//
// Ayumu 座標系: ティーが原点 (0, 0)
//   y > 0: ティー後方 (バックライン側)
//   y < 0: ティー前方 (デリバリー側)
//
// DC3 座標系: ティーが (0, 38.405)
//   変換: ayumu_y = dc3_y - 38.405
// ============================================================
int RolloutPolicy::getPositionIndex(float dc3_x, float dc3_y) const {
    // DC3 → Ayumu 座標変換
    float x = dc3_x;
    float y = dc3_y - DC3_TEE_Y;
    float r = std::sqrt(x * x + y * y);
    float th = std::atan2(x, y);  // Ayumu の慣例: atan2(x, y)

    // Active zone 判定
    if (y < 0) {  // ティー前方
        if (std::fabs(x) > FR_ACTIVE_LINE) return 0;
    } else {       // ティー後方
        if (r > FR_ACTIVE_LINE) return 0;
    }

    float ring_unit = HOUSE_PLUS_STONE / 3.0f;  // ≈0.658

    if (y > 0) {
        // ティー後方
        if (r < HOUSE_PLUS_STONE) {
            // ハウス内
            int i = 1;
            if (r <= ring_unit) {
                i += 2;  // 内リング: 3 or 4
            }
            if (std::fabs(th) > M_PI / 6.0) {
                i += 1;  // 大角度: 2 or 4
            }
            return i;  // 1-4
        } else {
            return 15;  // ティー後方、ハウス外
        }
    } else {
        // ティー前方 (y <= 0)
        if (r < HOUSE_PLUS_STONE) {
            // ハウス内
            int i = 5;
            if (r > ring_unit) {
                i += 2;  // 中リング: 7 or 8
                if (r > 2.0f * ring_unit) {
                    i += 2;  // 外リング: 9 or 10
                }
            }
            if (std::fabs(th) < 5.0 * M_PI / 6.0) {
                i += 1;
            }
            return i;  // 5-10
        } else {
            // ガードゾーン
            int i = 11;
            if (y < -2.0f * FR_HOUSE_RAD) {
                i += 2;  // 遠方: 13 or 14
            }
            if (std::fabs(x) < HOUSE_PLUS_STONE * std::sin(M_PI / 6.0)) {
                i += 1;  // 中央: 12 or 14
            }
            return i;  // 11-14
        }
    }
}

// ============================================================
// 盤面スコア計算 (ハウス内の No.1 石チームの得点数)
// ============================================================
int RolloutPolicy::countScore(const dc::GameState& state, dc::Team perspective_team) const {
    constexpr float house_r = 1.829f + 0.145f;

    struct StoneEntry {
        float dist;
        dc::Team team;
    };
    std::vector<StoneEntry> house_stones;

    for (int t = 0; t < 2; ++t) {
        for (int s = 0; s < 8; ++s) {
            auto& stone = state.stones[t][s];
            if (!stone) continue;
            float dx = stone->position.x;
            float dy = stone->position.y - DC3_TEE_Y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < house_r) {
                house_stones.push_back({dist, static_cast<dc::Team>(t)});
            }
        }
    }

    if (house_stones.empty()) return 0;

    std::sort(house_stones.begin(), house_stones.end(),
              [](const StoneEntry& a, const StoneEntry& b) { return a.dist < b.dist; });

    dc::Team scoring_team = house_stones[0].team;
    int score = 0;
    for (auto& entry : house_stones) {
        if (entry.team == scoring_team) {
            score++;
        } else {
            break;
        }
    }
    // perspective_team から見た符号付きスコア
    return (scoring_team == perspective_team) ? score : -score;
}

// ============================================================
// ブランクエンドが有利かの判定
// ============================================================
bool RolloutPolicy::isCleanBetter(bool is_second, int rel_score) const {
    // 後攻(ハンマー持ち)かつ同点以上ならブランクが有利
    if (is_second) return (rel_score >= 0);
    // 先攻で勝っているならブランクが有利
    return (rel_score > 0);
}

// ============================================================
// 候補手のスコア計算
// ============================================================
std::vector<double> RolloutPolicy::scoreCandidates(
    const dc::GameState& state,
    const std::vector<CandidateShot>& candidates,
    int shot_num,
    dc::Team my_team,
    int end_index,
    int rel_score)
{
    const int turn_idx = toTurnIndex(shot_num);
    const bool is_second = (shot_num % 2 == 1);  // 奇数ショット = 後攻
    const bool clean = isCleanBetter(is_second, rel_score);

    // 盤面のスコア (自チーム視点)
    int board_score = countScore(state, my_team);
    int clamped_score = std::max(-2, std::min(2, board_score));

    // opp2ndScore は簡易版では 0 とする
    int opp_2nd_score = 0;
    int clamped_2nd = std::max(-2, std::min(2, opp_2nd_score));

    // 盤面上の石情報を事前計算
    auto board_stones = buildBoardStones(state);

    // 簡易ゾーン配列 (POL_1STONE用)
    std::vector<int> stone_zones;
    for (auto& bs : board_stones) {
        stone_zones.push_back(getPositionIndex(
            bs.info.x + DC3_TEE_Y * 0.0f + bs.info.x,  // dc3_x = ayumu_x
            bs.info.y + DC3_TEE_Y));                     // dc3_y = ayumu_y + TEE_Y
    }

    std::vector<double> scores(candidates.size());

    for (size_t m = 0; m < candidates.size(); ++m) {
        double s = 0.0;
        int type = toAyumuType(candidates[m].type);
        int type_cat = toTypeCategory(type);
        // スピン変換: ylab CW=1→Ayumu RIGHT=0, ylab CCW=0→Ayumu LEFT=1
        int ayumu_spin = 1 - candidates[m].spin;

        // ---- グループ1: POL_1STONE ----
        for (size_t si = 0; si < board_stones.size(); ++si) {
            int zone = getPositionIndex(
                board_stones[si].info.x,            // Ayumu x (= DC3 x)
                board_stones[si].info.y + DC3_TEE_Y // DC3 y に戻す
            );
            int idx = IDX_1STONE
                + turn_idx * N_STANDARD_MOVES * N_ZONES
                + zone * N_STANDARD_MOVES
                + type;
            s += params_[idx];
        }

        // ---- グループ2: POL_2STONES ----
        if (type >= TYPE_BOUNDARY_WHOLE && type < TYPE_BOUNDARY_ONE) {
            // Draw系 (type 1-3): drawPositionInfo[type] を基準
            const StoneInfoDC3& dp = drawPosInfo_[type];
            int dp_zone = getPositionIndex(dp.x, dp.y + DC3_TEE_Y);

            for (auto& bs : board_stones) {
                RelStoneInfoDC3 rel;
                rel.set(dp, bs.info);
                if (rel.rp < 0) continue;
                int idx = IDX_2STONES
                    + turn_idx * N_STANDARD_MOVES * 2 * N_ZONES * N_ZONES
                    + type * 2 * N_ZONES * N_ZONES
                    + ayumu_spin * N_ZONES * N_ZONES
                    + dp_zone * N_ZONES
                    + rel.rp;
                if (idx >= 0 && idx < N_PARAMS_ALL) s += params_[idx];
            }
        } else if (type >= TYPE_BOUNDARY_ONE && type < TYPE_BOUNDARY_TWO) {
            // 1石系 (type 4-10): target_index の石を基準
            int target_idx = candidates[m].target_index;
            if (target_idx >= 0) {
                // target_index に対応する BoardStoneInfo を探す
                const StoneInfoDC3* target_info = nullptr;
                for (auto& bs : board_stones) {
                    if (bs.index == target_idx) {
                        target_info = &bs.info;
                        break;
                    }
                }
                if (target_info) {
                    int target_zone = getPositionIndex(
                        target_info->x, target_info->y + DC3_TEE_Y);
                    for (auto& bs : board_stones) {
                        if (bs.index == target_idx) continue;  // 対象石自身はスキップ
                        RelStoneInfoDC3 rel;
                        rel.set(*target_info, bs.info);
                        if (rel.rp < 0) continue;
                        int idx = IDX_2STONES
                            + turn_idx * N_STANDARD_MOVES * 2 * N_ZONES * N_ZONES
                            + type * 2 * N_ZONES * N_ZONES
                            + ayumu_spin * N_ZONES * N_ZONES
                            + target_zone * N_ZONES
                            + rel.rp;
                        if (idx >= 0 && idx < N_PARAMS_ALL) s += params_[idx];
                    }
                }
            }
        } else if (type >= TYPE_BOUNDARY_TWO) {
            // 2石系 (type 11-12): 第1対象石を基準
            int target_idx = candidates[m].target_index;
            if (target_idx >= 0) {
                const StoneInfoDC3* target_info = nullptr;
                for (auto& bs : board_stones) {
                    if (bs.index == target_idx) {
                        target_info = &bs.info;
                        break;
                    }
                }
                if (target_info) {
                    int target_zone = getPositionIndex(
                        target_info->x, target_info->y + DC3_TEE_Y);
                    for (auto& bs : board_stones) {
                        if (bs.index == target_idx) continue;
                        RelStoneInfoDC3 rel;
                        rel.set(*target_info, bs.info);
                        if (rel.rp < 0) continue;
                        int idx = IDX_2STONES
                            + turn_idx * N_STANDARD_MOVES * 2 * N_ZONES * N_ZONES
                            + type * 2 * N_ZONES * N_ZONES
                            + ayumu_spin * N_ZONES * N_ZONES
                            + target_zone * N_ZONES
                            + rel.rp;
                        if (idx >= 0 && idx < N_PARAMS_ALL) s += params_[idx];
                    }
                }
            }
        }
        // PASS (type 0) は POL_2STONES 特徴なし

        // ---- グループ4: POL_END_SCORE_TO_TYPE ----
        {
            int idx = IDX_END_SCORE
                + turn_idx * (2 * 5 * 4)
                + (clean ? 1 : 0) * (5 * 4)
                + (clamped_score + 2) * 4
                + type_cat;
            s += params_[idx];
        }

        // ---- グループ5: POL_2ND_SCORE_TO_TYPE ----
        {
            int idx = IDX_2ND_SCORE
                + turn_idx * (2 * 5 * 4)
                + (clean ? 1 : 0) * (5 * 4)
                + (clamped_2nd + 2) * 4
                + type_cat;
            s += params_[idx];
        }

        scores[m] = s;
    }

    return scores;
}

// ============================================================
// 候補手の中から1手を確率的に選択 (softmax sampling)
// ============================================================
int RolloutPolicy::selectShot(
    const dc::GameState& state,
    const std::vector<CandidateShot>& candidates,
    int shot_num,
    dc::Team my_team,
    int end_index,
    int rel_score)
{
    if (candidates.empty()) return -1;
    if (candidates.size() == 1) return 0;

    auto scores = scoreCandidates(state, candidates, shot_num, my_team, end_index, rel_score);

    // deterministic モード: argmax 選択（再現可能）
    if (deterministic_) {
        return static_cast<int>(
            std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    // softmax with temperature
    double max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<double> probs(scores.size());
    double sum = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        probs[i] = std::exp((scores[i] - max_score) / temperature_);
        sum += probs[i];
    }

    // 確率分布からサンプリング
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng_);
    for (size_t i = 0; i < probs.size(); ++i) {
        r -= probs[i];
        if (r <= 0.0) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size() - 1);
}
