#include "pool_clustering_experiment.h"
#include "pool_experiment.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>

namespace dc = digitalcurling3;

// --- ClusterAnalysis ---

float ClusterAnalysis::purity() const {
    int total = static_cast<int>(member_indices.size());
    if (total == 0) return 0.0f;
    int max_count = std::max({count_draw, count_hit, count_freeze, count_guard, count_other});
    return static_cast<float>(max_count) / total;
}

std::string ClusterAnalysis::dominantType() const {
    int max_count = std::max({count_draw, count_hit, count_freeze, count_guard, count_other});
    if (max_count == count_draw) return "DRAW";
    if (max_count == count_hit) return "HIT";
    if (max_count == count_freeze) return "FREEZE";
    if (max_count == count_guard) return "GUARD";
    return "OTHER";
}

// --- PoolClusteringResult ---

float PoolClusteringResult::weightedPurity() const {
    float total_purity = 0.0f;
    int total_members = 0;
    for (auto& c : clusters) {
        int size = static_cast<int>(c.member_indices.size());
        total_purity += c.purity() * size;
        total_members += size;
    }
    return total_members > 0 ? total_purity / total_members : 0.0f;
}

float PoolClusteringResult::typeCoverage() const {
    std::set<std::string> dominant_types;
    for (auto& c : clusters) {
        dominant_types.insert(c.dominantType());
    }
    return static_cast<float>(dominant_types.size()) / n_clusters;
}

// --- PoolClusteringExperiment ---

PoolClusteringExperiment::PoolClusteringExperiment(dc::GameSetting const& game_setting)
    : game_setting_(game_setting)
{
}

std::vector<dc::GameState> PoolClusteringExperiment::createTestStates() {
    std::vector<dc::GameState> states;
    test_state_names_.clear();

    // shot番号を配置済み石数から自動計算（偶数=team0の手番）
    auto calcShot = [](const dc::GameState& s) -> int {
        int total = 0;
        for (int t = 0; t < 2; t++)
            for (int i = 0; i < 8; i++)
                if (s.stones[t][i].has_value()) total++;
        return (total % 2 == 0) ? total : total + 1;
    };

    // テスト盤面1: 空場（第1投）
    {
        dc::GameState s(game_setting_);
        s.shot = 0;
        states.push_back(s);
        test_state_names_.push_back("empty");
    }

    // テスト盤面2: 相手石がティー近くに1個
    {
        dc::GameState s(game_setting_);
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("opp1_tee");
    }

    // テスト盤面3: 相手石2個 + 自分石1個
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(-0.5f, kHouseCenterY + 0.3f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY - 0.2f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY + 0.8f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("opp2_my1");
    }

    // テスト盤面4: 混雑した盤面（3対3）
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(-1.0f, kHouseCenterY - 0.5f), 0.f));
        s.stones[0][2].emplace(dc::Transform(dc::Vector2(0.5f, kHouseCenterY + 1.0f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY + 0.1f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.3f, kHouseCenterY + 1.5f), 0.f));
        s.stones[1][2].emplace(dc::Transform(dc::Vector2(0.8f, kHouseCenterY - 0.8f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("crowded_3v3");
    }

    // テスト盤面5: センターガード（相手石がガードゾーンに1個）
    {
        dc::GameState s(game_setting_);
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY - 2.5f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("center_guard");
    }

    // テスト盤面6: ダブルガード（ガードゾーンに2個）
    {
        dc::GameState s(game_setting_);
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.1f, kHouseCenterY - 2.0f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY - 3.0f), 0.f));
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.5f, kHouseCenterY + 0.5f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("double_guard");
    }

    // テスト盤面7: ボタン争い（両チームがティー付近に1個ずつ）
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(-0.15f, kHouseCenterY + 0.1f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.1f, kHouseCenterY - 0.05f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("button_fight");
    }

    // テスト盤面8: フリーズ狙い（相手石がハウス中心、自分石なし）
    {
        dc::GameState s(game_setting_);
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY + 0.05f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("freeze_target");
    }

    // テスト盤面9: 相手3点取り状態（自分不利）
    {
        dc::GameState s(game_setting_);
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY + 0.2f), 0.f));
        s.stones[1][2].emplace(dc::Transform(dc::Vector2(-0.2f, kHouseCenterY - 0.1f), 0.f));
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(1.5f, kHouseCenterY + 0.8f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("opp_scoring_3");
    }

    // テスト盤面10: コーナーガード（左右のガード）
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(-1.5f, kHouseCenterY - 1.0f), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(1.5f, kHouseCenterY - 1.2f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY + 0.3f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("corner_guards");
    }

    // テスト盤面11: ブランクエンド狙い（終盤、石が少ない）
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY - 0.3f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(-0.8f, kHouseCenterY + 1.2f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("blank_end");
    }

    // テスト盤面12: 左寄り配置
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(-1.0f, kHouseCenterY + 0.5f), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(-0.8f, kHouseCenterY - 0.3f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(-1.2f, kHouseCenterY + 0.1f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("left_heavy");
    }

    // テスト盤面13: 自分有利（自分2点状態）
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.05f, kHouseCenterY + 0.05f), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(-0.3f, kHouseCenterY + 0.4f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.8f, kHouseCenterY - 0.6f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("my_scoring_2");
    }

    // テスト盤面14: 密集ハウス（4対4、終盤）
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.1f, kHouseCenterY + 0.1f), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(-0.5f, kHouseCenterY + 0.8f), 0.f));
        s.stones[0][2].emplace(dc::Transform(dc::Vector2(0.7f, kHouseCenterY - 0.4f), 0.f));
        s.stones[0][3].emplace(dc::Transform(dc::Vector2(-0.2f, kHouseCenterY - 1.0f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY - 0.05f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(0.4f, kHouseCenterY + 0.5f), 0.f));
        s.stones[1][2].emplace(dc::Transform(dc::Vector2(-0.8f, kHouseCenterY - 0.3f), 0.f));
        s.stones[1][3].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY + 1.5f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("packed_house_4v4");
    }

    // テスト盤面15: ガード+ドロー混在
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY - 2.5f), 0.f));  // ガード
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY + 0.2f), 0.f));  // ハウス内
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.5f, kHouseCenterY - 2.0f), 0.f)); // ガード
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("guard_and_draw");
    }

    // テスト盤面16: スプリット配置（左右に分散）
    {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(-1.5f, kHouseCenterY + 0.5f), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(1.5f, kHouseCenterY + 0.3f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(-1.2f, kHouseCenterY - 0.2f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(1.0f, kHouseCenterY - 0.5f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("split_lr");
    }

    // === プログラム生成盤面（17〜100） ===
    // 決定的な擬似乱数で多様な盤面を生成
    auto pseudoRand = [](int seed) -> float {
        // 簡易ハッシュベース乱数 [-1, 1]
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return (seed % 2001 - 1000) / 1000.0f;
    };

    // calcShotは上で定義済み

    // カテゴリ1: 相手石1個（様々な位置）(17-26)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        float x = pseudoRand(i * 7 + 1) * 1.5f;
        float y = kHouseCenterY + pseudoRand(i * 7 + 2) * 2.0f;
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(x, y), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("opp1_pos" + std::to_string(i));
    }

    // カテゴリ2: 自分1石+相手1石(27-36)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        float mx = pseudoRand(i * 11 + 3) * 1.2f;
        float my = kHouseCenterY + pseudoRand(i * 11 + 4) * 1.5f;
        float ox = pseudoRand(i * 11 + 5) * 1.2f;
        float oy = kHouseCenterY + pseudoRand(i * 11 + 6) * 1.5f;
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(mx, my), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(ox, oy), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("1v1_pos" + std::to_string(i));
    }

    // カテゴリ3: ガード配置バリエーション(37-46)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        float gx = pseudoRand(i * 13 + 7) * 0.8f;
        float gy = kHouseCenterY - 2.0f - pseudoRand(i * 13 + 8) * 1.5f;
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(gx, gy), 0.f));
        if (i >= 3) {
            float hx = pseudoRand(i * 13 + 9) * 0.5f;
            float hy = kHouseCenterY + pseudoRand(i * 13 + 10) * 0.8f;
            s.stones[1][1].emplace(dc::Transform(dc::Vector2(hx, hy), 0.f));
        }
        if (i >= 6) {
            float mx2 = pseudoRand(i * 13 + 11) * 0.6f;
            float my2 = kHouseCenterY + pseudoRand(i * 13 + 12) * 0.5f;
            s.stones[0][0].emplace(dc::Transform(dc::Vector2(mx2, my2), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("guard_var" + std::to_string(i));
    }

    // カテゴリ4: ハウス内密集（2v2〜3v3）(47-56)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        int n_my = 1 + (i % 3);
        int n_opp = 1 + ((i + 1) % 3);
        for (int j = 0; j < n_my && j < 4; j++) {
            float x = pseudoRand(i * 17 + j * 3 + 20) * 1.0f;
            float y = kHouseCenterY + pseudoRand(i * 17 + j * 3 + 21) * 1.2f;
            s.stones[0][j].emplace(dc::Transform(dc::Vector2(x, y), 0.f));
        }
        for (int j = 0; j < n_opp && j < 4; j++) {
            float x = pseudoRand(i * 17 + j * 3 + 30) * 1.0f;
            float y = kHouseCenterY + pseudoRand(i * 17 + j * 3 + 31) * 1.2f;
            s.stones[1][j].emplace(dc::Transform(dc::Vector2(x, y), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("house_dense" + std::to_string(i));
    }

    // カテゴリ5: 得点盤面（自分有利/不利）(57-66)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        bool my_advantage = (i < 5);
        int scoring_team = my_advantage ? 0 : 1;
        int other_team = 1 - scoring_team;
        float r1 = 0.1f + pseudoRand(i * 19 + 40) * 0.3f;
        float a1 = pseudoRand(i * 19 + 41) * 3.14159f;
        s.stones[scoring_team][0].emplace(dc::Transform(
            dc::Vector2(r1 * std::cos(a1), kHouseCenterY + r1 * std::sin(a1)), 0.f));
        if (i % 3 > 0) {
            float r2 = 0.3f + pseudoRand(i * 19 + 42) * 0.5f;
            float a2 = pseudoRand(i * 19 + 43) * 3.14159f;
            s.stones[scoring_team][1].emplace(dc::Transform(
                dc::Vector2(r2 * std::cos(a2), kHouseCenterY + r2 * std::sin(a2)), 0.f));
        }
        float r3 = 0.8f + pseudoRand(i * 19 + 44) * 0.8f;
        float a3 = pseudoRand(i * 19 + 45) * 3.14159f;
        s.stones[other_team][0].emplace(dc::Transform(
            dc::Vector2(r3 * std::cos(a3), kHouseCenterY + r3 * std::sin(a3)), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("scoring" + std::to_string(i));
    }

    // カテゴリ6: 終盤密集（3v3〜4v4）(67-76)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        int n_stones = 3 + (i % 2);
        for (int j = 0; j < n_stones && j < 4; j++) {
            float x0 = pseudoRand(i * 23 + j * 5 + 50) * 1.3f;
            float y0 = kHouseCenterY + pseudoRand(i * 23 + j * 5 + 51) * 1.5f;
            s.stones[0][j].emplace(dc::Transform(dc::Vector2(x0, y0), 0.f));
            float x1 = pseudoRand(i * 23 + j * 5 + 52) * 1.3f;
            float y1 = kHouseCenterY + pseudoRand(i * 23 + j * 5 + 53) * 1.5f;
            s.stones[1][j].emplace(dc::Transform(dc::Vector2(x1, y1), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("endgame" + std::to_string(i));
    }

    // カテゴリ7: フリーズ対象バリエーション(77-84)
    for (int i = 0; i < 8; i++) {
        dc::GameState s(game_setting_);
        float tx = pseudoRand(i * 29 + 60) * 0.4f;
        float ty = kHouseCenterY + pseudoRand(i * 29 + 61) * 0.3f;
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(tx, ty), 0.f));
        if (i >= 4) {
            float mx = pseudoRand(i * 29 + 62) * 0.8f;
            float my = kHouseCenterY + pseudoRand(i * 29 + 63) * 0.6f;
            s.stones[0][0].emplace(dc::Transform(dc::Vector2(mx, my), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("freeze_var" + std::to_string(i));
    }

    // カテゴリ8: 左右非対称・エッジケース(85-92)
    for (int i = 0; i < 8; i++) {
        dc::GameState s(game_setting_);
        float side = (i % 2 == 0) ? 1.0f : -1.0f;
        s.stones[0][0].emplace(dc::Transform(
            dc::Vector2(side * (0.8f + pseudoRand(i * 31 + 70) * 0.5f),
                        kHouseCenterY + pseudoRand(i * 31 + 71) * 1.0f), 0.f));
        s.stones[1][0].emplace(dc::Transform(
            dc::Vector2(side * (0.5f + pseudoRand(i * 31 + 72) * 0.3f),
                        kHouseCenterY + pseudoRand(i * 31 + 73) * 0.8f), 0.f));
        if (i >= 3) {
            s.stones[0][1].emplace(dc::Transform(
                dc::Vector2(-side * 0.3f, kHouseCenterY - 2.0f), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("asym" + std::to_string(i));
    }

    // カテゴリ9: 混合戦略（ガード+ハウス）(93-100)
    for (int i = 0; i < 8; i++) {
        dc::GameState s(game_setting_);
        // ガード
        s.stones[1][0].emplace(dc::Transform(
            dc::Vector2(pseudoRand(i * 37 + 80) * 0.5f,
                        kHouseCenterY - 2.5f + pseudoRand(i * 37 + 81) * 0.5f), 0.f));
        // ハウス内（相手）
        s.stones[1][1].emplace(dc::Transform(
            dc::Vector2(pseudoRand(i * 37 + 82) * 0.6f,
                        kHouseCenterY + pseudoRand(i * 37 + 83) * 0.5f), 0.f));
        // ハウス内（自分）
        s.stones[0][0].emplace(dc::Transform(
            dc::Vector2(pseudoRand(i * 37 + 84) * 0.7f,
                        kHouseCenterY + pseudoRand(i * 37 + 85) * 0.8f), 0.f));
        if (i >= 4) {
            s.stones[0][1].emplace(dc::Transform(
                dc::Vector2(pseudoRand(i * 37 + 86) * 0.4f,
                            kHouseCenterY - 2.0f + pseudoRand(i * 37 + 87) * 0.3f), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("mixed" + std::to_string(i));
    }

    // === 追加カテゴリ（101〜200） ===

    // カテゴリ10: 相手2石バリエーション(101-115)
    for (int i = 0; i < 15; i++) {
        dc::GameState s(game_setting_);
        float x1 = pseudoRand(i * 41 + 100) * 1.5f;
        float y1 = kHouseCenterY + pseudoRand(i * 41 + 101) * 1.8f;
        float x2 = pseudoRand(i * 41 + 102) * 1.2f;
        float y2 = kHouseCenterY + pseudoRand(i * 41 + 103) * 1.5f;
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(x1, y1), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(x2, y2), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("opp2_var" + std::to_string(i));
    }

    // カテゴリ11: 自分2石+相手1石(116-130)
    for (int i = 0; i < 15; i++) {
        dc::GameState s(game_setting_);
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(
            pseudoRand(i * 43 + 110) * 1.0f,
            kHouseCenterY + pseudoRand(i * 43 + 111) * 1.2f), 0.f));
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(
            pseudoRand(i * 43 + 112) * 0.8f,
            kHouseCenterY + pseudoRand(i * 43 + 113) * 1.0f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(
            pseudoRand(i * 43 + 114) * 1.3f,
            kHouseCenterY + pseudoRand(i * 43 + 115) * 1.5f), 0.f));
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("my2_opp1_" + std::to_string(i));
    }

    // カテゴリ12: ティー周辺の接戦(131-145)
    for (int i = 0; i < 15; i++) {
        dc::GameState s(game_setting_);
        float r = 0.1f + pseudoRand(i * 47 + 120) * 0.5f;
        float a = pseudoRand(i * 47 + 121) * 3.14159f;
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(
            r * std::cos(a), kHouseCenterY + r * std::sin(a)), 0.f));
        float r2 = 0.15f + pseudoRand(i * 47 + 122) * 0.4f;
        float a2 = pseudoRand(i * 47 + 123) * 3.14159f;
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(
            r2 * std::cos(a2), kHouseCenterY + r2 * std::sin(a2)), 0.f));
        if (i >= 5) {
            float r3 = 0.3f + pseudoRand(i * 47 + 124) * 0.6f;
            float a3 = pseudoRand(i * 47 + 125) * 3.14159f;
            s.stones[1][1].emplace(dc::Transform(dc::Vector2(
                r3 * std::cos(a3), kHouseCenterY + r3 * std::sin(a3)), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("tee_fight" + std::to_string(i));
    }

    // カテゴリ13: ダブルガード配置(146-155)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        float gx1 = pseudoRand(i * 53 + 130) * 0.3f;
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(gx1, kHouseCenterY - 2.5f), 0.f));
        float gx2 = pseudoRand(i * 53 + 131) * 0.4f;
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(gx2, kHouseCenterY - 3.0f), 0.f));
        if (i >= 3) {
            s.stones[1][2].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i * 53 + 132) * 0.5f,
                kHouseCenterY + pseudoRand(i * 53 + 133) * 0.4f), 0.f));
        }
        if (i >= 6) {
            s.stones[0][0].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i * 53 + 134) * 0.4f,
                kHouseCenterY + pseudoRand(i * 53 + 135) * 0.6f), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("dbl_guard" + std::to_string(i));
    }

    // カテゴリ14: 2v2ハウス内バトル(156-170)
    for (int i = 0; i < 15; i++) {
        dc::GameState s(game_setting_);
        for (int j = 0; j < 2; j++) {
            float mx = pseudoRand(i * 59 + j * 4 + 140) * 1.2f;
            float my = kHouseCenterY + pseudoRand(i * 59 + j * 4 + 141) * 1.3f;
            s.stones[0][j].emplace(dc::Transform(dc::Vector2(mx, my), 0.f));
            float ox = pseudoRand(i * 59 + j * 4 + 142) * 1.2f;
            float oy = kHouseCenterY + pseudoRand(i * 59 + j * 4 + 143) * 1.3f;
            s.stones[1][j].emplace(dc::Transform(dc::Vector2(ox, oy), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("2v2_house" + std::to_string(i));
    }

    // カテゴリ15: ハウス外の石がある盤面(171-180)
    for (int i = 0; i < 10; i++) {
        dc::GameState s(game_setting_);
        // ハウス外の石（ガードゾーンや遠方）
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(
            pseudoRand(i * 61 + 150) * 0.5f,
            kHouseCenterY - 3.0f + pseudoRand(i * 61 + 151) * 1.0f), 0.f));
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(
            pseudoRand(i * 61 + 152) * 0.6f,
            kHouseCenterY - 2.0f + pseudoRand(i * 61 + 153) * 0.5f), 0.f));
        if (i >= 4) {
            s.stones[1][1].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i * 61 + 154) * 0.3f,
                kHouseCenterY + pseudoRand(i * 61 + 155) * 0.5f), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("outside" + std::to_string(i));
    }

    // カテゴリ16: 3v2非対称(181-195)
    for (int i = 0; i < 15; i++) {
        dc::GameState s(game_setting_);
        for (int j = 0; j < 3; j++) {
            s.stones[0][j].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i * 67 + j * 3 + 160) * 1.2f,
                kHouseCenterY + pseudoRand(i * 67 + j * 3 + 161) * 1.5f), 0.f));
        }
        for (int j = 0; j < 2; j++) {
            s.stones[1][j].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i * 67 + j * 3 + 170) * 1.0f,
                kHouseCenterY + pseudoRand(i * 67 + j * 3 + 171) * 1.3f), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("3v2_asym" + std::to_string(i));
    }

    // カテゴリ17: 第1投〜第4投の序盤(196-200)
    for (int i = 0; i < 5; i++) {
        dc::GameState s(game_setting_);
        if (i >= 1) {
            s.stones[1][0].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i * 71 + 180) * 1.0f,
                kHouseCenterY + pseudoRand(i * 71 + 181) * 1.5f), 0.f));
        }
        if (i >= 3) {
            s.stones[0][0].emplace(dc::Transform(dc::Vector2(
                pseudoRand(i * 71 + 182) * 0.8f,
                kHouseCenterY + pseudoRand(i * 71 + 183) * 1.0f), 0.f));
        }
        s.shot = calcShot(s);
        states.push_back(s);
        test_state_names_.push_back("early" + std::to_string(i));
    }

    return states;
}

bool PoolClusteringExperiment::isInHouse(const std::optional<dc::Transform>& stone) const {
    float x = stone->position.x;
    float y = stone->position.y;
    return std::pow(x, 2) + std::pow(y - kHouseCenterY, 2) <= std::pow(2 * kHouseRadius / 3, 2);
}

float PoolClusteringExperiment::dist(const dc::GameState& a, const dc::GameState& b) const {
    // Clusteringクラスと同じ距離関数
    int v = 0;
    for (size_t team = 0; team < 2; team++) {
        for (size_t index = 0; index < 8; index++) {
            if (a.stones[team][index] && b.stones[team][index]) v++;
        }
    }
    if (v == 0) return 100.0f;

    float distance = 0.0f;
    for (size_t team = 0; team < 2; team++) {
        for (size_t index = 0; index < 8; index++) {
            if (a.stones[team][index] && b.stones[team][index]) {
                float dx = a.stones[team][index]->position.x - b.stones[team][index]->position.x;
                float dy = a.stones[team][index]->position.y - b.stones[team][index]->position.y;
                distance += std::sqrt(dx * dx + dy * dy);
                if (isInHouse(a.stones[team][index]) != isInHouse(b.stones[team][index])) {
                    distance += 8.0f;
                }
            }
            else if (b.stones[team][index]) {
                float dx = b.stones[team][index]->position.x;
                float dy = b.stones[team][index]->position.y - kHouseCenterY;
                distance += std::sqrt(dx * dx + dy * dy);
                if (isInHouse(b.stones[team][index])) {
                    distance += 8.0f;
                }
            }
            else if (a.stones[team][index]) {
                float dx = a.stones[team][index]->position.x;
                float dy = a.stones[team][index]->position.y - kHouseCenterY;
                distance += std::sqrt(dx * dx + dy * dy);
                if (isInHouse(a.stones[team][index])) {
                    distance += 8.0f;
                }
            }
        }
    }

    // No.1ストーンのチーム一致判定
    float closest_a = std::numeric_limits<float>::max();
    float closest_b = std::numeric_limits<float>::max();
    int team_a = -1, team_b = -1;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (a.stones[t][i]) {
                float d = std::sqrt(std::pow(a.stones[t][i]->position.x, 2) +
                          std::pow(a.stones[t][i]->position.y - kHouseCenterY, 2));
                if (d < closest_a) { closest_a = d; team_a = t; }
            }
            if (b.stones[t][i]) {
                float d = std::sqrt(std::pow(b.stones[t][i]->position.x, 2) +
                          std::pow(b.stones[t][i]->position.y - kHouseCenterY, 2));
                if (d < closest_b) { closest_b = d; team_b = t; }
            }
        }
    }
    if (team_a >= 0 && team_b >= 0 && team_a != team_b) {
        distance += 12.0f;
    }

    return distance;
}

std::vector<std::vector<float>> PoolClusteringExperiment::makeDistanceTable(
    const std::vector<dc::GameState>& states
) {
    int n = static_cast<int>(states.size());
    std::vector<std::vector<float>> table(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float d = dist(states[i], states[j]);
            table[i][j] = d;
            table[j][i] = d;
        }
        table[i][i] = -1.0f;
    }
    return table;
}

// ゾーン判定: 石の位置がどの戦略的ゾーンに属するか
// Zone 0: ハウス内（得点圏）
// Zone 1: ガードゾーン（ハウス手前、y < HouseCenterY - HouseRadius）
// Zone 2: 遠方（シート外に近い、または横方向に大きく外れた位置）
int PoolClusteringExperiment::getZone(const std::optional<dc::Transform>& stone) const {
    if (!stone) return -1;
    float x = stone->position.x;
    float y = stone->position.y;
    float dist_to_tee = std::sqrt(x * x + (y - kHouseCenterY) * (y - kHouseCenterY));

    if (dist_to_tee <= kHouseRadius) return 0;  // ハウス内
    if (y < kHouseCenterY - kHouseRadius && y > kHouseCenterY - 3.0f * kHouseRadius) return 1;  // ガードゾーン
    return 2;  // 遠方
}

// 盤面の得点評価（カーリングの公式ルール準拠）
// team0の視点: 正=team0有利, 負=team1有利
float PoolClusteringExperiment::evaluateBoard(const dc::GameState& state) const {
    struct StoneInfo { float dist; int team; };
    std::vector<StoneInfo> in_house;

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (!state.stones[t][i]) continue;
            float dx = state.stones[t][i]->position.x;
            float dy = state.stones[t][i]->position.y - kHouseCenterY;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= kHouseRadius + 0.145f) {  // ハウス半径 + 石の半径
                in_house.push_back({d, t});
            }
        }
    }
    if (in_house.empty()) return 0.0f;

    std::sort(in_house.begin(), in_house.end(),
              [](auto& a, auto& b) { return a.dist < b.dist; });

    int scoring_team = in_house[0].team;
    int score = 0;
    for (auto& s : in_house) {
        if (s.team == scoring_team) score++;
        else break;
    }
    return scoring_team == 0 ? static_cast<float>(score) : -static_cast<float>(score);
}

float PoolClusteringExperiment::distDelta(
    const dc::GameState& input,
    const dc::GameState& a,
    const dc::GameState& b
) const {
    // 改良版デルタ距離関数 v2
    // 基本方針: 入力盤面からの「変化」を多次元で比較
    //
    // 改良点:
    // 1. 盤面スコア差ペナルティ: 戦略的結果の違いを直接反映
    // 2. 石インタラクションペナルティ: 既存石を動かしたか否かの違い
    //    (Draw=非接触 vs TOUCH/Freeze=接触 の区別)
    // 3. ゾーンペナルティ増大: Guard vs House Draw の分離を強化
    // 4. 新石の近接度: Freeze(密着) vs Draw(非密着) の区別

    constexpr float MOVE_THRESHOLD = 0.01f;        // 1cm以下は「不変」
    constexpr float PENALTY_EXISTENCE = 30.0f;      // 石の有無が異なるペナルティ
    constexpr float PENALTY_ZONE = 12.0f;           // ゾーン差ペナルティ (6→12)
    constexpr float NEW_STONE_WEIGHT = 4.0f;        // 新石位置差の重み (3→4)
    constexpr float MOVED_STONE_WEIGHT = 2.0f;      // 押された石の移動差
    constexpr float PENALTY_INTERACTION = 15.0f;    // 石接触有無の差ペナルティ (NEW)
    constexpr float INTERACTION_THRESHOLD = 0.03f;  // 3cm以上動いたら「接触」
    constexpr float SCORE_WEIGHT = 8.0f;            // 盤面スコア差の重み (NEW)
    constexpr float PROXIMITY_WEIGHT = 5.0f;        // 新石近接度差の重み (NEW)

    float distance = 0.0f;
    float max_displacement_a = 0.0f;
    float max_displacement_b = 0.0f;
    int new_stone_team = -1, new_stone_idx = -1;

    for (int team = 0; team < 2; team++) {
        for (int idx = 0; idx < 8; idx++) {
            bool in_input = input.stones[team][idx].has_value();
            bool in_a = a.stones[team][idx].has_value();
            bool in_b = b.stones[team][idx].has_value();

            if (in_input) {
                // === 入力盤面に存在した石 ===
                if (in_a && in_b) {
                    float dx_a = a.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dy_a = a.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float dx_b = b.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dy_b = b.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float move_a = std::sqrt(dx_a * dx_a + dy_a * dy_a);
                    float move_b = std::sqrt(dx_b * dx_b + dy_b * dy_b);

                    // 最大変位を追跡（インタラクション判定用）
                    max_displacement_a = std::max(max_displacement_a, move_a);
                    max_displacement_b = std::max(max_displacement_b, move_b);

                    if (move_a < MOVE_THRESHOLD && move_b < MOVE_THRESHOLD) {
                        continue;  // 両方不変 → 距離0
                    }

                    float ddx = dx_a - dx_b;
                    float ddy = dy_a - dy_b;
                    distance += MOVED_STONE_WEIGHT * std::sqrt(ddx * ddx + ddy * ddy);

                    int zone_a = getZone(a.stones[team][idx]);
                    int zone_b = getZone(b.stones[team][idx]);
                    if (zone_a != zone_b) {
                        distance += PENALTY_ZONE;
                    }
                }
                else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            }
            else {
                // === 新規配置石 ===
                if (in_a && in_b) {
                    new_stone_team = team;
                    new_stone_idx = idx;

                    float dx = a.stones[team][idx]->position.x - b.stones[team][idx]->position.x;
                    float dy = a.stones[team][idx]->position.y - b.stones[team][idx]->position.y;
                    distance += NEW_STONE_WEIGHT * std::sqrt(dx * dx + dy * dy);

                    int zone_a = getZone(a.stones[team][idx]);
                    int zone_b = getZone(b.stones[team][idx]);
                    if (zone_a != zone_b) {
                        distance += PENALTY_ZONE;
                    }
                }
                else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            }
        }
    }

    // (1) 石インタラクションペナルティ: 既存石を動かしたか否か
    // Draw/Guard=非接触, TOUCH/Freeze=接触 を区別
    bool interacted_a = max_displacement_a > INTERACTION_THRESHOLD;
    bool interacted_b = max_displacement_b > INTERACTION_THRESHOLD;
    if (interacted_a != interacted_b) {
        distance += PENALTY_INTERACTION;
    }

    // (2) 新石の近接度: 結果盤面で新石が既存石にどれだけ近いか
    // Freeze(密着≈0.29m) vs Draw(非密着≈1m+) の区別に有効
    if (new_stone_team >= 0) {
        auto computeMinProximity = [&](const dc::GameState& state) -> float {
            float min_dist = std::numeric_limits<float>::max();
            float nx = state.stones[new_stone_team][new_stone_idx]->position.x;
            float ny = state.stones[new_stone_team][new_stone_idx]->position.y;
            for (int t = 0; t < 2; t++) {
                for (int i = 0; i < 8; i++) {
                    if (t == new_stone_team && i == new_stone_idx) continue;
                    if (!state.stones[t][i].has_value()) continue;
                    float dx = nx - state.stones[t][i]->position.x;
                    float dy = ny - state.stones[t][i]->position.y;
                    min_dist = std::min(min_dist, std::sqrt(dx * dx + dy * dy));
                }
            }
            return min_dist;
        };

        float prox_a = computeMinProximity(a);
        float prox_b = computeMinProximity(b);
        // 両方に既存石がある場合のみ比較（空場ではスキップ）
        if (prox_a < 100.0f || prox_b < 100.0f) {
            distance += PROXIMITY_WEIGHT * std::abs(prox_a - prox_b);
        }
    }

    // (3) 盤面スコア差ペナルティ: 戦略的結果の違いを直接反映
    float score_a = evaluateBoard(a);
    float score_b = evaluateBoard(b);
    distance += SCORE_WEIGHT * std::abs(score_a - score_b);

    // (4) No.1ストーンのチーム一致判定
    float closest_a = std::numeric_limits<float>::max();
    float closest_b = std::numeric_limits<float>::max();
    int team_a = -1, team_b = -1;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (a.stones[t][i]) {
                float d = std::sqrt(std::pow(a.stones[t][i]->position.x, 2) +
                          std::pow(a.stones[t][i]->position.y - kHouseCenterY, 2));
                if (d < closest_a) { closest_a = d; team_a = t; }
            }
            if (b.stones[t][i]) {
                float d = std::sqrt(std::pow(b.stones[t][i]->position.x, 2) +
                          std::pow(b.stones[t][i]->position.y - kHouseCenterY, 2));
                if (d < closest_b) { closest_b = d; team_b = t; }
            }
        }
    }
    if (team_a >= 0 && team_b >= 0 && team_a != team_b) {
        distance += 10.0f;
    }

    return distance;
}

std::vector<std::vector<float>> PoolClusteringExperiment::makeDistanceTableDelta(
    const dc::GameState& input_state,
    const std::vector<dc::GameState>& result_states
) {
    int n = static_cast<int>(result_states.size());
    std::vector<std::vector<float>> table(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float d = distDelta(input_state, result_states[i], result_states[j]);
            table[i][j] = d;
            table[j][i] = d;
        }
        table[i][i] = -1.0f;
    }
    return table;
}

std::vector<std::set<int>> PoolClusteringExperiment::runClustering(
    const std::vector<std::vector<float>>& dist_table,
    int n_desired_clusters
) {
    int n = static_cast<int>(dist_table.size());
    std::vector<std::set<int>> clusters(n);
    for (int i = 0; i < n; i++) {
        clusters[i].insert(i);
    }

    while (static_cast<int>(clusters.size()) > n_desired_clusters) {
        // 最も近いクラスタペアを探す（平均連結法）
        float min_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;

        for (int i = 0; i < static_cast<int>(clusters.size()); i++) {
            for (int j = i + 1; j < static_cast<int>(clusters.size()); j++) {
                float total = 0.0f;
                int count = 0;
                for (int a : clusters[i]) {
                    for (int b : clusters[j]) {
                        total += dist_table[a][b];
                        count++;
                    }
                }
                float avg = total / count;
                if (avg < min_dist) {
                    min_dist = avg;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i == -1) break;

        // マージ
        clusters[best_i].insert(clusters[best_j].begin(), clusters[best_j].end());
        clusters.erase(clusters.begin() + best_j);
    }

    return clusters;
}

std::vector<int> PoolClusteringExperiment::calculateMedoids(
    const std::vector<std::vector<float>>& dist_table,
    const std::vector<std::set<int>>& clusters
) {
    std::vector<int> medoids;
    for (auto& cluster : clusters) {
        if (cluster.empty()) {
            medoids.push_back(-1);
            continue;
        }
        if (cluster.size() == 1) {
            medoids.push_back(*cluster.begin());
            continue;
        }
        float min_total = std::numeric_limits<float>::max();
        int best = -1;
        for (int c : cluster) {
            float total = 0.0f;
            for (int o : cluster) {
                if (c != o) total += dist_table[c][o];
            }
            if (total < min_total) {
                min_total = total;
                best = c;
            }
        }
        medoids.push_back(best);
    }
    return medoids;
}

std::vector<int> PoolClusteringExperiment::greedyFarthestPointSampling(
    const dc::GameState& input_state,
    const std::vector<dc::GameState>& result_states,
    int k
) {
    int n = static_cast<int>(result_states.size());
    if (k >= n) {
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }
    if (k <= 0) return {};

    // 最初の点: 先頭の候補を選ぶ（評価関数に依存しない公平な初期化）
    int first = 0;

    std::vector<int> selected;
    selected.push_back(first);

    // min_dist[i] = selectedの中で最も近い点との距離
    std::vector<float> min_dist(n, std::numeric_limits<float>::max());
    min_dist[first] = -1.0f;  // 選択済み

    for (int iter = 1; iter < k; iter++) {
        // 最後に選ばれた点との距離で min_dist を更新
        int last = selected.back();
        for (int i = 0; i < n; i++) {
            if (min_dist[i] < 0.0f) continue;  // 選択済み
            float d = distDelta(input_state, result_states[i], result_states[last]);
            min_dist[i] = std::min(min_dist[i], d);
        }

        // min_distが最大の点を選出
        int farthest = -1;
        float max_min_dist = -1.0f;
        for (int i = 0; i < n; i++) {
            if (min_dist[i] < 0.0f) continue;
            if (min_dist[i] > max_min_dist) {
                max_min_dist = min_dist[i];
                farthest = i;
            }
        }
        if (farthest < 0) break;
        selected.push_back(farthest);
        min_dist[farthest] = -1.0f;
    }

    return selected;
}

PoolClusteringResult PoolClusteringExperiment::analyzeClusterComposition(
    const std::string& state_name,
    const CandidatePool& pool,
    const std::vector<std::set<int>>& clusters,
    const std::vector<int>& medoids
) {
    PoolClusteringResult result;
    result.state_name = state_name;
    result.n_candidates = static_cast<int>(pool.candidates.size());
    result.n_clusters = static_cast<int>(clusters.size());

    for (int c = 0; c < static_cast<int>(clusters.size()); c++) {
        ClusterAnalysis ca;
        ca.cluster_id = c;
        ca.medoid_index = medoids[c];
        if (ca.medoid_index >= 0) {
            ca.medoid_type = pool.candidates[ca.medoid_index].type;
            ca.medoid_label = pool.candidates[ca.medoid_index].label;
        }

        for (int idx : clusters[c]) {
            ca.member_indices.push_back(idx);
            ShotType type = pool.candidates[idx].type;
            ca.member_types.push_back(type);
            ca.member_labels.push_back(pool.candidates[idx].label);

            switch (type) {
                case ShotType::DRAW: ca.count_draw++; break;
                case ShotType::HIT: ca.count_hit++; break;
                case ShotType::FREEZE: ca.count_freeze++; break;
                case ShotType::PREGUARD: case ShotType::POSTGUARD: ca.count_guard++; break;
                default: ca.count_other++; break;
            }
        }
        result.clusters.push_back(ca);
    }

    return result;
}

void PoolClusteringExperiment::printResult(const PoolClusteringResult& result) {
    std::cout << "\n=== " << result.state_name
              << " (candidates=" << result.n_candidates
              << ", clusters=" << result.n_clusters << ") ===" << std::endl;
    std::cout << "  Weighted Purity: " << std::fixed << std::setprecision(3)
              << result.weightedPurity() << std::endl;
    std::cout << "  Type Coverage: " << result.typeCoverage()
              << " (" << static_cast<int>(result.typeCoverage() * result.n_clusters)
              << "/" << result.n_clusters << " distinct dominant types)" << std::endl;

    for (auto& ca : result.clusters) {
        std::cout << "\n  Cluster " << ca.cluster_id
                  << " (size=" << ca.member_indices.size()
                  << ", purity=" << std::setprecision(2) << ca.purity()
                  << ", dominant=" << ca.dominantType() << ")" << std::endl;
        std::cout << "    Type breakdown: Draw=" << ca.count_draw
                  << " Hit=" << ca.count_hit
                  << " Freeze=" << ca.count_freeze
                  << " Guard=" << ca.count_guard
                  << " Other=" << ca.count_other << std::endl;
        std::cout << "    Medoid: [" << ca.medoid_index << "] " << ca.medoid_label << std::endl;

        // メンバー一覧（ラベル表示）
        std::cout << "    Members: ";
        for (size_t i = 0; i < ca.member_labels.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << ca.member_labels[i];
        }
        std::cout << std::endl;
    }
}

void PoolClusteringExperiment::exportResultCSV(
    const PoolClusteringResult& result,
    const std::string& output_dir
) {
    std::string filename = output_dir + "/cluster_analysis_" + result.state_name
                         + "_k" + std::to_string(result.n_clusters) + ".csv";
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return;
    }

    ofs << "candidate_index,cluster_id,shot_type,label,is_medoid,spin,target_index,param,vx,vy,rot" << std::endl;

    for (auto& ca : result.clusters) {
        for (size_t i = 0; i < ca.member_indices.size(); i++) {
            int idx = ca.member_indices[i];
            auto& cand = result.state_name.empty() ? throw std::runtime_error("") :
                         // dummy - we need pool reference
                         ca.member_labels[i]; // just for label
            (void)cand;

            ofs << idx << ","
                << ca.cluster_id << ","
                << static_cast<int>(ca.member_types[i]) << ","
                << "\"" << ca.member_labels[i] << "\","
                << (idx == ca.medoid_index ? 1 : 0)
                << std::endl;
        }
    }

    std::cout << "  Exported to " << filename << std::endl;
}

void PoolClusteringExperiment::run() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  3-Method Comparison: FullPool vs DeltaClustered vs GreedyFPS" << std::endl;
    std::cout << "================================================================" << std::endl;

    auto test_states = createTestStates();
    auto& state_names = test_state_names_;
    std::cout << "Test states: " << test_states.size() << std::endl;

    std::string output_dir = "experiments/pool_clustering_results";
    std::filesystem::create_directories(output_dir);

    ShotGenerator generator(game_setting_);
    auto grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    // 保持率ベースのK値（候補数に応じて動的に決定）
    std::vector<float> retention_ratios = {0.1f, 0.2f, 0.3f, 0.5f, 0.7f};

    auto classifyType = [](ShotType t) -> std::string {
        switch (t) {
            case ShotType::DRAW: return "Draw";
            case ShotType::HIT: return "Hit";
            case ShotType::FREEZE: return "Freeze";
            case ShotType::PREGUARD: case ShotType::POSTGUARD: return "Guard";
            case ShotType::PASS: return "Pass";
            default: return "Other";
        }
    };

    // サマリー用（手法ごとに1行）
    struct ResultRow {
        std::string state;
        int total_candidates;
        float ratio;
        int k;
        std::string method;  // "DeltaClustered" or "GreedyFPS"
        // 最良手比較
        std::string pool_best_label;
        std::string pool_best_type;
        float pool_best_score;
        std::string sel_best_label;
        std::string sel_best_type;
        float sel_best_score;
        bool same_shot;
        bool same_type;
        float score_diff;
        int dist_computations;  // 距離計算回数
        // クラスタ分析（DeltaClustered専用）
        float weighted_purity;   // 加重平均純度（タイプとクラスタの一致度）
        bool pool_best_in_same_cluster;  // 歩の最良手がDCの選んだ手と同じクラスタ内か
    };
    std::vector<ResultRow> results;

    for (size_t s = 0; s < test_states.size(); s++) {
        auto& state = test_states[s];
        dc::Team my_team = dc::Team::k0;

        auto pool = generator.generatePool(state, my_team);
        int n = static_cast<int>(pool.candidates.size());

        if (n <= 2) continue;  // 候補が少なすぎる場合スキップ

        // === 歩方式: 全候補を評価して最良手 ===
        int best_pool_idx = -1;
        float best_pool_score = -std::numeric_limits<float>::max();
        for (int i = 0; i < n; i++) {
            float score = evaluateBoard(pool.result_states[i]);
            if (score > best_pool_score) {
                best_pool_score = score;
                best_pool_idx = i;
            }
        }
        std::string best_pool_type = classifyType(pool.candidates[best_pool_idx].type);
        std::string best_pool_label = pool.candidates[best_pool_idx].label;

        std::cout << "\n[" << (s + 1) << "/" << test_states.size() << "] " << state_names[s]
                  << " (N=" << n << ") Pool best: " << best_pool_label
                  << "(" << best_pool_type << ",score=" << std::fixed << std::setprecision(1) << best_pool_score << ")" << std::endl;

        // Delta距離テーブル構築（DeltaClustered用）
        auto dist_delta = makeDistanceTableDelta(state, pool.result_states);

        for (float ratio : retention_ratios) {
            int k = std::max(2, static_cast<int>(std::round(n * ratio)));
            if (k >= n) continue;

            // === 方式1: DeltaClustered（階層的クラスタリング + メドイド）===
            auto clusters = runClustering(dist_delta, k);
            auto medoids = calculateMedoids(dist_delta, clusters);

            int best_dc_idx = -1;
            float best_dc_score = -std::numeric_limits<float>::max();
            for (int m : medoids) {
                if (m < 0) continue;
                float score = evaluateBoard(pool.result_states[m]);
                if (score > best_dc_score) { best_dc_score = score; best_dc_idx = m; }
            }

            // === クラスタ純度（Purity）計算 ===
            // 各クラスタ内の最多タイプの割合の加重平均
            float total_purity = 0.0f;
            int total_members = 0;
            for (auto& cluster : clusters) {
                if (cluster.empty()) continue;
                std::map<std::string, int> type_counts;
                for (int idx : cluster) {
                    type_counts[classifyType(pool.candidates[idx].type)]++;
                }
                int max_count = 0;
                for (auto& [t, c] : type_counts) {
                    max_count = std::max(max_count, c);
                }
                int sz = static_cast<int>(cluster.size());
                total_purity += static_cast<float>(max_count);  // 最多タイプのメンバー数
                total_members += sz;
            }
            float weighted_purity = total_members > 0 ? total_purity / total_members : 0.0f;

            // === 歩の最良手がDCの選んだ手と同じクラスタにいるか ===
            int pool_best_cluster = -1;
            int dc_best_cluster = -1;
            for (int ci = 0; ci < static_cast<int>(clusters.size()); ci++) {
                if (clusters[ci].count(best_pool_idx)) pool_best_cluster = ci;
                if (clusters[ci].count(best_dc_idx)) dc_best_cluster = ci;
            }
            bool pool_best_in_same_cluster = (pool_best_cluster == dc_best_cluster && pool_best_cluster >= 0);

            // === 方式2: GreedyFPS（貪欲最遠点サンプリング）===
            auto fps_selected = greedyFarthestPointSampling(state, pool.result_states, k);

            int best_fps_idx = -1;
            float best_fps_score = -std::numeric_limits<float>::max();
            for (int idx : fps_selected) {
                float score = evaluateBoard(pool.result_states[idx]);
                if (score > best_fps_score) { best_fps_score = score; best_fps_idx = idx; }
            }

            // 結果蓄積
            auto makeRow = [&](const std::string& method, int best_idx, float best_score, int dist_comp,
                               float purity, bool in_same_cluster) {
                ResultRow row;
                row.state = state_names[s]; row.total_candidates = n;
                row.ratio = ratio; row.k = k; row.method = method;
                row.pool_best_label = best_pool_label; row.pool_best_type = best_pool_type;
                row.pool_best_score = best_pool_score;
                row.sel_best_label = (best_idx >= 0) ? pool.candidates[best_idx].label : "N/A";
                row.sel_best_type = (best_idx >= 0) ? classifyType(pool.candidates[best_idx].type) : "N/A";
                row.sel_best_score = best_score;
                row.same_shot = (best_idx == best_pool_idx);
                row.same_type = (row.sel_best_type == best_pool_type);
                row.score_diff = best_score - best_pool_score;
                row.dist_computations = dist_comp;
                row.weighted_purity = purity;
                row.pool_best_in_same_cluster = in_same_cluster;
                return row;
            };

            int dc_dist_comp = n * (n - 1) / 2;
            int fps_dist_comp = n * (k - 1);
            results.push_back(makeRow("DeltaClustered", best_dc_idx, best_dc_score, dc_dist_comp,
                                      weighted_purity, pool_best_in_same_cluster));
            results.push_back(makeRow("GreedyFPS", best_fps_idx, best_fps_score, fps_dist_comp,
                                      0.0f, false));

            int ratio_pct = static_cast<int>(std::round(ratio * 100));
            auto dcLabel = (best_dc_idx >= 0) ? pool.candidates[best_dc_idx].label : "N/A";
            auto fpsLabel = (best_fps_idx >= 0) ? pool.candidates[best_fps_idx].label : "N/A";
            std::cout << "  " << ratio_pct << "%(K=" << k << ")"
                      << "  DC:" << (best_dc_idx == best_pool_idx ? "EXACT" : (classifyType(pool.candidates[best_dc_idx >= 0 ? best_dc_idx : 0].type) == best_pool_type ? "Same" : "DIFF"))
                      << "(diff=" << std::showpos << std::setprecision(0) << (best_dc_score - best_pool_score) << std::noshowpos << ")"
                      << "  FPS:" << (best_fps_idx == best_pool_idx ? "EXACT" : (classifyType(pool.candidates[best_fps_idx >= 0 ? best_fps_idx : 0].type) == best_pool_type ? "Same" : "DIFF"))
                      << "(diff=" << std::showpos << (best_fps_score - best_pool_score) << std::noshowpos << ")"
                      << std::endl;

            // === 詳細JSON出力（全盤面、保持率20%のみ）===
            if (ratio_pct == 20) {
                std::string json_path = output_dir + "/detail_" + state_names[s] + "_r" + std::to_string(ratio_pct) + ".json";
                std::ofstream jf(json_path);
                jf << std::fixed << std::setprecision(4);
                jf << "{\n";

                // 初期盤面
                jf << "  \"state_name\": \"" << state_names[s] << "\",\n";
                jf << "  \"retention_ratio\": " << ratio << ",\n";
                jf << "  \"k\": " << k << ",\n";
                jf << "  \"n_candidates\": " << n << ",\n";
                jf << "  \"initial_stones\": [\n";
                bool first_stone = true;
                for (int t = 0; t < 2; t++) {
                    for (int idx = 0; idx < 8; idx++) {
                        if (!state.stones[t][idx]) continue;
                        if (!first_stone) jf << ",\n";
                        first_stone = false;
                        jf << "    {\"team\":" << t << ",\"index\":" << idx
                           << ",\"x\":" << state.stones[t][idx]->position.x
                           << ",\"y\":" << state.stones[t][idx]->position.y << "}";
                    }
                }
                jf << "\n  ],\n";

                // 全候補手 + クラスタ割当 + 結果盤面の新石位置
                jf << "  \"candidates\": [\n";
                for (int ci = 0; ci < n; ci++) {
                    auto& cand = pool.candidates[ci];
                    auto& res = pool.result_states[ci];

                    // クラスタID検索
                    int cluster_id = -1;
                    for (int cj = 0; cj < static_cast<int>(clusters.size()); cj++) {
                        if (clusters[cj].count(ci)) { cluster_id = cj; break; }
                    }

                    // 新石の位置（入力に無かった石）
                    float new_x = 0.0f, new_y = 0.0f;
                    bool found_new = false;
                    for (int t = 0; t < 2; t++) {
                        for (int idx = 0; idx < 8; idx++) {
                            if (!state.stones[t][idx] && res.stones[t][idx]) {
                                new_x = res.stones[t][idx]->position.x;
                                new_y = res.stones[t][idx]->position.y;
                                found_new = true;
                            }
                        }
                    }

                    // 結果盤面の全ストーン
                    jf << "    {\"index\":" << ci
                       << ",\"type\":\"" << classifyType(cand.type) << "\""
                       << ",\"label\":\"" << cand.label << "\""
                       << ",\"vx\":" << cand.shot.vx << ",\"vy\":" << cand.shot.vy
                       << ",\"rot\":" << cand.shot.rot
                       << ",\"cluster_id\":" << cluster_id
                       << ",\"is_medoid\":" << (std::find(medoids.begin(), medoids.end(), ci) != medoids.end() ? 1 : 0)
                       << ",\"score\":" << evaluateBoard(res)
                       << ",\"new_stone_x\":" << (found_new ? new_x : 0.0f)
                       << ",\"new_stone_y\":" << (found_new ? new_y : 0.0f)
                       << ",\"new_stone_found\":" << (found_new ? 1 : 0)
                       << ",\"result_stones\":[";

                    bool first_rs = true;
                    for (int t = 0; t < 2; t++) {
                        for (int idx = 0; idx < 8; idx++) {
                            if (!res.stones[t][idx]) continue;
                            if (!first_rs) jf << ",";
                            first_rs = false;
                            jf << "{\"team\":" << t << ",\"index\":" << idx
                               << ",\"x\":" << res.stones[t][idx]->position.x
                               << ",\"y\":" << res.stones[t][idx]->position.y << "}";
                        }
                    }
                    jf << "]}";
                    if (ci < n - 1) jf << ",";
                    jf << "\n";
                }
                jf << "  ],\n";

                // 最良手インデックス
                jf << "  \"pool_best_idx\": " << best_pool_idx << ",\n";
                jf << "  \"dc_best_idx\": " << best_dc_idx << ",\n";
                jf << "  \"pool_best_score\": " << best_pool_score << ",\n";
                jf << "  \"dc_best_score\": " << best_dc_score << "\n";
                jf << "}\n";
                jf.close();
                std::cout << "  >> Detail JSON exported: " << json_path << std::endl;
            }
        }
    }

    // ========== サマリーテーブル ==========
    std::cout << "\n========================================================================" << std::endl;
    std::cout << "  Best Shot Agreement by Retention Ratio & Method" << std::endl;
    std::cout << "========================================================================" << std::endl;

    // 保持率 × 手法 別の集計
    struct AggKey {
        int ratio_pct;
        std::string method;
        bool operator<(const AggKey& o) const {
            if (ratio_pct != o.ratio_pct) return ratio_pct < o.ratio_pct;
            return method < o.method;
        }
    };
    std::map<AggKey, int> agg_count, agg_exact, agg_same_type;
    std::map<AggKey, float> agg_score_diff_sum;
    std::map<AggKey, long long> agg_dist_comp_sum;
    // DeltaClustered専用集計
    std::map<int, float> dc_purity_sum;
    std::map<int, int> dc_same_cluster_count, dc_count;

    for (auto& row : results) {
        AggKey key{static_cast<int>(std::round(row.ratio * 100)), row.method};
        agg_count[key]++;
        if (row.same_shot) agg_exact[key]++;
        if (row.same_type) agg_same_type[key]++;
        agg_score_diff_sum[key] += std::abs(row.score_diff);
        agg_dist_comp_sum[key] += row.dist_computations;

        if (row.method == "DeltaClustered") {
            int rp = static_cast<int>(std::round(row.ratio * 100));
            dc_purity_sum[rp] += row.weighted_purity;
            if (row.pool_best_in_same_cluster) dc_same_cluster_count[rp]++;
            dc_count[rp]++;
        }
    }

    std::cout << std::setw(8) << "Ratio"
              << std::setw(16) << "Method"
              << std::setw(16) << "Exact Match"
              << std::setw(16) << "Same Type"
              << std::setw(14) << "Avg|SDiff|"
              << std::setw(14) << "AvgDistComp" << std::endl;
    std::cout << std::string(84, '-') << std::endl;

    for (auto& [key, cnt] : agg_count) {
        int exact = agg_exact[key];
        int same = agg_same_type[key];
        float avg_diff = agg_score_diff_sum[key] / cnt;
        long long avg_dist = agg_dist_comp_sum[key] / cnt;
        std::cout << std::setw(6) << key.ratio_pct << "%"
                  << std::setw(16) << key.method
                  << std::setw(6) << exact << "/" << cnt
                  << " (" << std::setw(3) << std::fixed << std::setprecision(0)
                  << (100.0f * exact / cnt) << "%)"
                  << std::setw(6) << same << "/" << cnt
                  << " (" << std::setw(3) << (100.0f * same / cnt) << "%)"
                  << std::setw(10) << std::setprecision(2) << avg_diff
                  << std::setw(14) << avg_dist
                  << std::endl;
    }

    // ========== DeltaClustered: Purity & Same Cluster分析 ==========
    std::cout << "\n========================================================================" << std::endl;
    std::cout << "  DeltaClustered: Cluster Purity & Same Cluster Analysis" << std::endl;
    std::cout << "========================================================================" << std::endl;
    std::cout << "  Purity: クラスタ内の最多タイプの割合（1.0=全メンバーが同一タイプ）" << std::endl;
    std::cout << "  Same Cluster: 歩の最良手とDCの最良手が同じクラスタ内にある割合" << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(8) << "Ratio"
              << std::setw(16) << "Avg Purity"
              << std::setw(20) << "Same Cluster"
              << std::setw(16) << "Exact Match" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    for (auto& [rp, cnt] : dc_count) {
        float avg_purity = dc_purity_sum[rp] / cnt;
        int same_cl = dc_same_cluster_count[rp];
        AggKey dc_key{rp, "DeltaClustered"};
        int exact = agg_exact[dc_key];
        std::cout << std::setw(6) << rp << "%"
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_purity
                  << std::setw(10) << same_cl << "/" << cnt
                  << " (" << std::setw(3) << std::setprecision(0) << (100.0f * same_cl / cnt) << "%)"
                  << std::setw(7) << exact << "/" << cnt
                  << " (" << std::setw(3) << (100.0f * exact / cnt) << "%)"
                  << std::endl;
    }
    std::cout << "\n  解釈:" << std::endl;
    std::cout << "  - Purity≈1.0: クラスタ≒タイプ分類 → クラスタリング不要" << std::endl;
    std::cout << "  - Purity<1.0: クラスタがタイプを横断 → クラスタリングに独自の価値あり" << std::endl;
    std::cout << "  - Same Cluster > Exact Match: 歩の手を含むクラスタの代表を選べている" << std::endl;

    // CSVエクスポート
    {
        std::string csv_path = output_dir + "/three_method_comparison.csv";
        std::ofstream ofs(csv_path);
        ofs << "state,total,ratio,k,method,"
            << "pool_best_label,pool_best_type,pool_best_score,"
            << "sel_best_label,sel_best_type,sel_best_score,"
            << "same_shot,same_type,score_diff,dist_computations,"
            << "weighted_purity,pool_best_in_same_cluster" << std::endl;
        for (auto& row : results) {
            ofs << row.state << "," << row.total_candidates << ","
                << std::fixed << std::setprecision(2) << row.ratio << "," << row.k << ","
                << row.method << ","
                << "\"" << row.pool_best_label << "\"," << row.pool_best_type << ","
                << std::setprecision(1) << row.pool_best_score << ","
                << "\"" << row.sel_best_label << "\"," << row.sel_best_type << ","
                << row.sel_best_score << ","
                << (row.same_shot ? 1 : 0) << "," << (row.same_type ? 1 : 0) << ","
                << row.score_diff << "," << row.dist_computations << ","
                << std::setprecision(4) << row.weighted_purity << ","
                << (row.pool_best_in_same_cluster ? 1 : 0) << std::endl;
        }
        std::cout << "\nCSV exported to: " << csv_path << std::endl;
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  3-Method Comparison Complete" << std::endl;
    std::cout << "  Total test states: " << test_states.size() << std::endl;
    std::cout << "================================================================" << std::endl;
}
