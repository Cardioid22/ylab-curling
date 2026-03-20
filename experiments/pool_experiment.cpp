#include "pool_experiment.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <iomanip>

namespace dc = digitalcurling3;

static constexpr float kHouseCenterX = 0.0f;
static constexpr float kHouseCenterY = 38.405f;
static constexpr float kHouseRadius = 1.829f;

PoolExperiment::PoolExperiment(dc::GameSetting const& game_setting)
    : game_setting_(game_setting)
{
}

std::vector<Position> PoolExperiment::makeGrid(int m, int n) {
    const float x_min = -kHouseRadius;
    const float x_max = kHouseRadius;
    const float y_min = 36.0f;
    const float y_max = 40.0f;

    float x_grid = (x_max - x_min) / (m - 1);
    float y_grid = (y_max - y_min) / (n - 1);
    float x_start = -((m - 1) / 2.0f) * x_grid;

    std::vector<Position> result;
    for (int j = 0; j < n; ++j) {
        float y = y_max - j * y_grid;
        for (int i = 0; i < m; ++i) {
            float x = x_start + i * x_grid;
            result.push_back({ x, y });
        }
    }
    return result;
}

std::vector<dc::GameState> PoolExperiment::createTestStates() {
    std::vector<dc::GameState> states;

    // テスト盤面1: 空場 (End 0, Shot 0)
    {
        dc::GameState s(game_setting_);
        s.shot = 0;
        states.push_back(s);
    }

    // テスト盤面2: 相手石がティー近くに1個
    {
        dc::GameState s(game_setting_);
        s.shot = 2;  // 先手2投目
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY), 0.f));
        states.push_back(s);
    }

    // テスト盤面3: 相手石2個 + 自分石1個 (典型的な中盤)
    {
        dc::GameState s(game_setting_);
        s.shot = 4;  // team0の手番 (偶数 = team0)
        // 自分(team0)の石1個
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(-0.5f, kHouseCenterY + 0.3f), 0.f));
        // 相手(team1)の石2個
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.3f, kHouseCenterY - 0.2f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.1f, kHouseCenterY + 0.8f), 0.f));
        states.push_back(s);
    }

    // テスト盤面4: 混雑した盤面 (多くの石)
    {
        dc::GameState s(game_setting_);
        s.shot = 10;
        // 自分石3個
        s.stones[0][0].emplace(dc::Transform(dc::Vector2(0.0f, kHouseCenterY), 0.f));       // ティー
        s.stones[0][1].emplace(dc::Transform(dc::Vector2(-1.0f, kHouseCenterY - 0.5f), 0.f));
        s.stones[0][2].emplace(dc::Transform(dc::Vector2(0.5f, kHouseCenterY + 1.0f), 0.f));
        // 相手石3個
        s.stones[1][0].emplace(dc::Transform(dc::Vector2(0.2f, kHouseCenterY + 0.1f), 0.f));
        s.stones[1][1].emplace(dc::Transform(dc::Vector2(-0.3f, kHouseCenterY + 1.5f), 0.f));
        s.stones[1][2].emplace(dc::Transform(dc::Vector2(0.8f, kHouseCenterY - 0.8f), 0.f));
        states.push_back(s);
    }

    return states;
}

void PoolExperiment::exportPoolToCSV(
    const CandidatePool& pool,
    const std::string& filename
) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return;
    }

    // ヘッダー
    ofs << "index,type,spin,target_index,param,label,vx,vy,rot";
    // 結果盤面の石の位置も出力
    ofs << ",result_shot_num";
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < 8; ++i) {
            ofs << ",team" << t << "_stone" << i << "_x"
                << ",team" << t << "_stone" << i << "_y"
                << ",team" << t << "_stone" << i << "_active";
        }
    }
    ofs << "\n";

    for (size_t c = 0; c < pool.candidates.size(); ++c) {
        auto& cand = pool.candidates[c];
        auto& result = pool.result_states[c];

        ofs << c << ","
            << static_cast<int>(cand.type) << ","
            << cand.spin << ","
            << cand.target_index << ","
            << cand.param << ","
            << "\"" << cand.label << "\","
            << std::fixed << std::setprecision(6)
            << cand.shot.vx << ","
            << cand.shot.vy << ","
            << cand.shot.rot << ","
            << result.shot;

        // 各石の位置
        for (int t = 0; t < 2; ++t) {
            for (int i = 0; i < 8; ++i) {
                auto& stone = result.stones[t][i];
                if (stone.has_value()) {
                    ofs << "," << stone->position.x
                        << "," << stone->position.y
                        << ",1";
                } else {
                    ofs << ",0,0,0";
                }
            }
        }
        ofs << "\n";
    }

    std::cout << "  Exported " << pool.candidates.size() << " candidates to " << filename << std::endl;
}

void PoolExperiment::runPoolGeneration() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Extended Candidate Pool Experiment" << std::endl;
    std::cout << "========================================" << std::endl;

    auto grid = makeGrid(4, 4);
    auto test_states = createTestStates();

    // 出力ディレクトリ
    std::string output_dir = "experiments/pool_experiment_results";
    std::filesystem::create_directories(output_dir);

    ShotGenerator generator(game_setting_);

    std::string state_names[] = { "empty", "opp1_tee", "opp2_my1", "crowded" };

    for (size_t s = 0; s < test_states.size(); ++s) {
        auto& state = test_states[s];
        dc::Team my_team = dc::Team::k0;

        std::cout << "\n--- Test State " << s << " (" << state_names[s] << ") ---" << std::endl;
        std::cout << "  Shot number: " << state.shot << std::endl;

        // 候補手生成のみの時間計測
        auto t0 = std::chrono::high_resolution_clock::now();
        auto candidates = generator.generateCandidates(state, my_team, grid);
        auto t1 = std::chrono::high_resolution_clock::now();
        double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "  Candidates generated: " << candidates.size() << " (" << gen_ms << " ms)" << std::endl;

        // タイプ別の集計
        int count_draw = 0, count_hit = 0, count_freeze = 0, count_guard = 0, count_other = 0;
        for (auto& c : candidates) {
            switch (c.type) {
                case ShotType::DRAW: ++count_draw; break;
                case ShotType::HIT: ++count_hit; break;
                case ShotType::FREEZE: ++count_freeze; break;
                case ShotType::PREGUARD: case ShotType::POSTGUARD: ++count_guard; break;
                default: ++count_other; break;
            }
        }
        std::cout << "  Breakdown: Draw=" << count_draw
                  << " Hit=" << count_hit
                  << " Freeze=" << count_freeze
                  << " Guard=" << count_guard
                  << " Other=" << count_other << std::endl;

        // シミュレーション付きプール生成
        auto t2 = std::chrono::high_resolution_clock::now();
        auto pool = generator.generatePool(state, my_team, grid);
        auto t3 = std::chrono::high_resolution_clock::now();
        double pool_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << "  Pool with simulation: " << pool_ms << " ms"
                  << " (" << std::fixed << std::setprecision(1)
                  << pool_ms / pool.candidates.size() << " ms/candidate)" << std::endl;

        // CSV出力
        std::string csv_path = output_dir + "/pool_" + state_names[s] + ".csv";
        exportPoolToCSV(pool, csv_path);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Pool Experiment Complete" << std::endl;
    std::cout << "========================================" << std::endl;
}
