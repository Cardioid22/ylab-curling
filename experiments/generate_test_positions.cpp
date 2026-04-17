#include "generate_test_positions.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace dc = digitalcurling3;

GenerateTestPositionsExperiment::GenerateTestPositionsExperiment(
    dc::GameSetting const& game_setting)
    : game_setting_(game_setting)
{
    policy_ = std::make_unique<RolloutPolicy>();
    shot_gen_ = std::make_unique<ShotGenerator>(game_setting);
    rng_.seed(std::random_device{}());
}

// ============================================================
// 1ゲーム分を自己対戦し、全局面を記録
// 序盤 opening_random_ 手はランダム選択 → 盤面多様化
// ============================================================
std::vector<GenerateTestPositionsExperiment::PositionRecord>
GenerateTestPositionsExperiment::playOneGame(int match_id, SimulatorWrapper& sim)
{
    std::vector<PositionRecord> positions;

    dc::GameState state(game_setting_);
    state.end = 0;
    state.shot = 0;

    int max_shots = game_setting_.max_end * 16;
    int global_shot_counter = 0;

    for (int total_shot = 0; total_shot < max_shots && !state.IsGameOver(); ++total_shot) {
        int end = static_cast<int>(state.end);
        int shot_num = static_cast<int>(state.shot);
        dc::Team current_team = (shot_num % 2 == 0) ? dc::Team::k0 : dc::Team::k1;

        // 局面を記録
        PositionRecord rec;
        rec.match_id = match_id;
        rec.end = end;
        rec.shot_num = shot_num;
        rec.current_team = static_cast<int>(current_team);
        rec.state = state;
        positions.push_back(rec);

        // ロールアウト用簡易候補生成（高速）
        auto candidates = shot_gen_->generateRolloutCandidates(state, current_team);

        // Pass除外
        std::vector<CandidateShot> filtered;
        for (auto& c : candidates) {
            if (c.type != ShotType::PASS) filtered.push_back(c);
        }
        if (filtered.empty()) filtered = candidates;

        ShotInfo shot;
        if (global_shot_counter < opening_random_) {
            // 序盤: ランダム選択（盤面多様化のため）
            std::uniform_int_distribution<int> d(0, static_cast<int>(filtered.size()) - 1);
            shot = filtered[d(rng_)].shot;
        } else {
            // 中盤以降: gPolicyで選択
            int sel = policy_->selectShot(state, filtered, shot_num, current_team, end, 0);
            shot = filtered[sel].shot;
        }

        state = sim.run_single_simulation(state, shot);
        global_shot_counter++;
    }

    return positions;
}

// ============================================================
// バッチをCSVに出力
// ============================================================
void GenerateTestPositionsExperiment::exportBatch(
    const std::vector<PositionRecord>& positions,
    const std::string& dir, int batch_num)
{
    std::ostringstream oss;
    oss << dir << "/batch_" << std::setw(4) << std::setfill('0') << batch_num << ".csv";
    std::string path = oss.str();

    std::ofstream ofs(path);
    // ヘッダ: match_id, end, shot_num, team,
    //         各石 (team × 8) × (in_play, x, y) = 48カラム
    ofs << "match_id,end,shot_num,team";
    for (int t = 0; t < 2; ++t) {
        for (int s = 0; s < 8; ++s) {
            ofs << ",t" << t << "s" << s << "_inplay"
                << ",t" << t << "s" << s << "_x"
                << ",t" << t << "s" << s << "_y";
        }
    }
    ofs << std::endl;

    for (auto& p : positions) {
        ofs << p.match_id << "," << p.end << "," << p.shot_num << "," << p.current_team;
        for (int t = 0; t < 2; ++t) {
            for (int s = 0; s < 8; ++s) {
                auto& stone = p.state.stones[t][s];
                if (stone) {
                    ofs << ",1," << stone->position.x << "," << stone->position.y;
                } else {
                    ofs << ",0,0,0";
                }
            }
        }
        ofs << std::endl;
    }
    std::cout << "  Exported batch " << batch_num << ": " << positions.size()
              << " positions → " << path << std::endl;
}

// ============================================================
// タイムスタンプ付き出力ディレクトリ名
// ============================================================
std::string GenerateTestPositionsExperiment::makeOutputDir() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[20];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);

    std::string dir = std::string("test_positions_") + ts;
    std::filesystem::create_directories(dir);
    return dir;
}

// ============================================================
// メイン実行
// ============================================================
void GenerateTestPositionsExperiment::run() {
    std::cout << "=== Generate Test Positions ===" << std::endl;
    std::cout << "  Total games: " << total_games_ << std::endl;
    std::cout << "  Batch size: " << batch_size_ << std::endl;
    std::cout << "  Opening random shots: " << opening_random_ << std::endl;

    // gPolicy 読み込み
    if (!policy_->load("data/policy_param.dat")) {
        std::cerr << "Failed to load gPolicy. Aborting." << std::endl;
        return;
    }

    std::string dir = makeOutputDir();
    std::cout << "  Output directory: " << dir << std::endl << std::endl;

    SimulatorWrapper sim(dc::Team::k0, game_setting_);

    std::vector<PositionRecord> batch_buffer;
    int batch_num = 1;
    int positions_extracted = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int match = 0; match < total_games_; ++match) {
        // 1ゲーム自己対戦
        auto game_positions = playOneGame(match, sim);

        // ランダムに1局面を抽出
        if (!game_positions.empty()) {
            std::uniform_int_distribution<int> dist(0, static_cast<int>(game_positions.size()) - 1);
            int picked = dist(rng_);
            batch_buffer.push_back(game_positions[picked]);
            positions_extracted++;
        }

        // 進捗表示
        if ((match + 1) % 100 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            double rate = (match + 1) / elapsed;
            double eta = (total_games_ - match - 1) / rate;
            std::cout << "  [" << (match + 1) << "/" << total_games_ << "] "
                      << "elapsed=" << elapsed << "s, rate=" << rate << " games/s, "
                      << "ETA=" << eta << "s" << std::endl;
        }

        // バッチサイズに達したらCSV出力
        if (static_cast<int>(batch_buffer.size()) >= batch_size_) {
            exportBatch(batch_buffer, dir, batch_num);
            batch_buffer.clear();
            batch_num++;
        }
    }

    // 残りを出力
    if (!batch_buffer.empty()) {
        exportBatch(batch_buffer, dir, batch_num);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start).count();

    std::cout << "\n=== Complete ===" << std::endl;
    std::cout << "  Total matches: " << total_games_ << std::endl;
    std::cout << "  Positions extracted: " << positions_extracted << std::endl;
    std::cout << "  Total time: " << total_time << "s (" << total_time/60 << " min)" << std::endl;
    std::cout << "  Output directory: " << dir << std::endl;
}
