#include "trajectory_export.h"
#include "../src/simulator.h"
#include "../src/structure.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dc = digitalcurling3;

namespace {

struct LoadedRow {
    int game_id = 0;
    int end = 0;
    int shot_num = 0;
    int team = 0;
    dc::GameState state;
};

bool loadRow(const std::string& csv_path, int row_index,
             dc::GameSetting const& gs, LoadedRow& out)
{
    std::ifstream ifs(csv_path);
    if (!ifs) {
        std::cerr << "Cannot open CSV: " << csv_path << std::endl;
        return false;
    }
    std::string header;
    std::getline(ifs, header);

    std::string line;
    int i = 0;
    while (std::getline(ifs, line)) {
        if (i == row_index) break;
        ++i;
    }
    if (line.empty()) {
        std::cerr << "Row " << row_index << " not found in " << csv_path << std::endl;
        return false;
    }

    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) cols.push_back(col);
    if (cols.size() < 4 + 16 * 3) {
        std::cerr << "Row has too few columns: " << cols.size() << std::endl;
        return false;
    }

    out.game_id = std::stoi(cols[0]);
    out.end = std::stoi(cols[1]);
    out.shot_num = std::stoi(cols[2]);
    out.team = std::stoi(cols[3]);

    dc::GameState st(gs);
    st.end = static_cast<std::uint8_t>(out.end);
    st.shot = static_cast<std::uint8_t>(out.shot_num);
    for (int t = 0; t < 2; ++t)
        for (int s = 0; s < 8; ++s)
            st.stones[t][s].reset();

    int idx = 4;
    for (int t = 0; t < 2; ++t) {
        for (int s = 0; s < 8; ++s) {
            int inplay = std::stoi(cols[idx++]);
            float x = std::stof(cols[idx++]);
            float y = std::stof(cols[idx++]);
            if (inplay == 1) st.stones[t][s].emplace(dc::Vector2(x, y), 0.0f);
        }
    }
    out.state = st;
    return true;
}

struct NamedShot {
    std::string name;
    dc::Vector2 target;       // 絶対座標 (sheet 上)
    float target_speed;       // 目標位置での残速度 [m/s]
    dc::moves::Shot::Rotation rot;
};

} // anonymous namespace

TrajectoryExportExperiment::TrajectoryExportExperiment(
    dc::GameSetting const& game_setting,
    const std::string& csv_path,
    int row_index,
    const std::string& output_dir)
    : game_setting_(game_setting), csv_path_(csv_path),
      row_index_(row_index), output_dir_(output_dir) {}

void TrajectoryExportExperiment::run() {
    std::cout << "=== Trajectory Export ===\n"
              << "  CSV: " << csv_path_ << "\n"
              << "  Row: " << row_index_ << "\n"
              << "  Out: " << output_dir_ << std::endl;

    LoadedRow row;
    if (!loadRow(csv_path_, row_index_, game_setting_, row)) return;

    std::filesystem::create_directories(output_dir_);

    // 投擲チームに合わせてシミュレータを初期化
    dc::Team thrower = (row.team == 0) ? dc::Team::k0 : dc::Team::k1;
    SimulatorWrapper sim(thrower, game_setting_);
    sim.setDeterministic(true);  // ノイズ無しで再現可能

    // 代表的なショット (絶対座標、HouseCenter = (0, 38.405))
    std::vector<NamedShot> shots = {
        {"draw_button",   dc::Vector2(0.0f,  38.405f), 0.0f, dc::moves::Shot::Rotation::kCCW},
        {"guard_front",   dc::Vector2(0.0f,  36.30f),  0.0f, dc::moves::Shot::Rotation::kCCW},
        {"draw_corner_l", dc::Vector2(-1.0f, 38.405f), 0.0f, dc::moves::Shot::Rotation::kCW},
        {"draw_corner_r", dc::Vector2(1.0f,  38.405f), 0.0f, dc::moves::Shot::Rotation::kCCW},
        {"takeout_center",dc::Vector2(0.0f,  38.405f), 2.5f, dc::moves::Shot::Rotation::kCCW},
    };

    // 出力 CSV
    std::string out_csv = output_dir_ + "/trajectories.csv";
    std::ofstream ofs(out_csv);
    ofs << "shot_id,shot_name,frame,stone_id,team,stone_idx,x,y\n";

    // 初期盤面 (frame=-1, shot_id=-1) を最初に出力
    for (int t = 0; t < 2; ++t) {
        for (int s = 0; s < 8; ++s) {
            if (row.state.stones[t][s]) {
                auto p = row.state.stones[t][s]->position;
                ofs << -1 << ",initial," << -1 << ","
                    << (t * 8 + s) << "," << t << "," << s << ","
                    << p.x << "," << p.y << "\n";
            }
        }
    }

    int shot_id = 0;
    for (auto const& sh : shots) {
        dc::Vector2 v = sim.EstimateShotVelocityFCV1(sh.target, sh.target_speed, sh.rot);
        ShotInfo info;
        info.vx = v.x;
        info.vy = v.y;
        info.rot = (sh.rot == dc::moves::Shot::Rotation::kCW) ? 1 : 0;

        std::vector<SimulatorWrapper::TrajectoryFrame> frames;
        // frame_stride=4: ~30 fps 相当に間引いて軽量化 (FCV1 は ~120 Hz)
        sim.run_single_simulation_with_trajectory(row.state, info, frames, 4);

        std::cout << "  " << sh.name << ": " << frames.size() << " frames\n";

        for (size_t f = 0; f < frames.size(); ++f) {
            for (int i = 0; i < 16; ++i) {
                if (frames[f].stones[i]) {
                    int t = i / 8;
                    int s = i % 8;
                    auto p = *frames[f].stones[i];
                    ofs << shot_id << "," << sh.name << "," << f << ","
                        << i << "," << t << "," << s << ","
                        << p.x << "," << p.y << "\n";
                }
            }
        }
        ++shot_id;
    }

    ofs.close();
    std::cout << "Wrote " << out_csv << std::endl;
}
