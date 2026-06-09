#pragma once
#ifndef _TRAJECTORY_EXPORT_H_
#define _TRAJECTORY_EXPORT_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include <string>

namespace dc = digitalcurling3;

// Load one row from a test_positions batch CSV, run a few representative shots
// (draw / guard / takeout) from that game state, and write per-frame stone
// positions of every stone to a single CSV consumable by Python plotting.
//
// Output: <output_dir>/trajectories.csv with columns
//   shot_id,shot_name,frame,stone_id,team,stone_idx,x,y
// Stones not in play emit no row for that frame.
class TrajectoryExportExperiment {
public:
    TrajectoryExportExperiment(dc::GameSetting const& game_setting,
                               const std::string& csv_path,
                               int row_index,
                               const std::string& output_dir);
    void run();

private:
    dc::GameSetting game_setting_;
    std::string csv_path_;
    int row_index_;
    std::string output_dir_;
};

#endif
