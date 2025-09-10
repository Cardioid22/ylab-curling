#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <set>
#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#ifndef _Analysis_H
#define _Analysis_H

namespace dc = digitalcurling3;

class Analysis {
public:
Analysis(int grid_size_m, int grid_size_n);
void SaveSimilarityTableToCSV(const std::vector<std::vector<float>>& table, int shot_number) const;
void export_best_shot_comparison_to_csv(
    const ShotInfo& best,
    const ShotInfo& best_allgrid,
    int best_state,
    int best_allgrid_state,
    int shot_num,
    int iter,
    const std::string& filename
);
void cluster_id_to_state_csv(std::vector<std::vector<int>> cluster_id_to_state, int shot_num, int iter) const;
void OutputClusterGridToCSV(const std::vector<int>& state_index_to_cluster,
    int rows, int cols,
    const std::string& filename, const int shot_num);
void ExportStoneCoordinatesToCSV(const dc::GameState& game_state, const std::string& filename);
void ExportStonesByCluster(
    const std::vector<int>& state_index_to_cluster,
    const std::vector<dc::GameState>& all_game_states, const int shot_num);
void IntraToCSV(const std::vector<float>& scores, const int shot_num);
void SilhouetteToCSV(float score, int shot_num, int k_cluster);
float ComputeSilhouetteScore(
    const std::vector<std::vector<float>>& distance_matrix,
    const std::vector<int>& state_index_to_cluster
);
float ComputeIntraClusterDistance(
    const std::vector<std::vector<float>>& dist,
    const std::vector<std::set<int>>& clsters);

private:
int GridSize_M = 10;
int GridSize_N = 10;
};

#endif