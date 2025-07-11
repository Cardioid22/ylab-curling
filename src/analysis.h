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
void LinkageMatrixToCSV(const LinkageMatrix& linkage) const;
void SaveSimilarityTableToCSV(const std::vector<std::vector<float>>& table, int shot_number);
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
int GridSize_M = 4;
int GridSize_N = 4;
};

#endif