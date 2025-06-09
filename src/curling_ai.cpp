#include "digitalcurling3/digitalcurling3.hpp"
#include "clustering.h"
#include "structure.h"
#include "mcts.h"
#include "simulator.h"
#include <iostream>

namespace dc = digitalcurling3;

CurlingAI::CurlingAI(dc::Team team,
    const dc::GameSetting& game_setting,
    std::unique_ptr<dc::ISimulator> simulator,
    std::array<std::unique_ptr<dc::IPlayer>, 4> players)
    : team_(team), game_setting_(game_setting), simulator_(std::move(simulator)), players_(std::move(players)) {
}

void CurlingAI::Initialize() {
    grid_ = MakeGrid(GridSize_M, GridSize_N);
    shotData_.resize(GridSize_M, std::vector<ShotInfo>(GridSize_N));

    SimulatorWrapper simWrapper;
    for (int i = 0; i < GridSize_M; ++i) {
        for (int j = 0; j < GridSize_N; ++j) {
            shotData_[i][j] = simWrapper.FindShot(grid_[i][j]);
        }
    }
}

dc::Move CurlingAI::DecideMove(const dc::GameState& game_state) {
    grid_states_.clear();
    std::vector<ShotInfo> all_shots;

    SimulatorWrapper simWrapper;

    for (int i = 0; i < GridSize_M; ++i) {
        for (int j = 0; j < GridSize_N; ++j) {
            ShotInfo shot = shotData_[i][j];
            dc::GameState result = run(game_state, shot); // simulate one outcome
            grid_states_.push_back(result);
            all_shots.push_back(shot);
        }
    }

    // --- Clustering ---
    Clustering clustering;
    auto dist_table = clustering.MakeDistanceTable(grid_states_);
    auto linkage = clustering.hierarchicalClustering(dist_table, clustering.clusters, clustering.n_desired_clusters);

    std::vector<int> state_index_to_cluster(grid_states_.size());
    for (int i = 0; i < clustering.clusters.size(); ++i) {
        for (int idx : clustering.clusters[i]) {
            state_index_to_cluster[idx] = i;
        }
    }

    ExportStonesByCluster(state_index_to_cluster, grid_states_, game_state.shot);
    OutputClusterGridToCSV(state_index_to_cluster, GridSize_M, GridSize_N, "cluster_distribution_test", game_state.shot);

    // --- Extract Representative Shots ---
    std::vector<int> recommended_idxs = clustering.getRecommndedStates(clustering.clusters);
    std::vector<ShotInfo> candidate_shots;
    for (int idx : recommended_idxs) {
        candidate_shots.push_back(all_shots[idx]);
    }

    // --- MCTS Search ---
    MCTS mcts(game_state, candidate_shots);
    mcts.grow_tree(/*max_iter=*/500, /*max_limited_time=*/1.0); // adjust as needed

    ShotInfo best = mcts.get_best_shot();

    dc::moves::Shot final_shot;
    final_shot.velocity.x = best.vx;
    final_shot.velocity.y = best.vy;
    final_shot.rotation = best.rot == 1 ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;

    std::cout << "MCTS Selected Shot: " << best.vx << ", " << best.vy << "\n";
    return final_shot;
}