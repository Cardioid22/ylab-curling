#pragma once
#ifndef _MCTS_H_
#define _MCTS_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include "clustering-v2.h"
#include "simulator.h"
#include <fstream>

namespace dc = digitalcurling3;

class MCTS_Node {

public:
    MCTS_Node* parent;
    std::vector<std::unique_ptr<MCTS_Node>> children;
    int visits = 0;
    int wins = 0;
    float score = 0.0;
    dc::GameState state;
    ShotInfo selected_shot;
    bool terminal = false;
    bool selected = false;
    int degree = 0;
    int label = 0;

    std::unique_ptr<std::vector<ShotInfo>> untried_shots;  // Make it optional/lazy
    std::shared_ptr<SimulatorWrapper> simulator;
    NodeSource source = NodeSource::Clustered;
    std::vector<std::vector<int>> clusters_id_to_state;

    MCTS_Node(
        MCTS_Node* parent,
        dc::GameState const& game_state,
        NodeSource node_source,
        std::shared_ptr<SimulatorWrapper> shared_sim,
        int gridM,
        int gridN,
        int cluster_num,
        int num_rollout_sims = 10,
        std::optional<std::vector<ShotInfo>> shot_candidates = std::nullopt,
        std::optional<ShotInfo> selected_shot = std::nullopt
    );
	bool is_fully_expanded() const;
	MCTS_Node* select_best_child(double c = 1.41);
    MCTS_Node* select_worst_child(double c = 1.41);
	MCTS_Node* select_most_visited_child();
	void expand(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table);
	void rollout();
	double calculate_winrate() const;
	void backpropagate(int w, int n);
    bool NextIsOpponentTurn() const;
    void print_tree(int indent = 0) const;
private:
    int GridSize_M_ = 10;
    int GridSize_N_ = 10;
    int max_degree = 6;
    int cluster_num_ = 4;  // Number of clusters for Clustered MCTS
    int num_rollout_simulations_ = 10;  // Number of simulations per rollout

	std::vector<ShotInfo> generate_possible_shots_after(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table) const;
	dc::GameState getNextState(ShotInfo shotinfo) const;
};

class MCTS {
public:
	MCTS(dc::GameState const& root_state,
        NodeSource node_source,
        std::vector<dc::GameState> states,
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        std::shared_ptr<SimulatorWrapper> simWrapper,
        int gridM,
        int gridN,
        int cluster_num = 4,
        int num_rollout_sims = 10
    );

	void grow_tree(int max_iter, double max_limited_time);  // main loop
	ShotInfo get_best_shot();
	double get_best_shot_winrate();
    void report_rollout_result() const;
    void export_rollout_result_to_csv(const std::string& filename, int shot_num, int grid_m, int grid_n, std::vector<ShotInfo> shotData) const;


private:
	std::unique_ptr<MCTS_Node> root_;
    std::vector<dc::GameState> all_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;
    std::shared_ptr<SimulatorWrapper> simulator_;
    MCTS_Node* best_child_ = nullptr;
    int max_iteration = 0;
    int clustered_rollouts = 0;
    double clustered_total_score = 0.0;

    int random_rollouts = 0;
    double random_total_score = 0.0;

    int cluster_num_ = 4;  // Number of clusters for Clustered MCTS
    int num_rollout_simulations_ = 10;  // Number of simulations per rollout

};

#endif // _MCTS_H_

