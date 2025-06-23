#pragma once
#ifndef _MCTS_H_
#define _MCTS_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include "clustering.h"
#include "simulator.h"

namespace dc = digitalcurling3;

enum class NodeSource {
    Clustered,
    Random
};


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

    MCTS_Node(
        MCTS_Node* parent,
        dc::GameState const& game_state,
        std::shared_ptr<SimulatorWrapper> shared_sim,
        std::optional<std::vector<ShotInfo>> shot_candidates = std::nullopt,
        std::optional<ShotInfo> selected_shot = std::nullopt
    );
	bool is_fully_expanded() const;
	MCTS_Node* select_best_child(double c = 1.41);
    MCTS_Node* select_worst_child(double c = 1.41);
	void expand(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table);
	void rollout();
	double calculate_winrate() const;
	void backpropagate(double w, int n);
    bool NextIsOpponentTurn() const;
    void print_tree(int indent = 0) const;
private:
    int max_degree = 4;

	std::vector<ShotInfo> generate_possible_shots_after(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table);
	dc::GameState getNextState(ShotInfo shotinfo) const;
};

class MCTS {
public:
	MCTS(dc::GameState const& root_state, 
        std::vector<dc::GameState> states, 
        std::unordered_map<int, ShotInfo> state_to_shot_table,
        std::unique_ptr<SimulatorWrapper> simWrapper
    );

	void grow_tree(int max_iter, double max_limited_time);  // main loop
	ShotInfo get_best_shot();

private:
	std::unique_ptr<MCTS_Node> root_;
    std::vector<dc::GameState> all_states_;
    std::unordered_map<int, ShotInfo> state_to_shot_table_;
    std::shared_ptr<SimulatorWrapper> simulator_;
    MCTS_Node* best_child_ = nullptr;
};

#endif // _MCTS_H_

