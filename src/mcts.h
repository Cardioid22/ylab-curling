#pragma once
#ifndef _MCTS_H_
#define _MCTS_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include "simulator.h"
#include <vector>
namespace dc = digitalcurling3;

class MCTS_Node {

public:
	MCTS_Node* parent;
	std::vector<std::unique_ptr<MCTS_Node>> children;
	int visits;
	int wins;
	double score;
	dc::GameState state;
	ShotInfo selected_shot;
	std::vector<ShotInfo> untried_shots;
	bool terminal;
	SimulatorWrapper simulator_wrapper;
	MCTS_Node(MCTS_Node* parent, std::vector<ShotInfo> shot_candidates, dc::GameState& const game_state);
	bool is_fully_expanded() const;
	MCTS_Node* select_best_child(double c = 1.41);
	void expand();
	void rollout();
	double calculate_winrate() const ;
	void backpropagate(double w, int n);
private:
	std::vector<ShotInfo> MCTS_Node::generate_possible_shots_after(const ShotInfo& shotinfo, dc::GameState& const game_state);
	dc::GameState MCTS_Node::getNextState(ShotInfo shotinfo);
};

class MCTS {
public:
	MCTS(MCTS_Node* root);
	MCTS_Node* select_best_child();
	void grow_tree(int max_iter, double max_limited_time);


};

#endif // _MCTS_H_

