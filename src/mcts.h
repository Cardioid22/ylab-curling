#pragma once
#ifndef _MCTS_H_
#define _MCTS_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include <vector>
namespace dc = digitalcurling3;

class MCTS_Node {

public:
	MCTS_Node* parent;
	std::vector<std::unique_ptr<MCTS_Node>> children;
	int visits;
	double score;
	dc::GameState node_state;
	std::vector<ShotInfo> shot_candidates;
	MCTS_Node(MCTS_Node* parent, std::vector<ShotInfo> shot_candidates);
	bool is_fully_expanded() const;
	MCTS_Node* select_best_child(double c = 1.41);
	void expand();
	double rollout();
	double calculate_winrate();
	void backpropagate(double w, int n);
	bool is_terminal() const;
};

class MCTS {
public:
	MCTS(MCTS_Node* root);
	MCTS_Node* select_best_child();
	void grow_tree(int max_iter, double max_limited_time);


};

#endif // _MCTS_H_

