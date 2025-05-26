#include <iostream>
#include "mcts.h"
#include "digitalcurling3/digitalcurling3.hpp"

MCTS_Node::MCTS_Node(MCTS_Node* parent, std::vector<ShotInfo> shot_candidates) {

}

bool MCTS_Node::is_fully_expanded() const{
	return children.empty();
}
MCTS_Node* MCTS_Node::select_best_child(double c = 1.41) {

}
void MCTS_Node::expand() {

}
double MCTS_Node::rollout() {

}
double MCTS_Node::calculate_winrate() {

}
void MCTS_Node::backpropagate(double w, int n) {

}
bool MCTS_Node::is_terminal() const {

}

MCTS::MCTS(MCTS_Node* root) {

}
MCTS_Node* MCTS::select_best_child() {

}
void MCTS::grow_tree(int max_iter, double max_limited_time) {

}