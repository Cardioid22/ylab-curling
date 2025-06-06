#include <iostream>
#include <limits>
#include "mcts.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include "simulator.h"

namespace dc = digitalcurling3;

MCTS_Node::MCTS_Node(MCTS_Node* parent, std::vector<ShotInfo> shot_candidates, dc::GameState& const game_state)
: parent(parent), untried_shots(shot_candidates), state(game_state), visits(0), wins(0), score(0.0) {
    selected_shot = shot_candidates.front();
	terminal = state.IsGameOver();
	
}

bool MCTS_Node::is_fully_expanded() const{
	return untried_shots.empty();
}
MCTS_Node* MCTS_Node::select_best_child(double c = 1.41) {
    MCTS_Node* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
        if (child->visits == 0) continue;
        if (score > best_score) {
            best_score = score;
            best = child.get();
        }
    }
    return best;
}
void MCTS_Node::expand() {
    if (terminal) {
        rollout();
        return;
    }
    if (is_fully_expanded()) {
        std::cerr << "Warning: Cannot expanded this node any more!" << "\n";
        return;
    }
    // Pick one untried shot and create a new child node
    ShotInfo shot = untried_shots.back();
    untried_shots.pop_back();
    dc::GameState next_state = getNextState(shot); // get next state through shotinfo
    std::vector<ShotInfo> next_shots = generate_possible_shots_after(shot, next_state);
    auto new_node = std::make_unique<MCTS_Node>(this, shot, next_state);
}
void MCTS_Node::rollout() {
    if (!terminal) {
        simulator_wrapper.run_simulation(state, selected_shot);
        
    }
}
double MCTS_Node::calculate_winrate() const {
    if (visits == 0) {
        return 0.0;
    }
    else {
        return wins / visits;
    }
}
void MCTS_Node::backpropagate(double w, int n) {
    score += w;
    visits += n;
    if (parent != NULL) {
        parent->visits++;
        parent->backpropagate(w, n);
    }
}

std::vector<ShotInfo> MCTS_Node::generate_possible_shots_after(const ShotInfo& shotinfo, dc::GameState& const game_state) {}
dc::GameState MCTS_Node::getNextState(ShotInfo shotinfo) {

}

MCTS::MCTS(MCTS_Node* root) {

}
MCTS_Node* MCTS::select_best_child() {

}
void MCTS::grow_tree(int max_iter, double max_limited_time) {

}