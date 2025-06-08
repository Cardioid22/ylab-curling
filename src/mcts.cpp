#include <iostream>
#include <limits>
#include "mcts.h"
#include "digitalcurling3/digitalcurling3.hpp"
#include "simulator.h"

namespace dc = digitalcurling3;

MCTS_Node::MCTS_Node(MCTS_Node* parent, std::vector<ShotInfo> shot_candidates, dc::GameState& const game_state)
: parent(parent), untried_shots(shot_candidates), state(game_state), visits(0), wins(0), score(0.0), state_to_shot_table(state_to_shot_table){
    selected_shot = shot_candidates.front();
	terminal = state.IsGameOver();
	
}

bool MCTS_Node::is_fully_expanded() const{
	return untried_shots.empty();
}
MCTS_Node* MCTS_Node::select_best_child(double c) {
    MCTS_Node* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
        if (child->visits == 0) continue;

        double exploit = child->wins / static_cast<double>(child->visits);
        double explore = std::sqrt(std::log(visits + 1) / static_cast<double>(child->visits));
        double uct_score = exploit + c * explore;

        if (uct_score > best_score) {
            best_score = uct_score;
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

    dc::GameState next_state = getNextState(shot);
    std::vector<ShotInfo> next_shots = generate_possible_shots_after(shot, next_state);
    auto new_node = std::make_unique<MCTS_Node>(this, next_shots, next_state);
    MCTS_Node* new_node_ptr = new_node.get();
    children.push_back(std::move(new_node));
    new_node_ptr->rollout(); // start rollout from this child
}
void MCTS_Node::rollout() {
    double game_score = terminal
        ? simulator_wrapper.evaluate(state)
        : simulator_wrapper.run_simulation(state, selected_shot);

    wins += game_score > 0 ? 1 : 0;
    visits += 1;

    backpropagate(game_score > 0 ? 1.0 : 0.0, 1);
}
double MCTS_Node::calculate_winrate() const {
    return visits == 0 ? 0.0 : static_cast<double>(wins) / visits;
}
void MCTS_Node::backpropagate(double w, int n) {
    score += w;
    visits += n;
    if (parent) {
        parent->backpropagate(w, n);
    }
}
// shot candidates for the next node
std::vector<ShotInfo> MCTS_Node::generate_possible_shots_after(const ShotInfo& shotinfo, dc::GameState& const game_state) {
    std::vector<ShotInfo> candidates;
    auto clusters = algo.getClusters();
    auto recommended_states = algo.getRecommndedStates(clusters);
    for (auto const& state : recommended_states) {
        auto shot = state_to_shot_table[state];
        candidates.push_back(shot);
    }
    return candidates;
}
// get next state through shotinfo
dc::GameState MCTS_Node::getNextState(ShotInfo shotinfo) {
    dc::GameState next_state = state;
    simulator_wrapper.run_single_simulation(next_state, shotinfo);
    return next_state;
}

MCTS::MCTS(dc::GameState root_state, std::vector<ShotInfo> root_shots) {
    root = std::make_unique<MCTS_Node>(nullptr, root_shots, root_state);
}
void MCTS::grow_tree(int max_iter, double max_limited_time) {
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();

    for (int iter = 0; iter < max_iter; ++iter) {
        if (max_limited_time > 0) {
            double elapsed = std::chrono::duration<double>(Clock::now() - start_time).count();
            if (elapsed >= max_limited_time) {
                std::cerr << "Early Stopped: " << iter << "iterations in " << elapsed << "seconds." << "\n";
                break;
            }
        }

        // 1. Selection
        MCTS_Node* node = root.get();
        while (!node->untried_shots.empty() && node->is_fully_expanded()) {
            node = node->select_best_child();  // Traverse to leaf
        }
        node->expand();
    }
}
MCTS_Node* MCTS::select_best_child() {
    MCTS_Node* best = nullptr;
    int max_score = -1;

    for (const auto& child : root->children) {
        if (child->score > max_score) {
            best = child.get();
            max_score = child->score;
        }
    }
    return best;
}
ShotInfo MCTS::get_best_shot() {
    MCTS_Node* best_node = select_best_child();
    if (!best_node) {
        std::cerr << "No children found after tree search" << "\n";
    }
    return best_node->selected_shot;
}