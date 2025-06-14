#include <iostream>
#include <limits>
#include <memory>
#include "mcts.h"
#include "digitalcurling3/digitalcurling3.hpp"

namespace dc = digitalcurling3;

MCTS_Node::MCTS_Node(
    MCTS_Node* parent,
    dc::GameState const& game_state,
    std::shared_ptr<SimulatorWrapper> shared_sim,
    std::optional<std::vector<ShotInfo>> shot_candidates,
    std::optional<ShotInfo> selected_shot
)
    : parent(parent),
    state(game_state),
    selected_shot(selected_shot.value_or(ShotInfo{ 0, 0, 0 })), // default dummy
    terminal(false),
    visits(0),
    wins(0),
    score(0.0)
{
    if (shared_sim) {
        simulator = shared_sim;
    }
    else {
        std::cerr << "Simulator is NULL!\n";
    }
    if (shot_candidates.has_value()) {
        untried_shots = std::make_unique<std::vector<ShotInfo>>(std::move(shot_candidates.value()));
    }
}

bool MCTS_Node::is_fully_expanded() const{
    return untried_shots == nullptr || untried_shots->empty();
}
MCTS_Node* MCTS_Node::select_best_child(double c) {
    MCTS_Node* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
        double exploit = child->visits > 0
            ? static_cast<double>(child->wins) / child->visits
            : 0.0;
        double explore = std::sqrt(std::log(visits + 1) / static_cast<double>(child->visits));
        double uct_score = exploit + c * explore;
        score = uct_score;
        std::cout << "Score: " << score << "\n";

        if (uct_score > best_score) {
            best_score = uct_score;
            best = child.get();
        }
    }
    return best;
}

void MCTS_Node::expand(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table) {
    std::cout << "MCTS_Node Expand Begin.\n";
    if (terminal) {
        //rollout();
        return;
    }
    if (!selected) {
        std::cout << "MCTS_Node was not selected before. Generating possible shots...\n";
        untried_shots = std::make_unique<std::vector<ShotInfo>>(
            generate_possible_shots_after(all_states, state_to_shot_table)
        );
        std::cout << "Possible shots generated properly.\n";
        selected = true;
    }
    if (selected && is_fully_expanded()) {
        std::cerr << "Warning: Cannot expanded this node any more!" << "\n";
        return;
    }
    // Pick one untried shot and create a new child node
    ShotInfo shot = untried_shots->back();
    untried_shots->pop_back();

    dc::GameState child_state = getNextState(shot);
    auto child_node = std::make_unique<MCTS_Node>(
        this,
        child_state, // should be child_state
        simulator,
        std::nullopt,         // child will generate their own if selected later
        shot
    );
    children.push_back(std::move(child_node));
    std::cout << "MCTS_Node New Node Generated Done. # of Children: " << children.size() << ", # of Untried Shots: " << untried_shots->size() << "\n";
}
void MCTS_Node::rollout() {
    std::cout << "MCTS_Node Rollout Begin.\n";
    double game_score = terminal
        ? simulator->evaluate(state)
        : simulator->run_simulation(state, selected_shot);
    std::cout << "MCTS_Node Rollout Done.\n";
    wins += game_score > 0 ? 1 : 0;
    visits += 1;
}
double MCTS_Node::calculate_winrate() const {
    return visits == 0 ? 0.0 : static_cast<double>(wins) / visits;
}
void MCTS_Node::backpropagate(double w, int n) {
    visits += n;
    if (parent) {
        parent->backpropagate(w, n);
    }
}
// shot candidates for the next node
std::vector<ShotInfo> MCTS_Node::generate_possible_shots_after(
    const std::vector<dc::GameState> all_states, 
    const std::unordered_map<int, ShotInfo> state_to_shot_table) 
{
    std::vector<ShotInfo> candidates;
    Clustering algo(4, all_states);
    auto clusters = algo.getClusters();
    auto recommended_states = algo.getRecommendedStates(clusters);
    for (int state_index : recommended_states) {
        auto it = state_to_shot_table.find(state_index);
        if (it != state_to_shot_table.end()) {
            candidates.push_back(it->second);
        }
    }
    return candidates;
}
 //get next state through shotinfo
dc::GameState MCTS_Node::getNextState(ShotInfo shotinfo) {
    dc::GameState next_state = state;
    simulator->run_single_simulation(next_state, shotinfo);
    return next_state;
}

MCTS::MCTS(dc::GameState const& root_state, 
    std::vector<dc::GameState> states, 
    std::unordered_map<int, ShotInfo> state_to_shot_table, 
    std::unique_ptr<SimulatorWrapper> simWrapper)
: state_to_shot_table_(std::move(state_to_shot_table)), simulator_(std::move(simWrapper))
{
    all_states_.resize(states.size());
    std::copy(states.begin(), states.end(), all_states_.begin());
    std::shared_ptr<SimulatorWrapper> shared_sim = simulator_;
    root_ = std::make_unique<MCTS_Node>(nullptr, root_state, shared_sim);
}
void MCTS::grow_tree(int max_iter, double max_limited_time) {
    using Clock = std::chrono::high_resolution_clock;
    auto start_time = Clock::now();
    std::cout << "MCTS Tree Begin.\n";
    for (int iter = 0; iter < max_iter; ++iter) {
        if (max_limited_time > 0) {
            double elapsed = std::chrono::duration<double>(Clock::now() - start_time).count();
            if (elapsed >= max_limited_time) {
                std::cerr << "Early Stopped: " << iter << "iterations in " << elapsed << "seconds." << "\n";
                break;
            }
        }

        MCTS_Node* node = root_.get();
        while (node->selected && node->is_fully_expanded()) {
            MCTS_Node* next = node->select_best_child();
            if (next == nullptr) {
                std::cerr << "No selectable child found.\n";
                break; // Prevent null dereference
            }
            node = next;
            std::cout << "Switched Root Node.\n";
        }
        node->expand(all_states_, state_to_shot_table_);
        //node->rollout();
        std::cout << "MCTS_Node Backpropagate Begin.\n";
        node->backpropagate(node->wins, 1);
        std::cout << "MCTS_Node Backpropagate Done.\n";
    }
}

MCTS_Node* MCTS::get_best_child() {
    MCTS_Node* best = nullptr;
    int max_score = -std::numeric_limits<double>::infinity();

    for (const auto& child : root_->children) {
        if (child->score > max_score) {
            best = child.get();
            max_score = child->score;
        }
    }
    return best;
}

ShotInfo MCTS::get_best_shot() {
    MCTS_Node* best_node = get_best_child();
    if (!best_node) {
        std::cerr << "No children found after tree search. Returning default shot." << "\n";
        return ShotInfo{ 0.0f, 0.0f, 0 };
    }
    else {
        return best_node->selected_shot;
    }
}