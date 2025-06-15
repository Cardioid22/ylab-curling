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
    score(0.0),
    degree(0),
    label(0)
{
    static int global_label = 0;
    label = global_label++; // for debugging

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
    if (untried_shots) {
        return untried_shots->empty() || degree == max_degree;
    }
    return degree == max_degree;
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
        child->score = uct_score;
        std::cout << "Score: " << uct_score << "\n";

        if (uct_score > best_score) {
            best_score = uct_score;
            best = child.get();
        }
    }
    return best;
}

MCTS_Node* MCTS_Node::select_worst_child(double c) {
    MCTS_Node* worst = nullptr;
    double worst_score = std::numeric_limits<double>::infinity();

    for (const auto& child : children) {
        double exploit = child->visits > 0
            ? static_cast<double>(child->wins) / child->visits
            : 0.0;
        double explore = std::sqrt(std::log(visits + 1) / static_cast<double>(child->visits));
        double uct_score = exploit + c * explore;
        child->score = uct_score;
        std::cout << "Score (opponent): " << uct_score << "\n";

        if (uct_score < worst_score) {
            worst_score = uct_score;
            worst = child.get();
        }
    }
    return worst;
}

void MCTS_Node::expand(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> vx_dist(-0.3f, 0.3f);
    std::uniform_real_distribution<float> vy_dist(2.3f, 2.5f);
    if (terminal) {
        std::cout << "Game Reached the End\n";
        return;
    }
    if (!selected) {
        std::cout << "MCTS_Node was not selected before. Generating possible shots...\n";
        if (!IsOpponentTurn()) {
            untried_shots = std::make_unique<std::vector<ShotInfo>>(
                generate_possible_shots_after(all_states, state_to_shot_table)
            );
        }
        std::cout << "Possible shots generated properly.\n";
        selected = true;
    }
    if (selected && is_fully_expanded()) {
        std::cerr << "Warning: Cannot expanded this node any more!" << "\n";
        return;
    }
    ShotInfo shot = { 0.3, 2.5, 0 };
    if (IsOpponentTurn()) {
        // randomly pick one shot and create a new child node
        std::cout << "Opponent Turn. Genrating Random Shots...\n";
        float vx = vx_dist(gen);
        float vy = vy_dist(gen);
        int rot = (vx > 0.0f) ? 0 : 1;
        ShotInfo random_shot = {vx, vy, rot};
        shot = random_shot;
    }
    else {
        // Pick one untried shot and create a new child node
        std::cout << "My Turn. Genrating Shot From Untried_Shots...\n";
        shot = untried_shots->back();
        untried_shots->pop_back();
    }
    //int c_shot = static_cast<int>(state.shot);
    //std::cout << "Current Shot is " << c_shot << "\n";
    dc::GameState next_state = getNextState(shot);
    //int n_shot = static_cast<int>(next_state.shot);
    //std::cout << "Next Shot is " << n_shot << "\n";
    auto child_node = std::make_unique<MCTS_Node>(
        this,
        next_state,
        simulator,
        std::nullopt,  // child will generate their own if selected later
        shot
    );
    child_node->rollout();
    if (children.size() <= max_degree) {
        children.push_back(std::move(child_node));
        degree++;
    }
    std::cout << "MCTS_Node New Node Generated Done. Degree: " << degree << ",# of children: " << children.size() << ", # of Untried Shots: " << untried_shots->size() << "\n";
}
void MCTS_Node::rollout() {
    std::cout << "Rollout from node #" << label << "\n";
    double game_score = terminal
        ? simulator->evaluate(state)
        : simulator->run_simulations(state, selected_shot);
    wins += game_score > 0 ? 1 : 0;
    visits += 1;
    backpropagate(wins, 1);
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
dc::GameState MCTS_Node::getNextState(ShotInfo shotinfo) const {
    dc::GameState next_state = simulator->run_single_simulation(state, shotinfo);
    return next_state;
}

bool MCTS_Node::IsOpponentTurn() const {
    //dc::Team my_team = simulator->g_team;
    //dc::Team o_team = dc::GetOpponentTeam(my_team);
    //int current_shot = static_cast<int>(state.shot);
    //if (o_team == state.hammer) {
    //    return current_shot % 2 == 1;
    //}
    //else {
    //    return current_shot % 2 == 0;
    //}
    return false;
}

void MCTS_Node::print_tree(int indent) const {
    std::cout << std::string(indent, ' ')
        << "Node #" << label
        << " (visits: " << visits << ", wins: " << wins
        << ")\n";
    for (const auto& child : children) {
        child->print_tree(indent + 2);
    }
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
            MCTS_Node* next;
            if (node->IsOpponentTurn()) {
                next = node->select_worst_child();
            }
            else {
                next = node->select_best_child();
            }
            if (next == nullptr) {
                std::cerr << "No selectable child found.\n";
                break;
            }
            node = next;
            std::cout << "Switched Root Node.\n";
        }
        std::cout << "Expand Node #" << node->label << "\n";
        node->expand(all_states_, state_to_shot_table_);
        //node->rollout();
        //node->backpropagate(node->wins, 1);
    }
    root_->print_tree();
    std::cout << "MCTS Iteration Done.\n";
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