#include <iostream>
#include <limits>
#include <memory>
#include <random>
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
    std::uniform_real_distribution<float> vx_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> vy_dist(2.4f, 2.6f);
    NodeSource shot_source;
    if (terminal) {
        return;
    }
    if (!selected) {
        std::cout << "MCTS_Node was not selected before.\n";
        if (NextIsOpponentTurn()) {
            std::cout << "Generating possible shots...\n";
            untried_shots = std::make_unique<std::vector<ShotInfo>>(
                generate_possible_shots_after(all_states, state_to_shot_table)
            );
            for (int i = 0; i < 3; i++) {
                float vx = vx_dist(gen);
                float vy = vy_dist(gen);
                int rot = (vx > 0.0f) ? 0 : 1;
                ShotInfo random_shot = { vx, vy, rot };
                untried_shots->push_back(random_shot);
            }
            untried_shots->push_back({ 0.1, 2.5, 0 });
            std::cout << "[After]Untried Shots Size: " << untried_shots->size() << "\n";
        }
        selected = true;
    }
    if (selected && is_fully_expanded()) {
        std::cerr << "Warning: Cannot expanded this node any more!" << "\n";
        return;
    }
    ShotInfo shot = { 0.3, 2.5, 0 };
    if (NextIsOpponentTurn()) {
        // Pick one untried shot and create a new child node
        std::cout << "My Turn. Genrating Shot From Untried_Shots...\n";
        if (untried_shots->size() > 4) {
            shot_source = NodeSource::Random;
        }
        else {
            shot_source = NodeSource::Clustered;
        }
        shot = untried_shots->back();
        untried_shots->pop_back();
        //std::cout << "Untried_Shots: " << untried_shots->size() << "\n";
    }
    else {
        // randomly pick one shot and create a new child node
        std::cout << "Opponent Turn. Genrating Random Shots...\n";
        float vx = vx_dist(gen);
        float vy = vy_dist(gen);
        int rot = (vx > 0.0f) ? 0 : 1;
        ShotInfo random_shot = { vx, vy, rot };
        shot = random_shot;
        shot_source = NodeSource::Random;
    }
    dc::GameState next_state = getNextState(shot);
    if (next_state.IsGameOver()) {
        terminal = true;
        std::cout << "New child state is gameover!\n";
        this->rollout();
        return;
    }
    auto child_node = std::make_unique<MCTS_Node>(
        this,
        next_state,
        simulator,
        std::nullopt,  // child will generate their own if selected later
        shot
    );
    child_node->source = shot_source;
    child_node->rollout();
    if (children.size() <= max_degree) {
        children.push_back(std::move(child_node));
        degree++;
    }
    std::cout << "MCTS_Node New Node Generated Done. Degree: " << degree << ",# of children: " << children.size() << "\n";
}
void MCTS_Node::rollout() {
    std::cout << "Rollout from node #" << label << "\n";
    double game_score = terminal
        ? simulator->evaluate(state)
        : simulator->run_simulations(state, selected_shot);
    wins = game_score > 0 ? 1 : 0;
    if (source == NodeSource::Clustered) {
        std::cout << "[Clustered] rollout score: " << game_score << "\n";
    }
    else {
        std::cout << "[Random] rollout score: " << game_score << "\n";
    }
    backpropagate(wins, 1); //if terminate is true, that node's visits will also +1 for now.
}
double MCTS_Node::calculate_winrate() const {
    return visits == 0 ? 0.0 : static_cast<double>(wins) / visits;
}
void MCTS_Node::backpropagate(double w, int n) {
    wins += w;
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
    std::vector<int> recommended_states;
    std::vector<ShotInfo> candidates;
    Clustering algo(4, all_states);
    recommended_states = algo.getRecommendedStates();
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

bool MCTS_Node::NextIsOpponentTurn() const {
    dc::Team current_move_team = state.GetNextTeam(); // not next, it's the team who decide move now.
    dc::Team my_team = simulator->g_team;
    return current_move_team == my_team;
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
    std::shared_ptr<SimulatorWrapper> simWrapper)
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
    max_iteration = max_iter;
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
            MCTS_Node* next = nullptr;
            if (node->NextIsOpponentTurn()) {
                std::cout << "Select Best Child\n";
                next = node->select_best_child();
            }
            else {
                std::cout << "Select Worst Child\n";
                next = node->select_worst_child();
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
    }
    best_child_ = root_->select_best_child(); // all root child has -inf for the shotinfo!!
    root_->print_tree();
    std::cout << "MCTS Iteration Done.\n";
}

ShotInfo MCTS::get_best_shot() {
    if (!best_child_) {
        std::cerr << "No children found after tree search. Returning default shot." << "\n";
        return ShotInfo{ 0.0f, 0.0f, 0 };
    }
    else {
        return best_child_->selected_shot;
    }
}

void MCTS::report_rollout_result() const {
    if (!root_) {
        std::cerr << "No root node.\n";
        return;
    }

    std::cout << "=== MCTS Rollout Result Summary ===\n";

    int clustered_count = 0, random_count = 0;
    float max_clustered_score = -1e9, max_random_score = -1e9;
    float total_clustered_score = 0.0, total_random_score = 0.0;

    for (const auto& child : root_->children) {
        std::string label;
        if (child->source == NodeSource::Clustered) {
            label = "Clustered";
            clustered_count++;
            total_clustered_score += child->score;
            max_clustered_score = std::max(max_clustered_score, child->score);
        }
        else if (child->source == NodeSource::Random) {
            label = "Random";
            random_count++;
            total_random_score += child->score;
            max_random_score = std::max(max_random_score, child->score);
        }
        else {
            label = "Unknown";
        }

        std::cout << "[" << label << "] "
            << "Visits: " << child->visits
            << ", Wins: " << child->wins
            << ", Score: " << child->score
            << ", Vx: " << child->selected_shot.vx
            << ", Vy: " << child->selected_shot.vy
            << ", Rotation: " << child->selected_shot.rot << "\n";
    }

    std::cout << "--- Summary ---\n";
    if (clustered_count > 0) {
        std::cout << "Clustered Avg Score: " << (total_clustered_score / clustered_count)
            << ", Max Score: " << max_clustered_score << "\n";
    }
    if (random_count > 0) {
        std::cout << "Random Avg Score: " << (total_random_score / random_count)
            << ", Max Score: " << max_random_score << "\n";
    }
    std::cout << "==============================\n";
}

void MCTS::export_rollout_result_to_csv(const std::string& filename, int shot_num) const {
    if (!root_) {
        std::cerr << "No root node to export.\n";
        return;
    }
    std::string folder = "../../MCTS_Output_" + std::to_string(max_iteration) + "_Iterations" + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    std::string new_filename = folder + filename + "_" + std::to_string(shot_num) + ".csv";
    std::ofstream file(new_filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    // Write CSV header
    file << "Type,Visits,Wins,Score\n";

    for (const auto& child : root_->children) {
        std::string label;
        if (child->source == NodeSource::Clustered) {
            label = "Clustered";
        }
        else if (child->source == NodeSource::Random) {
            label = "Random";
        }
        else {
            label = "Unknown";
        }

        file << label << ","
            << child->visits << ","
            << child->wins << ","
            << child->score << "\n";
    }

    file.close();
    std::cout << "Rollout result exported to: " << filename << "\n";
}
