#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <chrono>
#include "mcts.h"
#include "digitalcurling3/digitalcurling3.hpp"
#define DBL_EPSILON 2.2204460492503131e-016
namespace dc = digitalcurling3;

MCTS_Node::MCTS_Node(
    MCTS_Node* parent,
    dc::GameState const& game_state,
    NodeSource node_source,
    std::shared_ptr<SimulatorWrapper> shared_sim,
    int gridM,
    int gridN,
    int cluster_num,
    int num_rollout_sims,
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
    label(0),
    source(node_source),
    cluster_num_(cluster_num),
    num_rollout_simulations_(num_rollout_sims)
{
    static int global_label = 0;
    label = global_label++; // for debugging
    GridSize_M_ = gridM;
    GridSize_N_ = gridN;
    if (shared_sim) {
        simulator = shared_sim;
    }
    else {
        std::cerr << "Simulator is NULL!\n";
    }
    if (shot_candidates.has_value()) {
        untried_shots = std::make_unique<std::vector<ShotInfo>>(std::move(shot_candidates.value()));
    }
    if (node_source == NodeSource::AllGrid) {
        max_degree = GridSize_M_ * GridSize_N_;
    }
    else {
        max_degree = cluster_num_;  // Use configurable cluster_num
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

MCTS_Node* MCTS_Node::select_most_visited_child() {
    MCTS_Node* best = nullptr;
    int max_visits = -1;

    for (const auto& child : children) {
        std::cout << "Child visits: " << child->visits
                  << ", wins: " << child->wins
                  << ", winrate: " << (child->visits > 0 ? static_cast<double>(child->wins) / child->visits : 0.0)
                  << "\n";

        if (child->visits > max_visits) {
            max_visits = child->visits;
            best = child.get();
        }
    }

    if (best) {
        std::cout << "[INFO] Selected most visited child with " << max_visits
                  << " visits (winrate: " << (best->visits > 0 ? static_cast<double>(best->wins) / best->visits : 0.0)
                  << ")\n";
    }

    return best;
}

void MCTS_Node::expand(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table) {
    static std::random_device rd;
    static std::mt19937 gen(10);
    std::uniform_real_distribution<float> vx_dist(-0.24f, 0.24f);
    std::uniform_real_distribution<float> vy_dist(2.38f, 2.48f);
    NodeSource shot_source;
    if (terminal) {
        rollout();
        return;
    }
    if (!selected) {
        std::cout << "MCTS_Node was not selected before.\n";
        if (NextIsOpponentTurn()) {
            std::cout << "Generating possible shots...\n";
            if (source == NodeSource::Clustered) {
                untried_shots = std::make_unique<std::vector<ShotInfo>>(
                    generate_possible_shots_after(all_states, state_to_shot_table)
                );
            }
            else if(source == NodeSource::Random){
                untried_shots = std::make_unique<std::vector<ShotInfo>>();
                for (int i = 0; i < max_degree; i++) {
                    float vx = vx_dist(gen);
                    float vy = vy_dist(gen);
                    int rot = 0;
                    if (std::abs(vx) >= 0.05f) {
                        rot = vx > 0 ? 0 : 1;
                    }
                    else {
                        rot = vx > 0 ? 1 : 0;
                    }
                    ShotInfo random_shot = { vx, vy, rot };
                    untried_shots->push_back(random_shot);
                }
            }
            else if (source == NodeSource::AllGrid) {
                untried_shots = std::make_unique<std::vector<ShotInfo>>();
                std::vector<ShotInfo> initialShots = simulator->initialShotData;
                for (int i = 0; i < initialShots.size(); i++) {
                    untried_shots->push_back(initialShots[i]);
                }
            }
            //untried_shots->push_back({ 0.1, 2.5, 0 });
            std::cout << "[After]Untried Shots Size: " << untried_shots->size() << "\n";
        }
        selected = true;
    }
    if (selected && is_fully_expanded()) {
        std::cerr << "Warning: Cannot expanded this node any more!" << "\n";
        return;
    }
    ShotInfo shot = { 0.1, 2.5, 0 };
    if (NextIsOpponentTurn()) {
        // Pick one untried shot and create a new child node
        std::cout << "My Turn. Genrating Shot From Untried_Shots...\n";
        //if (untried_shots->size() > max_degree / 2) {
        //    shot_source = NodeSource::Random;
        //}
        //else {
        //    shot_source = NodeSource::Clustered;
        //}
        shot_source = source;
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
        shot_source,
        simulator,
        GridSize_M_,
        GridSize_N_,
        cluster_num_,
        num_rollout_simulations_,
        std::nullopt,  // child will generate their own if selected later
        shot
    );
    child_node->source = shot_source;
    child_node->set_parent_mcts(parent_mcts_);  // Inherit parent MCTS pointer
    child_node->rollout();
    if (children.size() <= max_degree) {
        children.push_back(std::move(child_node));
        degree++;
    }
    std::cout << "MCTS_Node New Node Generated Done. Degree: " << degree << ",# of children: " << children.size() << "\n";
}
void MCTS_Node::rollout() {
    std::cout << "Rollout from node #" << label << " with " << num_rollout_simulations_ << " simulations\n";

    auto rollout_start = std::chrono::high_resolution_clock::now();

    double game_score;
    if (terminal) {
        game_score = simulator->evaluate(state);
    } else {
        // Use multiple simulations with random grid policy
        game_score = simulator->run_multiple_simulations_with_random_policy(
            state,
            selected_shot,
            num_rollout_simulations_
        );
    }

    auto rollout_end = std::chrono::high_resolution_clock::now();
    double rollout_time = std::chrono::duration<double>(rollout_end - rollout_start).count();

    // Track rollout timing in parent MCTS
    if (parent_mcts_ != nullptr) {
        parent_mcts_->total_rollout_time_ += rollout_time;
        parent_mcts_->total_rollout_count_ += 1;
    }

    wins = game_score > 0 ? 1 : 0;

    if (source == NodeSource::Clustered) {
        std::cout << "[Clustered] rollout avg score: " << game_score << "\n";
    }
    else if(source == NodeSource::Random){
        std::cout << "[Random] rollout avg score: " << game_score << "\n";
    }
    else if (source == NodeSource::AllGrid) {
        std::cout << "[AllGrid] rollout avg score: " << game_score << "\n";
    }
    backpropagate(wins, 1); //if terminate is true, that node's visits will also +1 for now.
}
double MCTS_Node::calculate_winrate() const {
    return visits == 0 ? 0.0 : static_cast<double>(wins) / visits;
}
void MCTS_Node::backpropagate(int w, int n) {
    wins += w;
    visits += n;
    if (parent) {
        parent->backpropagate(w, n);
    }
}
// shot candidates for the next node
std::vector<ShotInfo> MCTS_Node::generate_possible_shots_after(
    const std::vector<dc::GameState> all_states,
    const std::unordered_map<int, ShotInfo> state_to_shot_table) const
{
    std::vector<int> recommended_states;
    std::vector<ShotInfo> candidates;

    // Measure clustering time
    auto clustering_start = std::chrono::high_resolution_clock::now();
    ClusteringV2 algo(cluster_num_, all_states, GridSize_M_, GridSize_N_, simulator->g_team);
    recommended_states = algo.getRecommendedStates();
    auto clustering_end = std::chrono::high_resolution_clock::now();
    double clustering_time = std::chrono::duration<double>(clustering_end - clustering_start).count();

    // Track clustering timing in parent MCTS
    if (parent_mcts_ != nullptr) {
        parent_mcts_->total_clustering_time_ += clustering_time;
        parent_mcts_->total_clustering_count_ += 1;
    }

    for (int state_index : recommended_states) {
        auto it = state_to_shot_table.find(state_index);
        if (it != state_to_shot_table.end() && candidates.size() < max_degree) {
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
    NodeSource node_source,
    std::vector<dc::GameState> states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    std::shared_ptr<SimulatorWrapper> simWrapper,
    int gridM,
    int gridN,
    int cluster_num,
    int num_rollout_sims)
: state_to_shot_table_(std::move(state_to_shot_table)),
  simulator_(std::move(simWrapper)),
  cluster_num_(cluster_num),
  num_rollout_simulations_(num_rollout_sims)
{
    all_states_.resize(states.size());
    std::copy(states.begin(), states.end(), all_states_.begin());
    std::shared_ptr<SimulatorWrapper> shared_sim = simulator_;
    root_ = std::make_unique<MCTS_Node>(nullptr, root_state, node_source, shared_sim, gridM, gridN, cluster_num_, num_rollout_sims);
    root_->set_parent_mcts(this);
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
    //best_child_ = root_->select_best_child();
    // Final decision: select child with most visits (not UCT score)
    best_child_ = root_->select_most_visited_child();
    //root_->print_tree();
    std::cout << "MCTS Iteration Done.\n";
}

ShotInfo MCTS::get_best_shot() {
    if (!best_child_) {
        std::cerr << "[ERROR] No children found after tree search. Returning default shot." << "\n";
        std::cerr << "[DEBUG] Root has " << root_->children.size() << " children" << "\n";
        return ShotInfo{ 0.0f, 0.0f, 0 };
    }
    else {
        std::cout << "[DEBUG] get_best_shot() returning: vx=" << best_child_->selected_shot.vx
                  << ", vy=" << best_child_->selected_shot.vy
                  << ", rot=" << best_child_->selected_shot.rot << "\n";
        return best_child_->selected_shot;
    }
}

double MCTS::get_best_shot_winrate() {
    if (!best_child_) {
        std::cerr << "No best child found. Returning 0.0 win rate." << "\n";
        return 0.0;
    }
    return best_child_->calculate_winrate();
}

void MCTS::report_rollout_result() const {
    if (!root_) {
        std::cerr << "No root node.\n";
        return;
    }

    std::cout << "=== MCTS Rollout Result Summary ===\n";

    int clustered_count = 0, random_count = 0, allgrid_count = 0;
    float max_clustered_score = -1e9, max_random_score = -1e9, max_allgrid_score = -1e9;
    float total_clustered_score = 0.0, total_random_score = 0.0, total_allgrid_score = 0.0;

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
        else if (child->source == NodeSource::AllGrid) {
            label = "AllGrid";
            allgrid_count++;
            total_allgrid_score += child->score;
            max_allgrid_score = std::max(max_allgrid_score, child->score);
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
    if (allgrid_count > 0) {
        std::cout << "AllGrid Avg Score: " << (total_allgrid_score / allgrid_count)
            << ", Max Score: " << max_allgrid_score << "\n";
    }
    std::cout << "==============================\n";
}

void MCTS::export_rollout_result_to_csv(const std::string& filename, int shot_num, int grid_m, int grid_n, std::vector<ShotInfo> shotData) const {
    if (!root_) {
        std::cerr << "No root node to export.\n";
        return;
    }
    std::string folder = "../Grid_" + std::to_string(grid_m) + "x" + std::to_string(grid_n) + "/Iter_" + std::to_string(max_iteration) + "/";
    std::filesystem::create_directories(folder); // Create the folder if it doesn't exist
    std::string new_filename = folder + filename + "_" + std::to_string(shot_num) + ".csv";
    std::ofstream file(new_filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    // Write CSV header
    file << "Type,Visits,Wins,Score,Vx,Vy,Rotation,StateId\n";
    std::cout << root_->children.size() << "\n";
    for (const auto& child : root_->children) {
        std::string label;
        if (child->source == NodeSource::Clustered) { 
            label = "Clustered";
        }
        else if (child->source == NodeSource::Random) {
            label = "Random";
        }
        else if (child->source == NodeSource::AllGrid) {
            label = "AllGrid";
        }
        else {
            label = "Unknown";
        }
        ShotInfo selected_shot = child->selected_shot;
        int state_id = 0;
        for (int id = 0; id < shotData.size(); id++) {
            if (fabs(shotData[id].vx - selected_shot.vx) <= DBL_EPSILON) {
                state_id = id;
            }
        }

        file << label << ","
            << child->visits << ","
            << child->wins << ","
            << child->score << "," 
            << selected_shot.vx << ","
            << selected_shot.vy << ","
            << selected_shot.rot << ","
            << state_id << "\n";
    }

    file.close();
    std::cout << "Rollout result exported to: " << filename << "\n";
}