#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <chrono>
#include <set>
#include <cmath>
#include <algorithm>
#include "mcts.h"
#include "digitalcurling3/digitalcurling3.hpp"
#define DBL_EPSILON 2.2204460492503131e-016
namespace dc = digitalcurling3;

// ========== デルタ距離関数 (Delta Distance Function) ==========
// pool_clustering_experiment.cpp の distDelta v2 と同等

static constexpr float kHouseCenterX_dc = 0.0f;
static constexpr float kHouseCenterY_dc = 38.405f;
static constexpr float kHouseRadius_dc = 1.829f;

static int getZoneDC(const std::optional<dc::Transform>& stone) {
    if (!stone) return -1;
    float x = stone->position.x;
    float y = stone->position.y;
    float d = std::sqrt(x * x + (y - kHouseCenterY_dc) * (y - kHouseCenterY_dc));
    if (d <= kHouseRadius_dc) return 0;
    if (y < kHouseCenterY_dc - kHouseRadius_dc && y > kHouseCenterY_dc - 3.0f * kHouseRadius_dc) return 1;
    return 2;
}

static float evaluateBoardDC(const dc::GameState& state) {
    struct SI { float dist; int team; };
    std::vector<SI> in_house;
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (!state.stones[t][i]) continue;
            float dx = state.stones[t][i]->position.x;
            float dy = state.stones[t][i]->position.y - kHouseCenterY_dc;
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= kHouseRadius_dc + 0.145f) {
                in_house.push_back({d, t});
            }
        }
    }
    if (in_house.empty()) return 0.0f;
    std::sort(in_house.begin(), in_house.end(), [](auto& a, auto& b) { return a.dist < b.dist; });
    int scoring_team = in_house[0].team;
    int score = 0;
    for (auto& s : in_house) {
        if (s.team == scoring_team) score++;
        else break;
    }
    return scoring_team == 0 ? static_cast<float>(score) : -static_cast<float>(score);
}

static float distDeltaDC(const dc::GameState& input, const dc::GameState& a, const dc::GameState& b) {
    constexpr float MOVE_THRESHOLD = 0.01f;
    constexpr float PENALTY_EXISTENCE = 30.0f;
    constexpr float PENALTY_ZONE = 12.0f;
    constexpr float NEW_STONE_WEIGHT = 4.0f;
    constexpr float MOVED_STONE_WEIGHT = 2.0f;
    constexpr float PENALTY_INTERACTION = 15.0f;
    constexpr float INTERACTION_THRESHOLD = 0.03f;
    constexpr float SCORE_WEIGHT = 8.0f;
    constexpr float PROXIMITY_WEIGHT = 5.0f;

    float distance = 0.0f;
    float max_disp_a = 0.0f, max_disp_b = 0.0f;
    int new_team = -1, new_idx = -1;

    for (int team = 0; team < 2; team++) {
        for (int idx = 0; idx < 8; idx++) {
            bool in_inp = input.stones[team][idx].has_value();
            bool in_a = a.stones[team][idx].has_value();
            bool in_b = b.stones[team][idx].has_value();

            if (in_inp) {
                if (in_a && in_b) {
                    float dxa = a.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dya = a.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float dxb = b.stones[team][idx]->position.x - input.stones[team][idx]->position.x;
                    float dyb = b.stones[team][idx]->position.y - input.stones[team][idx]->position.y;
                    float ma = std::sqrt(dxa*dxa + dya*dya);
                    float mb = std::sqrt(dxb*dxb + dyb*dyb);
                    max_disp_a = std::max(max_disp_a, ma);
                    max_disp_b = std::max(max_disp_b, mb);
                    if (ma < MOVE_THRESHOLD && mb < MOVE_THRESHOLD) continue;
                    float ddx = dxa - dxb, ddy = dya - dyb;
                    distance += MOVED_STONE_WEIGHT * std::sqrt(ddx*ddx + ddy*ddy);
                    if (getZoneDC(a.stones[team][idx]) != getZoneDC(b.stones[team][idx]))
                        distance += PENALTY_ZONE;
                } else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            } else {
                if (in_a && in_b) {
                    new_team = team; new_idx = idx;
                    float dx = a.stones[team][idx]->position.x - b.stones[team][idx]->position.x;
                    float dy = a.stones[team][idx]->position.y - b.stones[team][idx]->position.y;
                    distance += NEW_STONE_WEIGHT * std::sqrt(dx*dx + dy*dy);
                    if (getZoneDC(a.stones[team][idx]) != getZoneDC(b.stones[team][idx]))
                        distance += PENALTY_ZONE;
                } else if (in_a != in_b) {
                    distance += PENALTY_EXISTENCE;
                }
            }
        }
    }

    if ((max_disp_a > INTERACTION_THRESHOLD) != (max_disp_b > INTERACTION_THRESHOLD))
        distance += PENALTY_INTERACTION;

    if (new_team >= 0) {
        auto minProx = [&](const dc::GameState& st) -> float {
            float mn = 1e9f;
            float nx = st.stones[new_team][new_idx]->position.x;
            float ny = st.stones[new_team][new_idx]->position.y;
            for (int t = 0; t < 2; t++)
                for (int i = 0; i < 8; i++) {
                    if (t == new_team && i == new_idx) continue;
                    if (!st.stones[t][i]) continue;
                    float dx = nx - st.stones[t][i]->position.x;
                    float dy = ny - st.stones[t][i]->position.y;
                    mn = std::min(mn, std::sqrt(dx*dx + dy*dy));
                }
            return mn;
        };
        float pa = minProx(a), pb = minProx(b);
        if (pa < 100.0f || pb < 100.0f)
            distance += PROXIMITY_WEIGHT * std::abs(pa - pb);
    }

    distance += SCORE_WEIGHT * std::abs(evaluateBoardDC(a) - evaluateBoardDC(b));

    float ca = 1e9f, cb = 1e9f;
    int ta = -1, tb = -1;
    for (int t = 0; t < 2; t++)
        for (int i = 0; i < 8; i++) {
            if (a.stones[t][i]) { float d = std::sqrt(std::pow(a.stones[t][i]->position.x,2)+std::pow(a.stones[t][i]->position.y-kHouseCenterY_dc,2)); if(d<ca){ca=d;ta=t;} }
            if (b.stones[t][i]) { float d = std::sqrt(std::pow(b.stones[t][i]->position.x,2)+std::pow(b.stones[t][i]->position.y-kHouseCenterY_dc,2)); if(d<cb){cb=d;tb=t;} }
        }
    if (ta >= 0 && tb >= 0 && ta != tb) distance += 10.0f;

    return distance;
}

static std::vector<std::set<int>> hierarchicalClusteringDC(
    const std::vector<std::vector<float>>& dist_table, int n_clusters
) {
    int n = static_cast<int>(dist_table.size());
    std::vector<std::set<int>> clusters(n);
    for (int i = 0; i < n; i++) clusters[i].insert(i);

    while (static_cast<int>(clusters.size()) > n_clusters) {
        float min_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;
        for (int i = 0; i < static_cast<int>(clusters.size()); i++) {
            for (int j = i + 1; j < static_cast<int>(clusters.size()); j++) {
                float total = 0.0f; int count = 0;
                for (int a : clusters[i])
                    for (int b : clusters[j]) { total += dist_table[a][b]; count++; }
                float avg = total / count;
                if (avg < min_dist) { min_dist = avg; best_i = i; best_j = j; }
            }
        }
        if (best_i == -1) break;
        clusters[best_i].insert(clusters[best_j].begin(), clusters[best_j].end());
        clusters.erase(clusters.begin() + best_j);
    }
    return clusters;
}

static std::vector<int> calculateMedoidsDC(
    const std::vector<std::vector<float>>& dist_table,
    const std::vector<std::set<int>>& clusters
) {
    std::vector<int> medoids;
    for (auto& cluster : clusters) {
        if (cluster.empty()) { medoids.push_back(-1); continue; }
        if (cluster.size() == 1) { medoids.push_back(*cluster.begin()); continue; }
        float min_total = std::numeric_limits<float>::max();
        int best = -1;
        for (int c : cluster) {
            float total = 0.0f;
            for (int o : cluster) if (c != o) total += dist_table[c][o];
            if (total < min_total) { min_total = total; best = c; }
        }
        medoids.push_back(best);
    }
    return medoids;
}

// ==========================================================

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
    double best_winrate = -1.0;

    for (const auto& child : children) {
        double winrate = child->visits > 0
            ? static_cast<double>(child->wins) / child->visits
            : 0.0;

        std::cout << "Child visits: " << child->visits
                  << ", wins: " << child->wins
                  << ", winrate: " << winrate
                  << "\n";

        // Primary: most visits
        // Secondary (tiebreaker): highest winrate when visits are equal
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_winrate = winrate;
            best = child.get();
        }
        else if (child->visits == max_visits && winrate > best_winrate) {
            // Same visits, but higher winrate -> use as tiebreaker
            best_winrate = winrate;
            best = child.get();
        }
    }

    if (best) {
        std::cout << "[INFO] Selected most visited child with " << max_visits
                  << " visits (winrate: " << best_winrate
                  << ")\n";
    }

    return best;
}

void MCTS_Node::expand(std::vector<dc::GameState> all_states, std::unordered_map<int, ShotInfo> state_to_shot_table) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
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
            else if (source == NodeSource::DeltaClustered) {
                untried_shots = std::make_unique<std::vector<ShotInfo>>(
                    generate_delta_clustered_shots(all_states, state_to_shot_table)
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
    else if (source == NodeSource::DeltaClustered) {
        std::cout << "[DeltaClustered] rollout avg score: " << game_score << "\n";
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
// Delta Clustered shot generation using hierarchical clustering on delta distances
std::vector<ShotInfo> MCTS_Node::generate_delta_clustered_shots(
    const std::vector<dc::GameState>& all_states,
    const std::unordered_map<int, ShotInfo>& state_to_shot_table) const
{
    auto clustering_start = std::chrono::high_resolution_clock::now();

    int n = static_cast<int>(all_states.size());
    if (n == 0) return {};

    // Build delta distance table using current state as input (pre-simulation state)
    std::vector<std::vector<float>> dist_table(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float d = distDeltaDC(state, all_states[i], all_states[j]);
            dist_table[i][j] = d;
            dist_table[j][i] = d;
        }
    }

    // Hierarchical clustering (average linkage)
    auto clusters = hierarchicalClusteringDC(dist_table, cluster_num_);

    // Calculate medoids
    auto medoids = calculateMedoidsDC(dist_table, clusters);

    // Track clustering timing
    auto clustering_end = std::chrono::high_resolution_clock::now();
    double clustering_time = std::chrono::duration<double>(clustering_end - clustering_start).count();
    if (parent_mcts_ != nullptr) {
        parent_mcts_->total_clustering_time_ += clustering_time;
        parent_mcts_->total_clustering_count_ += 1;
    }

    // Map medoid indices to ShotInfo
    std::vector<ShotInfo> candidates;
    for (int medoid_idx : medoids) {
        if (medoid_idx < 0) continue;
        auto it = state_to_shot_table.find(medoid_idx);
        if (it != state_to_shot_table.end() && static_cast<int>(candidates.size()) < max_degree) {
            candidates.push_back(it->second);
        }
    }

    std::cout << "[DeltaClustered] Generated " << candidates.size()
              << " candidates from " << n << " states ("
              << clustering_time << "s)\n";

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

    int clustered_count = 0, delta_clustered_count = 0, random_count = 0, allgrid_count = 0;
    float max_clustered_score = -1e9, max_delta_clustered_score = -1e9, max_random_score = -1e9, max_allgrid_score = -1e9;
    float total_clustered_score = 0.0, total_delta_clustered_score = 0.0, total_random_score = 0.0, total_allgrid_score = 0.0;

    for (const auto& child : root_->children) {
        std::string label;
        if (child->source == NodeSource::Clustered) {
            label = "Clustered";
            clustered_count++;
            total_clustered_score += child->score;
            max_clustered_score = std::max(max_clustered_score, child->score);
        }
        else if (child->source == NodeSource::DeltaClustered) {
            label = "DeltaClustered";
            delta_clustered_count++;
            total_delta_clustered_score += child->score;
            max_delta_clustered_score = std::max(max_delta_clustered_score, child->score);
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
    if (delta_clustered_count > 0) {
        std::cout << "DeltaClustered Avg Score: " << (total_delta_clustered_score / delta_clustered_count)
            << ", Max Score: " << max_delta_clustered_score << "\n";
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
        else if (child->source == NodeSource::DeltaClustered) {
            label = "DeltaClustered";
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