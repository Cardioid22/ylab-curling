#include "mcts_with_tracking.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

MCTS_WithTracking::MCTS_WithTracking(
    dc::GameState const& root_state,
    NodeSource node_source,
    std::vector<dc::GameState> states,
    std::unordered_map<int, ShotInfo> state_to_shot_table,
    std::shared_ptr<SimulatorWrapper> simWrapper,
    int gridM,
    int gridN
) : convergence_threshold_(0.8),
    ground_truth_shot_{0.0f, 0.0f, 0},
    all_states_(std::move(states)),
    state_to_shot_table_(std::move(state_to_shot_table)) {

    mcts_ = std::make_unique<MCTS>(root_state, node_source, all_states_,
                                  state_to_shot_table_, simWrapper, gridM, gridN);
}

void MCTS_WithTracking::setGroundTruth(const ShotInfo& ground_truth) {
    ground_truth_shot_ = ground_truth;
    std::cout << "Ground truth set: vx=" << ground_truth.vx
              << ", vy=" << ground_truth.vy
              << ", rot=" << ground_truth.rot << std::endl;
}

MCTS_WithTracking::SearchResult MCTS_WithTracking::trackGroundTruthDiscovery(
    int max_iter, double max_time) {

    SearchResult result;
    result.first_discovery_iteration = -1;
    result.best_evaluation_iteration = -1;
    result.convergence_iteration = -1;
    result.success = false;
    result.final_score = -std::numeric_limits<double>::infinity();

    double best_ground_truth_score = -std::numeric_limits<double>::infinity();

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Starting MCTS tracking with max_iter=" << max_iter
              << ", max_time=" << max_time << "s" << std::endl;

    for (int iter = 0; iter < max_iter; ++iter) {
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - start_time).count();
        if (elapsed >= max_time) {
            std::cout << "Time limit reached at iteration " << iter << std::endl;
            break;
        }

        performOneMCTSIteration();

        double current_score = getScoreForShot(ground_truth_shot_);
        result.score_history.push_back(current_score);

        if (result.first_discovery_iteration == -1 &&
            isShotInCandidates(ground_truth_shot_)) {
            result.first_discovery_iteration = iter;
            std::cout << "Ground truth discovered at iteration: " << iter << std::endl;
        }

        if (current_score > best_ground_truth_score) {
            best_ground_truth_score = current_score;
            if (isBestShot(ground_truth_shot_)) {
                result.best_evaluation_iteration = iter;
                std::cout << "Ground truth became best at iteration: " << iter
                          << " (score: " << current_score << ")" << std::endl;
            }
        }

        if (result.convergence_iteration == -1 &&
            current_score >= convergence_threshold_) {
            result.convergence_iteration = iter;
            result.success = true;
            std::cout << "Ground truth converged at iteration: " << iter
                      << " (score: " << current_score << ")" << std::endl;
            break;
        }

        if ((iter + 1) % 1000 == 0) {
            std::cout << "Iteration " << (iter + 1) << "/" << max_iter
                      << ", current score: " << current_score << std::endl;
        }
    }

    result.final_score = best_ground_truth_score;

    std::cout << "MCTS tracking completed. Success: " << result.success
              << ", Final score: " << result.final_score << std::endl;

    return result;
}

bool MCTS_WithTracking::isGroundTruthShot(const ShotInfo& shot, double tolerance) const {
    return std::abs(shot.vx - ground_truth_shot_.vx) < tolerance &&
           std::abs(shot.vy - ground_truth_shot_.vy) < tolerance &&
           shot.rot == ground_truth_shot_.rot;
}

void MCTS_WithTracking::recordGroundTruthEvaluation(int iteration, double score) {
    evaluation_scores_.push_back(score);
}

double MCTS_WithTracking::getScoreForShot(const ShotInfo& shot) const {
    ShotInfo best_shot = mcts_->get_best_shot();

    if (isGroundTruthShot(best_shot)) {
        return 1.0;
    }

    return 0.0;
}

bool MCTS_WithTracking::isShotInCandidates(const ShotInfo& shot) const {
    ShotInfo best_shot = mcts_->get_best_shot();
    return isGroundTruthShot(best_shot);
}

bool MCTS_WithTracking::isBestShot(const ShotInfo& shot) const {
    ShotInfo best_shot = mcts_->get_best_shot();
    return isGroundTruthShot(best_shot);
}

void MCTS_WithTracking::performOneMCTSIteration() {
    mcts_->grow_tree(1, 0.001);
}