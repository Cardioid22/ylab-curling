// Headless self-play harness: two agents, an in-process referee simulator,
// per-shot JSONL records for training and end-result statistics.
#pragma once

#include "agent.h"

#include <string>

namespace gpw {

struct SelfplayConfig {
    int games = 10;
    int ends = 2;
    int threads = 8;
    double budget_a = 1.0;   // seconds per shot
    double budget_b = 1.0;
    std::string eval_a;      // EvalParams file (optional)
    std::string eval_b;
    std::string model_a;     // value model file (optional)
    std::string model_b;
    std::string out_jsonl;   // per-shot records
    bool verbose = false;
    bool alternate = true;   // swap colours every game
    double explore_eps = 0.0;  // exploration for both agents (data generation)
    unsigned seed = 1;
    bool refine_a = true, refine_b = true;
    int reply_shots_a = 0, reply_shots_b = 0;
    SearchConfig search_a, search_b;   // other per-side search settings (keep, min/max sims, reply keep)
};

struct SelfplaySummary {
    int games = 0;
    int a_wins = 0, b_wins = 0, draws = 0;
    double a_score_diff = 0;  // total (A - B)
    // Hammer-team end-result histogram, k = -4..4 (index k + 4).
    std::array<int, 9> hammer_hist{};
};

SelfplaySummary RunSelfplay(const SelfplayConfig& cfg);

}  // namespace gpw
