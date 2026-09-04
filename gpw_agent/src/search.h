// Noisy one-ply search with successive halving, plus a short deterministic
// opponent-reply look-ahead for the last stones of an end.
#pragma once

#include "candidates.h"
#include "curling.h"
#include "eval.h"
#include "sim.h"
#include "threadpool.h"

#include <memory>
#include <string>
#include <vector>

namespace gpw {

struct SearchConfig {
    int threads = 8;
    int prescreen_keep = 24;     // candidates that survive the deterministic pre-screen
    int min_sims = 4;            // noisy samples per candidate in the first round
    int max_sims = 32;           // samples per finalist before the survivors are frozen
    int hard_max_sims = 256;     // absolute cap per candidate while time remains
    int reply_last_shots = 4;    // opponent-reply look-ahead when (16 - shot) <= this
    int reply_keep = 8;          // candidates kept for the noisy rounds when the reply look-ahead is on
    int reply_min_sims = 2;      // first-round samples per candidate in reply mode
    double sim_cost_init = 0.03; // seconds per physics call (updated online)
    bool refine = false;         // local speed/angle refinement of the chosen shot (A/B 7-13, off)
    bool verbose = false;
};

struct CandStat {
    Candidate cand;
    double det_value = 0;  // deterministic pre-screen value
    double sum = 0;
    int n = 0;
    double Mean() const { return n > 0 ? sum / n : det_value; }
};

struct SearchResult {
    Shot shot;
    std::string label;
    double value = 0;
    double det_value = 0;
    int sims = 0;
    double elapsed = 0;
    int n_candidates = 0;
    bool used_reply = false;
    bool fallback = false;
    bool refined = false;
    std::vector<CandStat> stats;  // sorted by mean, descending
};

class Searcher {
public:
    Searcher(const dc::GameSetting& setting,
             const dc::ISimulatorFactory& sim_factory,
             const dc::IPlayerFactory& player_factory,
             const EvalParams& eval_params,
             const SearchConfig& cfg);

    SearchResult Search(const dc::GameState& s, dc::Team me, double budget_sec);

    Candidate Fallback(const dc::GameState& s) const { return gen_.Fallback(s); }
    const Evaluator& evaluator() const { return eval_; }
    double sim_cost() const { return sim_cost_; }
    int threads() const { return pool_.Size(); }

    // Direct physics access for harnesses (worker 0's simulator).
    dc::GameState Apply(const dc::GameState& s, const Shot& shot, bool noisy) { return sims_[0]->Apply(s, shot, noisy); }

private:
    dc::GameSetting setting_;
    SearchConfig cfg_;
    ThreadPool pool_;
    std::vector<std::unique_ptr<Sim>> sims_;
    VelocitySolver vs_;
    CandidateGenerator gen_;
    Evaluator eval_;
    double sim_cost_;

    double LeafValue(int worker, const dc::GameState& after, dc::Team me, bool reply);
    void Prewarm();
};

}  // namespace gpw
