// Game-level wrapper: time management, opening book, fallbacks, logging.
#pragma once

#include "curling.h"
#include "eval.h"
#include "search.h"
#include "timeman.h"

#include <fstream>
#include <memory>
#include <optional>
#include <string>

namespace gpw {

struct AgentOptions {
    int threads = 8;
    bool verbose = true;
    std::string log_path;        // per-shot log (text); empty = none
    double fixed_budget = 0.0;   // >0: ignore the clock and use this many seconds per shot
    bool use_book = true;
    double explore_eps = 0.0;    // self-play: with this probability play a random top-k candidate
    int explore_topk = 4;
    unsigned explore_seed = 12345;
    TimeConfig time_cfg;
    SearchConfig search_cfg;
    EvalParams eval_params;
    std::string name = "gpw_agent";
};

class Agent {
public:
    explicit Agent(const AgentOptions& opt);

    // Heavy initialisation (thread pool, simulators, drift cache).
    void Init(const dc::GameSetting& setting,
              const dc::ISimulatorFactory& sim_factory,
              const dc::IPlayerFactory& player_factory);

    void SetTeam(dc::Team team) { me_ = team; }
    dc::Team team() const { return me_; }

    // Never throws; always returns a shot.
    dc::Move Think(const dc::GameState& s);

    const SearchResult& last_result() const { return last_; }
    double last_budget() const { return last_budget_; }
    Searcher& searcher() { return *searcher_; }
    const AgentOptions& options() const { return opt_; }

private:
    AgentOptions opt_;
    dc::Team me_ = dc::Team::k0;
    dc::GameSetting setting_;
    std::unique_ptr<Searcher> searcher_;
    SearchResult last_;
    double last_budget_ = 0;
    std::ofstream log_;

    std::optional<Candidate> Book(const dc::GameState& s) const;
    void Log(const dc::GameState& s, const SearchResult& r, double budget, const std::string& note);
};

}  // namespace gpw
