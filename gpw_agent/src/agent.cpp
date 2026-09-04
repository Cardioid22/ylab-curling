#include "agent.h"

#include <chrono>
#include <random>
#include <iostream>
#include <sstream>

namespace gpw {

Agent::Agent(const AgentOptions& opt) : opt_(opt) {
    if (!opt_.log_path.empty()) {
        log_.open(opt_.log_path, std::ios::app);
    }
}

void Agent::Init(const dc::GameSetting& setting,
                 const dc::ISimulatorFactory& sim_factory,
                 const dc::IPlayerFactory& player_factory) {
    setting_ = setting;
    SearchConfig sc = opt_.search_cfg;
    sc.threads = opt_.threads;
    auto t0 = std::chrono::steady_clock::now();
    searcher_ = std::make_unique<Searcher>(setting, sim_factory, player_factory, opt_.eval_params, sc);
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (opt_.verbose) {
        std::cout << "[agent] init done in " << Fmt(static_cast<float>(dt), 2) << "s, threads=" << sc.threads
                  << ", eval: " << opt_.eval_params.Describe()
                  << ", model: " << searcher_->evaluator().model_info() << std::endl;
    }
}

std::optional<Candidate> Agent::Book(const dc::GameState& s) const {
    if (!opt_.use_book) return std::nullopt;
    auto stones = Stones(s);
    if (!stones.empty()) return std::nullopt;
    if (s.shot != 0) return std::nullopt;
    // First stone of the end, we are not the hammer team.
    int diff = ScoreDiff(s, me_);
    int ends_left = (s.end < setting_.max_end) ? setting_.max_end - s.end : 0;
    Candidate c;
    c.kind = Kind::Guard;
    if (diff >= 2 && ends_left <= 3) {
        // Protecting a lead: keep the house clean.
        c.shot = searcher_->Fallback(s).shot;
        c.label = "book draw tee (lead)";
        c.kind = Kind::Draw;
        return c;
    }
    return std::nullopt;  // let the search choose the guard placement
}

dc::Move Agent::Think(const dc::GameState& s) {
    auto t0 = std::chrono::steady_clock::now();
    double budget = opt_.fixed_budget > 0 ? opt_.fixed_budget
                                          : ShotBudgetSeconds(s, me_, setting_, opt_.time_cfg);
    last_budget_ = budget;

    if (auto b = Book(s)) {
        last_ = SearchResult();
        last_.shot = b->shot;
        last_.label = b->label;
        last_.elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Log(s, last_, budget, "book");
        return dc::Move{ToDc(b->shot)};
    }

    try {
        last_ = searcher_->Search(s, me_, budget);
        if (opt_.explore_eps > 0.0) {
            static thread_local std::mt19937 rng(opt_.explore_seed + static_cast<unsigned>(TeamIdx(me_)));
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (u(rng) < opt_.explore_eps) {
                int k = 0;
                for (const auto& st : last_.stats) { if (st.n > 0) ++k; }
                k = std::min(k, opt_.explore_topk);
                if (k > 1) {
                    std::uniform_int_distribution<int> pick(0, k - 1);
                    const auto& st = last_.stats[pick(rng)];
                    last_.shot = st.cand.shot;
                    last_.label = st.cand.label + " [explore]";
                    last_.value = st.Mean();
                }
            }
        }
    } catch (const std::exception& e) {
        last_ = SearchResult();
        Candidate fb = searcher_->Fallback(s);
        last_.shot = fb.shot;
        last_.label = "EXCEPTION fallback: " + std::string(e.what());
        last_.fallback = true;
    } catch (...) {
        last_ = SearchResult();
        Candidate fb = searcher_->Fallback(s);
        last_.shot = fb.shot;
        last_.label = "EXCEPTION fallback";
        last_.fallback = true;
    }
    Log(s, last_, budget, "");
    return dc::Move{ToDc(last_.shot)};
}

void Agent::Log(const dc::GameState& s, const SearchResult& r, double budget, const std::string& note) {
    std::ostringstream o;
    o << "E" << int(s.end) << " S" << int(s.shot) << " " << (s.hammer == me_ ? "H" : "-")
      << " diff=" << ScoreDiff(s, me_)
      << " clock=" << Fmt(static_cast<float>(s.thinking_time_remaining[TeamIdx(me_)].count() / 1000.0), 1)
      << " budget=" << Fmt(static_cast<float>(budget), 2) << " used=" << Fmt(static_cast<float>(r.elapsed), 2)
      << " sims=" << r.sims << " cands=" << r.n_candidates << (r.used_reply ? " reply" : "")
      << " | " << r.label << " v=" << Fmt(static_cast<float>(r.value), 3)
      << " det=" << Fmt(static_cast<float>(r.det_value), 3) << (r.fallback ? " FALLBACK" : "") << " " << note;
    if (opt_.verbose) {
        o << "\n";
        int k = 0;
        for (const auto& st : r.stats) {
            if (k++ >= 6) break;
            o << "    " << st.cand.label << " n=" << st.n << " mean=" << Fmt(static_cast<float>(st.Mean()), 3)
              << " det=" << Fmt(static_cast<float>(st.det_value), 3) << "\n";
        }
    }
    std::string line = o.str();
    if (opt_.verbose) std::cout << "[think] " << line << std::endl;
    if (log_) { log_ << line << std::endl; }
}

}  // namespace gpw
