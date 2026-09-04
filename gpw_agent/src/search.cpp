#include "search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>

namespace gpw {

namespace {
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
}  // namespace

Searcher::Searcher(const dc::GameSetting& setting,
                   const dc::ISimulatorFactory& sim_factory,
                   const dc::IPlayerFactory& player_factory,
                   const EvalParams& eval_params,
                   const SearchConfig& cfg)
    : setting_(setting), cfg_(cfg), pool_(cfg.threads), gen_(vs_), eval_(eval_params, setting),
      sim_cost_(cfg.sim_cost_init) {
    sims_.reserve(pool_.Size());
    for (int i = 0; i < pool_.Size(); ++i) {
        sims_.push_back(std::make_unique<Sim>(setting, sim_factory, player_factory));
    }
    Prewarm();
}

// Fill the drift cache for the speeds the candidate generator uses so that the
// first turn is not slowed down by inverse-kinematics simulations.
void Searcher::Prewarm() {
    const float speeds[] = {0.f, 0.9f, 2.2f, 3.3f};
    struct Job { dc::Vector2 target; float speed; bool cw; };
    std::vector<Job> jobs;
    for (float sp : speeds) {
        for (float r = 30.5f; r <= 42.5f; r += 0.1f) {
            jobs.push_back({dc::Vector2(0.f, r), sp, false});
            jobs.push_back({dc::Vector2(0.f, r), sp, true});
        }
    }
    pool_.ParallelFor(static_cast<int>(jobs.size()), [&](int i, int) {
        vs_.Solve(jobs[i].target, jobs[i].speed, jobs[i].cw);
    });
}

double Searcher::LeafValue(int worker, const dc::GameState& after, dc::Team me, bool reply) {
    if (!reply || after.IsGameOver() || after.shot == 0) return eval_.Value(after, me);
    auto replies = gen_.GenerateReplies(after, Opp(me));
    if (replies.empty()) return eval_.Value(after, me);
    double worst = 2.0;
    for (const auto& r : replies) {
        dc::GameState st = sims_[worker]->Apply(after, r.shot, false);
        double v = eval_.Value(st, me);
        if (v < worst) worst = v;
    }
    return worst;
}

SearchResult Searcher::Search(const dc::GameState& s, dc::Team me, double budget_sec) {
    auto t0 = Clock::now();
    auto deadline = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget_sec));
    SearchResult res;

    // ---- candidates ---------------------------------------------------------
    std::vector<Candidate> cands = gen_.Generate(s, me);
    if (cands.empty()) cands.push_back(gen_.Fallback(s));
    res.n_candidates = static_cast<int>(cands.size());
    bool use_reply = (kShotsPerEnd - s.shot) <= cfg_.reply_last_shots;
    res.used_reply = use_reply;

    // ---- deterministic pre-screen ------------------------------------------
    std::vector<CandStat> stats(cands.size());
    auto tp0 = Clock::now();
    pool_.ParallelFor(static_cast<int>(cands.size()), [&](int i, int w) {
        stats[i].cand = cands[i];
        dc::GameState after = sims_[w]->Apply(s, cands[i].shot, false);
        stats[i].det_value = eval_.Value(after, me);
    });
    {
        double dt = Seconds(tp0, Clock::now());
        double per = dt * pool_.Size() / std::max<int>(1, static_cast<int>(cands.size()));
        sim_cost_ = 0.7 * sim_cost_ + 0.3 * per;
    }
    std::sort(stats.begin(), stats.end(), [](const CandStat& a, const CandStat& b) { return a.det_value > b.det_value; });

    // Keep the best K plus the best of every kind for diversity.
    int keep = use_reply ? cfg_.reply_keep : cfg_.prescreen_keep;
    std::vector<int> alive;
    std::set<int> alive_set;
    for (int i = 0; i < static_cast<int>(stats.size()) && static_cast<int>(alive.size()) < keep; ++i) {
        alive.push_back(i);
        alive_set.insert(i);
    }
    if (!use_reply) {
        std::set<int> kinds_seen;
        for (int i : alive) kinds_seen.insert(static_cast<int>(stats[i].cand.kind));
        for (int i = 0; i < static_cast<int>(stats.size()); ++i) {
            int k = static_cast<int>(stats[i].cand.kind);
            if (kinds_seen.count(k)) continue;
            kinds_seen.insert(k);
            alive.push_back(i);
            alive_set.insert(i);
        }
    }

    const CandStat& det_best = stats[0];
    res.det_value = det_best.det_value;

    // ---- noisy successive halving --------------------------------------------
    int n_target = use_reply ? cfg_.reply_min_sims : cfg_.min_sims;
    int total_sims = 0;
    int reply_factor = 1;
    if (use_reply) reply_factor = 1 + static_cast<int>(gen_.GenerateReplies(s, Opp(me)).size());
    std::atomic<bool> stop{false};

    while (true) {
        double elapsed = Seconds(t0, Clock::now());
        double remaining = budget_sec - elapsed;
        if (remaining <= 0) break;

        struct Task { int idx; };
        std::vector<Task> tasks;
        for (int i : alive) {
            int need = n_target - stats[i].n;
            for (int k = 0; k < need; ++k) tasks.push_back({i});
        }
        if (tasks.empty()) break;

        double cost_per_task = sim_cost_ * reply_factor;
        double est = tasks.size() * cost_per_task / pool_.Size();
        if (est > remaining) {
            // Not enough time for everyone: shrink the alive set (it is ordered
            // best-first) so that each survivor still gets n_target samples.
            int allowed = static_cast<int>(remaining * pool_.Size() / cost_per_task);
            int per_needed = std::max(2, n_target);
            int k = allowed / per_needed;
            if (k < 2) break;
            if (k < static_cast<int>(alive.size())) alive.resize(k);
            tasks.clear();
            for (int i : alive) {
                int need = n_target - stats[i].n;
                for (int q = 0; q < need; ++q) tasks.push_back({i});
            }
            if (tasks.empty()) break;
        }

        std::vector<double> results(tasks.size(), 0.0);
        std::vector<char> done(tasks.size(), 0);
        auto tr0 = Clock::now();
        pool_.ParallelFor(static_cast<int>(tasks.size()), [&](int t, int w) {
            if (stop.load(std::memory_order_relaxed)) return;
            if (Clock::now() > deadline) { stop.store(true); return; }
            const Candidate& c = stats[tasks[t].idx].cand;
            dc::GameState after = sims_[w]->Apply(s, c.shot, true);
            results[t] = LeafValue(w, after, me, use_reply);
            done[t] = 1;
        });
        int n_done = 0;
        for (size_t t = 0; t < tasks.size(); ++t) {
            if (!done[t]) continue;
            stats[tasks[t].idx].sum += results[t];
            stats[tasks[t].idx].n += 1;
            ++n_done;
        }
        total_sims += n_done * reply_factor;
        if (n_done > 0) {
            double dt = Seconds(tr0, Clock::now());
            double per = dt * pool_.Size() / (n_done * reply_factor);
            sim_cost_ = 0.7 * sim_cost_ + 0.3 * per;
        }
        if (stop.load()) break;

        // Halve.
        std::sort(alive.begin(), alive.end(), [&](int a, int b) { return stats[a].Mean() > stats[b].Mean(); });
        if (alive.size() > 2) {
            alive.resize((alive.size() + 1) / 2);
        }
        if (n_target >= cfg_.hard_max_sims) {
            bool all_full = true;
            for (int i : alive) if (stats[i].n < cfg_.hard_max_sims) all_full = false;
            if (all_full) break;
        }
        n_target = std::min(cfg_.hard_max_sims, n_target * 2);
    }

    // ---- pick -----------------------------------------------------------------
    int best = -1;
    int need_n = std::max(1, std::min(cfg_.min_sims, 2));
    for (int i = 0; i < static_cast<int>(stats.size()); ++i) {
        if (stats[i].n < need_n) continue;
        if (best < 0 || stats[i].Mean() > stats[best].Mean() ||
            (stats[i].Mean() == stats[best].Mean() && stats[i].n > stats[best].n)) {
            best = i;
        }
    }
    if (best < 0) {
        best = 0;  // deterministic best
        res.fallback = true;
    }

    // ---- local refinement of the leader ----------------------------------------
    // Small speed / angle perturbations of the chosen shot, evaluated with the
    // same noisy procedure. Replaces the leader only on a clear improvement.
    if (cfg_.refine && !res.fallback && !stop.load()) {
        double elapsed = Seconds(t0, Clock::now());
        double remaining = budget_sec - elapsed;
        int n_ref = use_reply ? 8 : 16;
        const double dv[] = {-0.02, 0.02, 0.0, 0.0, -0.015, 0.015, -0.015, 0.015};
        const double da[] = {0.0, 0.0, -0.003, 0.003, -0.003, -0.003, 0.003, 0.003};
        int n_var = use_reply ? 4 : 8;
        double cost = static_cast<double>(n_var) * n_ref * sim_cost_ * reply_factor / pool_.Size();
        if (remaining > cost * 1.2) {
            const Shot base = stats[best].cand.shot;
            double speed = std::sqrt(base.vx * base.vx + base.vy * base.vy);
            double ang = std::atan2(base.vy, base.vx);
            std::vector<CandStat> vars(n_var);
            for (int v = 0; v < n_var; ++v) {
                double sp = std::min(static_cast<double>(kMaxSpeed), std::max(0.5, speed + dv[v]));
                double a = ang + da[v];
                vars[v].cand.shot.vx = static_cast<float>(sp * std::cos(a));
                vars[v].cand.shot.vy = static_cast<float>(sp * std::sin(a));
                vars[v].cand.shot.cw = base.cw;
                vars[v].cand.kind = stats[best].cand.kind;
                vars[v].cand.label = stats[best].cand.label + " ~(" + Fmt(static_cast<float>(dv[v]), 3) + "," + Fmt(static_cast<float>(da[v]), 3) + ")";
                vars[v].det_value = stats[best].det_value;
            }
            std::vector<double> rres(static_cast<size_t>(n_var) * n_ref, 0.0);
            std::vector<char> rdone(rres.size(), 0);
            pool_.ParallelFor(static_cast<int>(rres.size()), [&](int t, int w) {
                if (stop.load(std::memory_order_relaxed)) return;
                if (Clock::now() > deadline) { stop.store(true); return; }
                const Candidate& c = vars[t / n_ref].cand;
                dc::GameState after = sims_[w]->Apply(s, c.shot, true);
                rres[t] = LeafValue(w, after, me, use_reply);
                rdone[t] = 1;
            });
            for (size_t t = 0; t < rres.size(); ++t) {
                if (!rdone[t]) continue;
                vars[t / n_ref].sum += rres[t];
                vars[t / n_ref].n += 1;
                total_sims += reply_factor;
            }
            int bv = -1;
            for (int v = 0; v < n_var; ++v) {
                if (vars[v].n < n_ref / 2) continue;
                if (bv < 0 || vars[v].Mean() > vars[bv].Mean()) bv = v;
            }
            // Require a margin over the incumbent (which has n samples already).
            if (bv >= 0 && vars[bv].Mean() > stats[best].Mean() + 0.01) {
                stats.push_back(vars[bv]);
                best = static_cast<int>(stats.size()) - 1;
                res.refined = true;
            }
        }
    }

    res.shot = stats[best].cand.shot;
    res.label = stats[best].cand.label;
    res.value = stats[best].Mean();
    res.sims = total_sims + res.n_candidates;
    res.elapsed = Seconds(t0, Clock::now());

    std::sort(stats.begin(), stats.end(), [](const CandStat& a, const CandStat& b) {
        if (a.n != b.n && (a.n == 0 || b.n == 0)) return a.n > b.n;
        return a.Mean() > b.Mean();
    });
    res.stats = std::move(stats);
    return res;
}

}  // namespace gpw
