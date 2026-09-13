#include "selfplay.h"

#include "sim.h"

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

namespace gpw {

namespace {

struct Record {
    std::string who;
    dc::Team team;
    dc::GameState state;   // pre-shot state
    Shot shot;
    std::string label;
    double value, det;
    int sims;
    double budget, used;
    int end_index;
    int end_result_hammer = 0;  // filled when the end finishes
    std::array<double, 9> p_hand{};  // hand-crafted end-result distribution of the pre-shot state
};

void WriteRecords(std::ofstream& out, int game, int max_end, const std::vector<Record>& recs, dc::Team winner) {
    if (!out) return;
    for (const auto& r : recs) {
        const dc::GameState& s = r.state;
        int win = (winner == dc::Team::kInvalid) ? 0 : (winner == r.team ? 1 : -1);
        out << "{\"game\":" << game << ",\"max_end\":" << max_end << ",\"agent\":\"" << r.who << "\",\"team\":" << TeamIdx(r.team)
            << ",\"end\":" << int(s.end) << ",\"shot\":" << int(s.shot)
            << ",\"hammer\":" << TeamIdx(s.hammer)
            << ",\"score\":[" << s.GetTotalScore(dc::Team::k0) << "," << s.GetTotalScore(dc::Team::k1) << "]"
            << ",\"stones\":[";
        bool first = true;
        for (int t = 0; t < 2; ++t) {
            for (int i = 0; i < 8; ++i) {
                if (!first) out << ",";
                first = false;
                const auto& st = s.stones[t][i];
                if (st) out << "[" << st->position.x << "," << st->position.y << "]";
                else out << "null";
            }
        }
        out << "],\"shot_v\":[" << r.shot.vx << "," << r.shot.vy << "]," << "\"cw\":" << (r.shot.cw ? 1 : 0)
            << ",\"label\":\"" << r.label << "\",\"value\":" << r.value << ",\"det\":" << r.det
            << ",\"sims\":" << r.sims << ",\"budget\":" << r.budget << ",\"used\":" << r.used
            << ",\"end_result_hammer\":" << r.end_result_hammer << ",\"result\":" << win << ",\"p_hand\":[";
        for (int k = 0; k < 9; ++k) out << r.p_hand[k] << (k < 8 ? "," : "");
        out << "]}\n";
    }
}

}  // namespace

SelfplaySummary RunSelfplay(const SelfplayConfig& cfg) {
    SelfplaySummary sum;
    dc::GameSetting setting;
    setting.max_end = static_cast<std::uint8_t>(cfg.ends);
    setting.five_rock_rule = true;
    setting.sheet_width = 4.75f;
    setting.thinking_time[0] = setting.thinking_time[1] = std::chrono::milliseconds(1000LL * 60 * 60 * 24);
    setting.extra_end_thinking_time[0] = setting.extra_end_thinking_time[1] = std::chrono::milliseconds(1000LL * 60 * 60);
    if (cfg.real_clock) {
        setting.thinking_time[0] = setting.thinking_time[1] = std::chrono::milliseconds(static_cast<long long>(cfg.clock_sec * 1000));
        setting.extra_end_thinking_time[0] = setting.extra_end_thinking_time[1] = std::chrono::milliseconds(static_cast<long long>(cfg.extra_clock_sec * 1000));
    }

    dc::simulators::SimulatorFCV1Factory simf;
    dc::players::PlayerNormalDistFactory pf;  // tournament defaults

    auto make = [&](const std::string& eval_file, const std::string& model, double budget, const std::string& name,
                    bool refine, int reply_shots, const SearchConfig& sc) {
        AgentOptions o;
        o.search_cfg = sc;
        o.search_cfg.refine = refine;
        o.search_cfg.reply_last_shots = reply_shots;
        o.threads = cfg.threads;
        o.verbose = cfg.verbose;
        o.fixed_budget = cfg.real_clock ? 0.0 : budget;
        o.name = name;
        o.explore_eps = cfg.explore_eps;
        o.explore_seed = cfg.seed * 7919u + static_cast<unsigned>(name[0]);
        if (!eval_file.empty() && !o.eval_params.LoadFromFile(eval_file)) {
            std::cerr << "warning: could not load eval params from " << eval_file << std::endl;
        }
        if (!model.empty()) o.eval_params.model_path = model;
        auto a = std::make_unique<Agent>(o);
        a->Init(setting, simf, pf);
        return a;
    };
    auto A = make(cfg.eval_a, cfg.model_a, cfg.budget_a, "A", cfg.refine_a, cfg.reply_shots_a, cfg.search_a);
    auto B = make(cfg.eval_b, cfg.model_b, cfg.budget_b, "B", cfg.refine_b, cfg.reply_shots_b, cfg.search_b);
    Sim referee(setting, simf, pf);

    std::ofstream out;
    if (!cfg.out_jsonl.empty()) out.open(cfg.out_jsonl, std::ios::app);

    for (int g = 0; g < cfg.games; ++g) {
        std::vector<Record> recs;
        recs.reserve(16 * (cfg.ends + 2));
        bool a_is_0 = !cfg.alternate || (g % 2 == 0);
        A->SetTeam(a_is_0 ? dc::Team::k0 : dc::Team::k1);
        B->SetTeam(a_is_0 ? dc::Team::k1 : dc::Team::k0);
        dc::GameState state(setting);
        auto t0 = std::chrono::steady_clock::now();
        int shots = 0;
        while (!state.IsGameOver()) {
            dc::Team next = state.GetNextTeam();
            Agent* ag = (next == A->team()) ? A.get() : B.get();
            const std::string who = (ag == A.get()) ? "A" : "B";
            dc::GameState before = state;
            auto t_think = std::chrono::steady_clock::now();
            dc::Move mv = ag->Think(state);
            long long think_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_think).count();
            const SearchResult& r = ag->last_result();
            Shot shot;
            if (std::holds_alternative<dc::moves::Shot>(mv)) {
                const auto& sh = std::get<dc::moves::Shot>(mv);
                shot.vx = sh.velocity.x; shot.vy = sh.velocity.y;
                shot.cw = (sh.rotation == dc::moves::Shot::Rotation::kCW);
            }
            if (out) {
                Record rec;
                rec.who = who; rec.team = next; rec.state = before; rec.shot = shot; rec.label = r.label;
                rec.value = r.value; rec.det = r.det_value; rec.sims = r.sims; rec.budget = ag->last_budget();
                rec.used = r.elapsed; rec.end_index = before.end;
                rec.p_hand = ag->searcher().evaluator().HandDistribution(before);
                recs.push_back(std::move(rec));
            }
            state = cfg.real_clock ? referee.ApplyTimed(before, shot, true, think_ms + 150) : referee.Apply(before, shot, true);
            ++shots;
            // End finished: record the hammer team's result.
            if (state.shot == 0 && before.shot == 15) {
                int e = before.end;
                int h = TeamIdx(before.hammer);
                int s0 = 0, s1 = 0;
                if (e < setting.max_end) {
                    s0 = state.scores[0][e].value_or(0);
                    s1 = state.scores[1][e].value_or(0);
                } else {
                    s0 = state.extra_end_score[0].value_or(0);
                    s1 = state.extra_end_score[1].value_or(0);
                }
                int k = (h == 0) ? (s0 - s1) : (s1 - s0);
                k = std::clamp(k, -4, 4);
                sum.hammer_hist[k + 4]++;
                for (auto& rec : recs) if (rec.end_index == e) rec.end_result_hammer = k;
            }
            if (shots > 16 * (cfg.ends + 6)) { std::cerr << "runaway game, aborting" << std::endl; break; }
        }
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        WriteRecords(out, g, cfg.ends, recs, state.game_result ? state.game_result->winner : dc::Team::kInvalid);
        out.flush();
        int sa = static_cast<int>(state.GetTotalScore(A->team()));
        int sb = static_cast<int>(state.GetTotalScore(B->team()));
        sum.games++;
        sum.a_score_diff += sa - sb;
        std::string result = "draw";
        if (state.game_result && state.game_result->winner == A->team()) { sum.a_wins++; result = "A"; }
        else if (state.game_result && state.game_result->winner == B->team()) { sum.b_wins++; result = "B"; }
        else sum.draws++;
        if (state.game_result && state.game_result->reason == dc::GameResult::Reason::kTimeLimit) result += "(TIMEOUT)";
        if (cfg.real_clock) {
            result += " clockA=" + Fmt(static_cast<float>(state.thinking_time_remaining[TeamIdx(A->team())].count() / 1000.0), 1)
                    + " clockB=" + Fmt(static_cast<float>(state.thinking_time_remaining[TeamIdx(B->team())].count() / 1000.0), 1);
        }
        std::cout << "game " << g << ": A(" << (a_is_0 ? "team0" : "team1") << ") " << sa << " - " << sb
                  << " B  winner=" << result << "  [" << Fmt(static_cast<float>(dt), 1) << "s, "
                  << shots << " shots]  running A-B: " << sum.a_wins << "-" << sum.b_wins << "-" << sum.draws
                  << std::endl;
    }
    std::cout << "hammer end-result histogram (k=-4..4): ";
    for (int i = 0; i < 9; ++i) std::cout << sum.hammer_hist[i] << (i < 8 ? " " : "\n");
    return sum;
}

}  // namespace gpw
