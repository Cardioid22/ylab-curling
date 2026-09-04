// gpw_agent entry point.
//
//   gpw_agent <host> <port> [options]        tournament client (DigitalCurling3 protocol v1)
//   gpw_agent --selfplay [options]           headless self-play / A-B testing
//   gpw_agent --bench                        physics timing
//
// Options: --threads N  --name S  --log FILE  --eval FILE  --fixed-budget SEC  --quiet
//          --games N --ends E --budget-a S --budget-b S --eval-a F --eval-b F --out FILE

#include "agent.h"
#include "selfplay.h"
#include "sim.h"

#include <boost/asio.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace {

using gpw::Fmt;
namespace dc = digitalcurling3;

struct Args {
    std::string host, port;
    bool selfplay = false, bench = false;
    std::string check_model_records;
    int check_n = 5;
    std::string annotate_in, annotate_out;
    std::string fit_in, fit_out;
    gpw::AgentOptions agent;
    gpw::SelfplayConfig sp;
    std::string eval_file;
};

bool ParseArgs(int argc, char** argv, Args& a) {
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { std::cerr << "missing value for " << argv[i] << std::endl; std::exit(2); }
        return argv[++i];
    };
    int positional = 0;
    a.agent.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    a.sp.threads = a.agent.threads;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--selfplay") a.selfplay = true;
        else if (s == "--bench") a.bench = true;
        else if (s == "--check-model") a.check_model_records = need(i);
        else if (s == "--check-n") a.check_n = std::atoi(need(i));
        else if (s == "--annotate") { a.annotate_in = need(i); a.annotate_out = need(i); }
        else if (s == "--fit-eval") { a.fit_in = need(i); a.fit_out = need(i); }
        else if (s == "--threads") { a.agent.threads = std::atoi(need(i)); a.sp.threads = a.agent.threads; }
        else if (s == "--name") a.agent.name = need(i);
        else if (s == "--log") a.agent.log_path = need(i);
        else if (s == "--eval") a.eval_file = need(i);
        else if (s == "--model") a.agent.eval_params.model_path = need(i);
        else if (s == "--model-a") a.sp.model_a = need(i);
        else if (s == "--model-b") a.sp.model_b = need(i);
        else if (s == "--fixed-budget") a.agent.fixed_budget = std::atof(need(i));
        else if (s == "--quiet") { a.agent.verbose = false; a.sp.verbose = false; }
        else if (s == "--verbose") { a.agent.verbose = true; a.sp.verbose = true; }
        else if (s == "--no-book") a.agent.use_book = false;
        else if (s == "--no-refine") a.agent.search_cfg.refine = false;
        else if (s == "--refine") a.agent.search_cfg.refine = true;
        else if (s == "--no-refine-a") a.sp.refine_a = false;
        else if (s == "--no-refine-b") a.sp.refine_b = false;
        else if (s == "--reply-shots-a") a.sp.reply_shots_a = std::atoi(need(i));
        else if (s == "--reply-shots-b") a.sp.reply_shots_b = std::atoi(need(i));
        else if (s == "--games") a.sp.games = std::atoi(need(i));
        else if (s == "--ends") a.sp.ends = std::atoi(need(i));
        else if (s == "--budget-a") a.sp.budget_a = std::atof(need(i));
        else if (s == "--budget-b") a.sp.budget_b = std::atof(need(i));
        else if (s == "--eval-a") a.sp.eval_a = need(i);
        else if (s == "--eval-b") a.sp.eval_b = need(i);
        else if (s == "--out") a.sp.out_jsonl = need(i);
        else if (s == "--no-alternate") a.sp.alternate = false;
        else if (s == "--explore") a.sp.explore_eps = std::atof(need(i));
        else if (s == "--seed") a.sp.seed = static_cast<unsigned>(std::atoi(need(i)));
        else if (s == "--keep") a.agent.search_cfg.prescreen_keep = std::atoi(need(i));
        else if (s == "--max-sims") a.agent.search_cfg.max_sims = std::atoi(need(i));
        else if (s == "--reply-shots") a.agent.search_cfg.reply_last_shots = std::atoi(need(i));
        else if (s == "--safety") a.agent.time_cfg.safety = std::atof(need(i));
        else if (s.size() > 2 && s[0] == '-' && s[1] == '-') { std::cerr << "unknown option " << s << std::endl; return false; }
        else {
            if (positional == 0) a.host = s;
            else if (positional == 1) a.port = s;
            ++positional;
        }
    }
    if (!a.eval_file.empty() && !a.agent.eval_params.LoadFromFile(a.eval_file)) {
        std::cerr << "warning: could not load eval params from " << a.eval_file << std::endl;
    }
    return true;
}

int RunBench(const Args& a) {
    dc::GameSetting setting;
    setting.max_end = 10;
    setting.thinking_time[0] = setting.thinking_time[1] = std::chrono::milliseconds(219000);
    setting.extra_end_thinking_time[0] = setting.extra_end_thinking_time[1] = std::chrono::milliseconds(21900);
    dc::simulators::SimulatorFCV1Factory simf;
    dc::players::PlayerNormalDistFactory pf;
    gpw::Sim sim(setting, simf, pf);
    gpw::VelocitySolver vs;

    auto t0 = std::chrono::steady_clock::now();
    gpw::Shot draw = vs.Draw(dc::Vector2(0.f, gpw::kTeeY), false);
    double t_solve = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "solve draw tee: v=(" << draw.vx << "," << draw.vy << ") " << Fmt(static_cast<float>(t_solve * 1000), 1) << " ms\n";

    dc::GameState st(setting);
    int n = 20;
    t0 = std::chrono::steady_clock::now();
    dc::GameState after;
    for (int i = 0; i < n; ++i) after = sim.Apply(st, draw, true);
    double t_apply = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n;
    std::cout << "apply draw on empty board: " << Fmt(static_cast<float>(t_apply * 1000), 1) << " ms/shot; result stone at ";
    auto stones = gpw::Stones(after);
    if (!stones.empty()) std::cout << "(" << stones[0].p.x << "," << stones[0].p.y << ") d_tee=" << stones[0].d;
    std::cout << "\n";

    // Fill the house with 8 stones and time a take-out.
    dc::GameState busy = st;
    for (int k = 0; k < 8; ++k) {
        gpw::Shot s = vs.Draw(dc::Vector2(-1.2f + 0.35f * k, gpw::kTeeY - 0.8f + 0.2f * (k % 3)), k % 2 == 0);
        busy = sim.Apply(busy, s, true);
    }
    gpw::Shot hit = vs.Solve(dc::Vector2(0.f, gpw::kTeeY), 2.2f, false);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) after = sim.Apply(busy, hit, true);
    t_apply = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / n;
    std::cout << "apply hit with 8 stones: " << Fmt(static_cast<float>(t_apply * 1000), 1) << " ms/shot\n";

    // Full agent init + one search.
    gpw::AgentOptions o = a.agent;
    o.fixed_budget = o.fixed_budget > 0 ? o.fixed_budget : 2.0;
    gpw::Agent agent(o);
    t0 = std::chrono::steady_clock::now();
    agent.Init(setting, simf, pf);
    std::cout << "agent init: " << Fmt(static_cast<float>(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()), 2) << " s\n";
    agent.SetTeam(busy.GetNextTeam());
    t0 = std::chrono::steady_clock::now();
    agent.Think(busy);
    std::cout << "search (budget " << o.fixed_budget << "s): " << Fmt(static_cast<float>(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()), 2)
              << " s, sims=" << agent.last_result().sims << ", sim_cost=" << Fmt(static_cast<float>(agent.searcher().sim_cost() * 1000), 1) << " ms\n";
    return 0;
}

dc::GameSetting SettingFor(int max_end) {
    dc::GameSetting setting;
    setting.max_end = static_cast<std::uint8_t>(max_end);
    setting.thinking_time[0] = setting.thinking_time[1] = std::chrono::milliseconds(219000);
    setting.extra_end_thinking_time[0] = setting.extra_end_thinking_time[1] = std::chrono::milliseconds(21900);
    return setting;
}

dc::GameState StateFromRecord(const nlohmann::json& j, const dc::GameSetting& setting) {
    dc::GameState st(setting);
    st.end = static_cast<std::uint8_t>(j.at("end").get<int>());
    st.shot = static_cast<std::uint8_t>(j.at("shot").get<int>());
    st.hammer = static_cast<dc::Team>(j.at("hammer").get<int>());
    const auto& stones = j.at("stones");
    for (int k = 0; k < 16; ++k) {
        if (stones[k].is_null()) { st.stones[k / 8][k % 8] = std::nullopt; continue; }
        st.stones[k / 8][k % 8] = dc::Transform(dc::Vector2(stones[k][0].get<float>(), stones[k][1].get<float>()), 0.f);
    }
    st.scores[0][0] = static_cast<std::uint8_t>(j.at("score")[0].get<int>());
    st.scores[1][0] = static_cast<std::uint8_t>(j.at("score")[1].get<int>());
    return st;
}

// Fits the scalar parameters of the hand-crafted end-result model to self-play
// records by coordinate descent on the cross-entropy of HandDistribution vs the
// observed hammer end result. Writes an EvalParams file.
int RunFitEval(const Args& a) {
    std::ifstream in(a.fit_in);
    if (!in) { std::cerr << "cannot open " << a.fit_in << std::endl; return 1; }
    struct Rec { dc::GameState st; int k; };
    std::vector<Rec> recs;
    std::string line;
    int max_end = 10;
    std::array<double, 9> hist{};
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        nlohmann::json j = nlohmann::json::parse(line);
        max_end = j.value("max_end", 10);
        if (j.at("shot").get<int>() == 0) continue;  // start-of-end states use the table, not the model
        Rec r;
        r.st = StateFromRecord(j, SettingFor(max_end));
        r.k = std::clamp(j.value("end_result_hammer", 0), -4, 4);
        hist[r.k + 4] += 1;
        recs.push_back(std::move(r));
    }
    if (recs.empty()) { std::cerr << "no records" << std::endl; return 1; }
    dc::GameSetting setting = SettingFor(max_end);
    gpw::EvalParams p = a.agent.eval_params;
    p.model_path.clear();
    {
        double tot = 0; for (double v : hist) tot += v;
        for (int k = 0; k < 9; ++k) p.end_dist[k] = hist[k] / tot;
    }
    auto ce = [&](const gpw::EvalParams& q) {
        gpw::Evaluator ev(q, setting);
        double s = 0;
        for (const auto& r : recs) {
            auto d = ev.HandDistribution(r.st);
            s += -std::log(std::max(d[r.k + 4], 1e-6));
        }
        return s / recs.size();
    };
    struct Knob { const char* name; double gpw::EvalParams::*ptr; double lo, hi; };
    const Knob knobs[] = {
        {"hammer_base", &gpw::EvalParams::hammer_base, 0.0, 3.0},
        {"count_kappa", &gpw::EvalParams::count_kappa, 0.5, 12.0},
        {"sigma0", &gpw::EvalParams::sigma0, 0.2, 2.5},
        {"sigma_r", &gpw::EvalParams::sigma_r, 0.0, 0.4},
        {"quality_w", &gpw::EvalParams::quality_w, 0.0, 3.0},
        {"cover_bonus", &gpw::EvalParams::cover_bonus, 0.0, 2.0},
        {"exposed_penalty", &gpw::EvalParams::exposed_penalty, 0.0, 1.0},
        {"guard_value", &gpw::EvalParams::guard_value, 0.0, 1.0},
    };
    double best = ce(p);
    std::cout << "records=" << recs.size() << " initial ce=" << best << std::endl;
    for (int round = 0; round < 6; ++round) {
        double step_scale = std::pow(0.5, round);
        bool improved = false;
        for (const auto& kn : knobs) {
            double cur = p.*(kn.ptr);
            double step = (kn.hi - kn.lo) * 0.1 * step_scale;
            for (double cand : {cur - step, cur + step, cur - 3 * step, cur + 3 * step}) {
                if (cand < kn.lo || cand > kn.hi) continue;
                gpw::EvalParams q = p;
                q.*(kn.ptr) = cand;
                double v = ce(q);
                if (v < best - 1e-6) { best = v; p = q; improved = true; }
            }
        }
        std::cout << "round " << round << " ce=" << best << "  " << p.Describe() << std::endl;
        if (!improved && round >= 2) break;
    }
    p.SaveToFile(a.fit_out);
    std::cout << "wrote " << a.fit_out << " (ce=" << best << ")" << std::endl;
    return 0;
}

// Adds "p_hand" (hand-crafted end-result distribution) to records that lack it.
int RunAnnotate(const Args& a) {
    std::ifstream in(a.annotate_in);
    std::ofstream out(a.annotate_out);
    if (!in || !out) { std::cerr << "cannot open files" << std::endl; return 1; }
    std::string line;
    gpw::EvalParams hand = a.agent.eval_params;
    hand.model_path.clear();
    std::unique_ptr<gpw::Evaluator> ev;
    int cur_max_end = -1, n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        nlohmann::json j = nlohmann::json::parse(line);
        int max_end = j.value("max_end", 10);
        if (!ev || cur_max_end != max_end) { ev = std::make_unique<gpw::Evaluator>(hand, SettingFor(max_end)); cur_max_end = max_end; }
        auto p = ev->HandDistribution(StateFromRecord(j, SettingFor(max_end)));
        j["p_hand"] = p;
        out << j.dump() << "\n";
        ++n;
    }
    std::cout << "annotated " << n << " records -> " << a.annotate_out << std::endl;
    return 0;
}

// Prints the evaluator's end-result distribution for the first records of a
// self-play JSONL file (compare with python/check_model.py).
int RunCheckModel(const Args& a) {
    using nlohmann::json;
    std::ifstream in(a.check_model_records);
    if (!in) { std::cerr << "cannot open " << a.check_model_records << std::endl; return 1; }
    std::string line;
    int i = 0;
    std::unique_ptr<gpw::Evaluator> ev;
    int cur_max_end = -1;
    while (i < a.check_n && std::getline(in, line)) {
        if (line.empty()) continue;
        json j = json::parse(line);
        int max_end = j.value("max_end", 10);
        dc::GameSetting setting = SettingFor(max_end);
        if (!ev || cur_max_end != max_end) { ev = std::make_unique<gpw::Evaluator>(a.agent.eval_params, setting); cur_max_end = max_end; }
        dc::GameState st = StateFromRecord(j, setting);
        auto p = ev->EndDistribution(st);
        std::cout << "rec " << i << " end=" << int(st.end) << " shot=" << int(st.shot)
                  << " target=" << j.value("end_result_hammer", 0) << ": ";
        for (int k = 0; k < 9; ++k) std::cout << Fmt(static_cast<float>(p[k]), 4) << (k < 8 ? " " : "\n");
        ++i;
    }
    return 0;
}

int RunClient(const Args& a) {
    using boost::asio::ip::tcp;
    using nlohmann::json;
    constexpr int kProtocolMajor = 1;

    boost::asio::io_context io;
    tcp::socket socket(io);
    tcp::resolver resolver(io);
    boost::asio::connect(socket, resolver.resolve(a.host, a.port));

    auto read_line = [&socket, buf = std::string()]() mutable {
        if (buf.empty()) boost::asio::read_until(socket, boost::asio::dynamic_buffer(buf), '\n');
        auto pos = buf.find_first_of('\n');
        auto line = buf.substr(0, pos + 1);
        buf.erase(0, pos + 1);
        return line;
    };
    auto send = [&socket](const json& j) {
        auto msg = j.dump() + '\n';
        boost::asio::write(socket, boost::asio::buffer(msg));
    };
    auto expect = [](const json& j, const char* cmd) {
        if (j.at("cmd").get<std::string>() != cmd) {
            throw std::runtime_error(std::string("unexpected cmd: ") + j.at("cmd").get<std::string>() + " (expected " + cmd + ")");
        }
    };

    // dc
    {
        auto j = json::parse(read_line());
        expect(j, "dc");
        if (j.at("version").at("major").get<int>() != kProtocolMajor) throw std::runtime_error("protocol version");
        std::cout << "[in] dc game_id=" << j.at("game_id").get<std::string>() << std::endl;
    }
    send({{"cmd", "dc_ok"}, {"name", a.agent.name}});

    gpw::Agent agent(a.agent);
    dc::Team team = dc::Team::kInvalid;
    {
        auto j = json::parse(read_line());
        expect(j, "is_ready");
        if (j.at("game").at("rule").get<std::string>() != "normal") throw std::runtime_error("rule");
        team = j.at("team").get<dc::Team>();
        auto setting = j.at("game").at("setting").get<dc::GameSetting>();
        auto simf = j.at("game").at("simulator").get<std::unique_ptr<dc::ISimulatorFactory>>();
        const auto& jp = j.at("game").at("players").at(dc::ToString(team));
        auto pf = jp[0].get<std::unique_ptr<dc::IPlayerFactory>>();
        std::cout << "[in] is_ready team=" << dc::ToString(team)
                  << " ends=" << int(setting.max_end)
                  << " time=" << setting.thinking_time[gpw::TeamIdx(team)].count() / 1000.0 << "s"
                  << " extra=" << setting.extra_end_thinking_time[gpw::TeamIdx(team)].count() / 1000.0 << "s"
                  << " five_rock=" << setting.five_rock_rule << std::endl;
        agent.SetTeam(team);
        agent.Init(setting, *simf, *pf);
        send({{"cmd", "ready_ok"}, {"player_order", std::array<size_t, 4>{0, 1, 2, 3}}});
    }
    {
        auto j = json::parse(read_line());
        expect(j, "new_game");
        std::cout << "[in] new_game " << j.at("name").at("team0") << " vs " << j.at("name").at("team1") << std::endl;
    }

    dc::GameState state;
    while (true) {
        auto j = json::parse(read_line());
        expect(j, "update");
        state = j.at("state").get<dc::GameState>();
        if (state.game_result) break;
        if (state.GetNextTeam() == team) {
            auto t_think = std::chrono::steady_clock::now();
            dc::Move mv = agent.Think(state);
            // Never answer faster than ~150 ms: an instant reply (opening book) makes the
            // server emit two updates back-to-back, which crashed the opponent's client
            // (JSON "Extra data") in local tests. Costs nothing over 80 shots.
            auto dt = std::chrono::steady_clock::now() - t_think;
            if (dt < std::chrono::milliseconds(150)) std::this_thread::sleep_for(std::chrono::milliseconds(150) - dt);
            send({{"cmd", "move"}, {"move", mv}});
        }
    }
    {
        auto j = json::parse(read_line());
        expect(j, "game_over");
    }
    int s0 = static_cast<int>(state.GetTotalScore(dc::Team::k0));
    int s1 = static_cast<int>(state.GetTotalScore(dc::Team::k1));
    std::string res = "draw";
    if (state.game_result && state.game_result->winner == team) res = "WIN";
    else if (state.game_result && state.game_result->winner != dc::Team::kInvalid) res = "LOSS";
    std::cout << "[game_over] team0 " << s0 << " - " << s1 << " team1  (" << dc::ToString(team) << " " << res << ")" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Args a;
    if (!ParseArgs(argc, argv, a)) return 2;
    try {
        if (a.bench) return RunBench(a);
        if (!a.check_model_records.empty()) return RunCheckModel(a);
        if (!a.annotate_in.empty()) return RunAnnotate(a);
        if (!a.fit_in.empty()) return RunFitEval(a);
        if (a.selfplay) {
            auto s = gpw::RunSelfplay(a.sp);
            std::cout << "RESULT games=" << s.games << " A=" << s.a_wins << " B=" << s.b_wins << " draws=" << s.draws
                      << " A_score_diff_mean=" << (s.games ? s.a_score_diff / s.games : 0.0) << std::endl;
            return 0;
        }
        if (a.host.empty() || a.port.empty()) {
            std::cerr << "usage: gpw_agent <host> <port> [--threads N --name S --log FILE --eval FILE]\n"
                         "       gpw_agent --selfplay --games N --ends E --budget-a S --budget-b S [--eval-a F --eval-b F --out FILE]\n"
                         "       gpw_agent --bench\n";
            return 2;
        }
        return RunClient(a);
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << std::endl;
        return 1;
    }
}
