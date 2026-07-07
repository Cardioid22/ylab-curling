#include "selfplay_experiment.h"
#include "pool_experiment.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

SelfPlayExperiment::SelfPlayExperiment(
    const dc::GameSetting& game_setting, const SelfPlayConfig& config)
    : game_setting_(game_setting), config_(config)
{
}

// 歩(gPolicyポリシープレイヤー)の一手: ロールアウト候補からポリシーで選択 (探索なし)
static ShotInfo decideAyumu(const dc::GameState& state, dc::Team team,
                            ShotGenerator& gen, RolloutPolicy& policy)
{
    auto cands = gen.generateRolloutCandidates(state, team);
    std::vector<CandidateShot> filtered;
    for (auto& c : cands) if (c.type != ShotType::PASS) filtered.push_back(c);
    if (filtered.empty()) filtered = cands;
    if (filtered.empty()) return ShotInfo{0.0f, 0.0f, 1};  // 候補ゼロ (念のため)
    int sel = policy.selectShot(state, filtered, static_cast<int>(state.shot),
                                team, static_cast<int>(state.end), 0);
    if (sel < 0 || sel >= static_cast<int>(filtered.size())) sel = 0;
    return filtered[sel].shot;
}

// n_ends エンドを打ち切り、team0 視点の各エンド純得点を返す
// t0_ayumu/t1_ayumu が真なら その手番は歩(policy) が打つ (policy!=nullptr 前提)
std::vector<double> SelfPlayExperiment::playGame(
    ReinvestExperiment& e0, ReinvestExperiment& e1,
    RolloutPolicy* policy, bool t0_ayumu, bool t1_ayumu,
    SimulatorWrapper& sim, ShotGenerator& gen,
    std::mt19937& rng, uint64_t game_seed, int n_ends)
{
    dc::GameState state(game_setting_);
    state.end = 0;
    state.shot = 0;

    int total_shots = 0;
    int guard_max = 16 * n_ends + 8;
    while (static_cast<int>(state.end) < n_ends && !state.IsGameOver()
           && total_shots < guard_max) {
        // 手番はハンマー依存 (エンドをまたぐと先攻後攻が交代する)
        dc::Team team = state.GetNextTeam();
        bool is_ayumu = (team == dc::Team::k0) ? t0_ayumu : t1_ayumu;

        uint64_t sseed = game_seed
            ^ (static_cast<uint64_t>(total_shots) * 0x9E3779B97F4A7C15ULL);
        ShotInfo shot;
        if (is_ayumu && policy != nullptr) {
            shot = decideAyumu(state, team, gen, *policy);
        } else {
            ReinvestExperiment& e = (team == dc::Team::k0) ? e0 : e1;
            shot = e.decideShot(state, team, sseed, sim, gen, rng);
        }
        state = sim.run_single_simulation(state, shot);
        total_shots++;
    }

    std::vector<double> nets(n_ends, 0.0);
    for (int e = 0; e < n_ends; e++) {
        int s0 = (e < static_cast<int>(state.scores[0].size()) && state.scores[0][e])
                     ? static_cast<int>(*state.scores[0][e]) : 0;
        int s1 = (e < static_cast<int>(state.scores[1].size()) && state.scores[1][e])
                     ? static_cast<int>(*state.scores[1][e]) : 0;
        nets[e] = static_cast<double>(s0 - s1);  // team0 視点
    }
    return nets;
}

void SelfPlayExperiment::run()
{
    using clock = std::chrono::steady_clock;
    // 表示名: 歩なら "Ayumu(gPolicy)"、そうでなければ手法名
    auto disp = [](bool ayumu, const ReinvestConfig& c) -> std::string {
        if (ayumu) return "Ayumu(gPolicy, 探索なし)";
        return ReinvestExperiment::methodName(c.mode) + " depth"
             + std::to_string(c.depth) + " P=" + std::to_string(c.playouts)
             + " R=" + std::to_string(c.rollouts_per_visit);
    };
    std::cout << "\n=== Self-play: " << config_.label_a << " vs " << config_.label_b
              << " (" << config_.n_games << " games x " << config_.n_ends
              << " ends) ===" << std::endl;
    std::cout << "  A: " << disp(config_.a_is_ayumu, config_.arm_a) << std::endl;
    std::cout << "  B: " << disp(config_.b_is_ayumu, config_.arm_b) << std::endl;

    ReinvestExperiment expA(game_setting_, config_.arm_a);
    ReinvestExperiment expB(game_setting_, config_.arm_b);

    int N = std::max(1, config_.n_games);
    std::vector<std::vector<double>> netA_ends(N);  // 各ゲームの A 視点エンド別純得点
    std::vector<int> a_ham0(N, 0);                  // 端0でAが後攻(ハンマー)か
    std::atomic<int> next_idx{0};
    std::atomic<int> done{0};
    std::mutex log_mutex;
    auto start = clock::now();

    auto shared_grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    bool need_policy = config_.a_is_ayumu || config_.b_is_ayumu;

    auto worker = [&](int) {
        SimulatorWrapper sim(dc::Team::k0, game_setting_);
        ShotGenerator gen(game_setting_);
        sim.initialShotData.reserve(shared_grid.size());
        for (auto& pos : shared_grid) sim.initialShotData.push_back(sim.FindShot(pos));

        // 歩を使うならスレッド毎に gPolicy をロード (selectShot は内部状態を持つため共有しない)
        RolloutPolicy policy;
        if (need_policy) {
            if (!policy.load("data/policy_param.dat")) {
                std::lock_guard<std::mutex> lk(log_mutex);
                std::cerr << "[WARN] gPolicy 読み込み失敗 (data/policy_param.dat)"
                          << " -- 歩の手は候補先頭にフォールバック" << std::endl;
            }
        }

        while (true) {
            int g = next_idx.fetch_add(1);
            if (g >= N) break;

            // ハンマー入替: 偶数ゲームは A=team0, 奇数は A=team1
            bool swap = (g % 2 == 1);
            uint64_t game_seed = config_.seed
                ^ (static_cast<uint64_t>(g) * 0xD1B54A32D192ED03ULL);
            std::mt19937 rng(static_cast<uint32_t>(game_seed ^ (game_seed >> 32)));

            // team0/team1 が歩かどうか (swap で担当が入れ替わる)
            bool t0_ay = swap ? config_.b_is_ayumu : config_.a_is_ayumu;
            bool t1_ay = swap ? config_.a_is_ayumu : config_.b_is_ayumu;
            RolloutPolicy* pol = need_policy ? &policy : nullptr;

            std::vector<double> ends0 = swap
                ? playGame(expB, expA, pol, t0_ay, t1_ay, sim, gen, rng, game_seed, config_.n_ends)
                : playGame(expA, expB, pol, t0_ay, t1_ay, sim, gen, rng, game_seed, config_.n_ends);
            std::vector<double> aEnds(ends0.size());
            for (size_t i = 0; i < ends0.size(); i++) aEnds[i] = swap ? -ends0[i] : ends0[i];
            netA_ends[g] = aEnds;       // A 視点 エンド別
            a_ham0[g] = swap ? 1 : 0;   // 端0で A が後攻(=ハンマー)か (既定 hammer=k1)

            int d = ++done;
            if (d % 10 == 0 || d == N) {
                std::lock_guard<std::mutex> lk(log_mutex);
                auto el = std::chrono::duration<double>(clock::now() - start).count();
                std::cerr << "[" << d << "/" << N << "] elapsed=" << std::fixed
                          << std::setprecision(1) << el << "s eta="
                          << (el / d) * (N - d) << "s" << std::endl;
            }
        }
    };

    std::vector<std::thread> threads;
    int nt = std::max(1, config_.num_threads);
    for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    // 集計 (勝敗は全エンド合計で判定)
    int a_win = 0, b_win = 0, tie = 0;
    double sum = 0.0;
    std::vector<double> end_sum(config_.n_ends, 0.0);
    for (int g = 0; g < N; g++) {
        double tot = 0.0;
        for (int e = 0; e < config_.n_ends; e++) { tot += netA_ends[g][e]; end_sum[e] += netA_ends[g][e]; }
        sum += tot;
        if (tot > 0) a_win++; else if (tot < 0) b_win++; else tie++;
    }
    double mean_net = sum / N;
    int decisive = a_win + b_win;
    double win_rate = decisive > 0 ? static_cast<double>(a_win) / decisive : 0.5;

    // 二項検定 (引分除外, 帰無=0.5)
    double p_binom = 1.0;
    if (decisive > 0) {
        // 正規近似 (n が小さければ参考値)
        double z = (a_win - 0.5 * decisive) / std::sqrt(0.25 * decisive);
        p_binom = std::erfc(std::abs(z) / std::sqrt(2.0));  // 両側
    }

    auto total = std::chrono::duration<double>(clock::now() - start).count();
    std::cout << "\n=== Result (" << config_.label_a << " vs " << config_.label_b << ") ===" << std::endl;
    std::cout << "  games            = " << N << "  (" << total << "s)" << std::endl;
    std::cout << "  " << config_.label_a << " win / " << config_.label_b << " win / tie = "
              << a_win << " / " << b_win << " / " << tie << std::endl;
    std::cout << "  win rate (" << config_.label_a << ", 引分除外) = "
              << std::fixed << std::setprecision(3) << win_rate << std::endl;
    std::cout << "  mean net (" << config_.label_a << " 視点, 全" << config_.n_ends
              << "エンド計) = " << std::showpos << mean_net << std::noshowpos
              << " 点/ゲーム" << std::endl;
    if (config_.n_ends > 1) {
        std::cout << "  per-end mean net (A視点):";
        for (int e = 0; e < config_.n_ends; e++)
            std::cout << " end" << e << "=" << std::showpos << (end_sum[e] / N) << std::noshowpos;
        std::cout << std::endl;
    }
    std::cout << "  binomial p (両側, 正規近似) = " << p_binom << std::endl;

    if (!config_.output_dir.empty()) {
        std::filesystem::create_directories(config_.output_dir);
        std::string sm = config_.output_dir + "/selfplay_summary.csv";
        std::ofstream ofs(sm);
        ofs << "label_a,label_b,method_a,method_b,depth,playouts,rollouts,n_games,"
            << "a_win,b_win,tie,win_rate,mean_net,binom_p\n";
        std::string mname_a = config_.a_is_ayumu ? "Ayumu"
                                : ReinvestExperiment::methodName(config_.arm_a.mode);
        std::string mname_b = config_.b_is_ayumu ? "Ayumu"
                                : ReinvestExperiment::methodName(config_.arm_b.mode);
        ofs << config_.label_a << "," << config_.label_b << ","
            << mname_a << "," << mname_b << ","
            << config_.arm_a.depth << "," << config_.arm_a.playouts << ","
            << config_.arm_a.rollouts_per_visit << "," << N << ","
            << a_win << "," << b_win << "," << tie << ","
            << win_rate << "," << mean_net << "," << p_binom << "\n";
        // 各ゲームのエンド別純得点 (A視点) + 端0ハンマー。net_a=総純得点(後段集計と互換)
        std::ofstream og(config_.output_dir + "/selfplay_games.csv");
        og << "game,a_hammer_end0";
        for (int e = 0; e < config_.n_ends; e++) og << ",net_end" << e;
        og << ",net_a\n";
        for (int g = 0; g < N; g++) {
            og << g << "," << a_ham0[g];
            double tot = 0.0;
            for (int e = 0; e < config_.n_ends; e++) { og << "," << netA_ends[g][e]; tot += netA_ends[g][e]; }
            og << "," << tot << "\n";
        }
        std::cout << "  [csv] -> " << sm << std::endl;
    }
}
