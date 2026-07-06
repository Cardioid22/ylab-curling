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

// 単一エンド (end 0) を打ち切り、team0 の純得点を返す
double SelfPlayExperiment::playOneEnd(
    ReinvestExperiment& e0, ReinvestExperiment& e1,
    SimulatorWrapper& sim, ShotGenerator& gen,
    std::mt19937& rng, uint64_t game_seed)
{
    dc::GameState state(game_setting_);
    state.end = 0;
    state.shot = 0;

    int guard = 0;
    while (state.end == 0 && !state.IsGameOver() && guard < 64) {
        int shot_num = static_cast<int>(state.shot);
        dc::Team team = (shot_num % 2 == 0) ? dc::Team::k0 : dc::Team::k1;
        ReinvestExperiment& e = (team == dc::Team::k0) ? e0 : e1;

        uint64_t sseed = game_seed
            ^ (static_cast<uint64_t>(shot_num) * 0x9E3779B97F4A7C15ULL);
        ShotInfo shot = e.decideShot(state, team, sseed, sim, gen, rng);
        state = sim.run_single_simulation(state, shot);
        guard++;
    }

    int s0 = state.scores[0][0] ? static_cast<int>(*state.scores[0][0]) : 0;
    int s1 = state.scores[1][0] ? static_cast<int>(*state.scores[1][0]) : 0;
    return static_cast<double>(s0 - s1);
}

void SelfPlayExperiment::run()
{
    using clock = std::chrono::steady_clock;
    std::cout << "\n=== Self-play: " << config_.label_a << " vs " << config_.label_b
              << " (" << config_.n_games << " ends) ===" << std::endl;
    std::cout << "  A: " << ReinvestExperiment::methodName(config_.arm_a.mode)
              << " depth" << config_.arm_a.depth << " P=" << config_.arm_a.playouts
              << " R=" << config_.arm_a.rollouts_per_visit << std::endl;
    std::cout << "  B: " << ReinvestExperiment::methodName(config_.arm_b.mode)
              << " depth" << config_.arm_b.depth << " P=" << config_.arm_b.playouts
              << " R=" << config_.arm_b.rollouts_per_visit << std::endl;

    ReinvestExperiment expA(game_setting_, config_.arm_a);
    ReinvestExperiment expB(game_setting_, config_.arm_b);

    int N = std::max(1, config_.n_games);
    std::vector<double> netA(N, 0.0);  // 各ゲームでの A 視点の純得点
    std::atomic<int> next_idx{0};
    std::atomic<int> done{0};
    std::mutex log_mutex;
    auto start = clock::now();

    auto shared_grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    auto worker = [&](int) {
        SimulatorWrapper sim(dc::Team::k0, game_setting_);
        ShotGenerator gen(game_setting_);
        sim.initialShotData.reserve(shared_grid.size());
        for (auto& pos : shared_grid) sim.initialShotData.push_back(sim.FindShot(pos));

        while (true) {
            int g = next_idx.fetch_add(1);
            if (g >= N) break;

            // ハンマー入替: 偶数ゲームは A=team0, 奇数は A=team1
            bool swap = (g % 2 == 1);
            uint64_t game_seed = config_.seed
                ^ (static_cast<uint64_t>(g) * 0xD1B54A32D192ED03ULL);
            std::mt19937 rng(static_cast<uint32_t>(game_seed ^ (game_seed >> 32)));

            double net0;
            if (!swap) net0 = playOneEnd(expA, expB, sim, gen, rng, game_seed);
            else       net0 = playOneEnd(expB, expA, sim, gen, rng, game_seed);
            netA[g] = swap ? -net0 : net0;  // A 視点に統一

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

    // 集計
    int a_win = 0, b_win = 0, tie = 0;
    double sum = 0.0;
    for (double v : netA) {
        sum += v;
        if (v > 0) a_win++; else if (v < 0) b_win++; else tie++;
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
    std::cout << "  mean net (" << config_.label_a << " 視点) = "
              << std::showpos << mean_net << std::noshowpos << " 点/エンド" << std::endl;
    std::cout << "  binomial p (両側, 正規近似) = " << p_binom << std::endl;

    if (!config_.output_dir.empty()) {
        std::filesystem::create_directories(config_.output_dir);
        std::string sm = config_.output_dir + "/selfplay_summary.csv";
        std::ofstream ofs(sm);
        ofs << "label_a,label_b,method_a,method_b,depth,playouts,rollouts,n_games,"
            << "a_win,b_win,tie,win_rate,mean_net,binom_p\n";
        ofs << config_.label_a << "," << config_.label_b << ","
            << ReinvestExperiment::methodName(config_.arm_a.mode) << ","
            << ReinvestExperiment::methodName(config_.arm_b.mode) << ","
            << config_.arm_a.depth << "," << config_.arm_a.playouts << ","
            << config_.arm_a.rollouts_per_visit << "," << N << ","
            << a_win << "," << b_win << "," << tie << ","
            << win_rate << "," << mean_net << "," << p_binom << "\n";
        // 各ゲームの純得点も
        std::ofstream og(config_.output_dir + "/selfplay_games.csv");
        og << "game,net_a\n";
        for (int g = 0; g < N; g++) og << g << "," << netA[g] << "\n";
        std::cout << "  [csv] -> " << sm << std::endl;
    }
}
