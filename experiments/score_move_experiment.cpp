#include "score_move_experiment.h"
#include "pool_experiment.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

ScoreMoveExperiment::ScoreMoveExperiment(
    const dc::GameSetting& game_setting,
    const ScoreMoveConfig& config)
    : game_setting_(game_setting), config_(config)
{
}

void ScoreMoveExperiment::run() {
    using clock = std::chrono::steady_clock;

    std::cout << "\n=== Score-Move Referee ===" << std::endl;
    std::cout << "  n_states           = " << config_.n_states << std::endl;
    std::cout << "  score_rollouts (K) = " << config_.score_rollouts << std::endl;
    std::cout << "  resample_first_shot= " << (config_.resample_first_shot ? "true (execution-uncertainty-aware)" : "false (frozen landing)") << std::endl;
    std::cout << "  epsilon            = " << config_.epsilon << std::endl;
    std::cout << "  num_threads        = " << config_.num_threads << std::endl;
    std::cout << "  seed               = " << config_.seed << std::endl;
    std::cout << "  start_index        = " << config_.start_index << std::endl;
    std::cout << "  max_positions      = " << config_.max_positions << std::endl;
    std::cout << "  load_positions_dir = " << config_.load_positions_dir << std::endl;
    std::cout << "  output_dir         = " << config_.output_dir << std::endl;

    // 1. 盤面ロード
    auto all_records = mcts_shared::loadTestPositionsFromCSV(
        config_.load_positions_dir, game_setting_, -1);
    if (all_records.empty()) {
        std::cerr << "Error: no positions loaded from " << config_.load_positions_dir << std::endl;
        return;
    }

    // 2. サンプリング (depth_n_mcts と同じ規則。行数==n_states なら no-op で固定)
    auto sampled = mcts_shared::sampleTestPositions(
        all_records, config_.n_states, config_.seed);
    std::cout << "  Sampled " << sampled.size() << " / " << all_records.size()
              << " positions (seed=" << config_.seed << ")" << std::endl;

    // 2b. スライス。state_seed 用にサンプリング後グローバル index を保持
    std::vector<int> global_indices(sampled.size());
    std::iota(global_indices.begin(), global_indices.end(), 0);
    if (config_.start_index > 0 || config_.max_positions > 0) {
        int total = static_cast<int>(sampled.size());
        int s = std::min(std::max(0, config_.start_index), total);
        int e = (config_.max_positions < 0) ? total : std::min(total, s + config_.max_positions);
        sampled = std::vector<mcts_shared::TestPositionRecord>(sampled.begin() + s, sampled.begin() + e);
        global_indices = std::vector<int>(global_indices.begin() + s, global_indices.begin() + e);
        std::cout << "  Sliced to [" << s << ", " << e << ") = " << sampled.size() << " positions" << std::endl;
    }

    int P = static_cast<int>(sampled.size());
    if (P == 0) {
        std::cerr << "Error: no positions after slicing" << std::endl;
        return;
    }

    std::filesystem::create_directories(config_.output_dir);

    // ロールアウトに使う 4x4 グリッド (depth_n_mcts と同一)
    auto shared_grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    // 3. 候補プールを各局面ごとに事前生成 (シリアル。局面数が小さい前提)
    //    候補 index は depth_n_mcts の generatePool と同じ順序なので実験 CSV と整合する
    std::cout << "  Generating candidate pools for " << P << " positions..." << std::endl;
    std::vector<CandidatePool> pools(P);
    {
        ShotGenerator gen(game_setting_);
        for (int p = 0; p < P; p++) {
            pools[p] = gen.generatePool(sampled[p].state, sampled[p].current_team);
            std::cout << "    [" << (p + 1) << "/" << P << "] g=" << sampled[p].game_id
                      << " e=" << sampled[p].end << " s=" << sampled[p].shot_num
                      << " -> " << pools[p].candidates.size() << " candidates" << std::endl;
        }
    }

    // 4. (position, candidate) ジョブを平坦化して並列採点
    struct Job { int p; int c; };
    std::vector<Job> jobs;
    for (int p = 0; p < P; p++) {
        for (int c = 0; c < static_cast<int>(pools[p].candidates.size()); c++) {
            jobs.push_back({p, c});
        }
    }
    int total_jobs = static_cast<int>(jobs.size());
    std::cout << "  Total candidate-scoring jobs: " << total_jobs
              << " (K=" << config_.score_rollouts << " rollouts each)" << std::endl;

    std::vector<ScoredCandidate> results(total_jobs);
    std::atomic<int> next_job{0};
    std::atomic<int> done_count{0};
    std::mutex log_mutex;
    auto start = clock::now();

    auto worker = [&](int thread_id) {
        SimulatorWrapper sim(dc::Team::k0, game_setting_);
        sim.initialShotData.reserve(shared_grid.size());
        for (auto& pos : shared_grid) {
            sim.initialShotData.push_back(sim.FindShot(pos));
        }

        while (true) {
            int j = next_job.fetch_add(1);
            if (j >= total_jobs) break;

            int p = jobs[j].p;
            int c = jobs[j].c;
            const auto& rec = sampled[p];
            const auto& cand = pools[p].candidates[c];
            const dc::GameState& post_state = pools[p].result_states[c];
            dc::Team root_team = rec.current_team;

            // ジョブ固有の決定的シード (スレッド順に依存せず再現可能)
            uint64_t job_seed = config_.seed
                ^ (static_cast<uint64_t>(global_indices[p]) * 0x9E3779B97F4A7C15ULL)
                ^ (static_cast<uint64_t>(c) * 0x85EBCA6B029E4F31ULL);
            std::mt19937 rng(static_cast<uint32_t>(job_seed ^ (job_seed >> 32)));

            int K = std::max(1, config_.score_rollouts);
            double sum = 0.0, sumsq = 0.0;
            if (config_.resample_first_shot) {
                // 実行不確実性込み: 初手の物理シミュレーションを毎回振り直す。
                // Q_ref = E[着地ばらつき + 継続プレイ] = 「その手を試みる価値」
                for (int i = 0; i < K; i++) {
                    dc::GameState post = sim.run_single_simulation(rec.state, cand.shot);
                    double v;
                    if (post.end != rec.state.end || post.IsGameOver()) {
                        // 初手でエンドが終わった: 実スコアで評価
                        int e = rec.state.end;
                        int t0 = post.scores[0][e] ? static_cast<int>(*post.scores[0][e]) : 0;
                        int t1 = post.scores[1][e] ? static_cast<int>(*post.scores[1][e]) : 0;
                        double diff = static_cast<double>(t0 - t1);
                        v = (root_team == dc::Team::k0) ? diff : -diff;
                    } else {
                        int remaining = 16 - static_cast<int>(post.shot);
                        v = mcts_shared::rolloutFromState(
                            sim, post, remaining, root_team, rng, config_.epsilon);
                    }
                    sum += v;
                    sumsq += v * v;
                }
            } else {
                // 従来: 候補プール生成時の1回の着地で固定 (探索木の子ノードと同じ規約)
                int remaining = 16 - static_cast<int>(post_state.shot);
                for (int i = 0; i < K; i++) {
                    double v = mcts_shared::rolloutFromState(
                        sim, post_state, remaining, root_team, rng, config_.epsilon);
                    sum += v;
                    sumsq += v * v;
                }
            }
            double mean = sum / K;
            double var = std::max(0.0, sumsq / K - mean * mean);

            ScoredCandidate sc;
            sc.game_id = rec.game_id;
            sc.end = rec.end;
            sc.shot_num = rec.shot_num;
            sc.candidate_idx = c;
            sc.label = cand.label;
            sc.shot_type = static_cast<int>(cand.type);
            sc.q_ref_mean = mean;
            sc.q_ref_sd = std::sqrt(var);
            sc.n_rollouts = K;
            sc.resampled = config_.resample_first_shot ? 1 : 0;
            results[j] = sc;

            int d = ++done_count;
            if (d % 25 == 0 || d == total_jobs) {
                std::lock_guard<std::mutex> lk(log_mutex);
                auto elapsed = std::chrono::duration<double>(clock::now() - start).count();
                double per = elapsed / d;
                std::cerr << "[" << d << "/" << total_jobs << "] elapsed="
                          << std::fixed << std::setprecision(1) << elapsed << "s eta="
                          << per * (total_jobs - d) << "s" << std::endl;
            }
        }
    };

    int nt = std::max(1, config_.num_threads);
    std::vector<std::thread> threads;
    for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    auto total_elapsed = std::chrono::duration<double>(clock::now() - start).count();
    std::cout << "\n=== Scored " << total_jobs << " candidates in "
              << std::fixed << std::setprecision(1) << total_elapsed << "s ===" << std::endl;

    // 5. CSV 出力 (並列実行時は start_index でファイル名分離)
    std::string suffix = (config_.start_index > 0 || config_.max_positions > 0)
        ? "_idx" + std::to_string(config_.start_index) : "";
    std::string csv_path = config_.output_dir + "/score_move_qtable" + suffix + ".csv";
    writeCSV(results, csv_path);
    std::cout << "Wrote " << csv_path << " (" << results.size() << " rows)" << std::endl;
}

void ScoreMoveExperiment::writeCSV(
    const std::vector<ScoredCandidate>& rows, const std::string& path) const
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Error: cannot open " << path << " for writing" << std::endl;
        return;
    }
    ofs << "game_id,end,shot_num,candidate_idx,label,shot_type,"
        << "q_ref_mean,q_ref_sd,n_rollouts,resampled\n";
    ofs << std::setprecision(6);
    for (const auto& r : rows) {
        if (r.candidate_idx < 0) continue;
        ofs << r.game_id << "," << r.end << "," << r.shot_num << ","
            << r.candidate_idx << ",\"" << r.label << "\"," << r.shot_type << ","
            << r.q_ref_mean << "," << r.q_ref_sd << "," << r.n_rollouts << ","
            << r.resampled << "\n";
    }
}
