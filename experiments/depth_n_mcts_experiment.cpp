#include "depth_n_mcts_experiment.h"
#include "pool_experiment.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>

DepthNMctsExperiment::DepthNMctsExperiment(
    const dc::GameSetting& game_setting,
    const DepthNMctsConfig& config)
    : game_setting_(game_setting), config_(config)
{
}

// ========== 盤面ハッシュ ==========
// 展開キャッシュのキー用。衝突が起きても機能的には再計算するだけなので緩いハッシュでOK
uint64_t DepthNMctsExperiment::hashGameState(const dc::GameState& s) const {
    uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    auto mix = [&](uint64_t x) {
        h ^= x;
        h *= 1099511628211ULL;
    };
    mix(static_cast<uint64_t>(s.end));
    mix(static_cast<uint64_t>(s.shot));
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 8; i++) {
            if (s.stones[t][i]) {
                // 座標を1mm単位に量子化
                int qx = static_cast<int>(std::round(s.stones[t][i]->position.x * 1000.0f));
                int qy = static_cast<int>(std::round(s.stones[t][i]->position.y * 1000.0f));
                mix(static_cast<uint64_t>(qx) | (static_cast<uint64_t>(qy) << 32));
                mix(static_cast<uint64_t>((t << 8) | i));
            } else {
                mix(static_cast<uint64_t>((t << 8) | i | 0x10000));
            }
        }
    }
    return h;
}

// ========== ノード展開 ==========

void DepthNMctsExperiment::expandNode(
    TreeNode& node,
    MctsMode mode,
    ShotGenerator& gen,
    std::unordered_map<uint64_t, CandidateCacheEntry>& cache)
{
    if (node.expanded) return;

    // 候補手とシミュ結果をキャッシュから引く
    uint64_t key = hashGameState(node.state);
    auto it = cache.find(key);
    if (it != cache.end()) {
        node.candidates = it->second.candidates;
        node.result_states = it->second.result_states;
    } else {
        auto pool = gen.generatePool(node.state, node.to_play);
        node.candidates = pool.candidates;
        node.result_states = pool.result_states;
        cache[key] = { node.candidates, node.result_states };
    }

    int N = static_cast<int>(node.candidates.size());
    if (N == 0) {
        node.expanded = true;
        return;
    }

    if (mode == MctsMode::Proposed) {
        // 距離行列 → クラスタリング → メドイド
        auto dist_table = mcts_shared::makeDistanceTableDelta(node.state, node.result_states);
        int K = std::max(1, static_cast<int>(std::ceil(N * config_.retention_rate)));
        K = std::min(K, N);
        node.clusters = mcts_shared::runClustering(dist_table, K);
        node.medoid_indices = mcts_shared::calculateMedoids(dist_table, node.clusters);

        // -1 (空クラスタ) を除外
        node.medoid_indices.erase(
            std::remove_if(node.medoid_indices.begin(), node.medoid_indices.end(),
                           [](int m) { return m < 0; }),
            node.medoid_indices.end());
    } else {
        // AllGrid: 全候補を子ノードに（クラスタリングなし）
        node.medoid_indices.resize(N);
        std::iota(node.medoid_indices.begin(), node.medoid_indices.end(), 0);
    }

    node.children.resize(node.medoid_indices.size());
    node.expanded = true;
}

// ========== UCB1選択 ==========

int DepthNMctsExperiment::selectBestChildUCB(const TreeNode& node) const {
    int best = -1;
    double best_score = -1e18;
    int K = static_cast<int>(node.medoid_indices.size());
    for (int i = 0; i < K; i++) {
        double mean;
        int visits;
        if (node.children[i]) {
            mean = node.children[i]->mean();
            visits = node.children[i]->visits;
        } else {
            mean = 0.0;
            visits = 0;
        }
        double score = mcts_shared::ucb1Score(mean, visits, node.visits, config_.ucb_c);
        if (score > best_score) { best_score = score; best = i; }
    }
    return best;
}

int DepthNMctsExperiment::selectMostVisited(const TreeNode& node) const {
    int best = -1;
    int best_visits = -1;
    double best_mean = -1e18;
    int K = static_cast<int>(node.medoid_indices.size());
    for (int i = 0; i < K; i++) {
        if (!node.children[i]) continue;
        int v = node.children[i]->visits;
        double m = node.children[i]->mean();
        if (v > best_visits || (v == best_visits && m > best_mean)) {
            best_visits = v;
            best_mean = m;
            best = i;
        }
    }
    return best;
}

// ========== プレイアウト ==========

double DepthNMctsExperiment::runPlayout(
    TreeNode& node,
    int max_depth,
    MctsMode mode,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
    std::mt19937& rng,
    dc::Team root_team)
{
    int rollouts_per_visit = (mode == MctsMode::Proposed)
        ? config_.proposed_rollouts_per_visit
        : config_.allgrid_rollouts_per_visit;
    if (rollouts_per_visit < 1) rollouts_per_visit = 1;

    // 葉に到達したらロールアウト開始
    if (node.depth >= max_depth) {
        int remaining = 16 - static_cast<int>(node.state.shot);
        double sum = 0.0;
        for (int i = 0; i < rollouts_per_visit; i++) {
            sum += mcts_shared::rolloutFromState(
                sim, gen, node.state, remaining, root_team, rng, config_.epsilon);
        }
        double mean_reward = sum / rollouts_per_visit;
        node.visits++;
        node.total_reward += mean_reward;
        return mean_reward;
    }

    // 未展開なら展開
    if (!node.expanded) {
        expandNode(node, mode, gen, cache);
    }
    int K = static_cast<int>(node.medoid_indices.size());
    if (K == 0) {
        // 候補がない異常系: ロールアウトのみ
        int remaining = 16 - static_cast<int>(node.state.shot);
        double sum = 0.0;
        for (int i = 0; i < rollouts_per_visit; i++) {
            sum += mcts_shared::rolloutFromState(
                sim, gen, node.state, remaining, root_team, rng, config_.epsilon);
        }
        double mean_reward = sum / rollouts_per_visit;
        node.visits++;
        node.total_reward += mean_reward;
        return mean_reward;
    }

    // UCB1 で子を選択
    int idx = selectBestChildUCB(node);
    if (idx < 0) idx = 0;

    // 子ノード未作成なら作る
    if (!node.children[idx]) {
        auto child = std::make_unique<TreeNode>();
        child->state = node.result_states[node.medoid_indices[idx]];
        child->depth = node.depth + 1;
        // 次の手番（shot が 1 進んだ結果盤面の手番）
        child->to_play = (child->state.shot % 2 == 0) ? dc::Team::k0 : dc::Team::k1;
        node.children[idx] = std::move(child);
    }

    // 子に再帰
    double reward = runPlayout(*node.children[idx], max_depth, mode, sim, gen, cache, rng, root_team);

    // バックプロパゲーション
    node.visits++;
    node.total_reward += reward;
    return reward;
}

// ========== 木構築 ==========

void DepthNMctsExperiment::buildTree(
    TreeNode& root,
    MctsMode mode,
    int playouts,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
    std::mt19937& rng,
    dc::Team root_team)
{
    // まずルートを展開しておく
    if (!root.expanded) {
        expandNode(root, mode, gen, cache);
    }
    for (int p = 0; p < playouts; p++) {
        runPlayout(root, config_.depth, mode, sim, gen, cache, rng, root_team);
    }
}

// ========== 1盤面実行 ==========

DepthNComparisonResult DepthNMctsExperiment::runOneState(
    const mcts_shared::TestPositionRecord& rec,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    uint64_t thread_seed)
{
    using clock = std::chrono::steady_clock;

    DepthNComparisonResult r;
    r.game_id = rec.game_id;
    r.end = rec.end;
    r.shot_num = rec.shot_num;

    dc::Team root_team = rec.current_team;

    // スレッド内キャッシュ（Proposed→AllGrid 間で共有）
    std::unordered_map<uint64_t, CandidateCacheEntry> cache;

    // --- Proposed ---
    TreeNode proposed_root;
    proposed_root.state = rec.state;
    proposed_root.depth = 0;
    proposed_root.to_play = root_team;

    {
        std::mt19937 rng(thread_seed);  // 両手法で同じシード
        auto t0 = clock::now();
        buildTree(proposed_root, MctsMode::Proposed,
                  config_.proposed_playouts, sim, gen, cache, rng, root_team);
        auto t1 = clock::now();
        r.proposed_time_sec = std::chrono::duration<double>(t1 - t0).count();
    }

    r.num_candidates = static_cast<int>(proposed_root.candidates.size());
    r.num_clusters = static_cast<int>(proposed_root.medoid_indices.size());
    r.proposed_actual_playouts = proposed_root.visits;

    int proposed_idx = selectMostVisited(proposed_root);
    if (proposed_idx >= 0) {
        int cand_idx = proposed_root.medoid_indices[proposed_idx];
        r.proposed_best_idx = cand_idx;
        r.proposed_best_mean = proposed_root.children[proposed_idx]->mean();
        r.proposed_label = proposed_root.candidates[cand_idx].label;
    }

    // --- AllGrid ---
    TreeNode allgrid_root;
    allgrid_root.state = rec.state;
    allgrid_root.depth = 0;
    allgrid_root.to_play = root_team;

    {
        std::mt19937 rng(thread_seed);  // Proposed と同じシードで再スタート
        auto t0 = clock::now();
        buildTree(allgrid_root, MctsMode::AllGrid,
                  config_.allgrid_playouts, sim, gen, cache, rng, root_team);
        auto t1 = clock::now();
        r.allgrid_time_sec = std::chrono::duration<double>(t1 - t0).count();
    }

    r.allgrid_actual_playouts = allgrid_root.visits;

    int allgrid_idx = selectMostVisited(allgrid_root);
    if (allgrid_idx >= 0) {
        int cand_idx = allgrid_root.medoid_indices[allgrid_idx];
        r.allgrid_best_idx = cand_idx;
        r.allgrid_best_mean = allgrid_root.children[allgrid_idx]->mean();
        r.allgrid_label = allgrid_root.candidates[cand_idx].label;
    }

    // --- 比較指標 ---
    if (r.proposed_best_idx >= 0 && r.allgrid_best_idx >= 0) {
        r.exact_match = (r.proposed_best_idx == r.allgrid_best_idx);
        r.same_type = (proposed_root.candidates[r.proposed_best_idx].type
                       == allgrid_root.candidates[r.allgrid_best_idx].type);
        r.score_diff = std::abs(r.proposed_best_mean - r.allgrid_best_mean);

        // クラスタ一致: AllGrid 最良手候補が Proposed の最良メドイドが属するクラスタに含まれるか
        if (!proposed_root.clusters.empty() && proposed_idx >= 0) {
            // Proposed 最良メドイドが属するクラスタを探す
            int best_cluster_id = -1;
            int medoid_cand = proposed_root.medoid_indices[proposed_idx];
            for (int c = 0; c < static_cast<int>(proposed_root.clusters.size()); c++) {
                if (proposed_root.clusters[c].count(medoid_cand) > 0) {
                    best_cluster_id = c;
                    break;
                }
            }
            if (best_cluster_id >= 0) {
                r.same_cluster = (proposed_root.clusters[best_cluster_id]
                                      .count(r.allgrid_best_idx) > 0);
            }
        }
    }

    return r;
}

// ========== 進捗ログ ==========

void DepthNMctsExperiment::logProgress(int done, int total, double elapsed_sec) const {
    if (done <= 0) return;
    double per_state = elapsed_sec / done;
    double eta = per_state * (total - done);
    std::cerr << "[" << done << "/" << total << "] "
              << "elapsed=" << std::fixed << std::setprecision(1) << elapsed_sec << "s "
              << "per_state=" << per_state << "s "
              << "eta=" << eta << "s" << std::endl;
}

// ========== CSV 出力 ==========

void DepthNMctsExperiment::writeResultsCSV(
    const std::vector<DepthNComparisonResult>& results,
    const std::string& path) const
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Error: cannot open " << path << " for writing" << std::endl;
        return;
    }
    ofs << "game_id,end,shot_num,num_candidates,num_clusters,"
        << "proposed_idx,proposed_mean,proposed_label,proposed_playouts,proposed_time_sec,"
        << "allgrid_idx,allgrid_mean,allgrid_label,allgrid_playouts,allgrid_time_sec,"
        << "exact_match,same_cluster,same_type,score_diff\n";

    for (auto& r : results) {
        ofs << r.game_id << "," << r.end << "," << r.shot_num << ","
            << r.num_candidates << "," << r.num_clusters << ","
            << r.proposed_best_idx << "," << r.proposed_best_mean << ","
            << "\"" << r.proposed_label << "\"," << r.proposed_actual_playouts << ","
            << r.proposed_time_sec << ","
            << r.allgrid_best_idx << "," << r.allgrid_best_mean << ","
            << "\"" << r.allgrid_label << "\"," << r.allgrid_actual_playouts << ","
            << r.allgrid_time_sec << ","
            << (r.exact_match ? 1 : 0) << ","
            << (r.same_cluster ? 1 : 0) << ","
            << (r.same_type ? 1 : 0) << ","
            << r.score_diff << "\n";
    }
    std::cout << "  [csv] wrote " << results.size() << " records to " << path << std::endl;
}

// ========== 実験本体 ==========

void DepthNMctsExperiment::run() {
    using clock = std::chrono::steady_clock;

    std::cout << "\n=== Depth-" << config_.depth << " MCTS Experiment ===" << std::endl;
    std::cout << "  n_states                = " << config_.n_states << std::endl;
    std::cout << "  proposed_playouts       = " << config_.proposed_playouts << std::endl;
    std::cout << "  allgrid_playouts        = " << config_.allgrid_playouts << std::endl;
    std::cout << "  proposed_rollouts/visit = " << config_.proposed_rollouts_per_visit << std::endl;
    std::cout << "  allgrid_rollouts/visit  = " << config_.allgrid_rollouts_per_visit << std::endl;
    std::cout << "  retention_rate          = " << config_.retention_rate << std::endl;
    std::cout << "  ucb_c                   = " << config_.ucb_c << std::endl;
    std::cout << "  epsilon                 = " << config_.epsilon << std::endl;
    std::cout << "  num_threads             = " << config_.num_threads << std::endl;
    std::cout << "  seed                    = " << config_.seed << std::endl;
    std::cout << "  start_index             = " << config_.start_index << std::endl;
    std::cout << "  max_positions           = " << config_.max_positions << std::endl;
    std::cout << "  load_positions_dir      = " << config_.load_positions_dir << std::endl;
    std::cout << "  output_dir              = " << config_.output_dir << std::endl;

    // 1. 盤面ロード
    auto all_records = mcts_shared::loadTestPositionsFromCSV(
        config_.load_positions_dir, game_setting_, -1);
    if (all_records.empty()) {
        std::cerr << "Error: no positions loaded from " << config_.load_positions_dir << std::endl;
        return;
    }

    // 2. サンプリング
    auto sampled = mcts_shared::sampleTestPositions(
        all_records, config_.n_states, config_.seed);
    std::cout << "  Sampled " << sampled.size() << " / " << all_records.size()
              << " positions (seed=" << config_.seed << ")" << std::endl;

    // 2b. start_index / max_positions でスライス（並列実行用）
    //     state_seed 計算用に「サンプリング後の元index」を保持
    std::vector<int> global_indices(sampled.size());
    std::iota(global_indices.begin(), global_indices.end(), 0);
    if (config_.start_index > 0 || config_.max_positions > 0) {
        int total = static_cast<int>(sampled.size());
        int s = std::min(std::max(0, config_.start_index), total);
        int e = (config_.max_positions < 0)
                ? total
                : std::min(total, s + config_.max_positions);
        sampled = std::vector<mcts_shared::TestPositionRecord>(
            sampled.begin() + s, sampled.begin() + e);
        global_indices = std::vector<int>(
            global_indices.begin() + s, global_indices.begin() + e);
        std::cout << "  Sliced to [" << s << ", " << e << ") = "
                  << sampled.size() << " positions" << std::endl;
    }

    // 3. 出力ディレクトリ
    std::filesystem::create_directories(config_.output_dir);

    // 4. スレッド起動
    int N = static_cast<int>(sampled.size());
    std::vector<DepthNComparisonResult> results(N);
    std::atomic<int> next_idx{0};
    std::atomic<int> done_count{0};
    std::mutex log_mutex;
    auto start = clock::now();

    // ロールアウトで使うグリッド（全スレッド共通で構築してコピー）
    auto shared_grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    auto worker = [&](int thread_id) {
        SimulatorWrapper sim(dc::Team::k0, game_setting_);
        ShotGenerator gen(game_setting_);
        // initialShotData をスレッドごとに populate (depth1 と同じ)
        sim.initialShotData.reserve(shared_grid.size());
        for (auto& pos : shared_grid) {
            sim.initialShotData.push_back(sim.FindShot(pos));
        }

        while (true) {
            int idx = next_idx.fetch_add(1);
            if (idx >= N) break;

            const auto& rec = sampled[idx];
            // state_seed はスライス前のグローバルindexで決定
            // → プロセスを跨いでも同じ盤面なら同じ乱数で再現可能、シード衝突なし
            uint64_t state_seed = config_.seed ^ (static_cast<uint64_t>(global_indices[idx]) * 0x9E3779B97F4A7C15ULL);

            // 盤面開始ログ（実行中盤面の可視化）
            auto state_start = clock::now();
            {
                std::lock_guard<std::mutex> lk(log_mutex);
                auto elapsed = std::chrono::duration<double>(state_start - start).count();
                std::cerr << "[start " << (idx + 1) << "/" << N << "] "
                          << "thread=" << thread_id
                          << " global_idx=" << global_indices[idx]
                          << " (g=" << rec.game_id << ",e=" << rec.end << ",s=" << rec.shot_num << ") "
                          << "elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s"
                          << std::endl;
            }

            try {
                results[idx] = runOneState(rec, sim, gen, state_seed);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(log_mutex);
                std::cerr << "[thread " << thread_id << "] exception at idx=" << idx
                          << " (g=" << rec.game_id << ",e=" << rec.end << ",s=" << rec.shot_num
                          << "): " << e.what() << std::endl;
                // 最低限game_id等を記録
                results[idx].game_id = rec.game_id;
                results[idx].end = rec.end;
                results[idx].shot_num = rec.shot_num;
            }

            int d = ++done_count;
            {
                std::lock_guard<std::mutex> lk(log_mutex);
                auto now = clock::now();
                auto state_dur = std::chrono::duration<double>(now - state_start).count();
                auto elapsed = std::chrono::duration<double>(now - start).count();
                const auto& r = results[idx];
                std::cerr << "[done " << d << "/" << N << "] "
                          << "thread=" << thread_id
                          << " global_idx=" << global_indices[idx]
                          << " (g=" << r.game_id << ",e=" << r.end << ",s=" << r.shot_num << ") "
                          << "state_time=" << std::fixed << std::setprecision(1) << state_dur << "s"
                          << " p_time=" << r.proposed_time_sec << "s"
                          << " a_time=" << r.allgrid_time_sec << "s"
                          << " exact=" << (r.exact_match ? 1 : 0)
                          << " same_cluster=" << (r.same_cluster ? 1 : 0)
                          << " score_diff=" << std::setprecision(3) << r.score_diff
                          << std::endl;
                logProgress(d, N, elapsed);
            }
        }
    };

    std::vector<std::thread> threads;
    int nt = std::max(1, config_.num_threads);
    for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    auto total_elapsed = std::chrono::duration<double>(clock::now() - start).count();
    std::cout << "\n=== All " << N << " states done in "
              << std::fixed << std::setprecision(1) << total_elapsed << "s ===" << std::endl;

    // 5. CSV 書き出し（並列実行時は start_index でファイル名分離）
    std::string suffix = (config_.start_index > 0 || config_.max_positions > 0)
        ? "_idx" + std::to_string(config_.start_index)
        : "";
    std::string csv_path = config_.output_dir + "/depth" + std::to_string(config_.depth)
                         + "_results" + suffix + ".csv";
    writeResultsCSV(results, csv_path);

    // 6. サマリ出力
    int n_valid = 0, n_exact = 0, n_cluster = 0, n_type = 0;
    double sum_score_diff = 0.0;
    double sum_proposed_time = 0.0, sum_allgrid_time = 0.0;
    for (auto& r : results) {
        if (r.proposed_best_idx < 0 || r.allgrid_best_idx < 0) continue;
        n_valid++;
        if (r.exact_match) n_exact++;
        if (r.same_cluster) n_cluster++;
        if (r.same_type) n_type++;
        sum_score_diff += r.score_diff;
        sum_proposed_time += r.proposed_time_sec;
        sum_allgrid_time += r.allgrid_time_sec;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "  valid cases          = " << n_valid << " / " << N << std::endl;
    if (n_valid > 0) {
        auto pct = [&](int x) { return 100.0 * x / n_valid; };
        std::cout << "  exact_match          = " << n_exact << " (" << pct(n_exact) << "%)" << std::endl;
        std::cout << "  same_cluster         = " << n_cluster << " (" << pct(n_cluster) << "%)" << std::endl;
        std::cout << "  same_type            = " << n_type << " (" << pct(n_type) << "%)" << std::endl;
        std::cout << "  avg_score_diff       = " << sum_score_diff / n_valid << std::endl;
        std::cout << "  avg_proposed_time    = " << sum_proposed_time / n_valid << "s" << std::endl;
        std::cout << "  avg_allgrid_time     = " << sum_allgrid_time / n_valid << "s" << std::endl;
    }

    // サマリを別ファイルにも保存
    std::string summary_path = config_.output_dir + "/depth" + std::to_string(config_.depth)
                             + "_summary" + suffix + ".txt";
    std::ofstream sfs(summary_path);
    if (sfs) {
        sfs << "Depth-" << config_.depth << " MCTS Experiment Summary\n";
        sfs << "========================================\n";
        sfs << "n_states                = " << config_.n_states << "\n";
        sfs << "proposed_playouts       = " << config_.proposed_playouts << "\n";
        sfs << "allgrid_playouts        = " << config_.allgrid_playouts << "\n";
        sfs << "proposed_rollouts/visit = " << config_.proposed_rollouts_per_visit << "\n";
        sfs << "allgrid_rollouts/visit  = " << config_.allgrid_rollouts_per_visit << "\n";
        sfs << "retention_rate          = " << config_.retention_rate << "\n";
        sfs << "seed                    = " << config_.seed << "\n";
        sfs << "num_threads             = " << config_.num_threads << "\n";
        sfs << "start_index             = " << config_.start_index << "\n";
        sfs << "max_positions           = " << config_.max_positions << "\n";
        sfs << "total_elapsed_sec       = " << total_elapsed << "\n";
        sfs << "valid_cases             = " << n_valid << "\n";
        if (n_valid > 0) {
            auto pct = [&](int x) { return 100.0 * x / n_valid; };
            sfs << "exact_match_pct         = " << pct(n_exact) << "\n";
            sfs << "same_cluster_pct        = " << pct(n_cluster) << "\n";
            sfs << "same_type_pct           = " << pct(n_type) << "\n";
            sfs << "avg_score_diff          = " << sum_score_diff / n_valid << "\n";
            sfs << "avg_proposed_time_s     = " << sum_proposed_time / n_valid << "\n";
            sfs << "avg_allgrid_time_s      = " << sum_allgrid_time / n_valid << "\n";
        }
    }
}
