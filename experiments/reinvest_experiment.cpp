#include "reinvest_experiment.h"
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
#include <set>
#include <sstream>
#include <thread>

ReinvestExperiment::ReinvestExperiment(
    const dc::GameSetting& game_setting,
    const ReinvestConfig& config)
    : game_setting_(game_setting), config_(config)
{
}

namespace {
// ラベル "Draw(CW,5)" -> "Draw"。shotTypeToString は private なのでラベル接頭辞で代用。
// モード定義キー (shot_type) として cluster_table に出力する。
std::string labelToType(const std::string& label) {
    auto paren = label.find('(');
    return (paren == std::string::npos) ? label : label.substr(0, paren);
}
}  // namespace

std::string ReinvestExperiment::methodName(MctsMode m) {
    switch (m) {
        case MctsMode::AllGrid:  return "AllGrid";
        case MctsMode::Proposed: return "Proposed";
        case MctsMode::RandomK:  return "RandomK";
    }
    return "Proposed";
}

MctsMode parseMctsMode(const std::string& s) {
    if (s == "AllGrid" || s == "allgrid")   return MctsMode::AllGrid;
    if (s == "RandomK" || s == "randomk")   return MctsMode::RandomK;
    return MctsMode::Proposed;  // 既定 (Proposed / proposed / 未知)
}

// ========== 盤面ハッシュ ==========
// 展開キャッシュキー + RandomK の決定的乱択シード。depth_n と同一実装。
uint64_t ReinvestExperiment::hashGameState(const dc::GameState& s) const {
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

void ReinvestExperiment::expandNode(
    TreeNode& node,
    ShotGenerator& gen,
    std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
    uint64_t state_seed)
{
    if (node.expanded) return;

    // 候補手とシミュ結果をキャッシュから引く (盤面ハッシュ単位で再利用)
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

    if (config_.mode == MctsMode::Proposed) {
        // 距離行列 → 階層的クラスタリング → メドイド (歩相当の Proposed)
        auto dist_table = mcts_shared::makeDistanceTableDelta(node.state, node.result_states);
        int K = std::max(1, static_cast<int>(std::ceil(N * config_.retention_rate)));
        K = std::min(K, N);
        node.clusters = mcts_shared::runClustering(dist_table, K);
        node.medoid_indices = mcts_shared::calculateMedoids(dist_table, node.clusters);
        node.medoid_indices.erase(
            std::remove_if(node.medoid_indices.begin(), node.medoid_indices.end(),
                           [](int m) { return m < 0; }),
            node.medoid_indices.end());
    } else if (config_.mode == MctsMode::RandomK) {
        // 決定的乱択: 同じ retention_rate で K 個を「ランダムに」選ぶ。
        // クラスタリングと同じ削減率だが賢さ無し → A5 でクラスタリングの寄与を単離。
        // シードは (state_seed, 盤面ハッシュ) 由来でノード単位に決定的 (playout 順序非依存・再現可能)。
        int K = std::max(1, static_cast<int>(std::ceil(N * config_.retention_rate)));
        K = std::min(K, N);
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 sel_rng(static_cast<uint32_t>((state_seed ^ key) ^ ((state_seed ^ key) >> 32)));
        std::shuffle(idx.begin(), idx.end(), sel_rng);
        idx.resize(K);
        std::sort(idx.begin(), idx.end());  // 安定した子順序 (candidate_idx 昇順)
        node.medoid_indices = idx;
    } else {
        // AllGrid: 全候補を子ノードに (クラスタリングなし)
        node.medoid_indices.resize(N);
        std::iota(node.medoid_indices.begin(), node.medoid_indices.end(), 0);
    }

    node.children.resize(node.medoid_indices.size());
    node.expanded = true;
}

// ========== UCB1 選択 ==========

int ReinvestExperiment::selectBestChildUCB(const TreeNode& node) const {
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

int ReinvestExperiment::selectMostVisited(const TreeNode& node) const {
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

double ReinvestExperiment::runPlayout(
    TreeNode& node,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
    std::mt19937& rng,
    dc::Team root_team,
    uint64_t state_seed)
{
    int R = std::max(1, config_.rollouts_per_visit);

    // 葉に到達 → ロールアウト (R 回平均)。方策は全アーム共通の ε-greedy グリッド。
    if (node.depth >= config_.depth) {
        int remaining = 16 - static_cast<int>(node.state.shot);
        double sum = 0.0;
        for (int i = 0; i < R; i++) {
            sum += mcts_shared::rolloutFromState(
                sim, node.state, remaining, root_team, rng, config_.epsilon);
        }
        double mean_reward = sum / R;
        node.visits++;
        node.total_reward += mean_reward;
        return mean_reward;
    }

    if (!node.expanded) {
        expandNode(node, gen, cache, state_seed);
    }
    int K = static_cast<int>(node.medoid_indices.size());
    if (K == 0) {
        // 候補なし異常系: ロールアウトのみ
        int remaining = 16 - static_cast<int>(node.state.shot);
        double sum = 0.0;
        for (int i = 0; i < R; i++) {
            sum += mcts_shared::rolloutFromState(
                sim, node.state, remaining, root_team, rng, config_.epsilon);
        }
        double mean_reward = sum / R;
        node.visits++;
        node.total_reward += mean_reward;
        return mean_reward;
    }

    int idx = selectBestChildUCB(node);
    if (idx < 0) idx = 0;

    if (!node.children[idx]) {
        auto child = std::make_unique<TreeNode>();
        child->state = node.result_states[node.medoid_indices[idx]];
        child->depth = node.depth + 1;
        child->to_play = (child->state.shot % 2 == 0) ? dc::Team::k0 : dc::Team::k1;
        node.children[idx] = std::move(child);
    }

    double reward = runPlayout(*node.children[idx], sim, gen, cache, rng, root_team, state_seed);

    node.visits++;
    node.total_reward += reward;
    return reward;
}

// ========== 木構築 ==========

void ReinvestExperiment::buildTree(
    TreeNode& root,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
    std::mt19937& rng,
    dc::Team root_team,
    uint64_t state_seed)
{
    if (!root.expanded) {
        expandNode(root, gen, cache, state_seed);
    }
    for (int p = 0; p < config_.playouts; p++) {
        runPlayout(root, sim, gen, cache, rng, root_team, state_seed);
    }
}

// ========== 1 局面実行 ==========

ReinvestResult ReinvestExperiment::runOneState(
    const mcts_shared::TestPositionRecord& rec,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    uint64_t state_seed)
{
    using clock = std::chrono::steady_clock;

    ReinvestResult r;
    r.game_id = rec.game_id;
    r.end = rec.end;
    r.shot_num = rec.shot_num;

    dc::Team root_team = rec.current_team;

    std::unordered_map<uint64_t, CandidateCacheEntry> cache;

    TreeNode root;
    root.state = rec.state;
    root.depth = 0;
    root.to_play = root_team;

    // 等予算カウント: この局面が消費した実物理シミュ回数 (展開 + ロールアウト)。
    // g_physics_sim_count は thread_local なので、このワーカースレッド内の差分 = この局面分のみ。
    long long sims_before = g_physics_sim_count;

    std::mt19937 rng(static_cast<uint32_t>(state_seed ^ (state_seed >> 32)));
    auto t0 = clock::now();
    buildTree(root, sim, gen, cache, rng, root_team, state_seed);
    auto t1 = clock::now();

    r.time_sec = std::chrono::duration<double>(t1 - t0).count();
    r.actual_total_sims = g_physics_sim_count - sims_before;

    r.num_candidates = static_cast<int>(root.candidates.size());
    r.num_children = static_cast<int>(root.medoid_indices.size());
    r.actual_playouts = root.visits;

    int best_child = selectMostVisited(root);
    if (best_child >= 0) {
        int cand_idx = root.medoid_indices[best_child];
        r.best_idx = cand_idx;  // generatePool 順 index = 審判 Q テーブルとの join キー
        r.best_mean = root.children[best_child]->mean();
        r.label = root.candidates[cand_idx].label;
    }

    // ===== モード分離実験用: root のクラスタ割当をエクスポート =====
    // Proposed: 全候補をクラスタ + 代表点フラグつきで記録 (分離/被覆/collapse 診断の権威マップ)。
    // RandomK : クラスタ概念なし。選んだ K 個を代表として記録 (cluster_id = 選択順)。
    // AllGrid : 全候補が子なので分離対象外 → 空のまま (正解集合 A の供給側)。
    if (config_.mode == MctsMode::Proposed && !root.clusters.empty()) {
        std::set<int> rep_set(root.medoid_indices.begin(), root.medoid_indices.end());
        for (int cid = 0; cid < static_cast<int>(root.clusters.size()); cid++) {
            for (int cand_idx : root.clusters[cid]) {
                if (cand_idx < 0 || cand_idx >= static_cast<int>(root.candidates.size())) continue;
                ClusterAssign ca;
                ca.candidate_idx = cand_idx;
                ca.cluster_id = cid;
                ca.is_representative = (rep_set.count(cand_idx) > 0);
                ca.label = root.candidates[cand_idx].label;
                ca.shot_type = labelToType(ca.label);
                r.cluster_table.push_back(ca);
            }
        }
    } else if (config_.mode == MctsMode::RandomK) {
        for (int j = 0; j < static_cast<int>(root.medoid_indices.size()); j++) {
            int cand_idx = root.medoid_indices[j];
            if (cand_idx < 0 || cand_idx >= static_cast<int>(root.candidates.size())) continue;
            ClusterAssign ca;
            ca.candidate_idx = cand_idx;
            ca.cluster_id = j;
            ca.is_representative = true;
            ca.label = root.candidates[cand_idx].label;
            ca.shot_type = labelToType(ca.label);
            r.cluster_table.push_back(ca);
        }
    }

    return r;
}

// ========== CSV 出力 (§4 スキーマ厳守) ==========

void ReinvestExperiment::writeResultsCSV(
    const std::vector<ReinvestResult>& results,
    const std::string& path) const
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Error: cannot open " << path << " for writing" << std::endl;
        return;
    }
    ofs << "game_id,end,shot_num,method,depth,playouts,rollouts_per_visit,seed,"
        << "candidate_idx,label,actual_total_sims,time_sec\n";
    ofs << std::setprecision(6);

    std::string method = methodName(config_.mode);
    for (const auto& r : results) {
        if (r.game_id < 0) continue;  // 例外でスキップされた局面
        ofs << r.game_id << "," << r.end << "," << r.shot_num << ","
            << method << "," << config_.depth << "," << config_.playouts << ","
            << config_.rollouts_per_visit << "," << config_.seed << ","
            << r.best_idx << ",\"" << r.label << "\","
            << r.actual_total_sims << "," << r.time_sec << "\n";
    }
    std::cout << "  [csv] wrote " << results.size() << " records to " << path << std::endl;
}

// ========== クラスタ割当 CSV 出力 (モード分離実験用) ==========
// 1 候補 1 行。candidate_idx で reinvest_results (AllGrid 選択) / 審判 Q テーブルと join 可能。
void ReinvestExperiment::writeClusterTableCSV(
    const std::vector<ReinvestResult>& results,
    const std::string& path) const
{
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Error: cannot open " << path << " for writing" << std::endl;
        return;
    }
    ofs << "game_id,end,shot_num,method,seed,candidate_idx,cluster_id,"
        << "is_representative,shot_type,label\n";

    std::string method = methodName(config_.mode);
    long long rows = 0;
    for (const auto& r : results) {
        if (r.game_id < 0) continue;
        for (const auto& ca : r.cluster_table) {
            ofs << r.game_id << "," << r.end << "," << r.shot_num << ","
                << method << "," << config_.seed << ","
                << ca.candidate_idx << "," << ca.cluster_id << ","
                << (ca.is_representative ? 1 : 0) << ","
                << "\"" << ca.shot_type << "\",\"" << ca.label << "\"\n";
            rows++;
        }
    }
    std::cout << "  [csv] wrote " << rows << " cluster-assign rows to " << path << std::endl;
}

// ========== 実験本体 ==========

void ReinvestExperiment::run() {
    using clock = std::chrono::steady_clock;

    std::string method = methodName(config_.mode);
    std::cout << "\n=== Reinvestment Arm: "
              << (config_.arm_label.empty() ? method : config_.arm_label)
              << " (" << method << ", depth " << config_.depth << ") ===" << std::endl;
    std::cout << "  method             = " << method << std::endl;
    std::cout << "  depth              = " << config_.depth << std::endl;
    std::cout << "  playouts (P)       = " << config_.playouts << std::endl;
    std::cout << "  rollouts/visit (R) = " << config_.rollouts_per_visit << std::endl;
    std::cout << "  retention_rate     = " << config_.retention_rate << std::endl;
    std::cout << "  ucb_c              = " << config_.ucb_c << std::endl;
    std::cout << "  epsilon            = " << config_.epsilon << std::endl;
    std::cout << "  n_states           = " << config_.n_states << std::endl;
    std::cout << "  num_threads        = " << config_.num_threads << std::endl;
    std::cout << "  seed               = " << config_.seed << std::endl;
    std::cout << "  start_index        = " << config_.start_index << std::endl;
    std::cout << "  max_positions      = " << config_.max_positions << std::endl;
    std::cout << "  load_positions_dir = " << config_.load_positions_dir << std::endl;
    std::cout << "  output_dir         = " << config_.output_dir << std::endl;

    // 1. 局面ロード
    auto all_records = mcts_shared::loadTestPositionsFromCSV(
        config_.load_positions_dir, game_setting_, -1);
    if (all_records.empty()) {
        std::cerr << "Error: no positions loaded from " << config_.load_positions_dir << std::endl;
        return;
    }

    // 2. サンプリング (depth_n / 審判と同一規則。行数==n_states なら全件固定)
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

    std::filesystem::create_directories(config_.output_dir);

    int N = static_cast<int>(sampled.size());
    if (N == 0) {
        std::cerr << "Error: no positions after slicing" << std::endl;
        return;
    }
    std::vector<ReinvestResult> results(N);
    std::atomic<int> next_idx{0};
    std::atomic<int> done_count{0};
    std::mutex log_mutex;
    auto start = clock::now();

    // ロールアウト用 4x4 グリッド (depth_n / 審判と同一)
    auto shared_grid = PoolExperiment(game_setting_).makeGrid(4, 4);

    auto worker = [&](int thread_id) {
        SimulatorWrapper sim(dc::Team::k0, game_setting_);
        ShotGenerator gen(game_setting_);
        sim.initialShotData.reserve(shared_grid.size());
        for (auto& pos : shared_grid) {
            sim.initialShotData.push_back(sim.FindShot(pos));
        }

        while (true) {
            int idx = next_idx.fetch_add(1);
            if (idx >= N) break;

            const auto& rec = sampled[idx];
            // state_seed はスライス前グローバル index で決定 (プロセス跨ぎでも同局面=同乱数)
            uint64_t state_seed = config_.seed
                ^ (static_cast<uint64_t>(global_indices[idx]) * 0x9E3779B97F4A7C15ULL);

            auto state_start = clock::now();
            {
                std::lock_guard<std::mutex> lk(log_mutex);
                auto elapsed = std::chrono::duration<double>(state_start - start).count();
                std::cerr << "[start " << (idx + 1) << "/" << N << "] thread=" << thread_id
                          << " global_idx=" << global_indices[idx]
                          << " (g=" << rec.game_id << ",e=" << rec.end << ",s=" << rec.shot_num << ") "
                          << "elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s" << std::endl;
            }

            try {
                results[idx] = runOneState(rec, sim, gen, state_seed);
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(log_mutex);
                std::cerr << "[thread " << thread_id << "] exception at idx=" << idx
                          << " (g=" << rec.game_id << ",e=" << rec.end << ",s=" << rec.shot_num
                          << "): " << e.what() << std::endl;
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
                std::cerr << "[done " << d << "/" << N << "] thread=" << thread_id
                          << " global_idx=" << global_indices[idx]
                          << " (g=" << r.game_id << ",e=" << r.end << ",s=" << r.shot_num << ") "
                          << "state_time=" << std::fixed << std::setprecision(1) << state_dur << "s"
                          << " best_idx=" << r.best_idx
                          << " sims=" << r.actual_total_sims
                          << " N=" << r.num_candidates << " K=" << r.num_children;
                if (d > 0) {
                    double per = elapsed / d;
                    std::cerr << " eta=" << per * (N - d) << "s";
                }
                std::cerr << std::endl;
            }
        }
    };

    std::vector<std::thread> threads;
    int nt = std::max(1, config_.num_threads);
    for (int t = 0; t < nt; t++) threads.emplace_back(worker, t);
    for (auto& th : threads) th.join();

    auto total_elapsed = std::chrono::duration<double>(clock::now() - start).count();
    std::cout << "\n=== Arm done: " << N << " states in "
              << std::fixed << std::setprecision(1) << total_elapsed << "s ===" << std::endl;

    // 5. CSV 出力 (並列スライス時は _idx{start} で分離)
    std::string suffix = (config_.start_index > 0 || config_.max_positions > 0)
        ? "_idx" + std::to_string(config_.start_index) : "";
    std::string csv_path = config_.output_dir + "/reinvest_results" + suffix + ".csv";
    writeResultsCSV(results, csv_path);

    // モード分離実験用: クラスタ割当テーブル (Proposed/RandomK のみ中身あり)
    if (config_.mode == MctsMode::Proposed || config_.mode == MctsMode::RandomK) {
        std::string ct_path = config_.output_dir + "/cluster_table" + suffix + ".csv";
        writeClusterTableCSV(results, ct_path);
    }

    // 6. サマリ (実シミュ予算の揃い確認用)
    long long sum_sims = 0, min_sims = -1, max_sims = -1;
    double sum_time = 0.0;
    int n_valid = 0;
    for (const auto& r : results) {
        if (r.game_id < 0 || r.best_idx < 0) continue;
        n_valid++;
        sum_sims += r.actual_total_sims;
        sum_time += r.time_sec;
        if (min_sims < 0 || r.actual_total_sims < min_sims) min_sims = r.actual_total_sims;
        if (max_sims < 0 || r.actual_total_sims > max_sims) max_sims = r.actual_total_sims;
    }
    std::cout << "\n=== Summary (" << (config_.arm_label.empty() ? method : config_.arm_label) << ") ===" << std::endl;
    std::cout << "  valid cases        = " << n_valid << " / " << N << std::endl;
    if (n_valid > 0) {
        std::cout << "  avg actual_sims    = " << (sum_sims / n_valid)
                  << "  (min=" << min_sims << ", max=" << max_sims << ")" << std::endl;
        std::cout << "  avg time           = " << std::setprecision(2) << (sum_time / n_valid) << "s" << std::endl;
    }
}
