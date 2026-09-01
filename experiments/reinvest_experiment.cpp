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
        case MctsMode::AllGrid:     return "AllGrid";
        case MctsMode::Proposed:    return "Proposed";
        case MctsMode::RandomK:     return "RandomK";
        case MctsMode::ScoreScreen: return "ScoreScreen";
        case MctsMode::ScoreTopK:   return "ScoreTopK";
        case MctsMode::ClusterValue: return "ClusterValue";
        case MctsMode::ClusterValueDeep: return "ClusterValueDeep";
        case MctsMode::ClusterPW:   return "ClusterPW";
        case MctsMode::ClusterTS:   return "ClusterTS";
    }
    return "Proposed";
}

MctsMode parseMctsMode(const std::string& s) {
    if (s == "AllGrid" || s == "allgrid")         return MctsMode::AllGrid;
    if (s == "ClusterPW" || s == "clusterpw")     return MctsMode::ClusterPW;
    if (s == "ClusterTS" || s == "clusterts")     return MctsMode::ClusterTS;
    if (s == "RandomK" || s == "randomk")         return MctsMode::RandomK;
    if (s == "ScoreScreen" || s == "scorescreen") return MctsMode::ScoreScreen;
    if (s == "ScoreTopK" || s == "scoretopk")     return MctsMode::ScoreTopK;
    if (s == "ClusterValueDeep" || s == "clustervaluedeep") return MctsMode::ClusterValueDeep;
    if (s == "ClusterValue" || s == "clustervalue") return MctsMode::ClusterValue;
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
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
    std::mt19937& rng,
    dc::Team root_team,
    uint64_t state_seed)
{
    if (node.expanded) return;

    // 終局ノードは展開しない。generatePool が終了済み状態に着手すると
    // ApplyMove が "game is already over" を投げるため (2エンド目終盤で発生)。
    // 子なし(N=0)として扱い、runPlayout 側で終局値を評価させる。
    if (node.state.IsGameOver()) {
        node.expanded = true;
        return;
    }

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

    if ((config_.mode == MctsMode::ScoreScreen || config_.mode == MctsMode::ScoreTopK)
        && node.depth == 0) {
        // root: 得点スクリーン (A7: ①R_pre推定→②ε帯→③リスク多様性 / A8 TopK: ①のみ→E[score]上位K)
        node.medoid_indices = selectScoreScreen(node, sim, gen, rng, root_team);
    } else if (config_.mode == MctsMode::ClusterTS && node.depth == 0) {
        // A12: 階層ベイズ縮約 + Thompson。全クラスタ代表を子に (訪問配分は selectThompson が行う)
        node.medoid_indices = selectClusterTS(node, sim, gen, rng, root_team);
    } else if (((config_.mode == MctsMode::ClusterValue || config_.mode == MctsMode::ClusterPW) && node.depth == 0)
               || config_.mode == MctsMode::ClusterValueDeep) {
        // A9: root のみ / A10: 全深さ = distDeltaクラスタ + クラスタ平均E[score]価値付け
        //  → 価値上位クラスタの最良メンバー (A10 は手番視点で選択・相手番ノードは子数 cv_k_opp)
        // A11 (ClusterPW): 同じ順位付けで先頭 pw_k0 個だけ開き、残りは pw_queue に積む (runPlayout で widening)
        node.medoid_indices = selectClusterValue(node, sim, gen, rng, root_team);
    } else if (config_.mode == MctsMode::Proposed
               || config_.mode == MctsMode::ScoreScreen
               || config_.mode == MctsMode::ScoreTopK
               || config_.mode == MctsMode::ClusterValue
               || config_.mode == MctsMode::ClusterPW
               || config_.mode == MctsMode::ClusterTS) {
        // distDelta クラスタリング (Proposed 全ノード / ScoreScreen・ScoreTopK・ClusterValue の depth>0 ノード)
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

// ========== ScoreScreen: 得点スクリーン型の root 候補選別 (A7) ==========
// ① 各候補を R_pre 回ロールアウトして E[score]/SD を安価推定
// ② ε帯: 最良 E[score] から Δ 以内の「有望集合」に絞る (ジャンク除去)
// ③ |有望集合| > K_cap のときだけ、SD で 低/中/高 の3帯に分けて K_cap を配分
//    (「安全な手」と「博打の手」を両方残す = リスク多様性保持)。再利用なし。
void ReinvestExperiment::estimateRootScores(
    TreeNode& node,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::mt19937& rng,
    dc::Team root_team)
{
    // ① 安価な E[score]/SD 推定 (result_states[c] = simulateNoRand の決定的着地から継続)
    // R_pre は深さで逓減 (root:R_pre → 深さd: R_pre-d, 最低1)。深いノードほど数が多く
    // 残り手数も短いため、逓減でスクリーニング総コストを等予算圏に収める (A10)。
    int N = static_cast<int>(node.candidates.size());
    int R_pre = std::max(1, config_.score_screen_r_pre - node.depth);
    int cur_end = static_cast<int>(node.state.end);
    node.e_pre.assign(N, 0.0);
    node.sd_pre.assign(N, 0.0);
    for (int c = 0; c < N; c++) {
        const dc::GameState& rs = node.result_states[c];
        // 候補着手で既にこのエンドが終了 → 実エンド得点を使う (次エンドをロールアウトしない; 審判と同じ規約)
        bool end_done = (static_cast<int>(rs.end) != cur_end) || rs.IsGameOver();
        if (end_done) {
            double diff = 0.0;
            if (cur_end >= 0 && cur_end < static_cast<int>(rs.scores[0].size())) {
                int t0 = rs.scores[0][cur_end] ? static_cast<int>(*rs.scores[0][cur_end]) : 0;
                int t1 = rs.scores[1][cur_end] ? static_cast<int>(*rs.scores[1][cur_end]) : 0;
                diff = static_cast<double>(t0 - t1);
            }
            node.e_pre[c] = (root_team == dc::Team::k0) ? diff : -diff;
            node.sd_pre[c] = 0.0;  // 決定的着地でエンド確定 → 分散なし
            continue;
        }
        int remaining = 16 - static_cast<int>(rs.shot);
        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < R_pre; i++) {
            double v;
            if (config_.noisy_tree) {
                // 候補手自身も毎回外乱ありで打ち直す (審判の resample_first_shot と同じ規約)。
                // SD は実行リスク込みの分散になる (A7のリスク帯の意味も改善)
                dc::GameState start = sim.run_single_simulation(node.state, node.candidates[c].shot);
                if (static_cast<int>(start.end) != cur_end || start.IsGameOver()) {
                    double diff = 0.0;
                    if (cur_end >= 0 && cur_end < static_cast<int>(start.scores[0].size())) {
                        int t0 = start.scores[0][cur_end] ? static_cast<int>(*start.scores[0][cur_end]) : 0;
                        int t1 = start.scores[1][cur_end] ? static_cast<int>(*start.scores[1][cur_end]) : 0;
                        diff = static_cast<double>(t0 - t1);
                    }
                    v = (root_team == dc::Team::k0) ? diff : -diff;
                } else {
                    v = mcts_shared::rolloutFromState(
                        sim, gen, start, 16 - static_cast<int>(start.shot),
                        root_team, rng, config_.epsilon);
                }
            } else {
                v = mcts_shared::rolloutFromState(
                    sim, gen, rs, remaining, root_team, rng, config_.epsilon);
            }
            sum += v; sumsq += v * v;
        }
        double m = sum / R_pre;
        node.e_pre[c] = m;
        node.sd_pre[c] = std::sqrt(std::max(0.0, sumsq / R_pre - m * m));
    }
}

std::vector<int> ReinvestExperiment::selectScoreScreen(
    TreeNode& node,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::mt19937& rng,
    dc::Team root_team)
{
    int N = static_cast<int>(node.candidates.size());
    std::vector<int> selected;
    if (N == 0) return selected;

    estimateRootScores(node, sim, gen, rng, root_team);
    const std::vector<double>& e_pre = node.e_pre;
    const std::vector<double>& sd_pre = node.sd_pre;

    // K_cap: 予算連動 (子1個に最低 v_target 訪問させたい)
    int v_target = std::max(1, config_.score_screen_v_target);
    int K_cap = std::max(1, config_.playouts / v_target);

    auto by_escore_desc = [&](int a, int b) { return e_pre[a] > e_pre[b]; };

    // A8 (ScoreTopK): ε帯もリスク多様性も無し。E[score] 降順で上位 K_cap をそのまま採る。
    if (config_.mode == MctsMode::ScoreTopK) {
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), by_escore_desc);
        order.resize(std::min(N, K_cap));
        return order;
    }

    // ② ε帯 (有望集合)
    double e_star = *std::max_element(e_pre.begin(), e_pre.end());
    std::vector<int> promising;
    for (int c = 0; c < N; c++) {
        if (e_pre[c] >= e_star - config_.score_screen_band) promising.push_back(c);
    }

    if (static_cast<int>(promising.size()) <= K_cap) {
        std::sort(promising.begin(), promising.end(), by_escore_desc);
        return promising;  // 間引き不要
    }

    // ③ リスク多様性を保った間引き: SD 昇順に 3 等分 → 低/中/高 リスク帯
    std::vector<int> by_sd = promising;
    std::sort(by_sd.begin(), by_sd.end(),
              [&](int a, int b) { return sd_pre[a] < sd_pre[b]; });
    int n = static_cast<int>(by_sd.size());
    std::vector<std::vector<int>> tiers(3);
    for (int i = 0; i < n; i++) tiers[std::min(2, i * 3 / n)].push_back(by_sd[i]);
    for (auto& t : tiers) std::sort(t.begin(), t.end(), by_escore_desc);

    // 全体の E[score] 最良手を必ず確保
    std::set<int> chosen;
    int best = *std::max_element(promising.begin(), promising.end(),
                                [&](int a, int b) { return e_pre[a] < e_pre[b]; });
    selected.push_back(best);
    chosen.insert(best);

    // 各リスク帯からラウンドロビンで高 E[score] 順に採る
    std::vector<size_t> pos(3, 0);
    while (static_cast<int>(selected.size()) < K_cap) {
        bool added = false;
        for (int t = 0; t < 3 && static_cast<int>(selected.size()) < K_cap; t++) {
            while (pos[t] < tiers[t].size() && chosen.count(tiers[t][pos[t]])) pos[t]++;
            if (pos[t] < tiers[t].size()) {
                int idx = tiers[t][pos[t]++];
                selected.push_back(idx);
                chosen.insert(idx);
                added = true;
            }
        }
        if (!added) break;
    }
    return selected;
}

// ========== ClusterValue: クラスタ価値型の root 候補選別 (A9 = Proposed改) ==========
// Proposed の「結果盤面クラスタ = 外乱込みでその行動が引き起こす状態による行動表現」を
// 保持したまま、選択規則に価値を注入する:
//   ① R_pre ロールアウトで候補別 E[score] を推定 (A7 の①と同一)
//   ② distDelta で結果盤面クラスタリング (Proposed と同一; K = ceil(N*retention))
//   ③ クラスタ価値 = メンバー E[score] の平均 (エリア単位の平均化 = 候補単位ノイズの分散削減)
//   ④ 価値上位 K_cap クラスタから、各クラスタ内 E[score] 最大メンバーを子に。
//      全体最良 E[score] の候補が属するクラスタは必ず採用 (A7 の best確保と同型)。
// medoid(盤面的中心) を代表にしない点が Proposed との唯一の選択規則差。
std::vector<int> ReinvestExperiment::selectClusterValue(
    TreeNode& node,
    SimulatorWrapper& sim,
    ShotGenerator& gen,
    std::mt19937& rng,
    dc::Team root_team)
{
    int N = static_cast<int>(node.candidates.size());
    std::vector<int> selected;
    if (N == 0) return selected;

    // 手番視点の符号: e_pre/cluster_value は root_team 視点で格納するが、
    // 「良い手」の判定はこのノードで打つ側 (to_play) の視点で行う。
    // 相手番ノード (A10 の深さ1 等) では sign=-1 → 相手最良 = root視点最小 を選ぶ。
    // これを怠ると相手に悪手を打たせる楽観的な木になる。
    double sign = (node.to_play == root_team) ? 1.0 : -1.0;

    // ① 候補別 E[score]/SD
    estimateRootScores(node, sim, gen, rng, root_team);

    // ② 結果盤面クラスタ (Proposed と同一)
    auto dist_table = mcts_shared::makeDistanceTableDelta(node.state, node.result_states);
    int Kc = std::max(1, static_cast<int>(std::ceil(N * config_.retention_rate)));
    Kc = std::min(Kc, N);
    node.clusters = mcts_shared::runClustering(dist_table, Kc);

    // ③ クラスタ価値 μ_c (メンバー平均 E[score], root視点で格納) とプールSD σ_c
    //      σ_c² = (1/n) Σ_i [ sd_i² + (e_i − μ_c)² ]   (各候補 R_pre 標本の同数プール
    //      = 候補間の期待値ばらつき + 候補内の実行/継続ばらつき)
    //    リスク調整価値 (手番視点): adj_c = sign·μ_c − λ·σ_c、候補も adj_i = sign·e_i − λ·sd_i
    //    (λ = risk_lambda。λ=0 なら従来の A9 = 平均 E[score] のみ)
    const double lam = config_.risk_lambda;
    auto adj_cand = [&](int m) { return sign * node.e_pre[m] - lam * node.sd_pre[m]; };

    int C = static_cast<int>(node.clusters.size());
    node.cluster_value.assign(C, std::numeric_limits<double>::quiet_NaN());
    node.cluster_sd.assign(C, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> cluster_adj(C, -std::numeric_limits<double>::infinity());
    std::vector<int> best_member(C, -1);
    for (int cid = 0; cid < C; cid++) {
        double sum = 0.0; int cnt = 0;
        for (int m : node.clusters[cid]) {
            if (m < 0 || m >= N) continue;
            sum += node.e_pre[m]; cnt++;
            if (best_member[cid] < 0 || adj_cand(m) > adj_cand(best_member[cid]))
                best_member[cid] = m;
        }
        if (cnt == 0) continue;
        double mu = sum / cnt;
        double var = 0.0;
        for (int m : node.clusters[cid]) {
            if (m < 0 || m >= N) continue;
            double d = node.e_pre[m] - mu;
            var += node.sd_pre[m] * node.sd_pre[m] + d * d;
        }
        double sd = std::sqrt(std::max(0.0, var / cnt));
        node.cluster_value[cid] = mu;
        node.cluster_sd[cid] = sd;
        cluster_adj[cid] = sign * mu - lam * sd;
    }

    // ④ リスク調整価値上位 K クラスタ → 各クラスタの最良メンバー (順位も手番視点)
    //    自分番: K = playouts / v_target (予算連動; P=200 → 4)
    //    相手番: K = cv_k_opp (min側は広めに持ち、最善応手の取りこぼし = 楽観バイアスを防ぐ)
    int v_target = std::max(1, config_.score_screen_v_target);
    int K_cap = (sign > 0) ? std::max(1, config_.playouts / v_target)
                           : std::max(1, config_.cv_k_opp);

    std::vector<int> order;
    for (int cid = 0; cid < C; cid++) if (best_member[cid] >= 0) order.push_back(cid);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return cluster_adj[a] > cluster_adj[b]; });

    // 手番視点で全体最良 (リスク調整値) の候補が属するクラスタを先頭に (安全網: 平均が低くても最良手は残す)
    int g_best = 0;
    for (int c = 1; c < N; c++) if (adj_cand(c) > adj_cand(g_best)) g_best = c;
    int g_cid = -1;
    for (int cid = 0; cid < C; cid++) if (node.clusters[cid].count(g_best)) { g_cid = cid; break; }
    if (g_cid >= 0) {
        auto it = std::find(order.begin(), order.end(), g_cid);
        if (it != order.end()) order.erase(it);
        order.insert(order.begin(), g_cid);
        best_member[g_cid] = g_best;  // このクラスタの代表は全体最良で確定
    }

    if (config_.mode == MctsMode::ClusterPW && node.depth == 0) {
        // A11: 全クラスタの最良メンバーを価値順にキューへ。先頭 pw_k0 個だけ最初に開く
        node.pw_queue.clear();
        for (int cid : order) node.pw_queue.push_back(best_member[cid]);
        int k0 = std::min(static_cast<int>(node.pw_queue.size()), std::max(1, config_.pw_k0));
        selected.assign(node.pw_queue.begin(), node.pw_queue.begin() + k0);
        node.pw_opened = k0;
        return selected;
    }

    for (int i = 0; i < static_cast<int>(order.size()) && static_cast<int>(selected.size()) < K_cap; i++) {
        selected.push_back(best_member[order[i]]);
    }
    return selected;
}

// ========== ClusterTS: 階層ベイズ縮約 + Thompson sampling (A12) ==========
// ① R_pre で候補別 e_i, sd_i を推定 (estimateRootScores; noisy_tree なら実行ノイズ込み)
// ② distDelta で結果盤面クラスタリング (A9 と同一)
// ③ 縮約: 候補 i の真値 m_i に 事前 N(μ_c, τ_c²)、観測 e_i ~ N(m_i, sd_i²/R_pre) を置いた事後
//      m̃_i = (μ_c/τ_c² + e_i/σ_i²) / (1/τ_c² + 1/σ_i²),  ṽ_i = 1/(1/τ_c² + 1/σ_i²)
//    τ_c² = クラスタ内の e_i の分散 (singleton は全候補の分散で代用)、下限 0.04。
//    σ_i² = max(sd_i²/R_pre, 0.04)。少標本で上振れた候補ほど μ_c へ引き戻される (winner's curse 対策)。
// ④ 各クラスタの代表 = m̃ 最大メンバー。全クラスタを子とし (m̃ 降順)、
//    子の事前 = N(m̃_rep, ts_prior_scale·ṽ_rep) を保存 → 訪問配分は selectThompson。
std::vector<int> ReinvestExperiment::selectClusterTS(
    TreeNode& node, SimulatorWrapper& sim, ShotGenerator& gen,
    std::mt19937& rng, dc::Team root_team)
{
    int N = static_cast<int>(node.candidates.size());
    std::vector<int> selected;
    if (N == 0) return selected;

    estimateRootScores(node, sim, gen, rng, root_team);

    auto dist_table = mcts_shared::makeDistanceTableDelta(node.state, node.result_states);
    int Kc = std::max(1, static_cast<int>(std::ceil(N * config_.retention_rate)));
    Kc = std::min(Kc, N);
    node.clusters = mcts_shared::runClustering(dist_table, Kc);

    const double R = std::max(1, config_.score_screen_r_pre - node.depth);
    const double VAR_FLOOR = 0.04;  // (0.2 点)²
    auto obs_var = [&](int i) { return std::max(node.sd_pre[i] * node.sd_pre[i] / R, VAR_FLOOR); };

    // 全候補の e_i 分散 (singleton クラスタの τ² 代用)
    double gsum = 0.0, gsq = 0.0;
    for (int i = 0; i < N; i++) { gsum += node.e_pre[i]; gsq += node.e_pre[i] * node.e_pre[i]; }
    double gmean = gsum / N;
    double tau2_global = std::max(gsq / N - gmean * gmean, VAR_FLOOR);

    int C = static_cast<int>(node.clusters.size());
    node.cluster_value.assign(C, std::numeric_limits<double>::quiet_NaN());
    node.cluster_sd.assign(C, std::numeric_limits<double>::quiet_NaN());
    struct Rep { int cand; int cid; double post_mean; double post_var; };
    std::vector<Rep> reps;
    for (int cid = 0; cid < C; cid++) {
        double sum = 0.0, sq = 0.0; int cnt = 0;
        for (int m : node.clusters[cid]) {
            if (m < 0 || m >= N) continue;
            sum += node.e_pre[m]; sq += node.e_pre[m] * node.e_pre[m]; cnt++;
        }
        if (cnt == 0) continue;
        double mu = sum / cnt;
        double tau2 = (cnt >= 2) ? std::max(sq / cnt - mu * mu, VAR_FLOOR) : tau2_global;
        // 診断出力用 (μ_c と プールSD; A9 と同じ意味の列に格納)
        double pooled = 0.0;
        for (int m : node.clusters[cid]) {
            if (m < 0 || m >= N) continue;
            double d = node.e_pre[m] - mu;
            pooled += node.sd_pre[m] * node.sd_pre[m] + d * d;
        }
        node.cluster_value[cid] = mu;
        node.cluster_sd[cid] = std::sqrt(std::max(0.0, pooled / cnt));
        Rep best{-1, cid, -1e18, 0.0};
        for (int m : node.clusters[cid]) {
            if (m < 0 || m >= N) continue;
            double prec = 1.0 / tau2 + 1.0 / obs_var(m);
            double pm = (mu / tau2 + node.e_pre[m] / obs_var(m)) / prec;
            if (pm > best.post_mean) best = Rep{m, cid, pm, 1.0 / prec};
        }
        if (best.cand >= 0) reps.push_back(best);
    }
    std::sort(reps.begin(), reps.end(), [](const Rep& a, const Rep& b) { return a.post_mean > b.post_mean; });

    node.ts_prior_mean.clear();
    node.ts_prior_var.clear();
    for (const auto& r : reps) {
        selected.push_back(r.cand);
        node.ts_prior_mean.push_back(r.post_mean);
        node.ts_prior_var.push_back(std::max(config_.ts_prior_scale * r.post_var, VAR_FLOOR));
    }
    return selected;
}

// Thompson: 子 j の事後 N(post_mean, post_var) からサンプルし argmax を返す。
// 事後 = 事前 (スクリーン縮約) ⊕ 観測 (子の訪問統計; 1 プレイアウト報酬の分散 = ts_obs_var)
int ReinvestExperiment::selectThompson(const TreeNode& node, std::mt19937& rng) const {
    int K = static_cast<int>(node.medoid_indices.size());
    int best = -1;
    double best_draw = -1e18;
    for (int j = 0; j < K; j++) {
        double pm = node.ts_prior_mean[j], pv = node.ts_prior_var[j];
        int n = 0; double mean = 0.0;
        if (node.children[j]) { n = node.children[j]->visits; mean = node.children[j]->mean(); }
        double prec = 1.0 / pv + static_cast<double>(n) / config_.ts_obs_var;
        double post_mean = (pm / pv + n * mean / config_.ts_obs_var) / prec;
        double post_var = 1.0 / prec;
        std::normal_distribution<double> dist(post_mean, std::sqrt(post_var));
        double draw = dist(rng);
        if (draw > best_draw) { best_draw = draw; best = j; }
    }
    return best;
}

// 最終着手: 事後平均最大の子 (Thompson は訪問が最良手に集中しない場合があるため、訪問数でなく事後で決める)
int ReinvestExperiment::selectBestPosterior(const TreeNode& node) const {
    int K = static_cast<int>(node.medoid_indices.size());
    int best = -1;
    double best_pm = -1e18;
    for (int j = 0; j < K; j++) {
        double pm = node.ts_prior_mean[j], pv = node.ts_prior_var[j];
        int n = 0; double mean = 0.0;
        if (node.children[j]) { n = node.children[j]->visits; mean = node.children[j]->mean(); }
        double prec = 1.0 / pv + static_cast<double>(n) / config_.ts_obs_var;
        double post_mean = (pm / pv + n * mean / config_.ts_obs_var) / prec;
        if (post_mean > best_pm) { best_pm = post_mean; best = j; }
    }
    return best;
}

// ========== ClusterPW: root の progressive widening (A11) ==========
// 開く子数 k(N) = max(k0, ceil(C · N^α)) (N = root の訪問数)。増えた分だけ pw_queue から価値順に子を追加する。
// 新しい子は visits=0 なので UCB の未訪問優先で直後に 1 回は必ず調べられる (標準的な PW の挙動)。
void ReinvestExperiment::widenRoot(TreeNode& node) const {
    int total = static_cast<int>(node.pw_queue.size());
    if (node.pw_opened >= total) return;
    double target = config_.pw_c * std::pow(static_cast<double>(node.visits) + 1.0, config_.pw_alpha);
    int k = std::max(std::max(1, config_.pw_k0), static_cast<int>(std::ceil(target - 1e-9)));
    k = std::min(k, total);
    while (node.pw_opened < k) {
        node.medoid_indices.push_back(node.pw_queue[node.pw_opened]);
        node.children.push_back(nullptr);
        node.pw_opened++;
    }
}

// ========== UCB1 選択 ==========

int ReinvestExperiment::selectBestChildUCB(const TreeNode& node, dc::Team root_team) const {
    // negamax: 子の mean は root_team 視点で格納されている。このノードで打つのは
    // node.to_play なので、相手番ノードでは符号反転して「相手最良 = root視点最小」を選ぶ。
    // (修正前は全ノードで root視点最大化 = 相手が root に協力する max-max だった)
    double sign = (node.to_play == root_team) ? 1.0 : -1.0;
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
        double score = mcts_shared::ucb1Score(sign * mean, visits, node.visits, config_.ucb_c);
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
    uint64_t state_seed,
    const dc::GameState* actual)
{
    int R = std::max(1, config_.rollouts_per_visit);

    // 評価に使う状態: noisy_tree 時は外乱ありで辿った実状態、通常は参照状態 (決定的着地)
    const dc::GameState& cur = (config_.noisy_tree && actual) ? *actual : node.state;

    // 葉に到達 → ロールアウト (R 回平均)。方策は全アーム共通の ε-greedy 賢い候補。
    if (node.depth >= config_.depth) {
        int remaining = std::max(0, 16 - static_cast<int>(cur.shot));
        double sum = 0.0;
        for (int i = 0; i < R; i++) {
            sum += mcts_shared::rolloutFromState(
                sim, gen, cur, remaining, root_team, rng, config_.epsilon);
        }
        double mean_reward = sum / R;
        node.visits++;
        node.total_reward += mean_reward;
        return mean_reward;
    }

    if (!node.expanded) {
        expandNode(node, sim, gen, cache, rng, root_team, state_seed);
    }
    if (config_.mode == MctsMode::ClusterPW && node.depth == 0) {
        widenRoot(node);  // A11: 訪問数に応じて次のクラスタ代表を開く
    }
    int K = static_cast<int>(node.medoid_indices.size());
    if (K == 0) {
        // 候補なし異常系: ロールアウトのみ
        int remaining = std::max(0, 16 - static_cast<int>(cur.shot));
        double sum = 0.0;
        for (int i = 0; i < R; i++) {
            sum += mcts_shared::rolloutFromState(
                sim, gen, cur, remaining, root_team, rng, config_.epsilon);
        }
        double mean_reward = sum / R;
        node.visits++;
        node.total_reward += mean_reward;
        return mean_reward;
    }

    int idx = (config_.mode == MctsMode::ClusterTS && node.depth == 0)
                  ? selectThompson(node, rng)
                  : selectBestChildUCB(node, root_team);
    if (idx < 0) idx = 0;

    if (!node.children[idx]) {
        auto child = std::make_unique<TreeNode>();
        child->state = node.result_states[node.medoid_indices[idx]];
        child->depth = node.depth + 1;
        child->to_play = (child->state.shot % 2 == 0) ? dc::Team::k0 : dc::Team::k1;
        node.children[idx] = std::move(child);
    }

    // エンド跨ぎ終端の得点抽出 (現エンド ce の純得点, root_team視点)
    auto endScore = [&](const dc::GameState& s, int ce) {
        double diff = 0.0;
        if (ce >= 0 && ce < static_cast<int>(s.scores[0].size())) {
            int t0 = s.scores[0][ce] ? static_cast<int>(*s.scores[0][ce]) : 0;
            int t1 = s.scores[1][ce] ? static_cast<int>(*s.scores[1][ce]) : 0;
            diff = static_cast<double>(t0 - t1);
        }
        return (root_team == dc::Team::k0) ? diff : -diff;
    };
    // 跨ぎ子は「終端の葉」として子の統計も更新する (更新しないと visits=0 のまま
    // UCB優先度∞で同じ子だけが選ばれ続け、相手最終ショットの探索が壊れる)
    auto terminalVisit = [&](int child_idx, double reward_t) {
        node.children[child_idx]->visits++;
        node.children[child_idx]->total_reward += reward_t;
    };

    double reward;
    if (config_.noisy_tree) {
        // 開ループ: 選んだ手を「実状態」から外乱ありで打ち直し、サンプルされた次状態を子に運ぶ。
        dc::GameState next = sim.run_single_simulation(
            cur, node.candidates[node.medoid_indices[idx]].shot);
        if (static_cast<int>(next.end) != static_cast<int>(cur.end) || next.IsGameOver()) {
            // この一打でエンドが確定 (最終ショット跨ぎ): 次エンドを読まず実エンド得点を報酬に
            // (審判・R_pre と同じ規約)。noisy では毎訪問サンプルが異なる=外乱込みの終端分布
            reward = endScore(next, static_cast<int>(cur.end));
            terminalVisit(idx, reward);
        } else {
            reward = runPlayout(*node.children[idx], sim, gen, cache, rng, root_team, state_seed, &next);
        }
    } else {
        const dc::GameState& cs = node.children[idx]->state;
        if (static_cast<int>(cs.end) != static_cast<int>(cur.end) || cs.IsGameOver()) {
            // 【重要バグ修正】決定的木でもエンド跨ぎはここで打ち切り、実エンド得点を報酬にする。
            // 修正前は次エンドの盤面を展開して「次エンドのロールアウト得点」を価値にしていた
            // (現エンドの得点が価値に入らない上、得点するとハンマーを失うため
            //  「今得点しない消極手」を系統的に選好する誤った目的関数になっていた)。
            reward = endScore(cs, static_cast<int>(cur.end));
            terminalVisit(idx, reward);
        } else {
            reward = runPlayout(*node.children[idx], sim, gen, cache, rng, root_team, state_seed);
        }
    }

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
        expandNode(root, sim, gen, cache, rng, root_team, state_seed);
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

    int best_child = (config_.mode == MctsMode::ClusterTS) ? selectBestPosterior(root)
                                                           : selectMostVisited(root);
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
    const bool value_mode = (config_.mode == MctsMode::ClusterValue
                             || config_.mode == MctsMode::ClusterValueDeep
                             || config_.mode == MctsMode::ClusterPW
                             || config_.mode == MctsMode::ClusterTS);
    if ((config_.mode == MctsMode::Proposed || value_mode) && !root.clusters.empty()) {
        // ClusterValue/Deep/PW は加えて E[score]/クラスタ価値/無外乱着地座標 (エリア逆射影用) を出力
        int team_idx = static_cast<int>(root.to_play);
        int stone_idx = static_cast<int>(root.state.shot) / 2;
        // 代表手 → (開いた順, 子の訪問数)。PW では「何番目に開かれ、どれだけ調べられたか」の素データ
        std::unordered_map<int, std::pair<int, int>> rep_info;
        for (int j = 0; j < static_cast<int>(root.medoid_indices.size()); j++) {
            int v = (j < static_cast<int>(root.children.size()) && root.children[j]) ? root.children[j]->visits : 0;
            rep_info[root.medoid_indices[j]] = {j, v};
        }
        for (int cid = 0; cid < static_cast<int>(root.clusters.size()); cid++) {
            for (int cand_idx : root.clusters[cid]) {
                if (cand_idx < 0 || cand_idx >= static_cast<int>(root.candidates.size())) continue;
                ClusterAssign ca;
                ca.candidate_idx = cand_idx;
                ca.cluster_id = cid;
                auto ri = rep_info.find(cand_idx);
                ca.is_representative = (ri != rep_info.end());
                if (ca.is_representative) { ca.rep_rank = ri->second.first; ca.rep_visits = ri->second.second; }
                ca.label = root.candidates[cand_idx].label;
                ca.shot_type = labelToType(ca.label);
                if (value_mode) {
                    if (cand_idx < static_cast<int>(root.e_pre.size())) {
                        ca.e_score = root.e_pre[cand_idx];
                        ca.e_sd = root.sd_pre[cand_idx];
                    }
                    if (cid < static_cast<int>(root.cluster_value.size()))
                        ca.cluster_value = root.cluster_value[cid];
                    if (cid < static_cast<int>(root.cluster_sd.size())) {
                        ca.cluster_sd = root.cluster_sd[cid];
                        if (!std::isnan(ca.cluster_value) && !std::isnan(ca.cluster_sd))
                            ca.cluster_value_risk = ca.cluster_value - config_.risk_lambda * ca.cluster_sd;
                    }
                    // 投球石の無外乱着地 (アウトなら NaN のまま)
                    const auto& rs = root.result_states[cand_idx];
                    if (team_idx >= 0 && team_idx < 2 && stone_idx >= 0 && stone_idx < 8) {
                        const auto& st = rs.stones[team_idx][stone_idx];
                        if (st) { ca.land_x = st->position.x; ca.land_y = st->position.y; }
                    }
                }
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

// ========== 1局面の着手決定 (自己対戦ハーネス用) ==========
ShotInfo ReinvestExperiment::decideShot(
    const dc::GameState& state, dc::Team to_play, uint64_t state_seed,
    SimulatorWrapper& sim, ShotGenerator& gen, std::mt19937& rng)
{
    // フェーズ適応: 前半 (r > 閾値) は depth1×P_early、終盤 (r ≤ 閾値) は depth×P_late。
    // 局所インスタンス (adaptive=false) に委譲するのでスレッド安全・再帰なし。
    if (config_.adaptive) {
        int r = 16 - static_cast<int>(state.shot);
        ReinvestConfig c = config_;
        c.adaptive = false;
        if (r <= config_.adaptive_r_late) {
            c.depth = std::min(config_.depth, std::max(1, r));
            c.playouts = config_.adaptive_p_late;
        } else {
            c.depth = 1;
            c.playouts = config_.adaptive_p_early;
        }
        ReinvestExperiment sub(game_setting_, c);
        return sub.decideShot(state, to_play, state_seed, sim, gen, rng);
    }

    std::unordered_map<uint64_t, CandidateCacheEntry> cache;
    TreeNode root;
    root.state = state;
    root.depth = 0;
    root.to_play = to_play;
    buildTree(root, sim, gen, cache, rng, to_play, state_seed);  // root_team = to_play (自己最適化)

    int best = selectMostVisited(root);
    if (best >= 0 && best < static_cast<int>(root.medoid_indices.size())) {
        int idx = root.medoid_indices[best];
        if (idx >= 0 && idx < static_cast<int>(root.candidates.size()))
            return root.candidates[idx].shot;
    }
    if (!root.candidates.empty()) return root.candidates[0].shot;  // フォールバック
    return ShotInfo{0.0f, 0.0f, 1};  // 候補なし異常系 (実質到達しない)
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
        << "candidate_idx,label,actual_total_sims,time_sec,risk_lambda,num_candidates,num_children\n";
    ofs << std::setprecision(6);

    std::string method = methodName(config_.mode);
    for (const auto& r : results) {
        if (r.game_id < 0) continue;  // 例外でスキップされた局面
        ofs << r.game_id << "," << r.end << "," << r.shot_num << ","
            << method << "," << config_.depth << "," << config_.playouts << ","
            << config_.rollouts_per_visit << "," << config_.seed << ","
            << r.best_idx << ",\"" << r.label << "\","
            << r.actual_total_sims << "," << r.time_sec << ","
            << config_.risk_lambda << "," << r.num_candidates << "," << r.num_children << "\n";
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
        << "is_representative,shot_type,label,"
        << "e_score,e_sd,cluster_value,land_x,land_y,"
        << "cluster_sd,cluster_value_risk,risk_lambda,rep_rank,rep_visits\n";

    // NaN は空欄で出力 (ClusterValue 以外のモード / アウトになった投球石)
    auto num = [](double v) {
        return std::isnan(v) ? std::string() : std::to_string(v);
    };

    std::string method = methodName(config_.mode);
    long long rows = 0;
    for (const auto& r : results) {
        if (r.game_id < 0) continue;
        for (const auto& ca : r.cluster_table) {
            ofs << r.game_id << "," << r.end << "," << r.shot_num << ","
                << method << "," << config_.seed << ","
                << ca.candidate_idx << "," << ca.cluster_id << ","
                << (ca.is_representative ? 1 : 0) << ","
                << "\"" << ca.shot_type << "\",\"" << ca.label << "\","
                << num(ca.e_score) << "," << num(ca.e_sd) << ","
                << num(ca.cluster_value) << "," << num(ca.land_x) << "," << num(ca.land_y) << ","
                << num(ca.cluster_sd) << "," << num(ca.cluster_value_risk) << ","
                << config_.risk_lambda << ","
                << (ca.rep_rank >= 0 ? std::to_string(ca.rep_rank) : std::string()) << ","
                << (ca.rep_visits >= 0 ? std::to_string(ca.rep_visits) : std::string()) << "\n";
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
    std::cout << "  risk_lambda        = " << config_.risk_lambda
              << (config_.mode == MctsMode::ClusterValue || config_.mode == MctsMode::ClusterValueDeep
                  || config_.mode == MctsMode::ClusterPW
                  ? "" : "  (ClusterValue 系以外では無効)") << std::endl;
    if (config_.mode == MctsMode::ClusterTS) {
        std::cout << "  ts (obs_var, prior_scale) = " << config_.ts_obs_var << ", " << config_.ts_prior_scale << std::endl;
    }
    if (config_.mode == MctsMode::ClusterPW) {
        std::cout << "  pw (C, alpha, k0)  = " << config_.pw_c << ", " << config_.pw_alpha << ", " << config_.pw_k0
                  << "  -> k(N)=max(k0, ceil(C*N^alpha)); k(" << config_.playouts << ")="
                  << std::max(config_.pw_k0, static_cast<int>(std::ceil(config_.pw_c * std::pow(config_.playouts, config_.pw_alpha) - 1e-9)))
                  << std::endl;
    }
    std::cout << "  noisy_tree         = " << (config_.noisy_tree ? "true" : "false") << std::endl;
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

    // モード分離実験用: クラスタ割当テーブル (Proposed/RandomK/ClusterValue/Deep のみ中身あり)
    // ClusterValue/Deep は e_score/cluster_value/land_x,y 列 = エリア価値マップの素データ
    if (config_.mode == MctsMode::Proposed || config_.mode == MctsMode::RandomK
        || config_.mode == MctsMode::ClusterValue
        || config_.mode == MctsMode::ClusterValueDeep
        || config_.mode == MctsMode::ClusterPW
        || config_.mode == MctsMode::ClusterTS) {
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
