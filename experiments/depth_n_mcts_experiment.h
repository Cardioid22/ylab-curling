#pragma once
#ifndef _DEPTH_N_MCTS_EXPERIMENT_H_
#define _DEPTH_N_MCTS_EXPERIMENT_H_

// 深さN MCTS 実験（デフォルトN=3）
//   - Proposed: クラスタリングでK個のメドイドのみ子ノードに展開
//   - AllGrid : 全候補N個を子ノードに展開（歩相当の純UCT）
// ロールアウトは depth1 実験と同じε-greedyグリッドポリシー
//
// 乱数シードは両手法で一致させ、公平な比較を行う

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include "mcts_shared.h"
#include <vector>
#include <set>
#include <string>
#include <memory>
#include <random>
#include <optional>
#include <unordered_map>

namespace dc = digitalcurling3;

enum class MctsMode {
    Proposed,  // クラスタリングでK個のメドイドを子ノードに
    AllGrid    // 全候補を子ノードに（歩相当）
};

// 展開キャッシュ: 同じ盤面ハッシュなら候補手とシミュ結果を再利用
// 盤面ごと（1スレッド内）でProposed/AllGrid間共有
struct CandidateCacheEntry {
    std::vector<CandidateShot> candidates;
    std::vector<dc::GameState> result_states;
};

// MCTS 木のノード
struct TreeNode {
    dc::GameState state;
    int depth = 0;
    dc::Team to_play = dc::Team::k0;

    // 統計
    int visits = 0;
    double total_reward = 0.0;  // root_team 視点

    // 展開情報
    bool expanded = false;
    std::vector<CandidateShot> candidates;
    std::vector<dc::GameState> result_states;

    // 子ノードインデックス: medoid_indices[i] = candidates の index
    std::vector<int> medoid_indices;

    // クラスタ割当 (Proposed のみ、診断用)
    std::vector<std::set<int>> clusters;

    // 子ノード（lazy allocation）
    std::vector<std::unique_ptr<TreeNode>> children;

    double mean() const {
        return visits > 0 ? total_reward / static_cast<double>(visits) : 0.0;
    }
};

// 実験設定
struct DepthNMctsConfig {
    int depth = 3;                         // MCTS 木の深さ
    int n_states = 100;                    // テスト盤面数
    int proposed_playouts = 500;           // Proposed プレイアウト数
    int allgrid_playouts = 10000;          // AllGrid プレイアウト数
    int proposed_rollouts_per_visit = 20;  // Proposed: 葉到達時の平均化ロールアウト数
    int allgrid_rollouts_per_visit = 10;   // AllGrid: 葉到達時の平均化ロールアウト数
    double retention_rate = 0.20;          // Proposed 保持率
    double ucb_c = 1.41;                   // UCB1 の c (≒√2)
    double epsilon = 0.3;                  // ロールアウト ε
    int num_threads = 8;                   // スレッド数
    uint64_t seed = 42;                    // サンプリング＆ロールアウト用シード
    int start_index = 0;                   // 並列実行用: サンプリング後リストの開始index
    int max_positions = -1;                // 並列実行用: 担当盤面数 (-1=全部)
    std::string load_positions_dir;        // batch_*.csv があるディレクトリ
    std::string output_dir;                // 結果出力ディレクトリ
};

// 1盤面分の比較結果
struct DepthNComparisonResult {
    int game_id = -1;
    int end = -1;
    int shot_num = -1;
    int num_candidates = 0;
    int num_clusters = 0;

    // 選ばれた手 (候補indexで記録)
    int proposed_best_idx = -1;
    int allgrid_best_idx = -1;

    // 選択基準
    double proposed_best_mean = 0.0;   // Proposed で選んだ子の平均報酬
    double allgrid_best_mean = 0.0;

    // 一致指標
    bool exact_match = false;          // candidate index 一致
    bool same_cluster = false;         // AllGrid 最良手が Proposed 最良クラスタ所属
    bool same_type = false;            // ShotType 一致
    double score_diff = 0.0;           // |proposed_mean - allgrid_mean|

    // ショットタイプ（ログ用）
    std::string proposed_label;
    std::string allgrid_label;

    // 実行時間
    double proposed_time_sec = 0.0;
    double allgrid_time_sec = 0.0;

    // プレイアウト数（実際に走った数）
    int proposed_actual_playouts = 0;
    int allgrid_actual_playouts = 0;
};

class DepthNMctsExperiment {
public:
    DepthNMctsExperiment(const dc::GameSetting& game_setting,
                         const DepthNMctsConfig& config);

    // 実験本体: 盤面ロード → スレッド起動 → 結果集約 → CSV出力
    void run();

private:
    dc::GameSetting game_setting_;
    DepthNMctsConfig config_;

    // スレッド内で 1 盤面を処理する
    // thread_seed は盤面 index + base_seed から決まる
    DepthNComparisonResult runOneState(
        const mcts_shared::TestPositionRecord& rec,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        uint64_t thread_seed);

    // 木の構築（プレイアウト実行）
    // mode と playouts 以外は共通
    void buildTree(
        TreeNode& root,
        MctsMode mode,
        int playouts,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
        std::mt19937& rng,
        dc::Team root_team);

    // 1プレイアウト
    double runPlayout(
        TreeNode& node,
        int max_depth,
        MctsMode mode,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
        std::mt19937& rng,
        dc::Team root_team);

    // ノード展開（候補手生成 + 必要ならクラスタリング）
    void expandNode(
        TreeNode& node,
        MctsMode mode,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache);

    // UCB1 で子インデックスを選ぶ
    int selectBestChildUCB(const TreeNode& node) const;

    // 最多訪問の子を選ぶ（最終決定用）
    int selectMostVisited(const TreeNode& node) const;

    // 盤面ハッシュ（キャッシュキー）
    uint64_t hashGameState(const dc::GameState& s) const;

    // 結果を CSV に書き出す
    void writeResultsCSV(const std::vector<DepthNComparisonResult>& results,
                         const std::string& path) const;

    // 進捗ログを stderr に出す
    void logProgress(int done, int total, double elapsed_sec) const;
};

#endif  // _DEPTH_N_MCTS_EXPERIMENT_H_
