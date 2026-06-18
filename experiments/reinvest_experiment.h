#pragma once
#ifndef _REINVEST_EXPERIMENT_H_
#define _REINVEST_EXPERIMENT_H_

// 計算再投資実験 (GPW2026) — 単一アーム MCTS ランナー
//
// 1 プロセス = 1 アーム = (method, depth, playouts P, rollouts_per_visit R) の独立構成。
// すべてのアームを同一の「総物理シミュレーション予算 B」で走らせ、各アームが選んだ root 手を
// 共通審判 (score_move_experiment, Q_ref) で採点してリグレットを比較する。
//
//   問い: クラスタリングで浮いた計算を「深さ(3->5)」と「葉あたりロールアウト数」の
//         どちらに再投資すると等予算で手の質が上がるか。
//
// アーム:
//   A1 AllGrid  depth3              基準
//   A2 Proposed depth3              クラスタリング効果の単離 (A1 と同配分)
//   A3 Proposed depth5              深さ再投資
//   A4 Proposed depth3 (R 増, P 減) ロールアウト再投資
//   A5 RandomK  depth3              クラスタリング vs 単なる削減の単離
//
// 木の機構 (展開/UCB/ロールアウト/バックプロップ) は depth_n_mcts_experiment と同一系統。
// 全アームが本ファイルの同一コードを使うので、アーム間の比較は公平。
// ロールアウト方策・葉評価は mcts_shared 経由で全アーム共通 (ガイド §5)。
//
// 等予算カウント: src/structure.h の g_physics_sim_count (thread_local) を局面前後で差分。
// run_single_simulation (ロールアウト) と simulateNoRand (展開) の両方を計上する。

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include "depth_n_mcts_experiment.h"  // TreeNode, CandidateCacheEntry, MctsMode を再利用
#include "mcts_shared.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace dc = digitalcurling3;

// 単一アームの実験設定
struct ReinvestConfig {
    MctsMode mode = MctsMode::Proposed;  // アーム手法 (Proposed/AllGrid/RandomK)
    int depth = 3;                       // MCTS 木の深さ (3 or 5)
    int playouts = 500;                  // P: プレイアウト数
    int rollouts_per_visit = 20;         // R: 葉到達時の平均化ロールアウト数
    double retention_rate = 0.20;        // Proposed/RandomK: K = ceil(N * rate)
    double ucb_c = 1.41;                 // UCB1 の c (≒√2)
    double epsilon = 0.3;                // ロールアウト ε (全アーム共通)
    int n_states = 10;                   // テスト局面数
    int num_threads = 8;                 // スレッド数
    uint64_t seed = 42;                  // ベースシード (multi-seed の seed)
    int start_index = 0;                 // 並列スライス開始 index (サンプリング後)
    int max_positions = -1;              // 担当局面数 (-1 = 全部)
    std::string load_positions_dir;      // batch_*.csv のディレクトリ
    std::string output_dir;              // 出力ディレクトリ
    std::string arm_label;               // "A1".."A6" など (ファイル名/ログ用、空可)
};

// Proposed/RandomK のクラスタ割当 1 行 (候補 -> クラスタ + 代表点フラグ)
// モード分離実験用: 審判 Q テーブル・AllGrid 選択分布と candidate_idx で join する。
// 展開は simulateNoRand (決定的) なのでクラスタリングは局面ごと seed 非依存 = 審判と同一プール。
struct ClusterAssign {
    int candidate_idx = -1;          // generatePool 順 index (join キー)
    int cluster_id = -1;             // Proposed: 所属クラスタ; RandomK: 選択順 (membership 概念なし)
    bool is_representative = false;   // クラスタ代表 (medoid) / RandomK 選択手か
    std::string label;               // 候補ラベル ("Draw(CW,5)" 等)
    std::string shot_type;           // ShotType 文字列 ("Draw"/"Hit"/... = モード定義キー)
};

// 1 局面分の選択結果 (§4 出力スキーマに対応)
struct ReinvestResult {
    int game_id = -1;
    int end = -1;
    int shot_num = -1;

    int num_candidates = 0;   // generatePool の候補数 N
    int num_children = 0;     // 実際に展開した子数 K

    int best_idx = -1;        // 選んだ手の generatePool 順 index (審判 Q テーブルとの join キー)
    double best_mean = 0.0;   // 選んだ子の平均報酬 (root_team 視点)
    std::string label;        // 選んだ手のラベル

    long long actual_total_sims = 0;  // その局面で消費した実物理シミュ回数 (等予算検証用)
    double time_sec = 0.0;            // 壁時計
    int actual_playouts = 0;          // 実際に走ったプレイアウト数

    // モード分離実験用: root のクラスタ割当 (Proposed/RandomK のみ; AllGrid は空)
    std::vector<ClusterAssign> cluster_table;
};

class ReinvestExperiment {
public:
    ReinvestExperiment(const dc::GameSetting& game_setting, const ReinvestConfig& config);

    // 局面ロード → サンプリング → スライス → スレッド実行 → CSV 出力
    void run();

private:
    dc::GameSetting game_setting_;
    ReinvestConfig config_;

    // 1 局面を処理する (state_seed はスライス前グローバル index 由来で決定)
    ReinvestResult runOneState(
        const mcts_shared::TestPositionRecord& rec,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        uint64_t state_seed);

    // 木構築 (P 回プレイアウト)
    void buildTree(
        TreeNode& root,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
        std::mt19937& rng,
        dc::Team root_team,
        uint64_t state_seed);

    // 1 プレイアウト (選択 → 展開 → ロールアウト → バックプロップ)
    double runPlayout(
        TreeNode& node,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
        std::mt19937& rng,
        dc::Team root_team,
        uint64_t state_seed);

    // ノード展開 (候補生成 + Proposed:クラスタリング / RandomK:決定的乱択 / AllGrid:全候補)
    void expandNode(
        TreeNode& node,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
        uint64_t state_seed);

    int selectBestChildUCB(const TreeNode& node) const;
    int selectMostVisited(const TreeNode& node) const;
    uint64_t hashGameState(const dc::GameState& s) const;

    void writeResultsCSV(const std::vector<ReinvestResult>& results,
                         const std::string& path) const;

    // モード分離実験用: root のクラスタ割当を 1 候補 1 行で出力 (Proposed/RandomK)
    void writeClusterTableCSV(const std::vector<ReinvestResult>& results,
                              const std::string& path) const;

    static std::string methodName(MctsMode m);
};

// 文字列 -> MctsMode ("AllGrid"/"Proposed"/"RandomK")。未知は Proposed。
MctsMode parseMctsMode(const std::string& s);

#endif  // _REINVEST_EXPERIMENT_H_
