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
#include <limits>
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
    double retention_rate = 0.20;        // Proposed/RandomK: K = ceil(N * rate); ScoreScreen: depth>0 の子数
    // ScoreScreen (A7) 専用パラメータ
    int score_screen_r_pre = 3;          // root 候補ごとの E[score] 事前推定ロールアウト数 R_pre
    double score_screen_band = 1.0;      // ε帯 Δ: 最良E[score]から Δ 点以内を有望集合に残す
    int score_screen_v_target = 50;      // 子1個に割り当てたい最低訪問数 → K_cap = max(1, playouts / v_target)
    // ClusterValueDeep (A10) 専用: 相手番ノードの子数 (min側は広めに持ち最善応手の取りこぼしを防ぐ)
    int cv_k_opp = 8;
    // ClusterValue (A9) のリスク統合 (--risk-lambda λ):
    //   クラスタ価値 = μ_c − λ·σ_c。σ_c はメンバー全ロールアウト標本のプールSD
    //   (候補間の期待値ばらつき + 候補内の実行/継続ばらつきを両方含む)。
    //   代表手も e_i − λ·sd_i (リスク調整値) が最大の候補。λ=0 で従来の A9 と完全一致。
    double risk_lambda = 0.0;
    // 開ループ・リサンプリング木 (--noisy-tree): 木の構造(候補/クラスタ/スクリーン)は決定的着地で
    // 作るが、価値評価は毎訪問エッジを外乱ありで再シミュレーションした実状態で行う。
    // R_pre推定も候補手を毎回打ち直す (審判の resample_first_shot と同じ規約)。
    // 無外乱木が立てる「精密だが脆い計画」を、訪問ごとの実行ノイズで自然に罰する。
    bool noisy_tree = false;
    // フェーズ適応探索 (--adaptive, decideShot=自己対戦経路のみ):
    // treedepth の知見「深さが手を変えるのは終盤 (残りr≤4) だけ」に基づき、
    // 前半 (r > adaptive_r_late) は depth1×adaptive_p_early で安く流し、
    // 終盤 (r ≤ adaptive_r_late) は depth×adaptive_p_late に集中投資する
    // (K_cap = P/v_target なので終盤は子の数も自動で増える)。
    // 既定 100×6手 + 500×2手 = 1600 = 定数P200×8手 と同一総プレイアウト。
    bool adaptive = false;
    int adaptive_r_late = 4;     // 終盤とみなす残り手数の閾値
    int adaptive_p_early = 100;  // 前半の playouts (depth は 1 固定)
    int adaptive_p_late = 500;   // 終盤の playouts (depth は config.depth)
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
    bool is_representative = false;   // クラスタ代表 (medoid/価値最大メンバー) / RandomK 選択手か
    std::string label;               // 候補ラベル ("Draw(CW,5)" 等)
    std::string shot_type;           // ShotType 文字列 ("Draw"/"Hit"/... = モード定義キー)
    // --- ClusterValue (A9) 拡張列 (他モードは NaN のまま = CSV では空欄) ---
    double e_score = std::numeric_limits<double>::quiet_NaN();        // 候補の E[score] (R_pre 推定)
    double e_sd = std::numeric_limits<double>::quiet_NaN();           // 同 SD
    double cluster_value = std::numeric_limits<double>::quiet_NaN();  // 所属クラスタの平均 E[score]
    double land_x = std::numeric_limits<double>::quiet_NaN();         // 投球石の無外乱着地 x (エリア逆射影用)
    double land_y = std::numeric_limits<double>::quiet_NaN();         // 同 y (アウトなら NaN)
    // --- リスク統合 (--risk-lambda) ---
    double cluster_sd = std::numeric_limits<double>::quiet_NaN();          // 所属クラスタのプールSD σ_c
    double cluster_value_risk = std::numeric_limits<double>::quiet_NaN();  // μ_c − λ·σ_c (λ=0 なら cluster_value と同値)
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

    // 1局面 → この手法(config_.mode)での着手を返す (自己対戦ハーネス用)。
    // 木を構築し最多訪問子の候補手を返す。決定は to_play 視点で自己最適化。
    // メンバ状態を変更しないので、同一インスタンスを複数スレッドから呼んでよい。
    ShotInfo decideShot(const dc::GameState& state, dc::Team to_play, uint64_t state_seed,
                        SimulatorWrapper& sim, ShotGenerator& gen, std::mt19937& rng);

    static std::string methodName(MctsMode m);  // "AllGrid"/"Proposed"/"RandomK"/"ScoreScreen"

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
    // actual: noisy_tree 時にエッジを外乱ありで辿った「実状態」(nullptr = node.state を使う)。
    // 木の構造は node.state (決定的着地) 基準のまま、評価だけ実状態で行う開ループ方式。
    double runPlayout(
        TreeNode& node,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
        std::mt19937& rng,
        dc::Team root_team,
        uint64_t state_seed,
        const dc::GameState* actual = nullptr);

    // ノード展開 (候補生成 + Proposed:クラスタリング / RandomK:決定的乱択 / AllGrid:全候補 / ScoreScreen:得点スクリーン)
    // sim/rng は ScoreScreen の root で E[score] 事前推定ロールアウトに使う
    void expandNode(
        TreeNode& node,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::unordered_map<uint64_t, CandidateCacheEntry>& cache,
        std::mt19937& rng,
        dc::Team root_team,
        uint64_t state_seed);

    // root 候補ごとの E[score]/SD を R_pre ロールアウトで推定し node.e_pre/sd_pre に格納
    // (ScoreScreen/ScoreTopK/ClusterValue 共通の ①)
    void estimateRootScores(
        TreeNode& node,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::mt19937& rng,
        dc::Team root_team);

    // ScoreScreen の root 候補選別: ①R_pre 推定 →②ε帯 →③リスク多様性保持でK個。選んだ candidate idx を返す。
    // (ScoreTopK は ① の後 E[score] 降順上位 K_cap を返す)
    std::vector<int> selectScoreScreen(
        TreeNode& node,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::mt19937& rng,
        dc::Team root_team);

    // ClusterValue (A9) の root 候補選別: distDelta クラスタ + クラスタ平均 E[score] 価値付け
    // → 価値上位 K_cap クラスタから各クラスタ内 E[score] 最大の手を子に。
    // 全体最良 E[score] の候補が属するクラスタは必ず含める。
    std::vector<int> selectClusterValue(
        TreeNode& node,
        SimulatorWrapper& sim,
        ShotGenerator& gen,
        std::mt19937& rng,
        dc::Team root_team);

    // negamax UCB: 子の mean は root_team 視点。相手番ノード (to_play != root_team) では
    // 符号反転して「相手最良 = root視点最小」の子を選ぶ (max-max = 協力的相手モデルの修正)。
    int selectBestChildUCB(const TreeNode& node, dc::Team root_team) const;
    int selectMostVisited(const TreeNode& node) const;
    uint64_t hashGameState(const dc::GameState& s) const;

    void writeResultsCSV(const std::vector<ReinvestResult>& results,
                         const std::string& path) const;

    // モード分離実験用: root のクラスタ割当を 1 候補 1 行で出力 (Proposed/RandomK)
    void writeClusterTableCSV(const std::vector<ReinvestResult>& results,
                              const std::string& path) const;
};

// 文字列 -> MctsMode ("AllGrid"/"Proposed"/"RandomK")。未知は Proposed。
MctsMode parseMctsMode(const std::string& s);

#endif  // _REINVEST_EXPERIMENT_H_
