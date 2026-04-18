#pragma once
#ifndef _CLUSTERING_EFFECTIVENESS_EXPERIMENT_H_
#define _CLUSTERING_EFFECTIVENESS_EXPERIMENT_H_

// クラスタリング有効性実験
// AllGrid(全候補×gPolicyロールアウト) vs Clustered(クラスタリング後×gPolicyロールアウト)
// ショット番号・エンド番号ごとの一致率を計測

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include "../src/simulator.h"
#include "../src/policy.h"
#include <vector>
#include <set>
#include <string>
#include <chrono>
#include <random>

namespace dc = digitalcurling3;

// 1テスト盤面の結果
struct TestCaseResult {
    int game_id;
    int end;
    int shot_num;
    int n_candidates;          // 全候補手数
    int n_clustered;           // クラスタリング後の候補数

    // AllGrid の最良手
    int allgrid_best_idx;
    std::string allgrid_best_label;
    ShotType allgrid_best_type;
    double allgrid_best_score;
    double allgrid_time_sec;

    // Clustered の最良手
    int clustered_best_idx;
    std::string clustered_best_label;
    ShotType clustered_best_type;
    double clustered_best_score;
    double clustered_time_sec;

    // 一致指標
    bool exact_match;          // 完全一致
    bool same_cluster;         // 同じクラスタ内
    bool same_type;            // 同じショットタイプ
    double score_diff;         // スコア差 (AllGrid - Clustered)

    // クラスタリング情報
    double silhouette_score;
    int allgrid_cluster_id;    // AllGrid最良手が属するクラスタ
    int clustered_cluster_id;  // Clustered最良手が属するクラスタ

    // Simple クラスタリング (ベースライン: 速度ベクトルのグリッド分割)
    // 「打つ瞬間の情報で似ているものを削る」ポリシー
    int simple_best_idx;
    std::string simple_best_label;
    ShotType simple_best_type;
    double simple_best_score;
    double simple_time_sec;
    bool simple_exact_match;
    bool simple_same_cluster;
    bool simple_same_type;
    double simple_score_diff;   // AllGrid - Simple

    // ランダムクラスタリング (ベースライン)
    int random_best_idx;
    std::string random_best_label;
    ShotType random_best_type;
    double random_best_score;
    double random_time_sec;
    bool random_exact_match;
    bool random_same_cluster;      // AllGrid最良手と同じランダムクラスタ内
    bool random_same_type;
    double random_score_diff;  // AllGrid - Random
    double random_silhouette_score;
};

class ClusteringEffectivenessExperiment {
public:
    ClusteringEffectivenessExperiment(dc::GameSetting const& game_setting);

    // 実験パラメータ設定
    void setTestNum(int n) { test_games_ = n; }
    void setRolloutCount(int b) { rollout_count_ = b; }
    void setRetentionRates(const std::vector<int>& rates) { retention_rates_ = rates; }
    void setLoadPositionsDir(const std::string& d) { load_positions_dir_ = d; }
    void setMaxPositions(int n) { max_positions_ = n; }
    void setDeterministic(bool d) { deterministic_ = d; }
    void setStartIndex(int i) { start_index_ = i; }

    void run();

private:
    dc::GameSetting game_setting_;

    // 実験パラメータ
    int test_games_ = 10;           // テストゲーム数
    int rollout_count_ = 1;         // 各候補のロールアウト回数
    std::vector<int> retention_rates_ = {10, 20};  // 保持率(%)
    std::string load_positions_dir_;  // CSVから読み込むディレクトリ (空 = 自己対戦)
    int max_positions_ = -1;          // -1 = 全局面使用
    bool deterministic_ = false;       // true: PlayerIdentical + argmax (再現可能)
    int start_index_ = 0;              // 並列実行用: この index から開始 (先頭を skip)

    // コンポーネント
    std::unique_ptr<RolloutPolicy> policy_;
    std::unique_ptr<ShotGenerator> shot_gen_;

    static constexpr float kHouseCenterX = 0.0f;
    static constexpr float kHouseCenterY = 38.405f;
    static constexpr float kHouseRadius = 1.829f;

    // Phase 1: テスト盤面の生成 (gPolicyで自己対戦)
    struct GameRecord {
        int game_id;
        int end;
        int shot_num;
        dc::GameState state;
        dc::Team current_team;
    };
    std::vector<GameRecord> generateTestPositions();

    // CSVから読み込み（generate_test_positions が出力したバッチ形式）
    std::vector<GameRecord> loadTestPositionsFromCSV(const std::string& dir, int max_n = -1);

    // Phase 2: 各盤面で AllGrid vs Clustered を比較
    TestCaseResult evaluatePosition(
        const GameRecord& record,
        int retention_pct,
        SimulatorWrapper& sim
    );

    // 候補手のロールアウト評価 (1候補 → B回ロールアウトの平均スコア)
    double evaluateCandidate(
        const dc::GameState& state,
        const CandidateShot& candidate,
        SimulatorWrapper& sim
    );

    // Delta距離関数 (pool_clustering_experimentから移植)
    float evaluateBoard(const dc::GameState& state) const;
    int getZone(const std::optional<dc::Transform>& stone) const;
    float distDelta(const dc::GameState& input, const dc::GameState& a, const dc::GameState& b) const;
    std::vector<std::vector<float>> makeDistanceTableDelta(
        const dc::GameState& input_state,
        const std::vector<dc::GameState>& result_states);

    // 階層的クラスタリング
    std::vector<std::set<int>> runClustering(
        const std::vector<std::vector<float>>& dist_table,
        int n_desired_clusters);

    // ランダム選択 (ベースライン: ランダムにK手選ぶ)
    std::vector<int> selectRandom(int n_items, int k, std::mt19937& rng);
    // ランダムクラスタ割り当て (same_cluster指標用)
    std::vector<std::set<int>> assignRandomClusters(
        int n_items, int n_clusters, std::mt19937& rng);

    // Simple クラスタリング: 速度ベクトル(vx,vy,rot)のグリッド分割
    // 「打つ瞬間の情報で似ているものを削る」
    std::vector<int> selectByVelocityGrid(
        const std::vector<CandidateShot>& candidates, int k);

    std::vector<int> calculateMedoids(
        const std::vector<std::vector<float>>& dist_table,
        const std::vector<std::set<int>>& clusters);

    // シルエットスコア
    double calcSilhouetteScore(
        const std::vector<std::vector<float>>& dist_table,
        const std::vector<std::set<int>>& clusters);

    // クラスタ構成情報（1ポジションあたり）
    struct ClusterInfo {
        int game_id, end, shot_num;
        int n_candidates;
        std::string method;  // "proposed" or "random"
        int cluster_id;
        int cluster_size;
        int medoid_idx;
        std::string medoid_label;
    };
    std::vector<ClusterInfo> cluster_details_;

    // 結果出力
    void printSummary(const std::vector<TestCaseResult>& results, int retention_pct);
    void exportCSV(const std::vector<TestCaseResult>& results, int retention_pct);
    void exportClusterDetailsCSV(int retention_pct);
};

#endif
