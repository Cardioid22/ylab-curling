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
};

class ClusteringEffectivenessExperiment {
public:
    ClusteringEffectivenessExperiment(dc::GameSetting const& game_setting);

    // 実験パラメータ設定
    void setTestNum(int n) { test_games_ = n; }
    void setRolloutCount(int b) { rollout_count_ = b; }
    void setRetentionRates(const std::vector<int>& rates) { retention_rates_ = rates; }

    void run();

private:
    dc::GameSetting game_setting_;

    // 実験パラメータ
    int test_games_ = 10;           // テストゲーム数
    int rollout_count_ = 1;         // 各候補のロールアウト回数
    std::vector<int> retention_rates_ = {10, 20};  // 保持率(%)

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
    std::vector<int> calculateMedoids(
        const std::vector<std::vector<float>>& dist_table,
        const std::vector<std::set<int>>& clusters);

    // シルエットスコア
    double calcSilhouetteScore(
        const std::vector<std::vector<float>>& dist_table,
        const std::vector<std::set<int>>& clusters);

    // 結果出力
    void printSummary(const std::vector<TestCaseResult>& results, int retention_pct);
    void exportCSV(const std::vector<TestCaseResult>& results, int retention_pct);
};

#endif
