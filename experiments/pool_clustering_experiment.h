#pragma once
#ifndef _POOL_CLUSTERING_EXPERIMENT_H_
#define _POOL_CLUSTERING_EXPERIMENT_H_

#include "digitalcurling3/digitalcurling3.hpp"
#include "../src/shot_generator.h"
#include <vector>
#include <set>
#include <string>

namespace dc = digitalcurling3;

// クラスタの分析結果
struct ClusterAnalysis {
    int cluster_id;
    std::vector<int> member_indices;           // 候補手のインデックス
    std::vector<ShotType> member_types;        // 各メンバーのショットタイプ
    std::vector<std::string> member_labels;    // 各メンバーのラベル
    int medoid_index;                          // メドイドのインデックス
    ShotType medoid_type;                      // メドイドのショットタイプ
    std::string medoid_label;                  // メドイドのラベル

    // タイプ別集計
    int count_draw = 0;
    int count_hit = 0;
    int count_freeze = 0;
    int count_guard = 0;
    int count_other = 0;

    // 純度（最多タイプの割合）
    float purity() const;
    std::string dominantType() const;
};

// 全体の分析結果
struct PoolClusteringResult {
    std::string state_name;
    int n_candidates;
    int n_clusters;
    std::vector<ClusterAnalysis> clusters;

    // 全体純度（加重平均）
    float weightedPurity() const;
    // タイプカバレッジ（異なる支配タイプを持つクラスタ数 / クラスタ数）
    float typeCoverage() const;
};

class PoolClusteringExperiment {
public:
    PoolClusteringExperiment(dc::GameSetting const& game_setting);

    void run();

private:
    dc::GameSetting game_setting_;
    std::vector<std::string> test_state_names_;  // テスト盤面名のリスト

    static constexpr float kHouseCenterX = 0.0f;
    static constexpr float kHouseCenterY = 38.405f;
    static constexpr float kHouseRadius = 1.829f;

    // テスト盤面を生成
    std::vector<dc::GameState> createTestStates();

    // 距離計算
    bool isInHouse(const std::optional<dc::Transform>& stone) const;
    int getZone(const std::optional<dc::Transform>& stone) const;  // 0:ハウス内, 1:ガード, 2:遠方

    // 旧距離関数（絶対位置比較）
    float dist(const dc::GameState& a, const dc::GameState& b) const;
    std::vector<std::vector<float>> makeDistanceTable(const std::vector<dc::GameState>& states);

    // 盤面スコア評価（カーリングの得点ルール準拠）
    float evaluateBoard(const dc::GameState& state) const;

    // デルタ距離関数v2（入力盤面からの変化量 + スコア + インタラクション + 近接度）
    float distDelta(const dc::GameState& input, const dc::GameState& a, const dc::GameState& b) const;
    std::vector<std::vector<float>> makeDistanceTableDelta(
        const dc::GameState& input_state,
        const std::vector<dc::GameState>& result_states
    );

    // 階層的クラスタリング
    std::vector<std::set<int>> runClustering(
        const std::vector<std::vector<float>>& dist_table,
        int n_desired_clusters
    );

    // メドイド計算
    std::vector<int> calculateMedoids(
        const std::vector<std::vector<float>>& dist_table,
        const std::vector<std::set<int>>& clusters
    );

    // 貪欲最遠点サンプリング（距離テーブル不要、O(N*K)）
    std::vector<int> greedyFarthestPointSampling(
        const dc::GameState& input_state,
        const std::vector<dc::GameState>& result_states,
        int k
    );

    // クラスタ分析
    PoolClusteringResult analyzeClusterComposition(
        const std::string& state_name,
        const CandidatePool& pool,
        const std::vector<std::set<int>>& clusters,
        const std::vector<int>& medoids
    );

    // 結果出力
    void printResult(const PoolClusteringResult& result);
    void exportResultCSV(const PoolClusteringResult& result, const std::string& output_dir);
};

#endif
