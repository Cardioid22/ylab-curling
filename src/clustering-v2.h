#pragma once
#ifndef _CLUSTERING_V2_H
#define _CLUSTERING_V2_H

#include "digitalcurling3/digitalcurling3.hpp"
#include "structure.h"
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <array>

namespace dc = digitalcurling3;

/**
 * @brief 6領域のインデックス定義
 *
 * シートを以下のように分割:
 * Y軸: [Y_MAX, HouseCenterY, Y_MIN] の2区間
 * X軸: [X_MIN, -1, 1, X_MAX] の3区間
 *
 * 領域番号:
 *   0: 左上   (x < -1, y >= HouseCenterY)
 *   1: 中上   (-1 <= x <= 1, y >= HouseCenterY)
 *   2: 右上   (x > 1, y >= HouseCenterY)
 *   3: 左下   (x < -1, y < HouseCenterY)
 *   4: 中下   (-1 <= x <= 1, y < HouseCenterY)
 *   5: 右下   (x > 1, y < HouseCenterY)
 */
enum class GridRegion : int {
    LeftUpper = 0,
    CenterUpper = 1,
    RightUpper = 2,
    LeftLower = 3,
    CenterLower = 4,
    RightLower = 5,
    OutOfBounds = -1
};

/**
 * @brief 状態の特徴ベクトル
 *
 * 各GameStateを特徴量で表現し、効率的な比較・分類を可能にする
 */
struct StateFeature {
    // 基本情報
    int total_stones;                    // シート上の総石数
    int my_stones;                       // 自チームの石数
    int opponent_stones;                 // 相手チームの石数

    // 領域別分布 [6領域 × 2チーム]
    std::array<int, 6> my_team_distribution;        // 自チームの6領域別石数
    std::array<int, 6> opponent_team_distribution;  // 相手チームの6領域別石数

    // ハウス情報
    int my_stones_in_house;              // 自チームのハウス内石数
    int opponent_stones_in_house;        // 相手チームのハウス内石数

    // No.1ストーン情報
    bool has_no1_stone;                  // No.1ストーン(最も中心に近い)が存在するか
    int no1_team;                        // No.1ストーンのチーム (0 or 1, なければ-1)

    // 元の状態ID
    int original_state_id;               // グリッド内のインデックス

    StateFeature();
};

/**
 * @brief クラスタ情報
 *
 * 似た特徴を持つ状態のグループ
 */
struct Cluster {
    std::vector<int> state_ids;          // このクラスタに属する状態IDのリスト
    StateFeature centroid;               // クラスタの重心(平均特徴ベクトル)
    int representative_id;               // 代表状態のID (重心に最も近い状態)
    float internal_variance;             // クラスタ内分散 (品質評価用)

    Cluster();
};

/**
 * @brief クラスタリングV2クラス
 *
 * 新しいアプローチ:
 * 1. 各状態をシミュレーション後の石配置から特徴ベクトル化
 * 2. 総石数で粗分類
 * 3. 6領域分布の類似性で細分類
 * 4. 各クラスタの代表状態を選出
 */
class ClusteringV2 {
public:
    /**
     * @brief コンストラクタ
     *
     * @param k_clusters 目標クラスタ数 (通常はlog2(gridM * gridN))
     * @param all_states シミュレーション後の全状態
     * @param gridM グリッド行数
     * @param gridN グリッド列数
     * @param team 自チーム
     */
    ClusteringV2(
        int k_clusters,
        std::vector<dc::GameState> all_states,
        int gridM,
        int gridN,
        dc::Team team
    );

    /**
     * @brief クラスタリング実行 (メイン関数)
     *
     * @return クラスタのベクトル
     */
    std::vector<Cluster> getClusters();

    /**
     * @brief 各クラスタの代表状態IDを取得
     *
     * MCTSで使用する推奨状態のリスト
     *
     * @return 代表状態IDのベクトル
     */
    std::vector<int> getRecommendedStates();

    /**
     * @brief クラスタID→状態IDマッピングテーブルを取得
     *
     * デバッグ・可視化用
     *
     * @return clusters[i] = {state_id1, state_id2, ...}
     */
    std::vector<std::vector<int>> getClusterIdTable();

    /**
     * @brief クラスタリング品質評価
     *
     * シルエット係数やクラスタ内分散を計算
     *
     * @return 品質スコア (0-1, 高いほど良い)
     */
    float evaluateClusteringQuality() const;

private:
    // 定数
    static constexpr float HouseRadius_ = 1.829f;
    static constexpr float AreaMaxX_ = 2.375f;
    static constexpr float AreaMaxY_ = 40.234f;
    static constexpr float HouseCenterX_ = 0.0f;
    static constexpr float HouseCenterY_ = 38.405f;

    // メンバ変数
    int GridSize_M_;
    int GridSize_N_;
    int n_desired_clusters_;
    dc::Team g_team_;

    std::vector<dc::GameState> states_;
    std::vector<StateFeature> features_;     // 全状態の特徴ベクトル
    std::vector<Cluster> clusters_;          // 最終クラスタ
    bool clustering_done_;

    // === ステップ1: 特徴抽出 ===

    /**
     * @brief 全状態から特徴ベクトルを抽出
     */
    void extractFeatures();

    /**
     * @brief 単一状態から特徴ベクトルを計算
     *
     * @param state GameState
     * @param state_id 状態のインデックス
     * @return StateFeature
     */
    StateFeature computeFeature(const dc::GameState& state, int state_id);

    /**
     * @brief 石の位置から6領域のどれに属するか判定
     *
     * @param x X座標
     * @param y Y座標
     * @return GridRegion
     */
    GridRegion getRegion(float x, float y) const;

    /**
     * @brief 石がハウス内にあるか判定
     *
     * @param stone ストーンのTransform
     * @return ハウス内ならtrue
     */
    bool isInHouse(const std::optional<dc::Transform>& stone) const;

    // === ステップ2: 類似度計算 ===

    /**
     * @brief 2つの特徴ベクトル間の距離を計算
     *
     * 重み付きユークリッド距離:
     * - 総石数の差
     * - 6領域分布の差
     * - ハウス内石数の差
     * - No.1ストーンチームの一致/不一致
     *
     * @param f1 特徴ベクトル1
     * @param f2 特徴ベクトル2
     * @return 距離 (小さいほど類似)
     */
    float computeDistance(const StateFeature& f1, const StateFeature& f2) const;

    // === ステップ3: クラスタリング ===

    /**
     * @brief 粗分類: 総石数でグループ化
     *
     * 総石数が同じ状態をまとめる
     *
     * @return map[total_stones] = {state_id1, state_id2, ...}
     */
    std::map<int, std::vector<int>> coarseGrouping();

    /**
     * @brief 細分類: グループ内でk-meansクラスタリング
     *
     * 各グループを6領域分布の類似性でさらに分割
     *
     * @param group_ids グループ内の状態IDリスト
     * @param k クラスタ数
     * @return クラスタのベクトル
     */
    std::vector<Cluster> fineGrainedClustering(
        const std::vector<int>& group_ids,
        int k
    );

    /**
     * @brief k-meansアルゴリズムの実装
     *
     * @param feature_indices クラスタリング対象の特徴ベクトルのインデックス
     * @param k クラスタ数
     * @param max_iterations 最大反復回数
     * @return クラスタのベクトル
     */
    std::vector<Cluster> kMeansClustering(
        const std::vector<int>& feature_indices,
        int k,
        int max_iterations = 100
    );

    /**
     * @brief クラスタの重心を計算
     *
     * @param cluster クラスタ
     * @return 重心の特徴ベクトル
     */
    StateFeature computeCentroid(const Cluster& cluster) const;

    /**
     * @brief クラスタの代表状態を選出
     *
     * クラスタ内で最もスコアが高い状態を選ぶ
     *
     * @param cluster クラスタ
     */
    void selectRepresentative(Cluster& cluster);

    /**
     * @brief 状態のスコアを評価
     *
     * 自チームにとって有利な状態ほど高いスコアを返す
     *
     * @param state_id 状態のインデックス
     * @return スコア (高いほど有利)
     */
    float evaluateStateScore(int state_id) const;

    /**
     * @brief クラスタの内部分散を計算
     *
     * @param cluster クラスタ
     * @return 分散値
     */
    float computeVariance(const Cluster& cluster) const;

    // === ユーティリティ ===

    /**
     * @brief クラスタ数を調整
     *
     * 粗分類の結果、目標数より多い/少ない場合に調整
     */
    void adjustClusterCount();

    /**
     * @brief デバッグ情報を出力
     */
    void printDebugInfo() const;
};

#endif // _CLUSTERING_V2_H
