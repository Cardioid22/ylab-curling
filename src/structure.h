#pragma once
#ifndef _STRUCTURE_H
#define _STRUCTURE_H

#include <tuple>
#include <vector>

// 物理シミュレーション (= dc::ApplyMove 1回) のスレッドローカル累計カウンタ。
//   - SimulatorWrapper::run_single_simulation (ロールアウト / 審判のリサンプル) と
//     ShotGenerator::simulateNoRand (ノード展開 = generatePool) の両方で ++ する。
//   - これで「総物理シミュ呼び出し回数」(展開コストも含む) を計上できる。
//     深さ5は展開ノードが増えるぶん simulateNoRand が増えるため、深さ間の
//     等予算比較を公平にするにはこの両方を数える必要がある (計算再投資実験)。
//   - thread_local なので、各ワーカースレッドが 1 局面を最後まで処理する設計と
//     合わせて、局面処理の前後で差分を取れば「その局面が消費した実シミュ回数」を
//     スレッド間の汚染なしに正確に取得できる (atomic 不要・競合なし)。
inline thread_local long long g_physics_sim_count = 0;

struct Position {
    float x;
    float y;
};

struct ShotInfo {
    float vx;
    float vy;
    int rot; // 1: CW, 0: CCW.
};

using LinkageRow = std::tuple<int, int, float, int>;
// <i, j, d, n>
// 	Cluster index i merged at this step
//  Cluster index j merged at this step
//  Distance between cluster i and j (merge cost)
// Number of samples in the new merged cluster
using LinkageMatrix = std::vector<LinkageRow>;

enum class NodeSource {
    Clustered,
    DeltaClustered,
    Random,
    AllGrid
};

#endif
