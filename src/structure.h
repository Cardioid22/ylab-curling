#pragma once
#ifndef _STRUCTURE_H
#define _STRUCTURE_H

#include <tuple>
#include <vector>

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
    Random
};

#endif
