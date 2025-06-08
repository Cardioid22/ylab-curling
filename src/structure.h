#pragma once
#ifndef _STRUCTURE_H
#define _STRUCTURE_H

struct Position {
    float x;
    float y;
};
struct ShotInfo {
    float vx;
    float vy;
    int rot;
};
using LinkageRow = std::tuple<int, int, float, int>;
using LinkageMatrix = std::vector<LinkageRow>;

#endif
