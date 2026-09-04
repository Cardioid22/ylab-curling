// Basic curling geometry, constants and light-weight board helpers.
// All coordinates are in the "shot" coordinate system used by
// dc::GameState::stones: origin at the hack, +y toward the far house.
#pragma once

#include "digitalcurling3/digitalcurling3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace gpw {

namespace dc = digitalcurling3;

// ---- sheet geometry (shot coordinates) -------------------------------------
constexpr float kTeeX = 0.f;
constexpr float kTeeY = 38.405f;                 // centre of the far house
constexpr float kHouseR = 1.829f;                // 12-foot radius
constexpr float kStoneR = 0.145f;
constexpr float kHogY = 32.004f;                 // far hog line
constexpr float kBackY = 40.234f;                // back line (stones beyond are out)
constexpr float kHalfW = 2.375f;                 // half sheet width
constexpr float kMaxSpeed = 4.0f;                // player speed cap
constexpr int kShotsPerEnd = 16;

inline float Dist(dc::Vector2 a, dc::Vector2 b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}
inline float DistTee(dc::Vector2 p) { return Dist(p, dc::Vector2(kTeeX, kTeeY)); }

// A stone "counts" while any part of it touches the house.
inline bool InHouse(dc::Vector2 p) { return DistTee(p) <= kHouseR + kStoneR; }

// Free guard zone: between the far hog line and the tee line, not in the house.
inline bool InFGZ(dc::Vector2 p) {
    return p.y > kHogY && p.y < kTeeY && !InHouse(p) && std::fabs(p.x) < kHalfW;
}

// ---- shot parameters --------------------------------------------------------
struct Shot {
    float vx = 0.f;
    float vy = 0.f;
    bool cw = false;  // true: clockwise rotation
};

inline dc::moves::Shot ToDc(const Shot& s) {
    dc::moves::Shot m;
    m.velocity = dc::Vector2(s.vx, s.vy);
    m.rotation = s.cw ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
    return m;
}

inline dc::moves::Shot::Rotation Rot(bool cw) {
    return cw ? dc::moves::Shot::Rotation::kCW : dc::moves::Shot::Rotation::kCCW;
}

// ---- board view ---------------------------------------------------------------
struct StoneRef {
    int team = 0;          // 0 / 1 (dc::Team index)
    int idx = 0;           // index within team
    dc::Vector2 p;
    float d = 0.f;         // distance to tee
    bool in_house = false;
    bool in_fgz = false;
};

// All stones currently in play, sorted by distance to the tee.
inline std::vector<StoneRef> Stones(const dc::GameState& s) {
    std::vector<StoneRef> v;
    v.reserve(16);
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < 8; ++i) {
            const auto& st = s.stones[t][i];
            if (!st) continue;
            StoneRef r;
            r.team = t;
            r.idx = i;
            r.p = st->position;
            r.d = DistTee(r.p);
            r.in_house = InHouse(r.p);
            r.in_fgz = InFGZ(r.p);
            v.push_back(r);
        }
    }
    std::sort(v.begin(), v.end(), [](const StoneRef& a, const StoneRef& b) { return a.d < b.d; });
    return v;
}

// Score if the end finished now, signed for `team` (positive = team counts).
inline int CountNow(const std::vector<StoneRef>& sorted, int team) {
    if (sorted.empty() || !sorted.front().in_house) return 0;
    int t = sorted.front().team;
    int n = 0;
    for (const auto& s : sorted) {
        if (!s.in_house || s.team != t) break;
        ++n;
    }
    return t == team ? n : -n;
}

inline int TeamIdx(dc::Team t) { return static_cast<int>(t); }
inline dc::Team Opp(dc::Team t) { return dc::GetOpponentTeam(t); }

inline int ScoreDiff(const dc::GameState& s, dc::Team me) {
    return static_cast<int>(s.GetTotalScore(me)) - static_cast<int>(s.GetTotalScore(Opp(me)));
}

// Number of shots `team` still throws in the current end (including the current one
// if it is `team`'s turn).
inline int MyShotsLeftInEnd(const dc::GameState& s, dc::Team team) {
    if (s.hammer == dc::Team::kInvalid) return 0;
    bool hammer = (s.hammer == team);
    int n = 0;
    for (int k = s.shot; k < kShotsPerEnd; ++k) {
        bool hammer_throws = (k % 2 == 1);
        if (hammer_throws == hammer) ++n;
    }
    return n;
}

inline std::string Fmt(float v, int prec = 3) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

}  // namespace gpw
