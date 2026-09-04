// Per-shot thinking-time allocation.
#pragma once

#include "curling.h"

#include <algorithm>

namespace gpw {

struct TimeConfig {
    double safety = 0.85;        // fraction of the remaining clock we allow ourselves to plan with (0.90 left 14 s of 219 s)
    double fixed_overhead = 0.06;  // seconds reserved per shot for I/O, JSON, thread wake-ups
    double max_fraction = 0.35;  // never spend more than this fraction of the remaining clock on one shot
    double max_seconds = 20.0;   // absolute cap per shot
    double min_seconds = 0.03;
};

// Importance weights by my shot index within the end (0 = my first stone,
// 7 = my last stone). Normalised so the average is 1.
inline double ShotWeight(int my_shot_idx) {
    static const double w[8] = {0.55, 0.65, 0.80, 0.95, 1.10, 1.25, 1.50, 2.20};
    double mean = 0;
    for (double x : w) mean += x;
    mean /= 8.0;
    int i = std::clamp(my_shot_idx, 0, 7);
    return w[i] / mean;
}

// Returns the search budget (seconds) for the current shot.
inline double ShotBudgetSeconds(const dc::GameState& s, dc::Team me,
                                const dc::GameSetting& setting, const TimeConfig& cfg) {
    double remaining = s.thinking_time_remaining[TeamIdx(me)].count() / 1000.0;
    int left_this_end = MyShotsLeftInEnd(s, me);
    if (left_this_end < 1) left_this_end = 1;
    int ends_after = 0;
    if (s.end < setting.max_end) ends_after = setting.max_end - s.end - 1;  // extra ends reset the clock
    int my_total = left_this_end + 8 * ends_after;
    double base = remaining * cfg.safety / my_total;
    int my_idx = 8 - left_this_end;  // 0..7
    double budget = base * ShotWeight(my_idx) - cfg.fixed_overhead;
    budget = std::min(budget, remaining * cfg.max_fraction - cfg.fixed_overhead);
    budget = std::min(budget, cfg.max_seconds);
    budget = std::max(budget, cfg.min_seconds);
    return budget;
}

}  // namespace gpw
