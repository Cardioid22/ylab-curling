// Static position evaluation: board -> distribution of the end result ->
// win probability through a game-level table.
#pragma once

#include "curling.h"
#include "nn.h"

#include <memory>

#include <array>
#include <string>
#include <vector>

namespace gpw {

struct EvalParams {
    // Probability that the hammer team's end result is k = -4..4 on a neutral
    // (empty) board. Used to build the win-probability table and as the prior
    // for early-end positions. Replace with self-play statistics later.
    std::array<double, 9> end_dist = {0.00, 0.01, 0.05, 0.16, 0.12, 0.36, 0.22, 0.06, 0.02};
    double hammer_base = 0.75;    // expected end score of the hammer team on a neutral board
    double count_kappa = 2.5;     // persistence of the current count vs shots left for the trailing side
    double quality_w = 0.55;      // weight of the positional quality term
    double cover_bonus = 0.35;    // multiplier bonus for stones protected by a guard
    double exposed_penalty = 0.45;  // max discount for exposed stones the opponent can still remove
    double covered_penalty = 0.15;  // same for covered stones
    double sigma0 = 0.55;         // spread of the end-result distribution at 1 shot left
    double sigma_r = 0.06;        // additional spread per remaining shot
    double guard_value = 0.08;    // small credit for own guards early in the end
    double margin_w = 0.02;       // value bonus per point of score margin (anti-saturation)
    std::string model_path;       // learned end-result model (empty = hand-crafted evaluation)

    bool LoadFromFile(const std::string& path);
    std::string Describe() const;
};

// WP[diff][ends_left][hammer]. ends_left counts the ends still to be played
// including the one about to start; 0 means "regulation finished" (an extra
// end is played if tied).
class WinProbTable {
public:
    explicit WinProbTable(const std::array<double, 9>& end_dist, int max_diff = 24, int max_ends = 12);
    double WP(int diff, int ends_left, bool hammer) const;

private:
    int max_diff_, max_ends_;
    double extra_hammer_wp_ = 0.5;
    std::vector<double> t_;  // [(diff+max_diff) * (max_ends+1) + ends_left], hammer perspective
    double& At(int diff, int ends_left) { return t_[(diff + max_diff_) * (max_ends_ + 1) + ends_left]; }
    double At(int diff, int ends_left) const { return t_[(diff + max_diff_) * (max_ends_ + 1) + ends_left]; }
};

class Evaluator {
public:
    Evaluator(const EvalParams& params, const dc::GameSetting& setting);

    // Value for `me` in [-1, 1] (2*WP - 1).
    double Value(const dc::GameState& s, dc::Team me) const;

    struct EndEstimate {
        double mean = 0, sigma = 1;   // hammer-team end result
        int count_now = 0;            // signed for the hammer team
        double pos_term = 0;
    };
    EndEstimate EstimateEnd(const dc::GameState& s) const;

    const WinProbTable& table() const { return wp_; }
    const EvalParams& params() const { return p_; }
    bool has_model() const { return net_ && net_->loaded(); }
    std::string model_info() const { return has_model() ? net_->info() : std::string("hand-crafted"); }

    // End-result distribution (hammer perspective) used by Value().
    std::array<double, 9> EndDistribution(const dc::GameState& s) const;

private:
    EvalParams p_;
    dc::GameSetting setting_;
    WinProbTable wp_;
    std::shared_ptr<ValueNet> net_;
    int EndsLeft(const dc::GameState& s) const;
};

}  // namespace gpw
