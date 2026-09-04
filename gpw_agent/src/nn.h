// Tiny CPU inference for the learned end-result model (DeepSets over stones).
// The feature layout must stay in sync with python/train_value.py (FEATURE_VERSION).
#pragma once

#include "curling.h"

#include <array>
#include <string>
#include <vector>

namespace gpw {

constexpr int kNnStoneFeat = 6;
constexpr int kNnGlobalFeat = 10;
constexpr int kNnClasses = 9;  // hammer-team end result k = -4..4

struct NnFeatures {
    int n_stones = 0;
    float stone[16][kNnStoneFeat] = {};
    float global[kNnGlobalFeat] = {};
};

// Hammer-team perspective encoding of a mid-end state.
NnFeatures EncodeFeatures(const dc::GameState& s, int max_end);

struct Linear {
    int out = 0, in = 0;
    std::vector<float> w;  // out x in, row-major
    std::vector<float> b;  // out
    void Apply(const float* x, float* y) const;  // y = W x + b
};

class ValueNet {
public:
    bool Load(const std::string& path);
    bool loaded() const { return loaded_; }
    // Returns p(k) for k = -4..4 (index k + 4), hammer perspective.
    std::array<double, kNnClasses> EndDist(const NnFeatures& f) const;
    const std::string& info() const { return info_; }

private:
    bool loaded_ = false;
    std::string info_;
    Linear phi1_, phi2_, head1_, head2_, out_;
};

}  // namespace gpw
