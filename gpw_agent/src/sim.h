// Physics access: per-thread simulator instances and the inverse-velocity solver.
#pragma once

#include "curling.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace gpw {

// One simulator + one noisy player + one noiseless player. Not thread-safe:
// create one instance per worker thread.
class Sim {
public:
    Sim(const dc::GameSetting& setting,
        const dc::ISimulatorFactory& sim_factory,
        const dc::IPlayerFactory& player_factory);

    // Applies `shot` to a copy of `state` and returns the resulting state.
    // noisy=true adds the tournament execution noise (normal_dist player).
    dc::GameState Apply(const dc::GameState& state, const Shot& shot, bool noisy);
    // Same, but charges `thinking_ms` to the mover's clock (referee use).
    dc::GameState ApplyTimed(const dc::GameState& state, const Shot& shot, bool noisy, long long thinking_ms);

    const dc::GameSetting& setting() const { return setting_; }

private:
    dc::GameSetting setting_;
    std::unique_ptr<dc::ISimulator> sim_;
    std::unique_ptr<dc::IPlayer> noisy_player_;
    std::unique_ptr<dc::IPlayer> exact_player_;
};

// Inverse kinematics for the FCV1 simulator: initial velocity such that the
// stone passes through `target` (shot coordinates) with speed `target_speed`.
// target_speed = 0 gives a draw that stops at `target`.
// Thread-safe; results of the expensive drift simulation are cached.
class VelocitySolver {
public:
    Shot Solve(dc::Vector2 target, float target_speed, bool cw);

    // Convenience wrappers.
    Shot Draw(dc::Vector2 target, bool cw) { return Solve(target, 0.f, cw); }

    size_t CacheSize();

private:
    static float InitialSpeed(float target_r, float target_speed);
    dc::Vector2 Drift(float v0_speed, float target_speed, bool cw);

    std::mutex m_;
    std::unordered_map<std::uint64_t, dc::Vector2> cache_;
};

}  // namespace gpw
