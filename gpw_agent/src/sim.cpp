#include "sim.h"

#include <cassert>
#include <cmath>

namespace gpw {

// ---------------------------------------------------------------------------
// Sim
// ---------------------------------------------------------------------------
Sim::Sim(const dc::GameSetting& setting,
         const dc::ISimulatorFactory& sim_factory,
         const dc::IPlayerFactory& player_factory)
    : setting_(setting) {
    sim_ = sim_factory.CreateSimulator();
    noisy_player_ = player_factory.CreatePlayer();
    exact_player_ = dc::players::PlayerIdenticalFactory().CreatePlayer();
}

dc::GameState Sim::Apply(const dc::GameState& state, const Shot& shot, bool noisy) {
    dc::GameState next = state;
    if (next.IsGameOver()) return next;
    dc::Move move{ToDc(shot)};
    dc::ApplyMove(setting_, *sim_, noisy ? *noisy_player_ : *exact_player_, next, move,
                  std::chrono::milliseconds(0));
    return next;
}

dc::GameState Sim::ApplyTimed(const dc::GameState& state, const Shot& shot, bool noisy, long long thinking_ms) {
    dc::GameState next = state;
    if (next.IsGameOver()) return next;
    dc::Move move{ToDc(shot)};
    dc::ApplyMove(setting_, *sim_, noisy ? *noisy_player_ : *exact_player_, next, move,
                  std::chrono::milliseconds(thinking_ms));
    return next;
}

// ---------------------------------------------------------------------------
// VelocitySolver
// Piecewise fits of the FCV1 stopping-distance curve (initial speed as a
// function of travel distance and residual speed), followed by one single-stone
// simulation to measure the lateral drift caused by curl.
// ---------------------------------------------------------------------------
float VelocitySolver::InitialSpeed(float target_r, float target_speed) {
    // The low-speed fits contain log(target_r - 29.9): keep the argument positive.
    if (target_speed <= 1.f && target_r < 30.5f) target_r = 30.5f;
    if (target_speed <= 0.05f) {
        constexpr float kC0[] = {0.0005048122574925176f, 0.2756242531609261f};
        constexpr float kC1[] = {0.00046669575066030805f, -29.898958358378636f, -0.0014030973174948508f};
        constexpr float kC2[] = {0.13968687866736632f, 0.41120940058777616f};
        float c0 = kC0[0] * target_r + kC0[1];
        float c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
        float c2 = kC2[0] * target_r + kC2[1];
        return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
    } else if (target_speed <= 1.f) {
        constexpr float kC0[] = {-0.0014309170115803444f, 0.9858457898438147f};
        constexpr float kC1[] = {-0.0008339331735471273f, -29.86751291726946f, -0.19811799977982522f};
        constexpr float kC2[] = {0.13967323742978f, 0.42816312110477517f};
        float c0 = kC0[0] * target_r + kC0[1];
        float c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
        float c2 = kC2[0] * target_r + kC2[1];
        return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
    } else {
        constexpr float kC0[] = {1.0833113118071224e-06f, -0.00012132851917870833f, 0.004578093297561233f, 0.9767006869364527f};
        constexpr float kC1[] = {0.07950648211492622f, -8.228225657195706f, -0.05601306077702578f};
        constexpr float kC2[] = {0.14140440186382008f, 0.3875782508767419f};
        float c0 = kC0[0] * target_r * target_r * target_r + kC0[1] * target_r * target_r + kC0[2] * target_r + kC0[3];
        float c1 = -kC1[0] * std::log(target_r + kC1[1]) + kC1[2];
        float c2 = kC2[0] * target_r + kC2[1];
        return std::sqrt(c0 * target_speed * target_speed + c1 * target_speed + c2);
    }
}

dc::Vector2 VelocitySolver::Drift(float v0_speed, float target_speed, bool cw) {
    // Quantise the key so that nearby requests share one simulation.
    std::uint64_t kv = static_cast<std::uint64_t>(std::lround(v0_speed * 500.f));      // 2 mm/s
    std::uint64_t kt = static_cast<std::uint64_t>(std::lround(target_speed * 100.f));  // 1 cm/s
    std::uint64_t key = (kv << 24) | (kt << 1) | (cw ? 1u : 0u);
    {
        std::lock_guard<std::mutex> lk(m_);
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second;
    }
    float qv = static_cast<float>(kv) / 500.f;
    float qt = static_cast<float>(kt) / 100.f;
    float rotation_factor = cw ? -1.f : 1.f;

    thread_local std::unique_ptr<dc::ISimulator> sim;
    if (!sim) sim = dc::simulators::SimulatorFCV1Factory().CreateSimulator();

    dc::ISimulator::AllStones stones;
    stones[0].emplace(dc::Vector2(), 0.f, dc::Vector2(0.f, qv), 1.57f * rotation_factor);
    sim->SetStones(stones);
    dc::Vector2 delta;
    int steps = 0;
    while (true) {
        const auto& st = sim->GetStones();
        if (!st[0]) { delta = dc::Vector2(0.f, 0.f); break; }
        float speed = st[0]->linear_velocity.Length();
        if (!(speed > qt) || sim->AreAllStonesStopped()) { delta = st[0]->position; break; }
        if (++steps > 200000) { delta = st[0]->position; break; }  // safety net (200 s of simulated time)
        sim->Step();
    }
    if (!std::isfinite(delta.x) || !std::isfinite(delta.y)) delta = dc::Vector2(0.f, 0.f);
    std::lock_guard<std::mutex> lk(m_);
    cache_.emplace(key, delta);
    return delta;
}

Shot VelocitySolver::Solve(dc::Vector2 target, float target_speed, bool cw) {
    if (target_speed < 0.f) target_speed = 0.f;
    float target_r = target.Length();
    if (target_r < 1.f) target_r = 1.f;
    float v0 = InitialSpeed(target_r, target_speed);
    if (!std::isfinite(v0) || v0 <= 0.f) v0 = 2.4f;  // never propagate NaN into the physics
    if (v0 > kMaxSpeed) v0 = kMaxSpeed;
    dc::Vector2 delta = Drift(v0, target_speed, cw);
    float delta_angle = std::atan2(delta.x, delta.y);
    float target_angle = std::atan2(target.y, target.x);
    float v0_angle = target_angle + delta_angle;
    Shot s;
    s.vx = v0 * std::cos(v0_angle);
    s.vy = v0 * std::sin(v0_angle);
    s.cw = cw;
    return s;
}

size_t VelocitySolver::CacheSize() {
    std::lock_guard<std::mutex> lk(m_);
    return cache_.size();
}

}  // namespace gpw
