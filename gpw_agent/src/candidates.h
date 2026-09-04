// Shot vocabulary: turns a board into a list of concrete candidate shots.
#pragma once

#include "curling.h"
#include "sim.h"

#include <string>
#include <vector>

namespace gpw {

enum class Kind { Draw, Guard, ComeAround, Freeze, Hit, Peel, Raise, Through };

struct Candidate {
    Shot shot;
    Kind kind = Kind::Draw;
    std::string label;
};

struct CandidateConfig {
    float tap_speed = 0.9f;     // residual speed at the target for raises / taps
    float hit_speed = 2.2f;     // normal take-out weight
    float peel_speed = 3.3f;    // peel weight
    float hit_offset = 0.09f;   // lateral aim offset for hit-and-roll variants (m)
    int max_hit_targets = 3;    // opponent stones that get the full hit menu
    int max_freeze_targets = 2;
    int max_comearound_guards = 3;
};

class CandidateGenerator {
public:
    CandidateGenerator(VelocitySolver& solver, const CandidateConfig& cfg = CandidateConfig());

    // Full menu for the side to move (`me`).
    std::vector<Candidate> Generate(const dc::GameState& s, dc::Team me) const;

    // Small reply menu used for the opponent in shallow look-ahead.
    std::vector<Candidate> GenerateReplies(const dc::GameState& s, dc::Team mover) const;

    // A safe default shot (draw to the tee) for fallbacks.
    Candidate Fallback(const dc::GameState& s) const;

    // Sign of the lateral drift of a CCW stone (+1: curls to +x).
    int ccw_curl_sign() const { return ccw_curl_sign_; }

private:
    VelocitySolver& vs_;
    CandidateConfig cfg_;
    int ccw_curl_sign_ = 1;

    // Rotation that curls toward negative x if want_neg_x, else toward +x.
    bool RotFor(float drift_sign) const;
    // Rotation whose curl brings the stone in from the outside toward target x.
    bool InwardRot(float target_x) const;

    void AddDraw(std::vector<Candidate>& out, dc::Vector2 target, bool cw, Kind kind, const std::string& label) const;
    void AddHit(std::vector<Candidate>& out, dc::Vector2 stone, float speed, float offset, bool cw, Kind kind, const std::string& label) const;
};

}  // namespace gpw
