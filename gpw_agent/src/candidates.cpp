#include "candidates.h"

#include <algorithm>
#include <cmath>

namespace gpw {

namespace {

bool InsideSheet(dc::Vector2 p) {
    return std::fabs(p.x) < kHalfW - kStoneR - 0.02f && p.y > kHogY + 0.3f && p.y < kBackY - 0.05f;
}

std::string XY(dc::Vector2 p) {
    return "(" + Fmt(p.x, 2) + "," + Fmt(p.y - kTeeY, 2) + ")";
}

}  // namespace

CandidateGenerator::CandidateGenerator(VelocitySolver& solver, const CandidateConfig& cfg)
    : vs_(solver), cfg_(cfg) {
    // Probe the simulator once to learn which way a CCW stone curls.
    Shot s = vs_.Draw(dc::Vector2(0.f, kTeeY), false);
    // If a CCW draw to x=0 needs a negative initial vx, the stone drifts to +x.
    ccw_curl_sign_ = (s.vx < 0.f) ? 1 : -1;
}

bool CandidateGenerator::RotFor(float drift_sign) const {
    // Want drift in direction drift_sign. CCW drifts ccw_curl_sign_.
    bool want_pos = drift_sign > 0.f;
    bool ccw_pos = ccw_curl_sign_ > 0;
    return want_pos != ccw_pos;  // cw when CCW drifts the wrong way
}

bool CandidateGenerator::InwardRot(float target_x) const {
    // Come in from the outside: drift toward the centre line.
    if (std::fabs(target_x) < 0.05f) return RotFor(1.f);
    return RotFor(target_x > 0.f ? -1.f : 1.f);
}

void CandidateGenerator::AddDraw(std::vector<Candidate>& out, dc::Vector2 target, bool cw, Kind kind, const std::string& label) const {
    if (!InsideSheet(target)) return;
    Candidate c;
    c.shot = vs_.Draw(target, cw);
    c.kind = kind;
    c.label = label + (cw ? " cw" : " ccw");
    out.push_back(c);
}

void CandidateGenerator::AddHit(std::vector<Candidate>& out, dc::Vector2 stone, float speed, float offset, bool cw, Kind kind, const std::string& label) const {
    // Approach direction ~ from the hack to the stone. Offset laterally.
    float len = stone.Length();
    if (len < 1.f) return;
    dc::Vector2 perp(-stone.y / len, stone.x / len);
    dc::Vector2 aim(stone.x + perp.x * offset, stone.y + perp.y * offset);
    Candidate c;
    c.shot = vs_.Solve(aim, speed, cw);
    c.kind = kind;
    c.label = label + (offset == 0.f ? "" : (offset > 0 ? " +off" : " -off")) + (cw ? " cw" : " ccw");
    out.push_back(c);
}

Candidate CandidateGenerator::Fallback(const dc::GameState&) const {
    Candidate c;
    c.shot = vs_.Draw(dc::Vector2(0.f, kTeeY), false);
    c.kind = Kind::Draw;
    c.label = "fallback draw tee";
    return c;
}

std::vector<Candidate> CandidateGenerator::Generate(const dc::GameState& s, dc::Team me) const {
    std::vector<Candidate> out;
    out.reserve(96);
    auto stones = Stones(s);
    int my = TeamIdx(me);
    bool fgz_protected = (s.shot < 5);  // five-rock rule: opponent FGZ stones may not be removed

    // ---- draws into the house (both rotations) ----------------------------
    const dc::Vector2 draw_targets[] = {
        {0.f, kTeeY},
        {0.f, kTeeY - 1.2f}, {0.f, kTeeY + 1.0f},
        {-0.6f, kTeeY}, {0.6f, kTeeY},
        {-1.0f, kTeeY - 0.7f}, {1.0f, kTeeY - 0.7f},
        {-1.2f, kTeeY + 0.4f}, {1.2f, kTeeY + 0.4f},
    };
    for (const auto& t : draw_targets) {
        AddDraw(out, t, false, Kind::Draw, "draw" + XY(t));
        AddDraw(out, t, true, Kind::Draw, "draw" + XY(t));
    }

    // ---- guards -------------------------------------------------------------
    const dc::Vector2 guard_targets[] = {
        {0.f, kTeeY - 2.7f}, {0.f, kTeeY - 3.8f},
        {-1.1f, kTeeY - 3.0f}, {1.1f, kTeeY - 3.0f},
    };
    for (const auto& t : guard_targets) {
        AddDraw(out, t, InwardRot(t.x), Kind::Guard, "guard" + XY(t));
    }
    // Guard my best house stone.
    for (const auto& st : stones) {
        if (st.team == my && st.in_house) {
            dc::Vector2 t(st.p.x * 0.75f, kTeeY - 2.9f);
            AddDraw(out, t, InwardRot(t.x), Kind::Guard, "cover" + XY(t));
            break;
        }
    }

    // ---- come-arounds behind guards ----------------------------------------
    int n_guards = 0;
    for (const auto& g : stones) {
        if (!g.in_fgz) continue;
        if (n_guards++ >= cfg_.max_comearound_guards) break;
        for (float dy : {1.3f, 2.2f}) {
            dc::Vector2 t(g.p.x * 0.9f, g.p.y + dy);
            if (t.y > kTeeY + 0.9f) continue;
            if (!InHouse(t) && t.y < kTeeY - kHouseR) continue;
            AddDraw(out, t, InwardRot(t.x), Kind::ComeAround, "around" + XY(t));
            if (std::fabs(t.x) < 0.4f) AddDraw(out, t, !InwardRot(t.x), Kind::ComeAround, "around" + XY(t));
        }
    }

    // ---- freezes on opponent house stones -----------------------------------
    int n_freeze = 0;
    for (const auto& st : stones) {
        if (st.team == my || !st.in_house) continue;
        if (n_freeze++ >= cfg_.max_freeze_targets) break;
        dc::Vector2 t(st.p.x, st.p.y - 2.f * kStoneR - 0.03f);
        AddDraw(out, t, false, Kind::Freeze, "freeze" + XY(st.p));
        AddDraw(out, t, true, Kind::Freeze, "freeze" + XY(st.p));
    }

    // ---- hits / peels on opponent stones ------------------------------------
    int n_hit = 0;
    for (const auto& st : stones) {
        if (st.team == my) continue;
        bool removable = st.in_house || (st.in_fgz && !fgz_protected) || (!st.in_house && !st.in_fgz);
        if (!removable) continue;
        std::string lab = "hit" + XY(st.p);
        if (st.in_house) {
            if (n_hit < cfg_.max_hit_targets) {
                AddHit(out, st.p, cfg_.hit_speed, 0.f, false, Kind::Hit, lab);
                AddHit(out, st.p, cfg_.hit_speed, 0.f, true, Kind::Hit, lab);
                bool rot = InwardRot(st.p.x);
                AddHit(out, st.p, cfg_.hit_speed, cfg_.hit_offset, rot, Kind::Hit, lab);
                AddHit(out, st.p, cfg_.hit_speed, -cfg_.hit_offset, rot, Kind::Hit, lab);
                AddHit(out, st.p, cfg_.peel_speed, 0.f, rot, Kind::Peel, "peel" + XY(st.p));
            } else {
                AddHit(out, st.p, cfg_.hit_speed, 0.f, InwardRot(st.p.x), Kind::Hit, lab);
            }
            ++n_hit;
        } else {
            // Guard: peel it, or tick it lightly.
            AddHit(out, st.p, cfg_.peel_speed, 0.f, InwardRot(st.p.x), Kind::Peel, "peel" + XY(st.p));
            AddHit(out, st.p, cfg_.hit_speed, cfg_.hit_offset, InwardRot(st.p.x), Kind::Peel, "peel" + XY(st.p));
        }
    }

    // ---- raises of my own guards into the house -----------------------------
    for (const auto& st : stones) {
        if (st.team != my || !st.in_fgz) continue;
        if (st.p.y < kTeeY - 4.0f) continue;
        AddHit(out, st.p, cfg_.tap_speed, 0.f, InwardRot(st.p.x), Kind::Raise, "raise" + XY(st.p));
    }
    // Tap-back my own front house stones toward the tee.
    for (const auto& st : stones) {
        if (st.team != my || !st.in_house) continue;
        if (st.p.y > kTeeY - 0.5f) continue;
        AddHit(out, st.p, cfg_.tap_speed * 0.6f, 0.f, InwardRot(st.p.x), Kind::Raise, "tap" + XY(st.p));
        break;
    }

    // ---- throw-through (blank / keep the house clean) -----------------------
    {
        float lane = 1.6f;
        bool left_clear = true, right_clear = true;
        for (const auto& st : stones) {
            if (std::fabs(st.p.x + lane) < 0.45f) left_clear = false;
            if (std::fabs(st.p.x - lane) < 0.45f) right_clear = false;
        }
        float x = right_clear ? lane : (left_clear ? -lane : 0.f);
        if (right_clear || left_clear) {
            Candidate c;
            c.shot = vs_.Solve(dc::Vector2(x, kBackY + 1.5f), 1.5f, InwardRot(x));
            c.kind = Kind::Through;
            c.label = "through";
            out.push_back(c);
        }
    }

    // Clamp velocities to the player's cap (the player would clamp anyway).
    for (auto& c : out) {
        float sp = std::sqrt(c.shot.vx * c.shot.vx + c.shot.vy * c.shot.vy);
        if (sp > kMaxSpeed) {
            c.shot.vx *= kMaxSpeed / sp;
            c.shot.vy *= kMaxSpeed / sp;
        }
    }
    return out;
}

std::vector<Candidate> CandidateGenerator::GenerateReplies(const dc::GameState& s, dc::Team mover) const {
    std::vector<Candidate> out;
    auto stones = Stones(s);
    int my = TeamIdx(mover);
    bool fgz_protected = (s.shot < 5);

    AddDraw(out, dc::Vector2(0.f, kTeeY), InwardRot(0.f), Kind::Draw, "r-draw tee");
    AddDraw(out, dc::Vector2(0.f, kTeeY + 0.9f), InwardRot(0.f), Kind::Draw, "r-draw back");

    // Come around the front-most guard.
    for (const auto& g : stones) {
        if (!g.in_fgz) continue;
        dc::Vector2 t(g.p.x * 0.9f, g.p.y + 1.6f);
        if (t.y <= kTeeY + 0.9f) AddDraw(out, t, InwardRot(t.x), Kind::ComeAround, "r-around");
        break;
    }
    // Hit the two best opponent stones; freeze the best one.
    int n = 0;
    for (const auto& st : stones) {
        if (st.team == my) continue;
        bool removable = st.in_house || !fgz_protected;
        if (!removable) continue;
        AddHit(out, st.p, cfg_.hit_speed, 0.f, InwardRot(st.p.x), Kind::Hit, "r-hit");
        if (n == 0 && st.in_house) {
            AddHit(out, st.p, cfg_.peel_speed, 0.f, InwardRot(st.p.x), Kind::Peel, "r-peel");
            dc::Vector2 t(st.p.x, st.p.y - 2.f * kStoneR - 0.03f);
            AddDraw(out, t, InwardRot(t.x), Kind::Freeze, "r-freeze");
        }
        if (++n >= 2) break;
    }
    // Throw-through (blank option for the hammer).
    {
        Candidate c;
        c.shot = vs_.Solve(dc::Vector2(1.6f, kBackY + 1.5f), 1.5f, InwardRot(1.6f));
        c.kind = Kind::Through;
        c.label = "r-through";
        out.push_back(c);
    }
    for (auto& c : out) {
        float sp = std::sqrt(c.shot.vx * c.shot.vx + c.shot.vy * c.shot.vy);
        if (sp > kMaxSpeed) { c.shot.vx *= kMaxSpeed / sp; c.shot.vy *= kMaxSpeed / sp; }
    }
    return out;
}

}  // namespace gpw
