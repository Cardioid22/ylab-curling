#include "eval.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace gpw {

// ---------------------------------------------------------------------------
// EvalParams
// ---------------------------------------------------------------------------
bool EvalParams::LoadFromFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    std::string key;
    while (in >> key) {
        if (key.empty() || key[0] == '#') { std::string rest; std::getline(in, rest); continue; }
        if (key == "end_dist") { for (auto& v : end_dist) in >> v; }
        else if (key == "hammer_base") in >> hammer_base;
        else if (key == "count_kappa") in >> count_kappa;
        else if (key == "quality_w") in >> quality_w;
        else if (key == "cover_bonus") in >> cover_bonus;
        else if (key == "exposed_penalty") in >> exposed_penalty;
        else if (key == "covered_penalty") in >> covered_penalty;
        else if (key == "sigma0") in >> sigma0;
        else if (key == "sigma_r") in >> sigma_r;
        else if (key == "guard_value") in >> guard_value;
        else if (key == "margin_w") in >> margin_w;
        else if (key == "model") in >> model_path;
        else { std::string rest; std::getline(in, rest); }
    }
    return true;
}

bool EvalParams::SaveToFile(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;
    out << "# gpw_agent evaluation parameters\n";
    out << "end_dist";
    for (double v : end_dist) out << " " << v;
    out << "\n";
    out << "hammer_base " << hammer_base << "\n" << "count_kappa " << count_kappa << "\n"
        << "quality_w " << quality_w << "\n" << "cover_bonus " << cover_bonus << "\n"
        << "exposed_penalty " << exposed_penalty << "\n" << "covered_penalty " << covered_penalty << "\n"
        << "sigma0 " << sigma0 << "\n" << "sigma_r " << sigma_r << "\n" << "guard_value " << guard_value << "\n"
        << "margin_w " << margin_w << "\n";
    if (!model_path.empty()) out << "model " << model_path << "\n";
    return true;
}

std::string EvalParams::Describe() const {
    std::ostringstream o;
    o << "hammer_base=" << hammer_base << " kappa=" << count_kappa << " quality_w=" << quality_w
      << " cover=" << cover_bonus << " exposed=" << exposed_penalty << " sigma0=" << sigma0
      << " sigma_r=" << sigma_r << (model_path.empty() ? "" : " model=" + model_path);
    return o.str();
}

// ---------------------------------------------------------------------------
// WinProbTable
// ---------------------------------------------------------------------------
WinProbTable::WinProbTable(const std::array<double, 9>& end_dist, int max_diff, int max_ends)
    : max_diff_(max_diff), max_ends_(max_ends), t_((2 * max_diff + 1) * (max_ends + 1), 0.5) {
    double p_pos = 0, p_neg = 0;
    for (int k = -4; k <= 4; ++k) {
        double p = end_dist[k + 4];
        if (k > 0) p_pos += p;
        if (k < 0) p_neg += p;
    }
    extra_hammer_wp_ = (p_pos + p_neg) > 0 ? p_pos / (p_pos + p_neg) : 0.5;

    // ends_left = 0: regulation finished.
    for (int d = -max_diff_; d <= max_diff_; ++d) {
        At(d, 0) = d > 0 ? 1.0 : (d < 0 ? 0.0 : extra_hammer_wp_);
    }
    // Recurrence for the hammer team; the non-hammer value is 1 - WP(-d, n, hammer).
    for (int n = 1; n <= max_ends_; ++n) {
        for (int d = -max_diff_; d <= max_diff_; ++d) {
            double v = 0;
            for (int k = -4; k <= 4; ++k) {
                double p = end_dist[k + 4];
                if (p <= 0) continue;
                int nd = std::clamp(d + k, -max_diff_, max_diff_);
                double w;
                if (k > 0) w = 1.0 - At(-nd, n - 1);   // we scored: opponent gets hammer
                else w = At(nd, n - 1);                  // steal or blank: we keep hammer
                v += p * w;
            }
            At(d, n) = v;
        }
    }
}

double WinProbTable::WP(int diff, int ends_left, bool hammer) const {
    ends_left = std::clamp(ends_left, 0, max_ends_);
    if (hammer) return At(std::clamp(diff, -max_diff_, max_diff_), ends_left);
    return 1.0 - At(std::clamp(-diff, -max_diff_, max_diff_), ends_left);
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------
int Evaluator::EndsLeft(const dc::GameState& s) const {
    if (s.end < setting_.max_end) return setting_.max_end - s.end;
    return 0;  // extra end in progress
}

namespace {

double NormalCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

}  // namespace

Evaluator::Evaluator(const EvalParams& params, const dc::GameSetting& setting)
    : p_(params), setting_(setting), wp_(params.end_dist) {
    if (!p_.model_path.empty()) {
        net_ = std::make_shared<ValueNet>();
        if (!net_->Load(p_.model_path)) {
            net_.reset();
            throw std::runtime_error("could not load value model: " + p_.model_path);
        }
    }
}

std::array<double, 9> Evaluator::EndDistribution(const dc::GameState& s) const {
    std::array<double, 9> hand = HandDistribution(s);
    if (has_model()) return net_->EndDist(EncodeFeatures(s, setting_.max_end), hand);
    return hand;
}

std::array<double, 9> Evaluator::HandDistribution(const dc::GameState& s) const {
    std::array<double, 9> p{};
    EndEstimate est = EstimateEnd(s);
    double cdf_prev = 0;
    for (int k = -4; k <= 4; ++k) {
        double cdf = (k == 4) ? 1.0 : NormalCdf((k + 0.5 - est.mean) / est.sigma);
        p[k + 4] = cdf - cdf_prev;
        cdf_prev = cdf;
    }
    return p;
}

Evaluator::EndEstimate Evaluator::EstimateEnd(const dc::GameState& s) const {
    EndEstimate e;
    auto stones = Stones(s);
    int h = TeamIdx(s.hammer);
    int c = CountNow(stones, h);
    e.count_now = c;

    int r = kShotsPerEnd - s.shot;  // shots remaining in the end (1..16)
    if (r < 1) r = 1;
    int r_h = (r + 1) / 2;          // hammer throws the last stone: shots 1,3,..15
    int r_n = r - r_h;

    // Persistence of the current count depends on how many stones the side
    // that wants to change it still has.
    int changer = (c > 0) ? r_n : (c < 0 ? r_h : std::max(r_h, r_n));
    double alpha = std::exp(-changer / p_.count_kappa);

    // Positional quality (hammer perspective).
    double pos = 0;
    for (const auto& st : stones) {
        bool mine = (st.team == h);
        int other_left = mine ? r_n : r_h;
        double removal = 1.0 - std::exp(-other_left / 2.0);
        if (st.in_house) {
            double q = std::clamp(1.0 - st.d / (kHouseR + kStoneR), 0.0, 1.0);
            q = std::pow(q, 1.5);
            bool cov = CoveredProxy(st, stones);
            double m = cov ? (1.0 + p_.cover_bonus) : 1.0;
            m *= 1.0 - (cov ? p_.covered_penalty : p_.exposed_penalty) * removal;
            pos += (mine ? 1.0 : -1.0) * q * m;
        } else if (st.in_fgz) {
            // Guards mostly help the side that is behind in shots (non-hammer)
            // early in the end; give a small credit to the owner.
            double g = p_.guard_value * (1.0 - std::exp(-r / 6.0));
            pos += (mine ? 1.0 : -1.0) * g;
        }
    }
    e.pos_term = pos;

    double base = (r_h > 0) ? p_.hammer_base : 0.0;
    double mean = alpha * c + (1.0 - alpha) * (base + p_.quality_w * pos);
    e.mean = std::clamp(mean, -4.0, 4.0);
    e.sigma = p_.sigma0 + p_.sigma_r * (r - 1);
    return e;
}

double Evaluator::Value(const dc::GameState& s, dc::Team me) const {
    if (s.IsGameOver()) {
        if (s.game_result->winner == me) return 1.0;
        if (s.game_result->winner == Opp(me)) return -1.0;
        return 0.0;
    }
    int diff = ScoreDiff(s, me);
    int ends_left = EndsLeft(s);
    bool hammer_me = (s.hammer == me);

    if (s.shot == 0) {
        // Start of an end: the table is exact for our model. A small margin term
        // keeps a gradient when the win probability saturates.
        return 2.0 * wp_.WP(diff, ends_left, hammer_me) - 1.0 + p_.margin_w * diff;
    }

    std::array<double, 9> dist = EndDistribution(s);
    double v = 0;
    for (int k = -4; k <= 4; ++k) {
        double p = dist[k + 4];
        if (p <= 1e-9) continue;
        int my_k = hammer_me ? k : -k;
        int nd = diff + my_k;
        bool next_hammer_me;
        if (k > 0) next_hammer_me = !hammer_me;      // hammer scored, loses hammer
        else if (k < 0) next_hammer_me = hammer_me;  // steal: hammer keeps hammer
        else next_hammer_me = hammer_me;             // blank
        int n_left = ends_left > 0 ? ends_left - 1 : 0;
        v += p * (2.0 * wp_.WP(nd, n_left, next_hammer_me) - 1.0 + p_.margin_w * nd);
    }
    return v;
}

}  // namespace gpw
