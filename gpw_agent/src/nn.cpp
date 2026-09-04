#include "nn.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace gpw {

NnFeatures EncodeFeatures(const dc::GameState& s, int max_end) {
    NnFeatures f;
    int h = TeamIdx(s.hammer);
    auto stones = Stones(s);
    int n = 0, nh_house = 0, nn_house = 0;
    for (const auto& st : stones) {
        if (n >= 16) break;
        bool mine = (st.team == h);
        float* v = f.stone[n];
        v[0] = st.p.x;
        v[1] = st.p.y - kTeeY;
        v[2] = mine ? 1.f : -1.f;
        v[3] = st.in_house ? 1.f : 0.f;
        v[4] = st.d;
        v[5] = st.in_fgz ? 1.f : 0.f;
        if (st.in_house) { if (mine) ++nh_house; else ++nn_house; }
        ++n;
    }
    f.n_stones = n;
    int r = kShotsPerEnd - s.shot;
    int r_h = (r + 1) / 2;
    int r_n = r - r_h;
    int c = CountNow(stones, h);
    int diff_h = static_cast<int>(s.GetTotalScore(s.hammer)) - static_cast<int>(s.GetTotalScore(Opp(s.hammer)));
    int ends_left = (s.end < max_end) ? (max_end - s.end) : 0;
    f.global[0] = r / 16.f;
    f.global[1] = r_h / 8.f;
    f.global[2] = r_n / 8.f;
    f.global[3] = c / 4.f;
    f.global[4] = nh_house / 8.f;
    f.global[5] = nn_house / 8.f;
    f.global[6] = n / 16.f;
    f.global[7] = (s.shot < 5) ? 1.f : 0.f;
    f.global[8] = std::clamp(diff_h, -6, 6) / 6.f;
    f.global[9] = ends_left / 10.f;
    return f;
}

void Linear::Apply(const float* x, float* y) const {
    for (int o = 0; o < out; ++o) {
        const float* row = &w[static_cast<size_t>(o) * in];
        float acc = b[o];
        for (int i = 0; i < in; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

namespace {
bool ReadLinear(std::istream& in, const std::string& expect_name, Linear& L) {
    std::string name;
    if (!(in >> name >> L.out >> L.in)) return false;
    if (name != expect_name) return false;
    L.w.resize(static_cast<size_t>(L.out) * L.in);
    L.b.resize(L.out);
    for (auto& v : L.w) if (!(in >> v)) return false;
    for (auto& v : L.b) if (!(in >> v)) return false;
    return true;
}
inline void Relu(float* x, int n) { for (int i = 0; i < n; ++i) x[i] = x[i] > 0.f ? x[i] : 0.f; }
}  // namespace

bool ValueNet::Load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    std::string magic;
    in >> magic;
    if (magic != "gpw_value_v1") return false;
    int F, G, K;
    std::string tag;
    in >> tag >> F >> tag >> G >> tag >> K;
    if (F != kNnStoneFeat || G != kNnGlobalFeat || K != kNnClasses) return false;
    if (!ReadLinear(in, "phi1", phi1_)) return false;
    if (!ReadLinear(in, "phi2", phi2_)) return false;
    if (!ReadLinear(in, "head1", head1_)) return false;
    if (!ReadLinear(in, "head2", head2_)) return false;
    if (!ReadLinear(in, "out", out_)) return false;
    if (phi1_.in != F || head1_.in != 2 * phi2_.out + G || out_.out != K) return false;
    std::ostringstream o;
    o << path << " (phi " << phi1_.out << "/" << phi2_.out << ", head " << head1_.out << "/" << head2_.out << ")";
    info_ = o.str();
    loaded_ = true;
    return true;
}

std::array<double, kNnClasses> ValueNet::EndDist(const NnFeatures& f) const {
    const int H = phi2_.out;
    std::vector<float> h1(phi1_.out), h2(H), sum(H, 0.f), mx(H, -1e30f);
    for (int i = 0; i < f.n_stones; ++i) {
        phi1_.Apply(f.stone[i], h1.data());
        Relu(h1.data(), phi1_.out);
        phi2_.Apply(h1.data(), h2.data());
        Relu(h2.data(), H);
        for (int j = 0; j < H; ++j) { sum[j] += h2[j]; mx[j] = std::max(mx[j], h2[j]); }
    }
    if (f.n_stones == 0) std::fill(mx.begin(), mx.end(), 0.f);
    std::vector<float> x(2 * H + kNnGlobalFeat);
    for (int j = 0; j < H; ++j) { x[j] = sum[j]; x[H + j] = mx[j]; }
    for (int j = 0; j < kNnGlobalFeat; ++j) x[2 * H + j] = f.global[j];
    std::vector<float> a(head1_.out), b(head2_.out), z(kNnClasses);
    head1_.Apply(x.data(), a.data());
    Relu(a.data(), head1_.out);
    head2_.Apply(a.data(), b.data());
    Relu(b.data(), head2_.out);
    out_.Apply(b.data(), z.data());
    float m = *std::max_element(z.begin(), z.end());
    double tot = 0;
    std::array<double, kNnClasses> p{};
    for (int k = 0; k < kNnClasses; ++k) { p[k] = std::exp(z[k] - m); tot += p[k]; }
    for (auto& v : p) v /= tot;
    return p;
}

}  // namespace gpw
