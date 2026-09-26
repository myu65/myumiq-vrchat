#pragma once
#include <array>
#include <cmath>
#include <locale>
#include <sstream>
#include <string>

namespace myumiq {
struct Target {
    std::array<double, 3> position{0, 0, 1.6};
    std::array<double, 4> rotation{1, 0, 0, 0};
    double trigger = 0;
    double grip = 0;
};
inline std::array<double, 3> to_openvr(std::array<double, 3> p) {
    return {-p[1], p[2], -p[0]};
}
// v1 x y z qw qx qy qz trigger grip. Parsing is transactional.
inline bool parse(const std::string& request, Target& output) {
    if (request.size() > 512) return false;
    std::istringstream in(request);
    in.imbue(std::locale::classic());
    std::string version, extra;
    Target t;
    if (!(in >> version) || version != "v1") return false;
    for (auto& v : t.position) if (!(in >> v) || !std::isfinite(v) || std::abs(v) > 10) return false;
    double norm = 0;
    for (auto& v : t.rotation) {
        if (!(in >> v) || !std::isfinite(v)) return false;
        norm += v * v;
    }
    if (std::abs(norm - 1) > 0.01) return false;
    if (!(in >> t.trigger >> t.grip) || !std::isfinite(t.trigger) || !std::isfinite(t.grip)
        || t.trigger < 0 || t.trigger > 1 || t.grip < 0 || t.grip > 1 || (in >> extra)) return false;
    for (auto& v : t.rotation) v /= std::sqrt(norm);
    output = t;
    return true;
}
}
