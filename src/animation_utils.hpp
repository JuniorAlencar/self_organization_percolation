#ifndef ANIMATION_UTILS_HPP
#define ANIMATION_UTILS_HPP

#include <cmath>
#include <numeric>
#include <vector>

constexpr int ANIMATION_SPECIES_FACTOR = 10000000;

struct DecodedValue {
    bool never_activated = false;
    bool blocked = false;
    int color_1b = -1;
    int color_idx = -1;
    int time = -1;
};

inline DecodedValue decode_animation_value(const long long code, const int species_factor)
{
    DecodedValue out;
    if (code == -1) { out.never_activated = true; return out; }
    if (code == 0) { out.blocked = true; return out; }
    out.color_1b = static_cast<int>(code / species_factor);
    out.color_idx = out.color_1b - 1;
    out.time = static_cast<int>(code % species_factor);
    return out;
}

inline int color_to_active_value(const int num_colors, const int c) {
    return (num_colors == 1 ? 1 : (c + 2));
}

inline int color_to_negative_value(const int num_colors, const int c) {
    return (num_colors == 1 ? -1 : -(c + 2));
}

inline int value_to_color_index(const int num_colors, const int v) {
    return (num_colors == 1 ? 0 : (std::abs(v) - 2));
}

inline double mean_of(const std::vector<double>& x, const int begin, const int end)
{
    if (begin >= end) return 0.0;
    const double s = std::accumulate(x.begin() + begin, x.begin() + end, 0.0);
    return s / static_cast<double>(end - begin);
}

inline double std_of(const std::vector<double>& x, const int begin, const int end, const double mu)
{
    if (begin >= end) return 0.0;
    double acc = 0.0;
    for (int i = begin; i < end; ++i) {
        const double d = x[i] - mu;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(end - begin));
}

inline std::vector<double> moving_average(const std::vector<double>& x, const int window)
{
    if (x.empty() || window <= 1) return x;
    std::vector<double> out(x.size(), 0.0);
    double acc = 0.0;
    int left = 0;
    for (int right = 0; right < static_cast<int>(x.size()); ++right) {
        acc += x[right];
        while (right - left + 1 > window) {
            acc -= x[left];
            ++left;
        }
        out[right] = acc / static_cast<double>(right - left + 1);
    }
    return out;
}

inline std::vector<double> build_mean_p_series(const TimeSeries& ts)
{
    if (ts.t.empty()) throw std::runtime_error("build_mean_p_series: TimeSeries.t vazio");
    if (ts.num_colors <= 0) throw std::runtime_error("build_mean_p_series: TimeSeries.num_colors inválido");
    if (static_cast<int>(ts.p_t.size()) != ts.num_colors) throw std::runtime_error("build_mean_p_series: p_t.size() incompatível com num_colors");
    if (static_cast<int>(ts.f_t.size()) != ts.num_colors) throw std::runtime_error("build_mean_p_series: f_t.size() incompatível com num_colors");

    const std::size_t T = ts.t.size();
    std::vector<double> p_mean(T, 0.0);
    for (const auto& row : ts.p_t) {
        if (row.size() != T) throw std::runtime_error("build_mean_p_series: linhas de p_t com tamanhos diferentes de t");
        for (std::size_t i = 0; i < T; ++i) p_mean[i] += row[i];
    }
    return p_mean;
}

#endif // ANIMATION_UTILS_HPP
