#ifndef ANIMATION_UTILS_HPP
#define ANIMATION_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
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

inline std::vector<double> centered_moving_average(const std::vector<double>& x, const int window)
{
    if (x.empty() || window <= 1) return x;

    const int n = static_cast<int>(x.size());
    const int half_left = window / 2;
    const int half_right = window - half_left - 1;

    std::vector<double> prefix(static_cast<std::size_t>(n + 1), 0.0);
    for (int i = 0; i < n; ++i) {
        prefix[static_cast<std::size_t>(i + 1)] =
            prefix[static_cast<std::size_t>(i)] + x[static_cast<std::size_t>(i)];
    }

    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        const int a = std::max(0, i - half_left);
        const int b = std::min(n, i + half_right + 1);
        out[static_cast<std::size_t>(i)] =
            (prefix[static_cast<std::size_t>(b)] - prefix[static_cast<std::size_t>(a)]) /
            static_cast<double>(b - a);
    }

    return out;
}

inline void block_mean_regular_time(const std::vector<int>& t,
                                    const std::vector<double>& y,
                                    const int window_block,
                                    std::vector<double>& t_center,
                                    std::vector<double>& j_w)
{
    t_center.clear();
    j_w.clear();
    if (window_block < 1) {
        throw std::runtime_error("block_mean_regular_time: window_block deve ser >= 1");
    }

    const int n = std::min(static_cast<int>(t.size()), static_cast<int>(y.size()));
    const int n_blocks = n / window_block;
    if (n_blocks <= 0) return;

    for (int k = 0; k < n_blocks; ++k) {
        const int i0 = k * window_block;
        const int i1 = i0 + window_block;
        double st = 0.0;
        double sy = 0.0;
        int count = 0;
        for (int i = i0; i < i1; ++i) {
            const double yi = y[static_cast<std::size_t>(i)];
            if (!std::isfinite(yi)) continue;
            st += static_cast<double>(t[static_cast<std::size_t>(i)]);
            sy += yi;
            ++count;
        }
        if (count == 0) continue;
        t_center.push_back(st / static_cast<double>(count));
        j_w.push_back(sy / static_cast<double>(count));
    }
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
    const double inv = 1.0 / static_cast<double>(ts.p_t.size());
    for (double& v : p_mean) v *= inv;
    return p_mean;
}

#endif // ANIMATION_UTILS_HPP
