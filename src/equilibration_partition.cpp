#include "equilibration_partition.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct GridRegular {
    int dim;
    int SX;
    int SY;
    int SZ;
    int grow_axis;
    int stride_y;
    int stride_z;
    int total_size;

    GridRegular(const int dim_, const int L)
        : dim(dim_),
          SX(L),
          SY(dim_ >= 2 ? L : 1),
          SZ(dim_ == 3 ? L : 1),
          grow_axis(dim_ - 1),
          stride_y(SX),
          stride_z(SX * (dim_ >= 2 ? L : 1)),
          total_size(SX * (dim_ >= 2 ? L : 1) * (dim_ == 3 ? L : 1)) {}

    inline int x_of(const int idx) const { return idx % SX; }
    inline int y_of(const int idx) const { return (idx / SX) % SY; }
    inline int z_of(const int idx) const { return idx / (SX * SY); }

    inline int grow_coord(const int idx) const {
        return (dim == 2) ? y_of(idx) : z_of(idx);
    }

    inline int neighbor_xm(const int idx) const {
        const int x = idx % SX;
        if (grow_axis == 0) return (x == 0) ? -1 : (idx - 1);
        return (x == 0) ? (idx + (SX - 1)) : (idx - 1);
    }

    inline int neighbor_xp(const int idx) const {
        const int x = idx % SX;
        if (grow_axis == 0) return (x == SX - 1) ? -1 : (idx + 1);
        return (x == SX - 1) ? (idx - (SX - 1)) : (idx + 1);
    }

    inline int neighbor_ym(const int idx) const {
        const int y = (idx / SX) % SY;
        if (grow_axis == 1) return (y == 0) ? -1 : (idx - SX);
        return (y == 0) ? (idx + SX * (SY - 1)) : (idx - SX);
    }

    inline int neighbor_yp(const int idx) const {
        const int y = (idx / SX) % SY;
        if (grow_axis == 1) return (y == SY - 1) ? -1 : (idx + SX);
        return (y == SY - 1) ? (idx - SX * (SY - 1)) : (idx + SX);
    }

    inline int neighbor_zm(const int idx) const {
        if (dim != 3) return -1;
        const int plane = SX * SY;
        const int z = idx / plane;
        return (z == 0) ? -1 : (idx - plane);
    }

    inline int neighbor_zp(const int idx) const {
        if (dim != 3) return -1;
        const int plane = SX * SY;
        const int z = idx / plane;
        return (z == SZ - 1) ? -1 : (idx + plane);
    }

    inline void for_each_neighbor(const int idx, const std::function<void(int)>& fn) const {
        const int xm = neighbor_xm(idx);
        const int xp = neighbor_xp(idx);
        if (xm >= 0) fn(xm);
        if (xp >= 0) fn(xp);

        if (dim >= 2) {
            const int ym = neighbor_ym(idx);
            const int yp = neighbor_yp(idx);
            if (ym >= 0) fn(ym);
            if (yp >= 0) fn(yp);
        }

        if (dim == 3) {
            const int zm = neighbor_zm(idx);
            const int zp = neighbor_zp(idx);
            if (zm >= 0) fn(zm);
            if (zp >= 0) fn(zp);
        }
    }
};

struct DecodedValue {
    bool never_activated = false;
    bool blocked = false;
    int color_1b = -1;
    int color_idx = -1;
    int time = -1;
};

DecodedValue decode_animation_value(const long long code, const int species_factor)
{
    DecodedValue out;

    if (code == -1) {
        out.never_activated = true;
        return out;
    }

    if (code == 0) {
        out.blocked = true;
        return out;
    }

    out.color_1b = static_cast<int>(code / species_factor);
    out.color_idx = out.color_1b - 1;
    out.time = static_cast<int>(code % species_factor);
    return out;
}

double mean_of(const std::vector<double>& x, const int begin, const int end)
{
    if (begin >= end) return 0.0;
    const double s = std::accumulate(x.begin() + begin, x.begin() + end, 0.0);
    return s / static_cast<double>(end - begin);
}

double std_of(const std::vector<double>& x, const int begin, const int end, const double mu)
{
    if (begin >= end) return 0.0;
    double acc = 0.0;
    for (int i = begin; i < end; ++i) {
        const double d = x[i] - mu;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(end - begin));
}

std::vector<double> moving_average(const std::vector<double>& x, const int window)
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

std::vector<double> build_mean_p_series(const TimeSeries& ts)
{
    if (ts.t.empty()) return {};
    if (ts.p_t.empty()) {
        throw std::runtime_error("build_mean_p_series: p_t vazio");
    }

    const std::size_t T = ts.t.size();
    std::vector<double> out(T, 0.0);

    for (const auto& row : ts.p_t) {
        if (row.size() != T) {
            throw std::runtime_error("build_mean_p_series: linhas de p_t com tamanhos diferentes");
        }
        for (std::size_t i = 0; i < T; ++i) {
            out[i] += row[i];
        }
    }

    const double inv = 1.0 / static_cast<double>(ts.p_t.size());
    for (double& v : out) v *= inv;

    return out;
}

bool is_active_color_site(const NetworkPattern& net,
                          const int idx,
                          const int color_idx,
                          const int species_factor,
                          int* time_out = nullptr)
{
    const long long code = static_cast<long long>(net.data[static_cast<std::size_t>(idx)]);
    const DecodedValue dv = decode_animation_value(code, species_factor);

    if (dv.never_activated || dv.blocked) return false;
    if (dv.color_idx != color_idx) return false;

    if (time_out) *time_out = dv.time;
    return true;
}

std::vector<int> largest_component_nodes_single_color(
    const NetworkPattern& net,
    const GridRegular& grid,
    const int color_idx,
    const int species_factor)
{
    std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
    std::vector<int> stack;
    std::vector<int> best_nodes;
    std::vector<int> comp_nodes;

    for (int idx = 0; idx < grid.total_size; ++idx) {
        if (visited[static_cast<std::size_t>(idx)]) continue;
        if (!is_active_color_site(net, idx, color_idx, species_factor)) continue;

        stack.clear();
        comp_nodes.clear();

        stack.push_back(idx);
        visited[static_cast<std::size_t>(idx)] = 1;

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            comp_nodes.push_back(u);

            grid.for_each_neighbor(u, [&](const int v) {
                if (v < 0) return;
                if (visited[static_cast<std::size_t>(v)]) return;
                if (!is_active_color_site(net, v, color_idx, species_factor)) return;

                visited[static_cast<std::size_t>(v)] = 1;
                stack.push_back(v);
            });
        }

        if (comp_nodes.size() > best_nodes.size()) {
            best_nodes = comp_nodes;
        }
    }

    return best_nodes;
}

} // namespace

int estimate_t_eq(const TimeSeries& ts, const EquilibrationConfig& cfg)
{
    if (ts.t.empty()) {
        throw std::runtime_error("estimate_t_eq: TimeSeries.t vazio");
    }

    const std::vector<double> p_mean = build_mean_p_series(ts);
    const std::vector<double> p_smoothed = moving_average(p_mean, cfg.smoothing_window);

    const int n = static_cast<int>(p_smoothed.size());
    const int tail_len = std::max(cfg.min_stable_steps, std::min(n, std::max(20, n / 5)));
    const int tail_begin = std::max(0, n - tail_len);

    const double ref_mean = mean_of(p_smoothed, tail_begin, n);
    const double ref_std = std_of(p_smoothed, tail_begin, n, ref_mean);

    const double tol = std::max(
        cfg.abs_tol,
        std::max(cfg.rel_tol * std::abs(ref_mean), cfg.sigma_multiplier * ref_std));

    const int stable_len = std::max(5, cfg.min_stable_steps);

    for (int i = 0; i < n; ++i) {
        const int end = std::min(n, i + stable_len);

        bool stable = true;
        for (int k = i; k < end; ++k) {
            if (std::abs(p_smoothed[k] - ref_mean) > tol) {
                stable = false;
                break;
            }
        }

        if (!stable) continue;

        const double suffix_mean = mean_of(p_smoothed, i, n);
        if (std::abs(suffix_mean - ref_mean) > tol) continue;

        return ts.t[static_cast<std::size_t>(i)];
    }

    return ts.t[static_cast<std::size_t>(tail_begin)];
}

void compute_equilibration_partition_metrics(
    const NetworkPattern& encoded_net,
    const TimeSeries& ts,
    PercolationSeries& ps,
    const EquilibrationConfig& cfg)
{
    if (encoded_net.shape.empty()) {
        throw std::runtime_error("compute_equilibration_partition_metrics: shape vazio");
    }

    const int L = encoded_net.shape[0];
    const GridRegular grid(encoded_net.dim, L);
    const int num_colors = encoded_net.num_colors;

    ps.t_eq = estimate_t_eq(ts, cfg);

    ps.sp_lin_preteq.assign(num_colors, -1);
    ps.sp_lin_posteq.assign(num_colors, -1);
    ps.M_size_preteq.assign(num_colors, 0);
    ps.M_size_posteq.assign(num_colors, 0);

    for (int c = 0; c < num_colors; ++c) {
        const std::vector<int> comp_nodes =
            largest_component_nodes_single_color(encoded_net, grid, c, cfg.species_factor);

        int m_pre = 0;
        int m_post = 0;

        for (const int idx : comp_nodes) {
            int t_site = -1;
            const bool ok = is_active_color_site(
                encoded_net, idx, c, cfg.species_factor, &t_site);

            if (!ok) {
                throw std::runtime_error(
                    "compute_equilibration_partition_metrics: nó inválido no maior componente");
            }

            if (t_site <= ps.t_eq) ++m_pre;
            else ++m_post;
        }

        ps.M_size_preteq[c] = m_pre;
        ps.M_size_posteq[c] = m_post;

        if (c >= static_cast<int>(ps.sp_path_lin.size()) || ps.sp_len[c] < 0) {
            continue;
        }

        const std::vector<int>& path = ps.sp_path_lin[c];
        if (path.empty()) {
            continue;
        }

        int sp_pre = 0;
        int sp_post = 0;

        for (std::size_t k = 1; k < path.size(); ++k) {
            const int idx = path[k];

            int t_site = -1;
            const bool ok = is_active_color_site(
                encoded_net, idx, c, cfg.species_factor, &t_site);

            if (!ok) {
                throw std::runtime_error(
                    "compute_equilibration_partition_metrics: caminho global contém nó inválido");
            }

            if (t_site <= ps.t_eq) ++sp_pre;
            else ++sp_post;
        }

        ps.sp_lin_preteq[c] = sp_pre;
        ps.sp_lin_posteq[c] = sp_post;
    }
}