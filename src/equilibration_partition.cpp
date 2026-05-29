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

std::vector<double> centered_moving_average(const std::vector<double>& x, const int window)
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

void block_mean_regular_time(const std::vector<int>& t,
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

void validate_timeseries_layout(const TimeSeries& ts,
                                const std::string& caller,
                                const bool require_f_t)
{
    if (ts.t.empty()) {
        throw std::runtime_error(caller + ": TimeSeries.t vazio");
    }

    if (ts.num_colors <= 0) {
        throw std::runtime_error(caller + ": TimeSeries.num_colors inválido");
    }

    if (static_cast<int>(ts.p_t.size()) != ts.num_colors) {
        throw std::runtime_error(caller + ": p_t.size() incompatível com num_colors");
    }

    if (require_f_t && static_cast<int>(ts.f_t.size()) != ts.num_colors) {
        throw std::runtime_error(caller + ": f_t.size() incompatível com num_colors");
    }

    const std::size_t T = ts.t.size();

    for (const auto& row : ts.p_t) {
        if (row.size() != T) {
            throw std::runtime_error(caller + ": linhas de p_t com tamanhos diferentes de t");
        }
    }

    if (require_f_t) {
        for (const auto& row : ts.f_t) {
            if (row.size() != T) {
                throw std::runtime_error(caller + ": linhas de f_t com tamanhos diferentes de t");
            }
        }
    }
}

std::vector<double> build_mean_p_series(const TimeSeries& ts)
{
    validate_timeseries_layout(ts, "build_mean_p_series", true);

    const std::size_t T = ts.t.size();
    std::vector<double> out(T, 0.0);

    for (const auto& row : ts.p_t) {
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

std::vector<Point3D> extract_top_surface_points(
    const NetworkPattern& net,
    const int species_factor)
{
    if (net.shape.empty()) {
        throw std::runtime_error("extract_top_surface_points: shape vazio");
    }

    const int L = net.shape[0];
    const GridRegular grid(net.dim, L);

    auto is_active_site = [&](const int idx, int* color_idx_out = nullptr) -> bool {
        if (idx < 0 || idx >= grid.total_size) {
            return false;
        }

        const long long code =
            static_cast<long long>(net.data[static_cast<std::size_t>(idx)]);

        const DecodedValue dv = decode_animation_value(code, species_factor);

        if (dv.never_activated || dv.blocked) {
            return false;
        }

        if (color_idx_out) {
            *color_idx_out = dv.color_idx;
        }

        return true;
    };

    struct TopCell {
        bool valid = false;
        int idx = -1;
        int z = -1;
    };

    if (net.dim == 2) {
        std::vector<TopCell> top_by_x(static_cast<std::size_t>(grid.SX));

        for (int idx = 0; idx < grid.total_size; ++idx) {
            int color_idx = -1;
            if (!is_active_site(idx, &color_idx)) {
                continue;
            }

            const int x = grid.x_of(idx);
            const int y = grid.y_of(idx);

            TopCell& cell = top_by_x[static_cast<std::size_t>(x)];
            if (!cell.valid || y > cell.z) {
                cell.valid = true;
                cell.idx = idx;
                cell.z = y;
            }
        }

        std::vector<Point3D> out;
        out.reserve(static_cast<std::size_t>(grid.SX));

        for (const auto& cell : top_by_x) {
            if (!cell.valid) continue;

            int color_idx = -1;
            is_active_site(cell.idx, &color_idx);

            Point3D p;
            p.x = grid.x_of(cell.idx);
            p.y = grid.y_of(cell.idx);
            p.z = 0;
            p.color_index = color_idx;
            out.push_back(p);
        }

        return out;
    }

    std::vector<TopCell> top_by_xy(static_cast<std::size_t>(grid.SX * grid.SY));

    for (int idx = 0; idx < grid.total_size; ++idx) {
        int color_idx = -1;
        if (!is_active_site(idx, &color_idx)) {
            continue;
        }

        const int x = grid.x_of(idx);
        const int y = grid.y_of(idx);
        const int z = grid.z_of(idx);

        const int id_xy = x + grid.SX * y;

        TopCell& cell = top_by_xy[static_cast<std::size_t>(id_xy)];
        if (!cell.valid || z > cell.z) {
            cell.valid = true;
            cell.idx = idx;
            cell.z = z;
        }
    }

    std::vector<Point3D> out;
    out.reserve(static_cast<std::size_t>(grid.SX * grid.SY));

    for (const auto& cell : top_by_xy) {
        if (!cell.valid) continue;

        int color_idx = -1;
        is_active_site(cell.idx, &color_idx);

        Point3D p;
        p.x = grid.x_of(cell.idx);
        p.y = grid.y_of(cell.idx);
        p.z = grid.z_of(cell.idx);
        p.color_index = color_idx;
        out.push_back(p);
    }

    return out;
}

} // namespace

double estimate_t_eq(const TimeSeries& ts, const EquilibrationConfig& cfg)
{
    validate_timeseries_layout(ts, "estimate_t_eq", true);

    const std::vector<double> p_mean = build_mean_p_series(ts);
    const std::vector<double> p_smoothed = centered_moving_average(p_mean, cfg.smoothing_window);

    std::vector<double> t_j;
    std::vector<double> j_w;
    block_mean_regular_time(ts.t, p_smoothed, cfg.window_block, t_j, j_w);
    if (j_w.size() < 3) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int ns = static_cast<int>(j_w.size()) - 1;
    std::vector<double> s(static_cast<std::size_t>(ns), 0.0);
    std::vector<double> t_s(static_cast<std::size_t>(ns), 0.0);
    for (int i = 0; i < ns; ++i) {
        s[static_cast<std::size_t>(i)] =
            std::abs(j_w[static_cast<std::size_t>(i + 1)] - j_w[static_cast<std::size_t>(i)]);
        t_s[static_cast<std::size_t>(i)] =
            0.5 * (t_j[static_cast<std::size_t>(i)] + t_j[static_cast<std::size_t>(i + 1)]);
    }

    if (ns < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    for (int i = 0; i < ns; ++i) {
        double sp = std::numeric_limits<double>::quiet_NaN();
        if (i == 0) {
            const double dt = t_s[1] - t_s[0];
            if (dt != 0.0) sp = (s[1] - s[0]) / dt;
        } else if (i == ns - 1) {
            const double dt = t_s[static_cast<std::size_t>(i)] - t_s[static_cast<std::size_t>(i - 1)];
            if (dt != 0.0) {
                sp = (s[static_cast<std::size_t>(i)] - s[static_cast<std::size_t>(i - 1)]) / dt;
            }
        } else {
            const double dt =
                t_s[static_cast<std::size_t>(i + 1)] - t_s[static_cast<std::size_t>(i - 1)];
            if (dt != 0.0) {
                sp = (s[static_cast<std::size_t>(i + 1)] - s[static_cast<std::size_t>(i - 1)]) / dt;
            }
        }

        if (std::isfinite(sp) && sp < cfg.s_prime_threshold) {
            return t_s[static_cast<std::size_t>(i)];
        }
    }

    return std::numeric_limits<double>::quiet_NaN();
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

    validate_timeseries_layout(ts, "compute_equilibration_partition_metrics", true);

    const int L = encoded_net.shape[0];
    const GridRegular grid(encoded_net.dim, L);
    const int num_colors = encoded_net.num_colors;

    if (ts.num_colors != num_colors) {
        throw std::runtime_error(
            "compute_equilibration_partition_metrics: TimeSeries.num_colors incompatível com a rede");
    }

    ps.t_eq = std::isfinite(ts.t_eq) ? ts.t_eq : estimate_t_eq(ts, cfg);

    ps.sp_lin_preteq.assign(num_colors, -1);
    ps.sp_lin_posteq.assign(num_colors, -1);
    ps.M_size_preteq.assign(num_colors, 0);
    ps.M_size_posteq.assign(num_colors, 0);

    if (!std::isfinite(ps.t_eq)) {
        return;
    }

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

        if (c >= static_cast<int>(ps.sp_len.size()) ||
            c >= static_cast<int>(ps.sp_path_lin.size()) ||
            ps.sp_len[c] < 0) {
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

EquilibrationCutNetworks build_equilibration_cut_networks(
    const NetworkPattern& encoded_net,
    const double t_eq,
    const int species_factor)
{
    if (encoded_net.shape.empty()) {
        throw std::runtime_error(
            "build_equilibration_cut_networks: shape vazio");
    }

    if (!std::isfinite(t_eq)) {
        throw std::runtime_error(
            "build_equilibration_cut_networks: t_eq inválido");
    }

    if (species_factor <= 0) {
        throw std::runtime_error(
            "build_equilibration_cut_networks: species_factor deve ser > 0");
    }

    NetworkPattern pre_net(
        encoded_net.dim,
        encoded_net.shape,
        encoded_net.num_colors,
        encoded_net.rho
    );

    NetworkPattern post_net(
        encoded_net.dim,
        encoded_net.shape,
        encoded_net.num_colors,
        encoded_net.rho
    );

    pre_net.seed = encoded_net.seed;
    post_net.seed = encoded_net.seed;

    const std::size_t total_size = encoded_net.data.size();
    pre_net.data.assign(total_size, static_cast<NetworkPattern::state_t>(-1));
    post_net.data.assign(total_size, static_cast<NetworkPattern::state_t>(-1));

    for (std::size_t i = 0; i < total_size; ++i) {
        const long long code =
            static_cast<long long>(encoded_net.data[i]);

        if (code == -1) {
            pre_net.data[i] = static_cast<NetworkPattern::state_t>(-1);
            post_net.data[i] = static_cast<NetworkPattern::state_t>(-1);
            continue;
        }

        if (code == 0) {
            pre_net.data[i] = static_cast<NetworkPattern::state_t>(0);
            post_net.data[i] = static_cast<NetworkPattern::state_t>(0);
            continue;
        }

        const int color_1b = static_cast<int>(code / species_factor);
        const int time = static_cast<int>(code % species_factor);

        if (color_1b <= 0 || color_1b > encoded_net.num_colors) {
            throw std::runtime_error(
                "build_equilibration_cut_networks: código de espécie inválido");
        }

        if (time <= t_eq) {
            pre_net.data[i] = encoded_net.data[i];
            post_net.data[i] = static_cast<NetworkPattern::state_t>(-1);
        } else {
            pre_net.data[i] = static_cast<NetworkPattern::state_t>(-1);
            post_net.data[i] = encoded_net.data[i];
        }
    }

    return EquilibrationCutNetworks(pre_net, post_net);
}

SurfacesCuts extract_exposed_surfaces(
    const NetworkPattern& encoded_net,
    const EquilibrationCutNetworks& cuts,
    const int species_factor)
{
    if (species_factor <= 0) {
        throw std::runtime_error(
            "extract_exposed_surfaces: species_factor deve ser > 0");
    }

    SurfacesCuts out;
    out.surface_preteq = extract_top_surface_points(cuts.pre_teq, species_factor);
    out.surface_posteq = extract_top_surface_points(encoded_net, species_factor);
    return out;
}

SurfacesCuts extract_exposed_surfaces_from_cuts(
    const EquilibrationCutNetworks& cuts,
    const int species_factor)
{
    if (species_factor <= 0) {
        throw std::runtime_error(
            "extract_exposed_surfaces_from_cuts: species_factor deve ser > 0");
    }

    SurfacesCuts out;
    out.surface_preteq = extract_top_surface_points(cuts.pre_teq, species_factor);
    out.surface_posteq = extract_top_surface_points(cuts.post_teq, species_factor);
    return out;
}

SurfacesCuts build_equilibration_exposed_surfaces(
    const NetworkPattern& encoded_net,
    const double t_eq,
    const int species_factor)
{
    const EquilibrationCutNetworks cuts =
        build_equilibration_cut_networks(encoded_net, t_eq, species_factor);

    return extract_exposed_surfaces(encoded_net, cuts, species_factor);
}
