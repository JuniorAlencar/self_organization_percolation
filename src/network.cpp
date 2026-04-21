#include "network.hpp"

#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <algorithm>

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

    inline int lin_index(const int x, const int y = 0, const int z = 0) const {
        return x + SX * (y + SY * z);
    }

    inline int x_of(const int idx) const {
        return idx % SX;
    }

    inline int y_of(const int idx) const {
        return (idx / SX) % SY;
    }

    inline int z_of(const int idx) const {
        return idx / (SX * SY);
    }

    inline int grow_coord(const int idx) const {
        return (dim == 2) ? y_of(idx) : z_of(idx);
    }

    inline void coords_of(const int idx, int& x, int& y, int& z) const {
        x = x_of(idx);
        y = (dim >= 2 ? y_of(idx) : 0);
        z = (dim == 3 ? z_of(idx) : 0);
    }

    inline int neighbor_xm(const int idx) const {
        const int x = idx % SX;
        if (grow_axis == 0) {
            return (x == 0) ? -1 : (idx - 1);
        }
        return (x == 0) ? (idx + (SX - 1)) : (idx - 1);
    }

    inline int neighbor_xp(const int idx) const {
        const int x = idx % SX;
        if (grow_axis == 0) {
            return (x == SX - 1) ? -1 : (idx + 1);
        }
        return (x == SX - 1) ? (idx - (SX - 1)) : (idx + 1);
    }

    inline int neighbor_ym(const int idx) const {
        const int y = (idx / SX) % SY;
        if (grow_axis == 1) {
            return (y == 0) ? -1 : (idx - SX);
        }
        return (y == 0) ? (idx + SX * (SY - 1)) : (idx - SX);
    }

    inline int neighbor_yp(const int idx) const {
        const int y = (idx / SX) % SY;
        if (grow_axis == 1) {
            return (y == SY - 1) ? -1 : (idx + SX);
        }
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

inline int color_to_active_value(const int num_colors, const int c) {
    return (num_colors == 1 ? 1 : (c + 2));
}

inline int color_to_negative_value(const int num_colors, const int c) {
    return (num_colors == 1 ? -1 : -(c + 2));
}

inline int value_to_color_index(const int num_colors, const int v) {
    return (num_colors == 1 ? 0 : (std::abs(v) - 2));
}

inline long long compute_base_size(const GridRegular& grid) {
    return (grid.dim == 2) ? static_cast<long long>(grid.SX)
                           : static_cast<long long>(grid.SX) * static_cast<long long>(grid.SY);
}

struct DecodedValue {
    bool never_activated = false;
    bool blocked = false;
    int color_1b = -1;
    int color_idx = -1;
    int time = -1;
};

inline DecodedValue decode_animation_value(const long long code,
                                           const int species_factor)
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

inline bool is_active_color_site_encoded(const std::vector<NetworkPattern::state_t>& activation_code,
                                         const int idx,
                                         const int color_idx,
                                         const int species_factor,
                                         int* time_out = nullptr)
{
    const long long code =
        static_cast<long long>(activation_code[static_cast<std::size_t>(idx)]);

    const DecodedValue dv = decode_animation_value(code, species_factor);

    if (dv.never_activated || dv.blocked) return false;
    if (dv.color_idx != color_idx) return false;

    if (time_out) *time_out = dv.time;
    return true;
}

inline double mean_slice(const std::vector<double>& x, const int a, const int b)
{
    if (a >= b) return 0.0;
    double s = 0.0;
    for (int i = a; i < b; ++i) s += x[i];
    return s / static_cast<double>(b - a);
}

inline double std_slice(const std::vector<double>& x, const int a, const int b, const double mu)
{
    if (a >= b) return 0.0;
    double s = 0.0;
    for (int i = a; i < b; ++i) {
        const double d = x[i] - mu;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(b - a));
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
    if (ts.t.empty()) return {};
    if (ts.p_t.empty()) {
        throw std::runtime_error("build_mean_p_series: p_t vazio");
    }

    const std::size_t T = ts.t.size();
    std::vector<double> p_mean(T, 0.0);

    for (const auto& row : ts.p_t) {
        if (row.size() != T) {
            throw std::runtime_error("build_mean_p_series: linhas de p_t com tamanhos diferentes");
        }
        for (std::size_t i = 0; i < T; ++i) {
            p_mean[i] += row[i];
        }
    }

    const double inv = 1.0 / static_cast<double>(ts.p_t.size());
    for (double& v : p_mean) v *= inv;

    return p_mean;
}

inline int estimate_t_eq_from_timeseries(const TimeSeries& ts,
                                         const int smoothing_window = 25,
                                         const int min_stable_steps = 25,
                                         const double rel_tol = 2.0e-2,
                                         const double abs_tol = 1.0e-6,
                                         const double sigma_multiplier = 2.0)
{
    if (ts.t.empty()) {
        throw std::runtime_error("estimate_t_eq_from_timeseries: ts.t vazio");
    }

    const std::vector<double> p_mean = build_mean_p_series(ts);
    const std::vector<double> p_smoothed = moving_average(p_mean, smoothing_window);

    const int n = static_cast<int>(p_smoothed.size());
    const int tail_len = std::max(min_stable_steps, std::min(n, std::max(20, n / 5)));
    const int tail_begin = std::max(0, n - tail_len);

    const double ref_mean = mean_slice(p_smoothed, tail_begin, n);
    const double ref_std  = std_slice(p_smoothed, tail_begin, n, ref_mean);

    const double tol = std::max(
        abs_tol,
        std::max(rel_tol * std::abs(ref_mean), sigma_multiplier * ref_std)
    );

    const int stable_len = std::max(5, min_stable_steps);

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

        const double suffix_mean = mean_slice(p_smoothed, i, n);
        if (std::abs(suffix_mean - ref_mean) > tol) continue;

        return ts.t[static_cast<std::size_t>(i)];
    }

    return ts.t[static_cast<std::size_t>(tail_begin)];
}

} // anonymous namespace


double network::type_Nt_create(const int type_N_t, const int t_i, const double a, const double alpha){
    if(type_N_t == 0) return N_t;
    else if (type_N_t == 1) return a*pow(t_i, alpha);
    throw std::invalid_argument("Invalid type_N_t value: " + std::to_string(type_N_t));
}

double network::generate_p(const int type_N_t, const double p_t, const int t_i, const int N_current, const double k, const double a, const double alpha) {
    double N_T = type_Nt_create(type_N_t, t_i, a, alpha);
    double p_next = p_t + k * (N_T - N_current);

    if (p_next > 1.0) p_next = 1.0;
    if (p_next < 0.0) p_next = 0.0;

    return p_next;
}

NetworkPattern network::create_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double k, const double N_t, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng)
{
    this->N_t = N_t;

    const GridRegular grid(dim, lenght_network);
    const std::vector<int> shape = (dim == 2)
        ? std::vector<int>{lenght_network, lenght_network}
        : std::vector<int>{lenght_network, lenght_network, lenght_network};

    const bool is_node = (type_percolation == "node");
    const long long base_size = compute_base_size(grid);

    NetworkPattern net(dim, shape, num_colors, rho);

    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<int>>    Nt_series(num_colors);
    std::vector<int>                 t_list;
    t_list.reserve(num_of_samples);

    std::vector<double> p_curr = p0;
    std::vector<int>    N_current(num_colors, 0);
    std::vector<int>    max_heights(num_colors, 0);
    std::vector<int>    parent(grid.total_size, -2);
    std::vector<int>    frontier;
    std::vector<int>    next_frontier;
    
    frontier.reserve(static_cast<std::size_t>(base_size));
    next_frontier.reserve(static_cast<std::size_t>(base_size));

    auto commit_step = [&](const int t_k, const std::vector<double>& p_vec, const std::vector<int>& Nt_vec)
    {
        t_list.push_back(t_k);
        for (int c = 0; c < num_colors; ++c) {
            p_series[c].push_back(p_vec[c]);
            Nt_series[c].push_back(Nt_vec[c]);
        }
    };

    std::vector<int> seeds_quota(num_colors, 0);
    for (int c = 0; c < num_colors; ++c) {
        long long q = std::llround(P0 * rho[c] * base_size);
        if (q < 0) q = 0;
        if (q > base_size) q = base_size;
        seeds_quota[c] = static_cast<int>(q);
    }

    for (int c = 0; c < num_colors; ++c) {
        int activated = 0;
        int tries = 0;
        const int max_tries = static_cast<int>(base_size) * 20;
        const int prefer_neg = color_to_negative_value(num_colors, c);
        const int active_val = color_to_active_value(num_colors, c);

        while (activated < seeds_quota[c] && tries < max_tries) {
            const int x = rng.uniform_int(0, grid.SX - 1);
            const int y = (dim == 3 ? rng.uniform_int(0, grid.SY - 1) : 0);
            const int z = 0;
            const int idx = grid.lin_index(x, y, z);
            const int v = net.get(idx);

            if (v == prefer_neg || v == -1) {
                net.set(idx, active_val);
                frontier.push_back(idx);
                ++N_current[c];
                ++activated;
                parent[idx] = -1;
            }
            ++tries;
        }
    }

    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, std::vector<int>{});
    ps_out.color_percolation.clear();
    ps_out.percolation_order.clear();
    ps_out.M_size_at_perc.clear();

    commit_step(0, p_curr, N_current);

    int order_ctr = 0;
    std::vector<bool> percolated(num_colors, false);
    std::vector<bool> died(num_colors, false);
    std::vector<bool> finished(num_colors, false);

    // Guarda o sítio que atingiu o topo para cada cor
    std::vector<int> top_site_per_color(num_colors, -1);

    // Guarda a ordem bruta de chegada ao topo, mas só escreve no ps_out no final
    std::vector<int> percolation_rank(num_colors, -1);

    auto build_path_from_parent = [&](int end_idx) -> std::vector<int>
    {
        std::vector<int> path;

        if (end_idx < 0) {
            return path;
        }

        int cur = end_idx;
        while (cur >= 0) {
            path.push_back(cur);
            cur = parent[cur];
        }

        if (cur != -1) {
            path.clear();
            return path;
        }

        std::reverse(path.begin(), path.end());
        return path;
    };

    auto is_valid_percolating_path = [&](const std::vector<int>& path) -> bool
    {
        if (path.empty()) {
            return false;
        }

        const int top_coord = (dim == 2) ? (grid.SY - 1) : (grid.SZ - 1);

        const int start_idx = path.front();
        const int end_idx   = path.back();

        if (grid.grow_coord(start_idx) != 0) {
            return false;
        }

        if (grid.grow_coord(end_idx) != top_coord) {
            return false;
        }

        return true;
    };

    auto largest_component_single_color = [&](const int color_idx) -> int
    {
        const int active_val = color_to_active_value(num_colors, color_idx);

        std::vector<char> visited(grid.total_size, 0);
        std::vector<int> stack;
        int max_component = 0;

        for (int idx = 0; idx < grid.total_size; ++idx) {
            if (visited[idx]) continue;
            if (net.get(idx) != active_val) continue;

            int comp_size = 0;
            stack.clear();
            stack.push_back(idx);
            visited[idx] = 1;

            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                ++comp_size;

                grid.for_each_neighbor(u, [&](const int v) {
                    if (v < 0) return;
                    if (visited[v]) return;
                    if (net.get(v) != active_val) return;

                    visited[v] = 1;
                    stack.push_back(v);
                });
            }

            if (comp_size > max_component) {
                max_component = comp_size;
            }
        }

        return max_component;
    };

    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        next_frontier.clear();

        for (const int idx : frontier) {
            const int a_val = net.get(idx);
            if (a_val <= 0) continue;

            const int cor_idx = value_to_color_index(num_colors, a_val);
            if (cor_idx < 0 || cor_idx >= num_colors) continue;
            if (finished[cor_idx]) continue;

            const int new_val = color_to_active_value(num_colors, cor_idx);

            grid.for_each_neighbor(idx, [&](const int viz_idx) {
                if (viz_idx < 0) return;

                const int vv = net.get(viz_idx);
                if (vv >= 0) return;

                const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                const bool no_color   = (vv == -1);
                if (!same_color && !no_color) return;

                const double r = rng.uniform_real(0.0, 1.0);
                if (r < p_curr[cor_idx]) {
                    net.set(viz_idx, new_val);
                    next_frontier.push_back(viz_idx);
                    ++N_current[cor_idx];
                    parent[viz_idx] = idx;

                    const int h = grid.grow_coord(viz_idx);
                    if (h > max_heights[cor_idx]) {
                        max_heights[cor_idx] = h;
                    }

                    if (!percolated[cor_idx] && h == lenght_network - 1) {
                        percolated[cor_idx] = true;
                        finished[cor_idx] = true;
                        top_site_per_color[cor_idx] = viz_idx;
                        percolation_rank[cor_idx] = ++order_ctr;
                    }
                } else if (is_node) {
                    net.set(viz_idx, 0);
                }
            });
        }

        // Se a espécie não percolou e N(t)=0, ela morreu e deve ser ignorada
        for (int c = 0; c < num_colors; ++c) {
            if (!percolated[c] && N_current[c] == 0) {
                died[c] = true;
                finished[c] = true;
            }
        }

        bool all_dead = true;
        bool all_percolated = true;
        bool any_dead = false;
        bool any_percolated = false;
        bool all_finished = true;

        for (int c = 0; c < num_colors; ++c) {
            if (!died[c]) {
                all_dead = false;
            } else {
                any_dead = true;
            }

            if (!percolated[c]) {
                all_percolated = false;
            } else {
                any_percolated = true;
            }

            if (!finished[c]) {
                all_finished = false;
            }
        }

        // Condições de parada:
        // 1) todas morreram sem percolar
        const bool stop_all_dead = all_dead;

        // 2) todas percolaram
        const bool stop_all_percolated = all_percolated;

        // 3) algumas percolaram e o restante morreu
        const bool stop_partial_percolation =
            all_finished && any_percolated && any_dead;

        if (stop_all_dead || stop_all_percolated || stop_partial_percolation) {
            ps_out.color_percolation.clear();
            ps_out.percolation_order.clear();
            ps_out.M_size_at_perc.clear();

            struct ValidPercolation {
                int rank;
                int color_idx;
            };
            std::vector<ValidPercolation> valid_percs;
            valid_percs.reserve(num_colors);

            for (int c = 0; c < num_colors; ++c) {
                // morreu antes de percolar -> ignora
                if (died[c]) {
                    ps_out.sp_len[c] = -1;
                    ps_out.sp_path_lin[c].clear();
                    continue;
                }

                // nunca atingiu o topo -> ignora
                if (!percolated[c]) {
                    ps_out.sp_len[c] = -1;
                    ps_out.sp_path_lin[c].clear();
                    continue;
                }

                // monta caminho até o sítio que tocou o topo
                std::vector<int> path = build_path_from_parent(top_site_per_color[c]);

                // valida caminho base -> topo
                if (!is_valid_percolating_path(path)) {
                    ps_out.sp_len[c] = -1;
                    ps_out.sp_path_lin[c].clear();
                    continue;
                }

                ps_out.sp_path_lin[c] = std::move(path);
                ps_out.sp_len[c] = static_cast<int>(ps_out.sp_path_lin[c].size()) - 1;

                valid_percs.push_back({percolation_rank[c], c});
            }

            std::sort(valid_percs.begin(), valid_percs.end(),
                    [](const ValidPercolation& a, const ValidPercolation& b) {
                        return a.rank < b.rank;
                    });

            int valid_order_ctr = 0;
            for (const auto& vp : valid_percs) {
                const int c = vp.color_idx;

                ps_out.color_percolation.push_back(c + 1);
                ps_out.percolation_order.push_back(++valid_order_ctr);
                ps_out.M_size_at_perc.push_back(largest_component_single_color(c));
            }

            break;
        }

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c) {
                std::cout
                    << ", p" << (c + 1) << "(t)=" << p_curr[c]
                    << ", N_t" << (c + 1) << "(t)=" << N_current[c]
                    << " max" << (c + 1) << "(t)=" << max_heights[c];
            }
            std::cout << '\n';
        }

        std::vector<double> p_next(num_colors);
        for (int c = 0; c < num_colors; ++c) {
            p_next[c] = finished[c] ? p_curr[c]
                                    : generate_p(type_N_t, p_curr[c], t, N_current[c], k, a, alpha);
        }

        commit_step(t, p_next, N_current);
        frontier.swap(next_frontier);
        p_curr.swap(p_next);
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.Nt  = std::move(Nt_series);
    ts_out.t   = std::move(t_list);

    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < static_cast<int>(rho.size())) {
            ps_out.rho.push_back(rho[i]);
        }
    }

    return net;
}

NetworkPattern network::animate_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double k, const double N_t, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng)
{
    this->N_t = N_t;

    const GridRegular grid(dim, lenght_network);
    const std::vector<int> shape = (dim == 2)
        ? std::vector<int>{lenght_network, lenght_network}
        : std::vector<int>{lenght_network, lenght_network, lenght_network};

    const bool is_node = (type_percolation == "node");
    const long long base_size = compute_base_size(grid);

    NetworkPattern net(dim, shape, num_colors, rho);

    // Codificação:
    //   -1 : nunca ativado
    //    0 : bloqueado no caso node
    // NUMBER*(c+1) + t : espécie c ativada no tempo t
    const int SPECIES_FACTOR = 10000000;

    if (num_of_samples >= SPECIES_FACTOR) {
        throw std::runtime_error(
            "animate_network: num_of_samples deve ser menor que SPECIES_FACTOR "
            "para evitar colisao entre especie e tempo.");
    }

    const long long max_code_needed =
        static_cast<long long>(SPECIES_FACTOR) * num_colors + num_of_samples;

    if (max_code_needed >
        static_cast<long long>(std::numeric_limits<NetworkPattern::state_t>::max()))
    {
        throw std::runtime_error(
            "animate_network: codificacao excede NetworkPattern::state_t. "
            "Reduza SPECIES_FACTOR/num_colors/num_of_samples ou aumente state_t.");
    }

    std::vector<NetworkPattern::state_t> activation_code(
        static_cast<std::size_t>(grid.total_size),
        static_cast<NetworkPattern::state_t>(-1));

    std::vector<int> color_mul(num_colors, 0);
    for (int c = 0; c < num_colors; ++c) {
        color_mul[c] = (c + 1) * SPECIES_FACTOR;
    }

    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<int>>    Nt_series(num_colors);
    std::vector<int>                 t_list;
    t_list.reserve(num_of_samples);

    std::vector<double> p_curr = p0;
    std::vector<int>    N_current(num_colors, 0);
    std::vector<int>    max_heights(num_colors, 0);
    std::vector<int>    parent(grid.total_size, -2);
    std::vector<int>    top_site_per_color(num_colors, -1);
    std::vector<int>    frontier;
    std::vector<int>    next_frontier;

    frontier.reserve(static_cast<std::size_t>(base_size));
    next_frontier.reserve(static_cast<std::size_t>(base_size));

    auto commit_step = [&](const int t_k,
                           const std::vector<double>& p_vec,
                           const std::vector<int>& Nt_vec)
    {
        t_list.push_back(t_k);
        for (int c = 0; c < num_colors; ++c) {
            p_series[c].push_back(p_vec[c]);
            Nt_series[c].push_back(Nt_vec[c]);
        }
    };

    std::vector<int> seeds_quota(num_colors, 0);
    for (int c = 0; c < num_colors; ++c) {
        long long q = std::llround(P0 * rho[c] * base_size);
        if (q < 0) q = 0;
        if (q > base_size) q = base_size;
        seeds_quota[c] = static_cast<int>(q);
    }

    // métricas globais
    ps_out.color_percolation.clear();
    ps_out.percolation_order.clear();
    ps_out.M_size_at_perc.clear();
    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, std::vector<int>{});

    // novas métricas temporais
    ps_out.t_eq = -1;
    ps_out.sp_lin_preteq.assign(num_colors, -1);
    ps_out.sp_lin_posteq.assign(num_colors, -1);
    ps_out.M_size_preteq.assign(num_colors, 0);
    ps_out.M_size_posteq.assign(num_colors, 0);

    // Ativação inicial na base
    for (int c = 0; c < num_colors; ++c) {
        int activated = 0;
        int tries = 0;
        const int max_tries = static_cast<int>(base_size) * 20;
        const int prefer_neg = color_to_negative_value(num_colors, c);
        const int active_val = color_to_active_value(num_colors, c);

        while (activated < seeds_quota[c] && tries < max_tries) {
            const int x = rng.uniform_int(0, grid.SX - 1);
            const int y = (dim == 3 ? rng.uniform_int(0, grid.SY - 1) : 0);
            const int z = 0;
            const int idx = grid.lin_index(x, y, z);
            const int v = net.get(idx);

            if (v == prefer_neg || v == -1) {
                net.set(idx, active_val);
                activation_code[static_cast<std::size_t>(idx)] =
                    static_cast<NetworkPattern::state_t>(color_mul[c]);
                frontier.push_back(idx);
                ++N_current[c];
                ++activated;
                parent[idx] = -1;
            }
            ++tries;
        }
    }

    commit_step(0, p_curr, N_current);

    int order_ctr = 0;
    std::vector<bool> percolated(num_colors, false);
    std::vector<bool> died(num_colors, false);
    std::vector<bool> finished(num_colors, false);
    std::vector<int>  percolation_rank(num_colors, -1);

    // cache do maior componente global por cor
    std::vector<std::vector<int>> largest_comp_nodes_cache(num_colors);
    bool global_metrics_finalized = false;

    auto build_path_from_parent = [&](int end_idx) -> std::vector<int>
    {
        std::vector<int> path;

        if (end_idx < 0) {
            return path;
        }

        int cur = end_idx;
        while (cur >= 0) {
            path.push_back(cur);
            cur = parent[cur];
        }

        if (cur != -1) {
            path.clear();
            return path;
        }

        std::reverse(path.begin(), path.end());
        return path;
    };

    auto is_valid_percolating_path = [&](const std::vector<int>& path) -> bool
    {
        if (path.empty()) {
            return false;
        }

        const int top_coord = (dim == 2) ? (grid.SY - 1) : (grid.SZ - 1);

        const int start_idx = path.front();
        const int end_idx   = path.back();

        if (grid.grow_coord(start_idx) != 0) {
            return false;
        }

        if (grid.grow_coord(end_idx) != top_coord) {
            return false;
        }

        return true;
    };

    auto largest_component_nodes_single_color = [&](const int color_idx) -> std::vector<int>
    {
        const int active_val = color_to_active_value(num_colors, color_idx);

        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<int> stack;
        std::vector<int> comp_nodes;
        std::vector<int> best_nodes;

        for (int idx = 0; idx < grid.total_size; ++idx) {
            if (visited[static_cast<std::size_t>(idx)]) continue;
            if (net.get(idx) != active_val) continue;

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
                    if (net.get(v) != active_val) return;

                    visited[static_cast<std::size_t>(v)] = 1;
                    stack.push_back(v);
                });
            }

            if (comp_nodes.size() > best_nodes.size()) {
                best_nodes = comp_nodes;
            }
        }

        return best_nodes;
    };

    auto finalize_global_metrics = [&]()
    {
        ps_out.color_percolation.clear();
        ps_out.percolation_order.clear();
        ps_out.M_size_at_perc.clear();

        ps_out.sp_len.assign(num_colors, -1);
        ps_out.sp_path_lin.assign(num_colors, std::vector<int>{});

        for (int c = 0; c < num_colors; ++c) {
            largest_comp_nodes_cache[c].clear();
        }

        struct ValidPercolation {
            int rank;
            int color_idx;
        };

        std::vector<ValidPercolation> valid_percs;
        valid_percs.reserve(static_cast<std::size_t>(num_colors));

        for (int c = 0; c < num_colors; ++c) {
            if (died[c]) {
                ps_out.sp_len[c] = -1;
                ps_out.sp_path_lin[c].clear();
                continue;
            }

            if (!percolated[c]) {
                ps_out.sp_len[c] = -1;
                ps_out.sp_path_lin[c].clear();
                continue;
            }

            std::vector<int> path = build_path_from_parent(top_site_per_color[c]);

            if (!is_valid_percolating_path(path)) {
                ps_out.sp_len[c] = -1;
                ps_out.sp_path_lin[c].clear();
                continue;
            }

            ps_out.sp_path_lin[c] = std::move(path);
            ps_out.sp_len[c] = static_cast<int>(ps_out.sp_path_lin[c].size()) - 1;

            largest_comp_nodes_cache[c] = largest_component_nodes_single_color(c);

            valid_percs.push_back({percolation_rank[c], c});
        }

        std::sort(valid_percs.begin(), valid_percs.end(),
                  [](const ValidPercolation& a, const ValidPercolation& b) {
                      return a.rank < b.rank;
                  });

        int valid_order_ctr = 0;
        for (const auto& vp : valid_percs) {
            const int c = vp.color_idx;

            ps_out.color_percolation.push_back(c + 1);
            ps_out.percolation_order.push_back(++valid_order_ctr);
            ps_out.M_size_at_perc.push_back(
                static_cast<int>(largest_comp_nodes_cache[c].size()));
        }

        global_metrics_finalized = true;
    };

    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        next_frontier.clear();

        for (const int idx : frontier) {
            const int a_val = net.get(idx);
            if (a_val <= 0) continue;

            const int cor_idx = value_to_color_index(num_colors, a_val);
            if (cor_idx < 0 || cor_idx >= num_colors) continue;
            if (finished[cor_idx]) continue;

            const int new_val = color_to_active_value(num_colors, cor_idx);

            grid.for_each_neighbor(idx, [&](const int viz_idx) {
                if (viz_idx < 0) return;

                const int vv = net.get(viz_idx);
                if (vv >= 0) return;

                const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                const bool no_color   = (vv == -1);
                if (!same_color && !no_color) return;

                const double r = rng.uniform_real(0.0, 1.0);
                if (r < p_curr[cor_idx]) {
                    net.set(viz_idx, new_val);
                    activation_code[static_cast<std::size_t>(viz_idx)] =
                        static_cast<NetworkPattern::state_t>(color_mul[cor_idx] + t);
                    next_frontier.push_back(viz_idx);
                    ++N_current[cor_idx];
                    parent[viz_idx] = idx;

                    const int h = grid.grow_coord(viz_idx);
                    if (h > max_heights[cor_idx]) {
                        max_heights[cor_idx] = h;
                    }

                    if (!percolated[cor_idx] && h == lenght_network - 1) {
                        percolated[cor_idx] = true;
                        finished[cor_idx] = true;
                        top_site_per_color[cor_idx] = viz_idx;
                        percolation_rank[cor_idx] = ++order_ctr;
                    }
                } else if (is_node) {
                    net.set(viz_idx, 0);
                    activation_code[static_cast<std::size_t>(viz_idx)] =
                        static_cast<NetworkPattern::state_t>(0);
                }
            });
        }

        for (int c = 0; c < num_colors; ++c) {
            if (!percolated[c] && N_current[c] == 0) {
                died[c] = true;
                finished[c] = true;
            }
        }

        bool all_dead = true;
        bool all_percolated = true;
        bool any_dead = false;
        bool any_percolated = false;
        bool all_finished = true;

        for (int c = 0; c < num_colors; ++c) {
            if (!died[c]) {
                all_dead = false;
            } else {
                any_dead = true;
            }

            if (!percolated[c]) {
                all_percolated = false;
            } else {
                any_percolated = true;
            }

            if (!finished[c]) {
                all_finished = false;
            }
        }

        const bool stop_all_dead = all_dead;
        const bool stop_all_percolated = all_percolated;
        const bool stop_partial_percolation =
            all_finished && any_percolated && any_dead;

        if (stop_all_dead || stop_all_percolated || stop_partial_percolation) {
            finalize_global_metrics();
            break;
        }

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c) {
                std::cout
                    << ", p" << (c + 1) << "(t)=" << p_curr[c]
                    << ", N_t" << (c + 1) << "(t)=" << N_current[c]
                    << " max" << (c + 1) << "(t)=" << max_heights[c];
            }
            std::cout << '\n';
        }

        std::vector<double> p_next(num_colors);
        for (int c = 0; c < num_colors; ++c) {
            p_next[c] = finished[c]
                ? p_curr[c]
                : generate_p(type_N_t, p_curr[c], t, N_current[c], k, a, alpha);
        }

        commit_step(t, p_next, N_current);
        frontier.swap(next_frontier);
        p_curr.swap(p_next);
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.Nt  = std::move(Nt_series);
    ts_out.t   = std::move(t_list);

    if (!global_metrics_finalized) {
        finalize_global_metrics();
    }

    ps_out.t_eq = estimate_t_eq_from_timeseries(
        ts_out,
        25,      // smoothing_window
        25,      // min_stable_steps
        2.0e-2,  // rel_tol
        1.0e-6,  // abs_tol
        2.0      // sigma_multiplier
    );

    for (int c = 0; c < num_colors; ++c) {
        if (ps_out.sp_len[c] < 0 || ps_out.sp_path_lin[c].empty()) {
            ps_out.sp_lin_preteq[c] = -1;
            ps_out.sp_lin_posteq[c] = -1;
            ps_out.M_size_preteq[c] = 0;
            ps_out.M_size_posteq[c] = 0;
            continue;
        }

        // ------------------------------------------------------------
        // Decomposição temporal do shortest path global
        // ------------------------------------------------------------
        int sp_pre = 0;
        int sp_post = 0;

        const std::vector<int>& path = ps_out.sp_path_lin[c];
        for (std::size_t k = 1; k < path.size(); ++k) {
            const int idx = path[k];

            int t_site = -1;
            const bool ok = is_active_color_site_encoded(
                activation_code, idx, c, SPECIES_FACTOR, &t_site);

            if (!ok) {
                throw std::runtime_error(
                    "animate_network: shortest path contém nó inválido no encoded");
            }

            if (t_site <= ps_out.t_eq) ++sp_pre;
            else ++sp_post;
        }

        ps_out.sp_lin_preteq[c] = sp_pre;
        ps_out.sp_lin_posteq[c] = sp_post;

        // ------------------------------------------------------------
        // Decomposição temporal do maior componente global
        // ------------------------------------------------------------
        int m_pre = 0;
        int m_post = 0;

        const std::vector<int>& comp_nodes = largest_comp_nodes_cache[c];
        for (const int idx : comp_nodes) {
            int t_site = -1;
            const bool ok = is_active_color_site_encoded(
                activation_code, idx, c, SPECIES_FACTOR, &t_site);

            if (!ok) {
                throw std::runtime_error(
                    "animate_network: maior componente contém nó inválido no encoded");
            }

            if (t_site <= ps_out.t_eq) ++m_pre;
            else ++m_post;
        }

        ps_out.M_size_preteq[c] = m_pre;
        ps_out.M_size_posteq[c] = m_post;
    }

    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < static_cast<int>(rho.size())) {
            ps_out.rho.push_back(rho[i]);
        }
    }

    NetworkPattern net_animation(dim, shape, num_colors, rho);
    net_animation.data = std::move(activation_code);
    return net_animation;
}  

NetworkPattern network::create_shortest_paths_map(const NetworkPattern& net,
                                                  const PercolationSeries& ps_out)
{
    NetworkPattern sp_net(net.dim, net.shape, net.num_colors, net.rho);

    if (sp_net.data.size() != net.data.size()) {
        throw std::runtime_error("[create_shortest_paths_map] tamanho de data inconsistente");
    }

    std::fill(sp_net.data.begin(), sp_net.data.end(), 0);

    const int num_colors = net.num_colors;
    if (static_cast<int>(ps_out.sp_path_lin.size()) < num_colors ||
        static_cast<int>(ps_out.sp_len.size())      < num_colors) {
        throw std::runtime_error("[create_shortest_paths_map] PercolationSeries inconsistente");
    }

    const std::size_t N = sp_net.data.size();

    for (int c = 0; c < num_colors; ++c) {
        if (ps_out.sp_len[c] <= 0) continue;

        const std::vector<int>& path = ps_out.sp_path_lin[c];
        if (path.empty()) continue;

        const int color_label = (num_colors == 1 ? 1 : (c + 2));

        for (int idx : path) {
            if (idx < 0 || static_cast<std::size_t>(idx) >= N) {
                continue;
            }
            sp_net.data[idx] = color_label;
        }
    }

    return sp_net;
}
