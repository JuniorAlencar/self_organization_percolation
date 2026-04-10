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

                        ps_out.color_percolation.push_back(cor_idx + 1);
                        ps_out.percolation_order.push_back(++order_ctr);
                    }
                } else if (is_node) {
                    net.set(viz_idx, 0);
                }
            });
        }

        // Marca como morta a espécie que não percolou e não cresceu neste passo
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
            // Só calcula shortest path e maior componente
            // se ao menos uma cor tiver percolado.
            if (!ps_out.color_percolation.empty()) {
                ps_out.M_size_at_perc.clear();
                ps_out.M_size_at_perc.reserve(ps_out.color_percolation.size());

                for (const int color_1b : ps_out.color_percolation) {
                    const int c = color_1b - 1;

                    std::vector<int> path = build_path_from_parent(top_site_per_color[c]);
                    ps_out.sp_path_lin[c] = std::move(path);

                    if (!ps_out.sp_path_lin[c].empty()) {
                        ps_out.sp_len[c] = static_cast<int>(ps_out.sp_path_lin[c].size()) - 1;
                    } else {
                        ps_out.sp_len[c] = -1;
                    }

                    ps_out.M_size_at_perc.push_back(largest_component_single_color(c));
                }
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

    // No animate, não vamos calcular:
    // ps_out.sp_len
    // ps_out.sp_path_lin
    // ps_out.M_size_at_perc
    // shortest_path
    // largest_component
    ps_out.color_percolation.clear();
    ps_out.percolation_order.clear();
    ps_out.M_size_at_perc.clear();
    ps_out.sp_len.clear();
    ps_out.sp_path_lin.clear();

    // Ativação inicial na base: mesma lógica do create_network
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
                activation_code[idx] = static_cast<NetworkPattern::state_t>(color_mul[c]);
                frontier.push_back(idx);
                ++N_current[c];
                ++activated;
            }
            ++tries;
        }
    }

    commit_step(0, p_curr, N_current);

    int order_ctr = 0;
    std::vector<bool> percolated(num_colors, false);
    std::vector<bool> died(num_colors, false);
    std::vector<bool> finished(num_colors, false);

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
                    activation_code[viz_idx] =
                        static_cast<NetworkPattern::state_t>(color_mul[cor_idx] + t);
                    next_frontier.push_back(viz_idx);
                    ++N_current[cor_idx];

                    const int h = grid.grow_coord(viz_idx);
                    if (h > max_heights[cor_idx]) {
                        max_heights[cor_idx] = h;
                    }

                    if (!percolated[cor_idx] && h == lenght_network - 1) {
                        percolated[cor_idx] = true;
                        finished[cor_idx] = true;
                        ps_out.color_percolation.push_back(cor_idx + 1);
                        ps_out.percolation_order.push_back(++order_ctr);
                    }
                } else if (is_node) {
                    net.set(viz_idx, 0);
                    activation_code[viz_idx] =
                        static_cast<NetworkPattern::state_t>(0);
                }
            });
        }

        // Mesma regra correta do create_network:
        // se não percolou e não cresceu neste passo, morreu
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
