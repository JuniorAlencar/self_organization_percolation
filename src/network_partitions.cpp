#include "network_partitions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cnpy.h>
#include <nlohmann/json.hpp>

namespace {

using json = nlohmann::json;

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

    inline int lin_index(const int x, const int y, const int z) const {
        return x + SX * (y + SY * z);
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

int largest_component_single_color_sparse(
    const SparseSubgraph& net,
    const GridRegular& grid,
    const int color_idx)
{
    if (color_idx < 0 || color_idx >= net.num_colors) {
        throw std::runtime_error("largest_component_single_color_sparse: color_idx inválido");
    }

    const auto& active = net.active_idx_by_color[color_idx];
    if (active.empty()) return 0;

    std::unordered_set<int> visited;
    visited.reserve(active.size() * 2 + 1);

    std::vector<int> stack;
    stack.reserve(1024);

    int best = 0;

    for (const int idx : active) {
        if (visited.find(idx) != visited.end()) continue;

        int comp_size = 0;
        stack.clear();
        stack.push_back(idx);
        visited.insert(idx);

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();
            ++comp_size;

            grid.for_each_neighbor(u, [&](const int v) {
                if (v < 0) return;
                if (active.find(v) == active.end()) return;
                if (visited.find(v) != visited.end()) return;

                visited.insert(v);
                stack.push_back(v);
            });
        }

        if (comp_size > best) best = comp_size;
    }

    return best;
}

bool shortest_path_to_subgraph_top_single_color_sparse(
    const SparseSubgraph& net,
    const GridRegular& grid,
    const int color_idx,
    std::vector<int>& out_path,
    int& out_len)
{
    if (color_idx < 0 || color_idx >= net.num_colors) {
        throw std::runtime_error(
            "shortest_path_to_subgraph_top_single_color_sparse: color_idx inválido");
    }

    const auto& active = net.active_idx_by_color[color_idx];
    if (active.empty()) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    int base_subgraph = std::numeric_limits<int>::max();
    int top_subgraph  = -1;

    for (const int idx : active) {
        const int g = grid.grow_coord(idx);
        base_subgraph = std::min(base_subgraph, g);
        top_subgraph  = std::max(top_subgraph, g);
    }

    std::queue<int> q;
    std::unordered_set<int> visited;
    std::unordered_map<int, int> parent;

    visited.reserve(active.size() * 2 + 1);
    parent.reserve(active.size() * 2 + 1);

    for (const int idx : active) {
        if (grid.grow_coord(idx) != base_subgraph) continue;
        visited.insert(idx);
        parent[idx] = -1;
        q.push(idx);
    }

    if (q.empty()) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    int target = -1;

    while (!q.empty()) {
        const int u = q.front();
        q.pop();

        if (grid.grow_coord(u) == top_subgraph) {
            target = u;
            break;
        }

        grid.for_each_neighbor(u, [&](const int v) {
            if (v < 0) return;
            if (active.find(v) == active.end()) return;
            if (visited.find(v) != visited.end()) return;

            visited.insert(v);
            parent[v] = u;
            q.push(v);
        });
    }

    if (target < 0) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    out_path.clear();
    int cur = target;
    while (cur >= 0) {
        out_path.push_back(cur);
        cur = parent[cur];
    }
    std::reverse(out_path.begin(), out_path.end());

    out_len = static_cast<int>(out_path.size()) - 1;
    return true;
}

bool shortest_path_postteq_with_support_single_color_sparse(
    const SparseEncodedNetwork& full_net,
    const SparseSubgraph& post_net,
    const GridRegular& grid,
    const int color_idx,
    std::vector<int>& out_path,
    int& out_len)
{
    if (color_idx < 0 || color_idx >= post_net.num_colors ||
        color_idx >= full_net.num_colors) {
        throw std::runtime_error(
            "shortest_path_postteq_with_support_single_color_sparse: color_idx inválido");
    }

    const auto& post_active = post_net.active_idx_by_color[color_idx];
    const auto& full_active = full_net.active_idx_by_color[color_idx];

    if (post_active.empty() || full_active.empty()) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    int base_subgraph = std::numeric_limits<int>::max();
    int top_subgraph  = -1;

    for (const int idx : post_active) {
        const int g = grid.grow_coord(idx);
        base_subgraph = std::min(base_subgraph, g);
        top_subgraph  = std::max(top_subgraph, g);
    }

    std::queue<int> q;
    std::unordered_set<int> visited;
    std::unordered_map<int, int> parent;

    visited.reserve(full_active.size() * 2 + 1);
    parent.reserve(full_active.size() * 2 + 1);

    for (const int idx : post_active) {
        if (grid.grow_coord(idx) != base_subgraph) continue;
        visited.insert(idx);
        parent[idx] = -1;
        q.push(idx);
    }

    if (q.empty()) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    int target = -1;

    while (!q.empty()) {
        const int u = q.front();
        q.pop();

        if (grid.grow_coord(u) == top_subgraph &&
            post_active.find(u) != post_active.end()) {
            target = u;
            break;
        }

        grid.for_each_neighbor(u, [&](const int v) {
            if (v < 0) return;
            if (full_active.find(v) == full_active.end()) return;
            if (visited.find(v) != visited.end()) return;

            visited.insert(v);
            parent[v] = u;
            q.push(v);
        });
    }

    if (target < 0) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    out_path.clear();
    int cur = target;
    while (cur >= 0) {
        out_path.push_back(cur);
        cur = parent[cur];
    }
    std::reverse(out_path.begin(), out_path.end());

    out_len = static_cast<int>(out_path.size()) - 1;
    return true;
}

SubgraphAnalysis analyze_sparse_postteq_with_support(
    const SparseEncodedNetwork& full_net,
    const SparseSubgraph& post_net)
{
    if (post_net.shape.empty()) {
        throw std::runtime_error("analyze_sparse_postteq_with_support: shape vazio");
    }

    const int L = post_net.shape[0];
    const GridRegular grid(post_net.dim, L);

    SubgraphAnalysis analysis;
    analysis.net = post_net;

    analysis.color_percolation.clear();
    analysis.percolation_order.clear();

    analysis.largest_component.assign(post_net.num_colors, 0);
    analysis.sp_len.assign(post_net.num_colors, -1);
    analysis.sp_path_lin.assign(post_net.num_colors, {});

    for (int c = 0; c < post_net.num_colors; ++c) {
        analysis.largest_component[c] =
            largest_component_single_color_sparse(post_net, grid, c);

        std::vector<int> path;
        int path_len = -1;

        const bool has_path =
            shortest_path_postteq_with_support_single_color_sparse(
                full_net, post_net, grid, c, path, path_len);

        if (!has_path) continue;

        analysis.sp_len[c] = path_len;
        analysis.sp_path_lin[c] = std::move(path);
    }

    return analysis;
}

inline int color_to_active_value(const int num_colors, const int c) {
    return (num_colors == 1 ? 1 : (c + 2));
}

struct DecodedValue {
    bool never_activated = false;
    bool blocked = false;
    int color_1b = -1;
    int color_idx = -1;
    int time = -1;
};

DecodedValue decode_animation_value(
    const long long code,
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

template <typename T>
std::vector<T> copy_npy_vector(const cnpy::NpyArray& arr) {
    const T* ptr = arr.data<T>();
    const std::size_t n = arr.num_vals;
    return std::vector<T>(ptr, ptr + n);
}

int read_scalar_int(const cnpy::NpyArray& arr) {
    if (arr.word_size == sizeof(int)) {
        return static_cast<int>(arr.data<int>()[0]);
    }
    if (arr.word_size == sizeof(long long)) {
        return static_cast<int>(arr.data<long long>()[0]);
    }
    if (arr.word_size == sizeof(std::int64_t)) {
        return static_cast<int>(arr.data<std::int64_t>()[0]);
    }
    throw std::runtime_error("read_scalar_int: tipo inteiro nao suportado no npz");
}

std::vector<int> read_vector_int(const cnpy::NpyArray& arr) {
    if (arr.word_size == sizeof(int)) {
        return copy_npy_vector<int>(arr);
    }
    if (arr.word_size == sizeof(long long)) {
        std::vector<long long> tmp = copy_npy_vector<long long>(arr);
        std::vector<int> out(tmp.size());
        for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = static_cast<int>(tmp[i]);
        return out;
    }
    if (arr.word_size == sizeof(std::int64_t)) {
        std::vector<std::int64_t> tmp = copy_npy_vector<std::int64_t>(arr);
        std::vector<int> out(tmp.size());
        for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = static_cast<int>(tmp[i]);
        return out;
    }
    throw std::runtime_error("read_vector_int: tipo inteiro nao suportado no npz");
}

std::vector<double> read_vector_double(const cnpy::NpyArray& arr) {
    if (arr.word_size == sizeof(double)) {
        return copy_npy_vector<double>(arr);
    }
    if (arr.word_size == sizeof(float)) {
        std::vector<float> tmp = copy_npy_vector<float>(arr);
        return std::vector<double>(tmp.begin(), tmp.end());
    }
    throw std::runtime_error("read_vector_double: tipo real nao suportado no npz");
}

std::vector<long long> read_vector_ll_from_state(const cnpy::NpyArray& arr) {
    if (arr.word_size == sizeof(int)) {
        std::vector<int> tmp = copy_npy_vector<int>(arr);
        return std::vector<long long>(tmp.begin(), tmp.end());
    }
    if (arr.word_size == sizeof(long long)) {
        return copy_npy_vector<long long>(arr);
    }
    if (arr.word_size == sizeof(std::int64_t)) {
        std::vector<std::int64_t> tmp = copy_npy_vector<std::int64_t>(arr);
        return std::vector<long long>(tmp.begin(), tmp.end());
    }
    throw std::runtime_error("read_vector_ll_from_state: tipo de data nao suportado no npz");
}

double mean_of(const std::vector<double>& x, const int begin, const int end) {
    if (begin >= end) return 0.0;
    const double s = std::accumulate(x.begin() + begin, x.begin() + end, 0.0);
    return s / static_cast<double>(end - begin);
}

double std_of(const std::vector<double>& x, const int begin, const int end, const double mu) {
    if (begin >= end) return 0.0;
    double acc = 0.0;
    for (int i = begin; i < end; ++i) {
        const double d = x[i] - mu;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(end - begin));
}

std::vector<double> moving_average(const std::vector<double>& x, const int window) {
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

std::vector<double> build_mean_p_series(const TimeSeries& ts) {
    if (ts.p_t.empty()) {
        throw std::runtime_error("estimate_t_eq: TimeSeries p_t vazio");
    }

    const int num_colors = static_cast<int>(ts.p_t.size());
    const int n = static_cast<int>(ts.p_t[0].size());
    if (n == 0) {
        throw std::runtime_error("estimate_t_eq: series p_t vazias");
    }

    for (int c = 1; c < num_colors; ++c) {
        if (static_cast<int>(ts.p_t[c].size()) != n) {
            throw std::runtime_error("estimate_t_eq: p_t com comprimentos inconsistentes");
        }
    }

    std::vector<double> p_mean(n, 0.0);
    for (int k = 0; k < n; ++k) {
        for (int c = 0; c < num_colors; ++c) {
            p_mean[k] += ts.p_t[c][k];
        }
        p_mean[k] /= static_cast<double>(num_colors);
    }

    return p_mean;
}

std::vector<int> build_path(const std::vector<int>& parent, int target) {
    std::vector<int> path;
    while (target >= 0) {
        path.push_back(target);
        target = parent[target];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

int largest_component_single_color(
    const NetworkPattern& net,
    const GridRegular& grid,
    const int color_idx)
{
    const int active_val = color_to_active_value(net.num_colors, color_idx);

    std::vector<char> visited(net.data.size(), 0);
    std::vector<int> stack;
    int best = 0;

    for (int idx = 0; idx < static_cast<int>(net.data.size()); ++idx) {
        if (visited[idx]) continue;
        if (static_cast<int>(net.data[idx]) != active_val) continue;

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
                if (static_cast<int>(net.data[v]) != active_val) return;
                visited[v] = 1;
                stack.push_back(v);
            });
        }

        if (comp_size > best) best = comp_size;
    }

    return best;
}

bool shortest_path_to_subgraph_top_single_color(
    const NetworkPattern& net,
    const GridRegular& grid,
    const int color_idx,
    std::vector<int>& out_path,
    int& out_len)
{
    const int active_val = color_to_active_value(net.num_colors, color_idx);
    const int N = static_cast<int>(net.data.size());

    std::vector<char> visited(N, 0);
    std::vector<int> parent(N, -1);
    std::vector<int> dist(N, -1);
    std::queue<int> q;

    bool has_active_nodes = false;
    int base_subgraph = std::numeric_limits<int>::max();
    int top_subgraph  = -1;

    for (int idx = 0; idx < N; ++idx) {
        if (static_cast<int>(net.data[idx]) != active_val) continue;
        has_active_nodes = true;
        const int g = grid.grow_coord(idx);
        base_subgraph = std::min(base_subgraph, g);
        top_subgraph  = std::max(top_subgraph, g);
    }

    if (!has_active_nodes) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    for (int idx = 0; idx < N; ++idx) {
        if (static_cast<int>(net.data[idx]) != active_val) continue;
        if (grid.grow_coord(idx) != base_subgraph) continue;

        visited[idx] = 1;
        dist[idx] = 0;
        parent[idx] = -1;
        q.push(idx);
    }

    if (q.empty()) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    int target = -1;

    while (!q.empty()) {
        const int u = q.front();
        q.pop();

        if (grid.grow_coord(u) == top_subgraph) {
            target = u;
            break;
        }

        grid.for_each_neighbor(u, [&](const int v) {
            if (v < 0) return;
            if (visited[v]) return;
            if (static_cast<int>(net.data[v]) != active_val) return;

            visited[v] = 1;
            parent[v] = u;
            dist[v] = dist[u] + 1;
            q.push(v);
        });
    }

    if (target < 0) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    out_path = build_path(parent, target);
    out_len = static_cast<int>(out_path.size()) - 1;
    return true;
}

NetworkPattern make_empty_like(const NetworkPattern& net)
{
    NetworkPattern out(net.dim, net.shape, net.num_colors, net.rho);
    out.data.assign(net.data.size(), static_cast<NetworkPattern::state_t>(-1));
    return out;
}

NetworkPattern build_preteq_network(
    const NetworkPattern& encoded_net,
    const int t_eq,
    const int species_factor)
{
    NetworkPattern filtered = make_empty_like(encoded_net);

    for (std::size_t i = 0; i < encoded_net.data.size(); ++i) {
        const long long code = static_cast<long long>(encoded_net.data[i]);
        const DecodedValue dv = decode_animation_value(code, species_factor);

        // Para partição temporal dos sítios ativos:
        // só entram ativos com t <= t_eq.
        // Bloqueados e nunca ativados ficam fora desta decomposição temporal.
        if (dv.never_activated || dv.blocked) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(-1);
            continue;
        }

        if (dv.color_idx < 0 || dv.color_idx >= encoded_net.num_colors) {
            throw std::runtime_error("build_preteq_network: cor decodificada invalida");
        }

        if (dv.time <= t_eq) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(
                color_to_active_value(encoded_net.num_colors, dv.color_idx));
        } else {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(-1);
        }
    }

    return filtered;
}

NetworkPattern build_postteq_network(
    const NetworkPattern& encoded_net,
    const int t_eq,
    const int species_factor)
{
    NetworkPattern filtered = make_empty_like(encoded_net);

    for (std::size_t i = 0; i < encoded_net.data.size(); ++i) {
        const long long code = static_cast<long long>(encoded_net.data[i]);
        const DecodedValue dv = decode_animation_value(code, species_factor);

        // Para partição temporal dos sítios ativos:
        // só entram ativos com t > t_eq.
        // Bloqueados e nunca ativados ficam fora desta decomposição temporal.
        if (dv.never_activated || dv.blocked) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(-1);
            continue;
        }

        if (dv.color_idx < 0 || dv.color_idx >= encoded_net.num_colors) {
            throw std::runtime_error("build_postteq_network: cor decodificada invalida");
        }

        if (dv.time > t_eq) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(
                color_to_active_value(encoded_net.num_colors, dv.color_idx));
        } else {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(-1);
        }
    }

    return filtered;
}

SubgraphAnalysis analyze_sparse_isolated_subgraph(const SparseSubgraph& net)
{
    if (net.shape.empty()) {
        throw std::runtime_error("analyze_sparse_isolated_subgraph: shape vazio");
    }

    const int L = net.shape[0];
    const GridRegular grid(net.dim, L);

    SubgraphAnalysis analysis;
    analysis.net = net;

    analysis.color_percolation.clear();
    analysis.percolation_order.clear();

    analysis.largest_component.assign(net.num_colors, 0);
    analysis.sp_len.assign(net.num_colors, -1);
    analysis.sp_path_lin.assign(net.num_colors, {});

    for (int c = 0; c < net.num_colors; ++c) {
        analysis.largest_component[c] =
            largest_component_single_color_sparse(net, grid, c);

        std::vector<int> path;
        int path_len = -1;

        const bool has_path =
            shortest_path_to_subgraph_top_single_color_sparse(
                net, grid, c, path, path_len);

        if (!has_path) continue;

        analysis.sp_len[c] = path_len;
        analysis.sp_path_lin[c] = std::move(path);
    }

    return analysis;
}

} // namespace

TimeSeries load_timeseries_from_json(const std::string& json_path)
{
    std::ifstream fin(json_path);
    if (!fin) {
        throw std::runtime_error("Nao foi possivel abrir JSON: " + json_path);
    }

    json j;
    fin >> j;

    TimeSeries ts;

    if (j.contains("time_series") || j.contains("ts_out")) {
        const json* root = nullptr;

        if (j.contains("time_series")) {
            root = &j["time_series"];
        } else {
            root = &j["ts_out"];
        }

        if (!root->contains("t") || !root->contains("p_t")) {
            throw std::runtime_error(
                "JSON antigo encontrado, mas sem chaves 't' e 'p_t': " + json_path);
        }

        ts.t = (*root)["t"].get<std::vector<int>>();
        ts.p_t = (*root)["p_t"].get<std::vector<std::vector<double>>>();

        if (root->contains("Nt")) {
            ts.Nt = (*root)["Nt"].get<std::vector<std::vector<int>>>();
        } else {
            ts.Nt.assign(ts.p_t.size(), std::vector<int>(ts.t.size(), 0));
        }

        if (root->contains("num_colors")) {
            ts.num_colors = (*root)["num_colors"].get<int>();
        } else {
            ts.num_colors = static_cast<int>(ts.p_t.size());
        }

        return ts;
    }

    if (!j.contains("results") || !j["results"].is_object()) {
        throw std::runtime_error(
            "JSON em formato nao suportado (sem 'time_series', 'ts_out' ou 'results'): " + json_path);
    }

    const json& results = j["results"];

    int num_colors = 0;
    if (j.contains("meta") && j["meta"].contains("num_colors")) {
        num_colors = j["meta"]["num_colors"].get<int>();
    } else {
        for (auto it = results.begin(); it != results.end(); ++it) {
            if (!it.value().contains("data")) continue;
            const json& data = it.value()["data"];
            if (!data.contains("color")) continue;
            num_colors = std::max(num_colors, data["color"].get<int>());
        }
    }

    if (num_colors <= 0) {
        throw std::runtime_error(
            "Nao foi possivel determinar num_colors no JSON: " + json_path);
    }

    ts.num_colors = num_colors;
    ts.p_t.assign(num_colors, {});
    ts.Nt.assign(num_colors, {});
    std::vector<char> color_found(num_colors, 0);

    bool t_initialized = false;

    for (auto it = results.begin(); it != results.end(); ++it) {
        if (!it.value().contains("data")) {
            continue;
        }

        const json& data = it.value()["data"];

        if (!data.contains("color") || !data.contains("time") || !data.contains("pt")) {
            throw std::runtime_error(
                "Bloco em 'results' sem chaves obrigatorias ('color', 'time', 'pt'): " + it.key());
        }

        const int color_1b = data["color"].get<int>();
        const int color_idx = color_1b - 1;

        if (color_idx < 0 || color_idx >= num_colors) {
            throw std::runtime_error(
                "Cor invalida no JSON: color = " + std::to_string(color_1b));
        }

        const std::vector<int> time = data["time"].get<std::vector<int>>();
        const std::vector<double> pt = data["pt"].get<std::vector<double>>();

        std::vector<int> nt;
        if (data.contains("nt")) {
            nt = data["nt"].get<std::vector<int>>();
        } else {
            nt.assign(time.size(), 0);
        }

        if (time.size() != pt.size()) {
            throw std::runtime_error(
                "Comprimentos incompatíveis entre 'time' e 'pt' para color = " + std::to_string(color_1b));
        }

        if (time.size() != nt.size()) {
            throw std::runtime_error(
                "Comprimentos incompatíveis entre 'time' e 'nt' para color = " + std::to_string(color_1b));
        }

        if (!t_initialized) {
            ts.t = time;
            t_initialized = true;
        } else {
            if (time.size() != ts.t.size()) {
                throw std::runtime_error(
                    "Series 'time' com tamanhos diferentes entre cores no JSON: " + json_path);
            }
            if (time != ts.t) {
                throw std::runtime_error(
                    "Series 'time' diferentes entre cores no JSON: " + json_path);
            }
        }

        ts.p_t[color_idx] = pt;
        ts.Nt[color_idx] = nt;
        color_found[color_idx] = 1;
    }

    if (!t_initialized) {
        throw std::runtime_error(
            "Nenhum bloco valido encontrado em 'results' no JSON: " + json_path);
    }

    for (int c = 0; c < num_colors; ++c) {
        if (!color_found[c]) {
            ts.p_t[c].assign(ts.t.size(), 0.0);
            ts.Nt[c].assign(ts.t.size(), 0);
        }
    }

    return ts;
}

NetworkPattern load_encoded_network_from_npz(const std::string& npz_path)
{
    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    if (!npz.count("dim") || !npz.count("shape") || !npz.count("num_colors")) {
        throw std::runtime_error(
            "NPZ deve conter pelo menos: dim, shape, num_colors");
    }

    const int dim = read_scalar_int(npz["dim"]);
    const int num_colors = read_scalar_int(npz["num_colors"]);
    const std::vector<int> shape = read_vector_int(npz["shape"]);

    std::vector<double> rho;
    if (npz.count("rho")) {
        rho = read_vector_double(npz["rho"]);
    } else {
        rho.assign(num_colors, 1.0 / std::max(1, num_colors));
    }

    NetworkPattern net(dim, shape, num_colors, rho);

    // Como no save novo só existem sítios ativos,
    // todo o resto volta como -1 por padrão.
    std::size_t total_size = 1;
    for (int s : shape) {
        if (s <= 0) {
            throw std::runtime_error("load_encoded_network_from_npz: shape inválido");
        }
        total_size *= static_cast<std::size_t>(s);
    }

    net.data.assign(total_size, static_cast<NetworkPattern::state_t>(-1));

    // -------- modo antigo denso --------
    if (npz.count("data")) {
        const std::vector<long long> data_ll = read_vector_ll_from_state(npz["data"]);

        if (data_ll.size() != total_size) {
            throw std::runtime_error(
                "load_encoded_network_from_npz: tamanho de data incompatível com shape");
        }

        net.data.resize(data_ll.size());
        for (std::size_t i = 0; i < data_ll.size(); ++i) {
            net.data[i] = static_cast<NetworkPattern::state_t>(data_ll[i]);
        }

        return net;
    }

    // -------- modo novo esparso --------
    if (npz.count("active_idx") && npz.count("active_val")) {
        const std::vector<int> idx = read_vector_int(npz["active_idx"]);
        const std::vector<int> val = read_vector_int(npz["active_val"]);

        if (idx.size() != val.size()) {
            throw std::runtime_error(
                "load_encoded_network_from_npz: active_idx.size != active_val.size");
        }

        for (std::size_t k = 0; k < idx.size(); ++k) {
            const int lin = idx[k];
            const int v   = val[k];

            if (lin < 0 || static_cast<std::size_t>(lin) >= total_size) {
                throw std::runtime_error(
                    "load_encoded_network_from_npz: índice linear fora do intervalo");
            }

            if (v <= 0) {
                throw std::runtime_error(
                    "load_encoded_network_from_npz: active_val deve conter apenas valores > 0");
            }

            net.data[static_cast<std::size_t>(lin)] =
                static_cast<NetworkPattern::state_t>(v);
        }

        return net;
    }

    throw std::runtime_error(
        "NPZ não contém nem 'data' (modo antigo) nem 'active_idx'/'active_val' (modo esparso)");
}

int estimate_t_eq(const TimeSeries& ts, const ReanalysisConfig& cfg)
{
    if (ts.t.empty()) {
        throw std::runtime_error("estimate_t_eq: TimeSeries t vazio");
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

        return ts.t[i];
    }

    return ts.t[tail_begin];
}

int estimate_t_eq_from_json(const std::string& json_path, const ReanalysisConfig& cfg)
{
    const TimeSeries ts = load_timeseries_from_json(json_path);
    return estimate_t_eq(ts, cfg);
}

NetworkPattern rebuild_network_from_animation(
    const NetworkPattern& encoded_net,
    const int t_eq,
    const int species_factor)
{
    return build_preteq_network(encoded_net, t_eq, species_factor);
}

ReanalysisResult reanalyze_animation(
    const std::string& json_path,
    const std::string& npz_path,
    const ReanalysisConfig& cfg)
{
    const TimeSeries ts = load_timeseries_from_json(json_path);
    const SparseEncodedNetwork encoded_net =
        load_sparse_encoded_network_from_npz(npz_path, cfg.species_factor);

    if (encoded_net.shape.empty()) {
        throw std::runtime_error("reanalyze_animation: shape vazio");
    }

    ReanalysisResult result;
    result.t_eq = estimate_t_eq(ts, cfg);

    const SparseSubgraph net_pre =
        build_preteq_sparse_subgraph(
            encoded_net, result.t_eq, cfg.species_factor);

    const SparseSubgraph net_post =
        build_postteq_sparse_subgraph(
            encoded_net, result.t_eq, cfg.species_factor);

    result.pre_teq  = analyze_sparse_isolated_subgraph(net_pre);
    result.post_teq = analyze_sparse_postteq_with_support(encoded_net, net_post);

    return result;
}

SparseEncodedNetwork load_sparse_encoded_network_from_npz(
    const std::string& npz_path,
    const int species_factor)
{
    if (species_factor <= 0) {
        throw std::runtime_error(
            "load_sparse_encoded_network_from_npz: species_factor deve ser > 0");
    }

    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    if (!npz.count("dim") || !npz.count("shape") || !npz.count("num_colors")) {
        throw std::runtime_error(
            "NPZ deve conter pelo menos: dim, shape, num_colors");
    }

    SparseEncodedNetwork net;

    net.dim        = read_scalar_int(npz["dim"]);
    net.num_colors = read_scalar_int(npz["num_colors"]);
    net.shape      = read_vector_int(npz["shape"]);

    if (npz.count("seed")) {
        net.seed = read_scalar_int(npz["seed"]);
    } else {
        net.seed = 0;
    }

    if (npz.count("rho")) {
        net.rho = read_vector_double(npz["rho"]);
    } else {
        net.rho.assign(net.num_colors, 1.0 / std::max(1, net.num_colors));
    }

    net.total_size = 1;
    for (int s : net.shape) {
        if (s <= 0) {
            throw std::runtime_error(
                "load_sparse_encoded_network_from_npz: shape inválido");
        }
        net.total_size *= static_cast<std::size_t>(s);
    }

    net.active_idx_by_color.assign(net.num_colors, {});

    // modo novo esparso
    if (npz.count("active_idx") && npz.count("active_val")) {
        net.active_idx = read_vector_int(npz["active_idx"]);
        net.active_val = read_vector_int(npz["active_val"]);

        if (net.active_idx.size() != net.active_val.size()) {
            throw std::runtime_error(
                "load_sparse_encoded_network_from_npz: active_idx.size != active_val.size");
        }

        net.encoded_value_by_idx.reserve(net.active_idx.size() * 2 + 1);

        for (std::size_t k = 0; k < net.active_idx.size(); ++k) {
            const int lin = net.active_idx[k];
            const int val = net.active_val[k];

            if (lin < 0 || static_cast<std::size_t>(lin) >= net.total_size) {
                throw std::runtime_error(
                    "load_sparse_encoded_network_from_npz: índice linear fora do intervalo");
            }

            if (val <= 0) {
                throw std::runtime_error(
                    "load_sparse_encoded_network_from_npz: active_val deve conter apenas valores > 0");
            }

            const DecodedValue dv =
                decode_animation_value(static_cast<long long>(val), species_factor);

            if (dv.never_activated || dv.blocked) {
                throw std::runtime_error(
                    "load_sparse_encoded_network_from_npz: active_val não pode decodificar para -1/0");
            }

            if (dv.color_idx < 0 || dv.color_idx >= net.num_colors) {
                throw std::runtime_error(
                    "load_sparse_encoded_network_from_npz: cor decodificada inválida");
            }

            net.encoded_value_by_idx[lin] = val;
            net.active_idx_by_color[dv.color_idx].insert(lin);
        }

        return net;
    }

    // modo antigo denso
    if (npz.count("data")) {
        const std::vector<long long> data_ll = read_vector_ll_from_state(npz["data"]);

        if (data_ll.size() != net.total_size) {
            throw std::runtime_error(
                "load_sparse_encoded_network_from_npz: tamanho de data incompatível com shape");
        }

        net.active_idx.reserve(data_ll.size() / 16);
        net.active_val.reserve(data_ll.size() / 16);
        net.encoded_value_by_idx.reserve(data_ll.size() / 16);

        for (std::size_t i = 0; i < data_ll.size(); ++i) {
            const int val = static_cast<int>(data_ll[i]);

            if (val <= 0) continue;

            const DecodedValue dv =
                decode_animation_value(static_cast<long long>(val), species_factor);

            if (dv.never_activated || dv.blocked) continue;

            if (dv.color_idx < 0 || dv.color_idx >= net.num_colors) {
                throw std::runtime_error(
                    "load_sparse_encoded_network_from_npz: cor decodificada inválida no modo denso");
            }

            const int lin = static_cast<int>(i);

            net.active_idx.push_back(lin);
            net.active_val.push_back(val);
            net.encoded_value_by_idx[lin] = val;
            net.active_idx_by_color[dv.color_idx].insert(lin);
        }

        return net;
    }

    throw std::runtime_error(
        "NPZ não contém nem 'data' nem 'active_idx'/'active_val'");
}

SparseSubgraph make_empty_sparse_like(const SparseEncodedNetwork& encoded_net)
{
    SparseSubgraph out;
    out.dim = encoded_net.dim;
    out.num_colors = encoded_net.num_colors;
    out.shape = encoded_net.shape;
    out.rho = encoded_net.rho;
    out.total_size = encoded_net.total_size;
    out.active_idx_by_color.assign(out.num_colors, {});
    return out;
}

SparseSubgraph build_preteq_sparse_subgraph(
    const SparseEncodedNetwork& encoded_net,
    const int t_eq,
    const int species_factor)
{
    SparseSubgraph out = make_empty_sparse_like(encoded_net);

    out.active_idx.reserve(encoded_net.active_idx.size() / 2);
    out.active_val.reserve(encoded_net.active_val.size() / 2);
    out.value_by_idx.reserve(encoded_net.active_idx.size() / 2);

    for (std::size_t k = 0; k < encoded_net.active_idx.size(); ++k) {
        const int lin = encoded_net.active_idx[k];
        const int enc = encoded_net.active_val[k];

        const DecodedValue dv =
            decode_animation_value(static_cast<long long>(enc), species_factor);

        if (dv.never_activated || dv.blocked) continue;
        if (dv.color_idx < 0 || dv.color_idx >= encoded_net.num_colors) continue;
        if (dv.time > t_eq) continue;

        const int dec_val =
            color_to_active_value(encoded_net.num_colors, dv.color_idx);

        out.active_idx.push_back(lin);
        out.active_val.push_back(dec_val);
        out.value_by_idx[lin] = dec_val;
        out.active_idx_by_color[dv.color_idx].insert(lin);
    }

    return out;
}

SparseSubgraph build_postteq_sparse_subgraph(
    const SparseEncodedNetwork& encoded_net,
    const int t_eq,
    const int species_factor)
{
    SparseSubgraph out = make_empty_sparse_like(encoded_net);

    out.active_idx.reserve(encoded_net.active_idx.size() / 2);
    out.active_val.reserve(encoded_net.active_val.size() / 2);
    out.value_by_idx.reserve(encoded_net.active_idx.size() / 2);

    for (std::size_t k = 0; k < encoded_net.active_idx.size(); ++k) {
        const int lin = encoded_net.active_idx[k];
        const int enc = encoded_net.active_val[k];

        const DecodedValue dv =
            decode_animation_value(static_cast<long long>(enc), species_factor);

        if (dv.never_activated || dv.blocked) continue;
        if (dv.color_idx < 0 || dv.color_idx >= encoded_net.num_colors) continue;
        if (dv.time <= t_eq) continue;

        const int dec_val =
            color_to_active_value(encoded_net.num_colors, dv.color_idx);

        out.active_idx.push_back(lin);
        out.active_val.push_back(dec_val);
        out.value_by_idx[lin] = dec_val;
        out.active_idx_by_color[dv.color_idx].insert(lin);
    }

    return out;
}

