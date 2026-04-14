#include "animation_reanalysis.hpp"
#include "../src/write_save.hpp"

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

bool shortest_path_single_color(
    const NetworkPattern& net,
    const GridRegular& grid,
    const int color_idx,
    std::vector<int>& out_path,
    int& out_len)
{
    const int active_val = color_to_active_value(net.num_colors, color_idx);
    const int N = static_cast<int>(net.data.size());
    const int top_coord = (grid.dim == 2) ? (grid.SY - 1) : (grid.SZ - 1);

    std::vector<char> visited(N, 0);
    std::vector<int> parent(N, -1);
    std::vector<int> dist(N, -1);
    std::queue<int> q;

    int base_subgraph = std::numeric_limits<int>::max();
    bool has_active_nodes = false;

    for (int idx = 0; idx < N; ++idx) {
        if (static_cast<int>(net.data[idx]) != active_val) continue;
        has_active_nodes = true;
        base_subgraph = std::min(base_subgraph, grid.grow_coord(idx));
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

    int target = -1;

    while (!q.empty()) {
        const int u = q.front();
        q.pop();

        if (grid.grow_coord(u) == top_coord) {
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

} // namespace

TimeSeries load_timeseries_from_json(const std::string& json_path) {
    std::ifstream fin(json_path);
    if (!fin) {
        throw std::runtime_error("Nao foi possivel abrir JSON: " + json_path);
    }

    json j;
    fin >> j;

    TimeSeries ts;

    // =========================================================
    // FORMATO ANTIGO
    //   {
    //     "time_series": { "t": ..., "p_t": ..., "Nt": ... }
    //   }
    // ou
    //   {
    //     "ts_out": { "t": ..., "p_t": ..., "Nt": ... }
    //   }
    // =========================================================
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

    // =========================================================
    // FORMATO NOVO
    //   {
    //     "meta": { "num_colors": ..., "rho": [...] },
    //     "results": {
    //       "order_percolation 1": {
    //         "data": {
    //           "color": 7,
    //           "time": [...],
    //           "pt": [...],
    //           "nt": [...]
    //         }
    //       },
    //       ...
    //     }
    //   }
    // =========================================================
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

    // Se alguma cor nao aparecer no JSON, preenche com zeros
    for (int c = 0; c < num_colors; ++c) {
        if (!color_found[c]) {
            ts.p_t[c].assign(ts.t.size(), 0.0);
            ts.Nt[c].assign(ts.t.size(), 0);
        }
    }

    return ts;
}

NetworkPattern load_encoded_network_from_npz(const std::string& npz_path) {
    cnpy::npz_t npz = cnpy::npz_load(npz_path);

    if (!npz.count("dim") || !npz.count("shape") || !npz.count("num_colors") || !npz.count("data")) {
        throw std::runtime_error("NPZ deve conter pelo menos: dim, shape, num_colors, data");
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

    const std::vector<long long> data_ll = read_vector_ll_from_state(npz["data"]);
    net.data.resize(data_ll.size());

    for (std::size_t i = 0; i < data_ll.size(); ++i) {
        net.data[i] = static_cast<NetworkPattern::state_t>(data_ll[i]);
    }

    return net;
}

int estimate_t_eq(const TimeSeries& ts, const ReanalysisConfig& cfg) {
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

int estimate_t_eq_from_json(const std::string& json_path, const ReanalysisConfig& cfg) {
    const TimeSeries ts = load_timeseries_from_json(json_path);
    return estimate_t_eq(ts, cfg);
}

namespace {

NetworkPattern make_empty_like(const NetworkPattern& net)
{
    NetworkPattern out(net.dim, net.shape, net.num_colors, net.rho);
    out.data.assign(net.data.size(), static_cast<NetworkPattern::state_t>(-1));
    return out;
}





NetworkPattern build_cumulative_network_until_time(
    const NetworkPattern& encoded_net,
    const int t_cut,
    const int species_factor)
{
    NetworkPattern filtered = make_empty_like(encoded_net);

    for (std::size_t i = 0; i < encoded_net.data.size(); ++i) {
        const long long code = static_cast<long long>(encoded_net.data[i]);
        const DecodedValue dv = decode_animation_value(code, species_factor);

        if (dv.never_activated) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(-1);
            continue;
        }

        if (dv.blocked) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(0);
            continue;
        }

        if (dv.color_idx < 0 || dv.color_idx >= encoded_net.num_colors) {
            throw std::runtime_error("build_cumulative_network_until_time: cor decodificada invalida");
        }

        if (dv.time <= t_cut) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(
                color_to_active_value(encoded_net.num_colors, dv.color_idx));
        } else {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(-1);
        }
    }

    return filtered;
}

NetworkPattern build_posteq_increment_network(
    const NetworkPattern& encoded_net,
    const int t_eq,
    const int species_factor)
{
    NetworkPattern filtered = make_empty_like(encoded_net);

    for (std::size_t i = 0; i < encoded_net.data.size(); ++i) {
        const long long code = static_cast<long long>(encoded_net.data[i]);
        const DecodedValue dv = decode_animation_value(code, species_factor);

        if (dv.never_activated) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(-1);
            continue;
        }

        if (dv.blocked) {
            filtered.data[i] = static_cast<NetworkPattern::state_t>(0);
            continue;
        }

        if (dv.color_idx < 0 || dv.color_idx >= encoded_net.num_colors) {
            throw std::runtime_error("build_posteq_increment_network: cor decodificada invalida");
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



SubgraphAnalysis analyze_filtered_net(const NetworkPattern& net)
{
    if (net.shape.empty()) {
        throw std::runtime_error("analyze_filtered_net: shape vazio");
    }

    const int L = net.shape[0];
    const GridRegular grid(net.dim, L);

    SubgraphAnalysis analysis;
    analysis.net = net;
    analysis.largest_component.assign(net.num_colors, 0);
    analysis.sp_len.assign(net.num_colors, -1);
    analysis.sp_path_lin.assign(net.num_colors, {});

    int order = 0;

    for (int c = 0; c < net.num_colors; ++c) {
        analysis.largest_component[c] =
            largest_component_single_color(net, grid, c);

        std::vector<int> path;
        int path_len = -1;

        const bool percolates =
            shortest_path_single_color(net, grid, c, path, path_len);

        if (!percolates) continue;

        analysis.color_percolation.push_back(c + 1);
        analysis.percolation_order.push_back(++order);
        analysis.sp_len[c] = path_len;
        analysis.sp_path_lin[c] = std::move(path);
    }

    return analysis;
}

} // namespace

NetworkPattern rebuild_network_from_animation(
    const NetworkPattern& encoded_net,
    const int t_eq,
    const int species_factor)
{
    return build_cumulative_network_until_time(encoded_net, t_eq, species_factor);
}



ReanalysisResult reanalyze_animation(
    const std::string& json_path,
    const std::string& npz_path,
    const ReanalysisConfig& cfg)
{
    const TimeSeries ts = load_timeseries_from_json(json_path);
    const NetworkPattern encoded_net = load_encoded_network_from_npz(npz_path);

    if (encoded_net.shape.empty()) {
        throw std::runtime_error("shape vazio no NetworkPattern");
    }

    ReanalysisResult result;
    result.t_eq = estimate_t_eq(ts, cfg);

    const NetworkPattern net_pre =
        build_cumulative_network_until_time(
            encoded_net, result.t_eq, cfg.species_factor);

    const NetworkPattern net_post =
        build_posteq_increment_network(
            encoded_net, result.t_eq, cfg.species_factor);

    // pre_teq: salva apenas a rede antes/até t_eq
    result.pre_teq.net = net_pre;
    result.pre_teq.color_percolation.clear();
    result.pre_teq.percolation_order.clear();
    result.pre_teq.largest_component.assign(encoded_net.num_colors, 0);
    result.pre_teq.sp_len.assign(encoded_net.num_colors, -1);
    result.pre_teq.sp_path_lin.assign(encoded_net.num_colors, {});

    // post_teq: calcula shortest_path e largest_component aqui
    result.post_teq = analyze_filtered_net(net_post);

    return result;
}