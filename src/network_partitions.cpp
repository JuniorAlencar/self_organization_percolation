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

#include <nlohmann/json.hpp>
#include "animation_utils.hpp"

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

std::vector<int> build_path(const std::vector<int>& parent, const int target) {
    std::vector<int> path;
    if (target < 0) return path;
    int cur = target;
    while (cur >= 0) {
        path.push_back(cur);
        cur = parent[static_cast<std::size_t>(cur)];
        if (static_cast<std::size_t>(cur) >= parent.size() && cur != -1) break;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

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
        out_path.clear();
        out_len = -1;
        return false;
    }

    const auto& active_set = net.active_idx_by_color[color_idx];
    if (active_set.empty()) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    const std::size_t total = net.total_size;
    std::vector<char> visited(total, 0);
    std::vector<int> parent(static_cast<std::size_t>(total), -1);
    std::queue<int> q;

    int base_subgraph = std::numeric_limits<int>::max();
    int top_subgraph = std::numeric_limits<int>::min();
    for (const int idx : active_set) {
        const int g = grid.grow_coord(idx);
        base_subgraph = std::min(base_subgraph, g);
        top_subgraph = std::max(top_subgraph, g);
    }

    for (const int idx : active_set) {
        if (grid.grow_coord(idx) != base_subgraph) continue;
        visited[static_cast<std::size_t>(idx)] = 1;
        parent[static_cast<std::size_t>(idx)] = -1;
        q.push(idx);
    }

    if (q.empty()) {
        out_path.clear();
        out_len = -1;
        return false;
    }

    int target = -1;
    while (!q.empty()) {
        const int u = q.front(); q.pop();
        if (grid.grow_coord(u) == top_subgraph) {
            target = u;
            break;
        }

        grid.for_each_neighbor(u, [&](const int v) {
            if (v < 0) return;
            if (visited[static_cast<std::size_t>(v)]) return;
            if (active_set.find(v) == active_set.end()) return;
            visited[static_cast<std::size_t>(v)] = 1;
            parent[static_cast<std::size_t>(v)] = u;
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
    const double t_eq,
    const int species_factor)
{
    NetworkPattern filtered = make_empty_like(encoded_net);
    if (!std::isfinite(t_eq)) {
        return filtered;
    }

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
    const double t_eq,
    const int species_factor)
{
    NetworkPattern filtered = make_empty_like(encoded_net);
    if (!std::isfinite(t_eq)) {
        return filtered;
    }

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

// Minimal compatibility helper: analyze post-teq sparse subgraph (placeholder).
SubgraphAnalysis analyze_sparse_postteq_with_support(
    const SparseEncodedNetwork& /*encoded_net*/, const SparseSubgraph& post)
{
    return analyze_sparse_isolated_subgraph(post);
}

TimeSeries load_timeseries_from_json(const std::string& json_path)
{
    std::ifstream fin(json_path);
    if (!fin) {
        throw std::runtime_error("Nao foi possivel abrir JSON: " + json_path);
    }

    json j;
    fin >> j;

    TimeSeries ts;
    ts.num_colors = 0;
    ts.t_eq = std::numeric_limits<double>::quiet_NaN();

    auto get_matrix_double = [](const json& value) -> std::vector<std::vector<double>> {
        return value.get<std::vector<std::vector<double>>>();
    };

    auto get_vector_double = [](const json& value) -> std::vector<double> {
        return value.get<std::vector<double>>();
    };

    auto validate_timeseries_sizes = [&](const std::string& context) {
        if (ts.t.empty()) {
            throw std::runtime_error(context + ": vetor 't' vazio em " + json_path);
        }

        if (ts.p_t.empty()) {
            throw std::runtime_error(context + ": matriz 'p_t' vazia em " + json_path);
        }

        if (ts.f_t.empty()) {
            throw std::runtime_error(context + ": matriz 'f_t' vazia em " + json_path);
        }

        if (ts.p_t.size() != ts.f_t.size()) {
            throw std::runtime_error(
                context + ": numero de cores incompatível entre 'p_t' e 'f_t' em " + json_path);
        }

        const std::size_t ntimes = ts.t.size();
        for (std::size_t c = 0; c < ts.p_t.size(); ++c) {
            if (ts.p_t[c].size() != ntimes) {
                throw std::runtime_error(
                    context + ": comprimento de 'p_t' incompatível com 't' para uma cor em " + json_path);
            }
            if (ts.f_t[c].size() != ntimes) {
                throw std::runtime_error(
                    context + ": comprimento de 'f_t' incompatível com 't' para uma cor em " + json_path);
            }
        }

        if (ts.num_colors <= 0) {
            ts.num_colors = static_cast<int>(ts.p_t.size());
        }

        if (ts.num_colors != static_cast<int>(ts.p_t.size())) {
            throw std::runtime_error(
                context + ": num_colors incompatível com tamanho de 'p_t' em " + json_path);
        }
    };

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

        if (root->contains("f_t")) {
            ts.f_t = get_matrix_double((*root)["f_t"]);
        } else if (root->contains("ft")) {
            ts.f_t = get_matrix_double((*root)["ft"]);
        } else if (root->contains("Nt")) {
            // Compatibilidade apenas para leitura de arquivos antigos.
            // No formato novo, esta matriz representa f(t), não N(t).
            ts.f_t = get_matrix_double((*root)["Nt"]);
        } else {
            ts.f_t.assign(ts.p_t.size(), std::vector<double>(ts.t.size(), 0.0));
        }

        if (root->contains("num_colors")) {
            ts.num_colors = (*root)["num_colors"].get<int>();
        } else {
            ts.num_colors = static_cast<int>(ts.p_t.size());
        }

        if (root->contains("t_eq") && !(*root)["t_eq"].is_null()) {
            ts.t_eq = (*root)["t_eq"].get<double>();
        } else if (j.contains("t_eq") && !j["t_eq"].is_null()) {
            ts.t_eq = j["t_eq"].get<double>();
        }

        validate_timeseries_sizes("load_timeseries_from_json/time_series");
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
    ts.f_t.assign(num_colors, {});
    std::vector<char> color_found(num_colors, 0);

    if (j.contains("t_eq") && !j["t_eq"].is_null()) {
        ts.t_eq = j["t_eq"].get<double>();
    } else if (
        j.contains("percolation") &&
        j["percolation"].contains("t_eq") &&
        !j["percolation"]["t_eq"].is_null()
    ) {
        ts.t_eq = j["percolation"]["t_eq"].get<double>();
    }

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

        std::vector<double> ft;
        if (data.contains("ft")) {
            ft = get_vector_double(data["ft"]);
        } else if (data.contains("f_t")) {
            ft = get_vector_double(data["f_t"]);
        } else if (data.contains("nt")) {
            // Compatibilidade apenas para leitura de arquivos antigos.
            // No formato novo, esta série representa f(t), não N(t).
            ft = get_vector_double(data["nt"]);
        } else if (data.contains("Nt")) {
            ft = get_vector_double(data["Nt"]);
        } else {
            ft.assign(time.size(), 0.0);
        }

        if (time.size() != pt.size()) {
            throw std::runtime_error(
                "Comprimentos incompatíveis entre 'time' e 'pt' para color = " + std::to_string(color_1b));
        }

        if (time.size() != ft.size()) {
            throw std::runtime_error(
                "Comprimentos incompatíveis entre 'time' e 'ft' para color = " + std::to_string(color_1b));
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
        ts.f_t[color_idx] = ft;
        color_found[color_idx] = 1;
    }

    if (!t_initialized) {
        throw std::runtime_error(
            "Nenhum bloco valido encontrado em 'results' no JSON: " + json_path);
    }

    for (int c = 0; c < num_colors; ++c) {
        if (!color_found[c]) {
            ts.p_t[c].assign(ts.t.size(), 0.0);
            ts.f_t[c].assign(ts.t.size(), 0.0);
        }
    }

    validate_timeseries_sizes("load_timeseries_from_json/results");
    return ts;
}
NetworkPattern load_encoded_network_from_npz(const std::string& npz_path)
{
    // Read compact .bin produced by save_network_compact_bin and convert to
    // NetworkPattern encoded format (species*species_factor + activation_time).
    NetworkCompact cnet;
    if (!cnet.read_binary(npz_path)) {
        throw std::runtime_error("load_encoded_network_from_npz: falha lendo binario " + npz_path);
    }

    const uint64_t total = static_cast<uint64_t>(cnet.N);

    // Infer shape/dimension like other loaders: prefer cube then square.
    std::vector<int> shape;
    int dim = 3;
    int L3 = static_cast<int>(std::round(std::cbrt(static_cast<double>(total))));
    if (static_cast<uint64_t>(L3) * L3 * L3 == total) {
        dim = 3;
        shape = {L3, L3, L3};
    } else {
        int L2 = static_cast<int>(std::round(std::sqrt(static_cast<double>(total))));
        if (static_cast<uint64_t>(L2) * L2 == total) {
            dim = 2;
            shape = {L2, L2};
        } else {
            throw std::runtime_error("load_encoded_network_from_npz: não foi possível inferir dimensão/shape do .bin");
        }
    }

    int maxc = 0;
    for (size_t i = 0; i < cnet.species.size(); ++i) {
        maxc = std::max<int>(maxc, static_cast<int>(cnet.species[i]));
    }
    int num_colors = std::max(0, maxc);
    if (num_colors <= 0) num_colors = 1;

    std::vector<double> rho(num_colors, 0.0);
    NetworkPattern net(dim, shape, num_colors, rho);
    net.seed = 0;

    net.data.assign(static_cast<std::size_t>(total), static_cast<NetworkPattern::state_t>(-1));

    const int species_factor = 10000000;
    for (size_t i = 0; i < cnet.species.size(); ++i) {
        const int sp = static_cast<int>(cnet.species[i]);
        if (sp <= 0) continue;
        const uint32_t at = cnet.activation_time[i];
        long long enc = static_cast<long long>(sp) * static_cast<long long>(species_factor) + static_cast<long long>(at);
        if (enc > static_cast<long long>(std::numeric_limits<NetworkPattern::state_t>::max())) {
            enc = static_cast<long long>(std::numeric_limits<NetworkPattern::state_t>::max());
        }
        net.data[i] = static_cast<NetworkPattern::state_t>(enc);
    }

    return net;
}

double estimate_t_eq(const TimeSeries& ts, const ReanalysisConfig& cfg)
{
    if (ts.t.empty()) {
        throw std::runtime_error("estimate_t_eq: TimeSeries t vazio");
    }

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

    int stable_run = 0;
    const int stable_links_required =
        std::max(1, static_cast<int>(std::ceil(
            static_cast<double>(cfg.min_stable_steps) /
            static_cast<double>(std::max(1, cfg.window_block)))));

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

        const double p_scale = std::max(
            std::abs(j_w[static_cast<std::size_t>(i)]),
            std::abs(j_w[static_cast<std::size_t>(i + 1)]));
        const double s_threshold = std::max(cfg.abs_tol, cfg.rel_tol * p_scale);
        const bool stable =
            std::isfinite(sp) &&
            std::abs(sp) < cfg.s_prime_threshold &&
            s[static_cast<std::size_t>(i)] <= s_threshold;

        if (stable) {
            ++stable_run;
            if (stable_run >= stable_links_required) {
                const int first_stable = i - stable_run + 1;
                return t_s[static_cast<std::size_t>(first_stable)];
            }
        } else {
            stable_run = 0;
        }
    }

    return std::numeric_limits<double>::quiet_NaN();
}

double estimate_t_eq_from_json(const std::string& json_path, const ReanalysisConfig& cfg)
{
    const TimeSeries ts = load_timeseries_from_json(json_path);
    return estimate_t_eq(ts, cfg);
}

NetworkPattern rebuild_network_from_animation(
    const NetworkPattern& encoded_net,
    const double t_eq,
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

    // Read compact .bin produced by save_network_compact_bin.
    NetworkCompact cnet;
    if (!cnet.read_binary(npz_path)) {
        throw std::runtime_error("load_sparse_encoded_network_from_npz: falha lendo binario " + npz_path);
    }

    SparseEncodedNetwork net;
    const uint64_t total = static_cast<uint64_t>(cnet.N);

    // Infer shape/dimension
    int dim = 3;
    int L3 = static_cast<int>(std::round(std::cbrt(static_cast<double>(total))));
    if (static_cast<uint64_t>(L3) * L3 * L3 == total) {
        dim = 3;
        net.shape = {L3, L3, L3};
    } else {
        int L2 = static_cast<int>(std::round(std::sqrt(static_cast<double>(total))));
        if (static_cast<uint64_t>(L2) * L2 == total) {
            dim = 2;
            net.shape = {L2, L2};
        } else {
            throw std::runtime_error("load_sparse_encoded_network_from_npz: não foi possível inferir dimensão/shape do .bin");
        }
    }

    net.dim = dim;
    net.total_size = static_cast<std::size_t>(total);
    net.seed = 0;

    int maxc = 0;
    for (size_t i = 0; i < cnet.species.size(); ++i) {
        maxc = std::max<int>(maxc, static_cast<int>(cnet.species[i]));
    }
    net.num_colors = std::max(0, maxc);
    if (net.num_colors <= 0) net.num_colors = 1;
    net.rho.assign(net.num_colors, 0.0);

    net.active_idx_by_color.assign(net.num_colors, {});

    const int sf = species_factor;
    for (std::size_t i = 0; i < cnet.species.size(); ++i) {
        const int sp = static_cast<int>(cnet.species[i]);
        if (sp <= 0) continue;
        const int enc = sp * sf + static_cast<int>(cnet.activation_time[i]);
        net.active_idx.push_back(static_cast<int>(i));
        net.active_val.push_back(enc);
        net.encoded_value_by_idx[static_cast<int>(i)] = enc;
        const int cidx = sp - 1;
        if (cidx >= 0 && cidx < net.num_colors) net.active_idx_by_color[cidx].insert(static_cast<int>(i));
    }

    return net;
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
    const double t_eq,
    const int species_factor)
{
    SparseSubgraph out = make_empty_sparse_like(encoded_net);
    if (!std::isfinite(t_eq)) {
        return out;
    }

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
    const double t_eq,
    const int species_factor)
{
    SparseSubgraph out = make_empty_sparse_like(encoded_net);
    if (!std::isfinite(t_eq)) {
        return out;
    }

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
