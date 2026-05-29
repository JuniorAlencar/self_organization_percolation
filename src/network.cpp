#include "network.hpp"
#include "write_save.hpp"
#include "height_stop_config.hpp"

#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <random>
#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>

namespace {

constexpr int ANIMATION_SPECIES_FACTOR = 10000000;

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
          SY(dim_ >= 2 ? (dim_ == 2 ? HEIGHT_STOP_MULTIPLIER * L : L) : 1),
          SZ(dim_ == 3 ? HEIGHT_STOP_MULTIPLIER * L : 1),
          grow_axis(dim_ - 1),
          stride_y(SX),
          stride_z(SX * SY),
          total_size(SX * SY * SZ) {}

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

    inline int grow_top_coord() const {
        return (dim == 2) ? (SY - 1) : (SZ - 1);
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


struct FrontCandidate {
    std::uint32_t activator = std::numeric_limits<std::uint32_t>::max();
    std::uint8_t color_idx = std::numeric_limits<std::uint8_t>::max();
    std::uint64_t edge_bit = std::numeric_limits<std::uint64_t>::max();

    FrontCandidate() = default;

    FrontCandidate(const int activator_, const int color_idx_)
        : activator(static_cast<std::uint32_t>(activator_)),
          color_idx(static_cast<std::uint8_t>(color_idx_)) {}

    FrontCandidate(const int activator_, const int color_idx_, const std::uint64_t edge_bit_)
        : activator(static_cast<std::uint32_t>(activator_)),
          color_idx(static_cast<std::uint8_t>(color_idx_)),
          edge_bit(edge_bit_) {}
};

struct NeighborBond {
    int idx = -1;
    std::uint64_t edge_bit = std::numeric_limits<std::uint64_t>::max();
};

inline int collect_neighbors(const GridRegular& grid, const int idx, int out[6])
{
    int n = 0;

    const int xm = grid.neighbor_xm(idx);
    const int xp = grid.neighbor_xp(idx);
    if (xm >= 0) out[n++] = xm;
    if (xp >= 0) out[n++] = xp;

    if (grid.dim >= 2) {
        const int ym = grid.neighbor_ym(idx);
        const int yp = grid.neighbor_yp(idx);
        if (ym >= 0) out[n++] = ym;
        if (yp >= 0) out[n++] = yp;
    }

    if (grid.dim == 3) {
        const int zm = grid.neighbor_zm(idx);
        const int zp = grid.neighbor_zp(idx);
        if (zm >= 0) out[n++] = zm;
        if (zp >= 0) out[n++] = zp;
    }

    return n;
}

inline int collect_neighbor_bonds(const GridRegular& grid, const int idx, NeighborBond out[6])
{
    int n = 0;

    const int xm = grid.neighbor_xm(idx);
    const int xp = grid.neighbor_xp(idx);
    if (xm >= 0) out[n++] = NeighborBond{xm, static_cast<std::uint64_t>(xm)};
    if (xp >= 0) out[n++] = NeighborBond{xp, static_cast<std::uint64_t>(idx)};

    if (grid.dim >= 2) {
        const std::uint64_t y_offset = static_cast<std::uint64_t>(grid.total_size);
        const int ym = grid.neighbor_ym(idx);
        const int yp = grid.neighbor_yp(idx);
        if (ym >= 0) out[n++] = NeighborBond{ym, y_offset + static_cast<std::uint64_t>(ym)};
        if (yp >= 0) out[n++] = NeighborBond{yp, y_offset + static_cast<std::uint64_t>(idx)};
    }

    if (grid.dim == 3) {
        const std::uint64_t z_offset = 2ull * static_cast<std::uint64_t>(grid.total_size);
        const int zm = grid.neighbor_zm(idx);
        const int zp = grid.neighbor_zp(idx);
        if (zm >= 0) out[n++] = NeighborBond{zm, z_offset + static_cast<std::uint64_t>(zm)};
        if (zp >= 0) out[n++] = NeighborBond{zp, z_offset + static_cast<std::uint64_t>(idx)};
    }

    return n;
}

struct OpenBondGraph {
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> neighbors;

    OpenBondGraph() = default;

    OpenBondGraph(const int total_size,
                  const std::vector<std::pair<uint32_t, uint32_t>>& edge_pairs)
    {
        build(total_size, edge_pairs);
    }

    void build(const int total_size,
               const std::vector<std::pair<uint32_t, uint32_t>>& edge_pairs)
    {
        offsets.assign(static_cast<std::size_t>(total_size) + 1u, 0u);

        for (const auto& edge : edge_pairs) {
            if (edge.first >= static_cast<uint32_t>(total_size) ||
                edge.second >= static_cast<uint32_t>(total_size)) {
                continue;
            }
            ++offsets[static_cast<std::size_t>(edge.first) + 1u];
        }

        for (int i = 1; i <= total_size; ++i) {
            offsets[static_cast<std::size_t>(i)] += offsets[static_cast<std::size_t>(i - 1)];
        }

        neighbors.assign(edge_pairs.size(), 0u);
        std::vector<uint32_t> cursor(offsets.begin(), offsets.end());
        for (const auto& edge : edge_pairs) {
            if (edge.first >= static_cast<uint32_t>(total_size) ||
                edge.second >= static_cast<uint32_t>(total_size)) {
                continue;
            }
            neighbors[static_cast<std::size_t>(cursor[edge.first]++)] = edge.second;
        }
    }
};

struct ComponentSummary {
    int seed = -1;
    int size = 0;
};

struct TimeSplit {
    int pre = 0;
    int post = 0;
};

std::vector<int> shortest_open_bond_path_single_color(
    const GridRegular& grid,
    const OpenBondGraph& graph,
    const int active_val,
    const std::function<int(int)>& get_site)
{
    const int top_coord = (grid.dim == 2) ? (grid.SY - 1) : (grid.SZ - 1);
    std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
    std::vector<int> parent(static_cast<std::size_t>(grid.total_size), -2);
    std::vector<int> queue;
    queue.reserve(static_cast<std::size_t>(grid.total_size));

    for (int idx = 0; idx < grid.total_size; ++idx) {
        if (get_site(idx) != active_val) continue;
        if (grid.grow_coord(idx) != 0) continue;
        visited[static_cast<std::size_t>(idx)] = 1;
        parent[static_cast<std::size_t>(idx)] = -1;
        queue.push_back(idx);
    }

    int target = -1;
    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
        const int u = queue[qi];
        if (grid.grow_coord(u) == top_coord) {
            target = u;
            break;
        }

        const uint32_t begin = graph.offsets[static_cast<std::size_t>(u)];
        const uint32_t end = graph.offsets[static_cast<std::size_t>(u) + 1u];
        for (uint32_t k = begin; k < end; ++k) {
            const int v = static_cast<int>(graph.neighbors[static_cast<std::size_t>(k)]);
            if (visited[static_cast<std::size_t>(v)]) continue;
            if (get_site(v) != active_val) continue;
            visited[static_cast<std::size_t>(v)] = 1;
            parent[static_cast<std::size_t>(v)] = u;
            queue.push_back(v);
        }
    }

    std::vector<int> path;
    if (target < 0) return path;

    for (int cur = target; cur >= 0; cur = parent[static_cast<std::size_t>(cur)]) {
        path.push_back(cur);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<int> shortest_site_path_single_color(
    const GridRegular& grid,
    const int active_val,
    const std::function<int(int)>& get_site)
{
    std::vector<int> parent(static_cast<std::size_t>(grid.total_size), -2);
    std::vector<int> current;
    std::vector<int> next;
    current.reserve(static_cast<std::size_t>(compute_base_size(grid)));
    next.reserve(static_cast<std::size_t>(compute_base_size(grid)));

    for (int idx = 0; idx < grid.total_size; ++idx) {
        if (grid.grow_coord(idx) != 0) continue;
        if (get_site(idx) != active_val) continue;
        parent[static_cast<std::size_t>(idx)] = -1;
        current.push_back(idx);
    }

    int reached_top = -1;
    int neigh[6];

    while (!current.empty() && reached_top < 0) {
        next.clear();

        for (const int u : current) {
            if (grid.grow_coord(u) == grid.grow_top_coord()) {
                reached_top = u;
                break;
            }

            const int nneigh = collect_neighbors(grid, u, neigh);
            for (int ni = 0; ni < nneigh; ++ni) {
                const int v = neigh[ni];
                if (parent[static_cast<std::size_t>(v)] != -2) continue;
                if (get_site(v) != active_val) continue;

                parent[static_cast<std::size_t>(v)] = u;
                next.push_back(v);
            }
        }

        current.swap(next);
    }

    if (reached_top < 0) return {};

    std::vector<int> path;
    for (int at = reached_top; at != -1; at = parent[static_cast<std::size_t>(at)]) {
        path.push_back(at);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

inline std::uint64_t undirected_edge_key(const int u, const int v)
{
    const std::uint32_t a = static_cast<std::uint32_t>(std::min(u, v));
    const std::uint32_t b = static_cast<std::uint32_t>(std::max(u, v));
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

inline bool mark_bond_if_new(std::unordered_set<std::uint64_t>& tested_bonds,
                             const int u,
                             const int v)
{
    return tested_bonds.insert(undirected_edge_key(u, v)).second;
}

struct TestedBondBitmap {
    const GridRegular& grid;
    std::vector<std::uint64_t> words;

    explicit TestedBondBitmap(const GridRegular& grid_)
        : grid(grid_),
          words((static_cast<std::size_t>(grid_.dim) *
                     static_cast<std::size_t>(grid_.total_size) +
                 63u) / 64u,
                0u) {}

    std::uint64_t edge_index(const int u, const int v) const {
        const int lo = std::min(u, v);
        const int hi = std::max(u, v);
        const int diff = hi - lo;

        // X bonds occupy [0, total_size). X is periodic for the 2D/3D
        // geometries used here, so the wrap bond is anchored at x = SX - 1.
        if (lo / grid.SX == hi / grid.SX) {
            if (diff == 1) {
                return static_cast<std::uint64_t>(lo);
            }
            if (diff == grid.SX - 1) {
                return static_cast<std::uint64_t>(hi);
            }
        }

        if (grid.dim >= 2) {
            const std::uint64_t y_offset = static_cast<std::uint64_t>(grid.total_size);
            if (lo % grid.SX == hi % grid.SX) {
                if (diff == grid.SX) {
                    return y_offset + static_cast<std::uint64_t>(lo);
                }
                // Y is periodic only when growth is along Z (3D).
                if (grid.grow_axis != 1 &&
                    diff == grid.SX * (grid.SY - 1) &&
                    lo / grid.stride_z == hi / grid.stride_z) {
                    return y_offset + static_cast<std::uint64_t>(hi);
                }
            }
        }

        if (grid.dim == 3 && diff == grid.stride_z) {
            const std::uint64_t z_offset = 2ull * static_cast<std::uint64_t>(grid.total_size);
            return z_offset + static_cast<std::uint64_t>(lo);
        }

        throw std::runtime_error("TestedBondBitmap: non-neighbor bond");
    }

    bool mark_if_new(const int u, const int v) {
        const std::uint64_t bit = edge_index(u, v);
        return mark_if_new(bit);
    }

    bool mark_if_new(const std::uint64_t bit) {
        const std::size_t word_idx = static_cast<std::size_t>(bit >> 6u);
        const std::uint64_t mask = 1ull << (bit & 63u);
        std::uint64_t& word = words[word_idx];
        if ((word & mask) != 0u) return false;
        word |= mask;
        return true;
    }

    void mark_open(const int u, const int v) {
        const std::uint64_t bit = edge_index(u, v);
        mark_open(bit);
    }

    void mark_open(const std::uint64_t bit) {
        words[static_cast<std::size_t>(bit >> 6u)] |= (1ull << (bit & 63u));
    }

    bool contains(const int u, const int v) const {
        const std::uint64_t bit = edge_index(u, v);
        return contains(bit);
    }

    bool contains(const std::uint64_t bit) const {
        const std::uint64_t word = words[static_cast<std::size_t>(bit >> 6u)];
        return (word & (1ull << (bit & 63u))) != 0u;
    }
};

inline bool mark_bond_if_new(TestedBondBitmap& tested_bonds,
                             const int u,
                             const int v)
{
    return tested_bonds.mark_if_new(u, v);
}

std::vector<int> shortest_open_bond_path_single_color(
    const GridRegular& grid,
    const TestedBondBitmap& open_bonds,
    const int active_val,
    const std::function<int(int)>& get_site)
{
    std::vector<int> parent(static_cast<std::size_t>(grid.total_size), -2);
    std::vector<int> current;
    std::vector<int> next;
    current.reserve(static_cast<std::size_t>(compute_base_size(grid)));
    next.reserve(static_cast<std::size_t>(compute_base_size(grid)));

    for (int idx = 0; idx < grid.total_size; ++idx) {
        if (grid.grow_coord(idx) != 0) continue;
        if (get_site(idx) != active_val) continue;
        parent[static_cast<std::size_t>(idx)] = -1;
        current.push_back(idx);
    }

    int reached_top = -1;
    int neigh[6];

    while (!current.empty() && reached_top < 0) {
        next.clear();

        for (const int u : current) {
            if (grid.grow_coord(u) == grid.grow_top_coord()) {
                reached_top = u;
                break;
            }

            const int nneigh = collect_neighbors(grid, u, neigh);
            for (int ni = 0; ni < nneigh; ++ni) {
                const int v = neigh[ni];
                if (parent[static_cast<std::size_t>(v)] != -2) continue;
                if (get_site(v) != active_val) continue;
                if (!open_bonds.contains(u, v)) continue;

                parent[static_cast<std::size_t>(v)] = u;
                next.push_back(v);
            }
        }

        current.swap(next);
    }

    if (reached_top < 0) return {};

    std::vector<int> path;
    for (int at = reached_top; at != -1; at = parent[static_cast<std::size_t>(at)]) {
        path.push_back(at);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

inline double topology_uniform01(std::mt19937_64& rng)
{
    return std::generate_canonical<double, 53>(rng);
}

struct SparseFrontCandidates {
    static constexpr int max_candidates_per_target = 16;

    std::vector<int> touched_targets;
    std::unordered_map<int, int> slot_of_target;
    std::vector<std::array<std::uint32_t, max_candidates_per_target>> activators;
    std::vector<std::array<std::uint8_t, max_candidates_per_target>> colors;
    std::vector<std::array<std::uint64_t, max_candidates_per_target>> edge_bits;
    std::vector<unsigned char> counts;

    explicit SparseFrontCandidates(const int nsites = 0)
    {
        (void)nsites;
    }

    void reserve_frontier(const std::size_t expected_targets) {
        touched_targets.reserve(expected_targets);
        activators.reserve(expected_targets);
        colors.reserve(expected_targets);
        edge_bits.reserve(expected_targets);
        counts.reserve(expected_targets);
        slot_of_target.reserve(expected_targets);
    }

    void clear() {
        slot_of_target.clear();
        touched_targets.clear();
        activators.clear();
        colors.clear();
        edge_bits.clear();
        counts.clear();
    }

    void add(const int target, const FrontCandidate candidate) {
        int slot = -1;
        const auto it = slot_of_target.find(target);
        if (it == slot_of_target.end()) {
            slot = static_cast<int>(counts.size());
            slot_of_target.emplace(target, slot);
            touched_targets.push_back(target);
            activators.emplace_back();
            colors.emplace_back();
            edge_bits.emplace_back();
            counts.push_back(0);
        } else {
            slot = it->second;
        }

        unsigned char& count = counts[static_cast<std::size_t>(slot)];
        if (count < max_candidates_per_target) {
            activators[static_cast<std::size_t>(slot)][count] = candidate.activator;
            colors[static_cast<std::size_t>(slot)][count] = candidate.color_idx;
            edge_bits[static_cast<std::size_t>(slot)][count] = candidate.edge_bit;
            ++count;
        }
    }

    FrontCandidate candidate(const std::size_t slot, const int idx) const {
        return FrontCandidate{
            static_cast<int>(activators[slot][static_cast<std::size_t>(idx)]),
            static_cast<int>(colors[slot][static_cast<std::size_t>(idx)]),
            edge_bits[slot][static_cast<std::size_t>(idx)]
        };
    }
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

    t_center.reserve(static_cast<std::size_t>(n_blocks));
    j_w.reserve(static_cast<std::size_t>(n_blocks));

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
    if (ts.t.empty()) {
        throw std::runtime_error("build_mean_p_series: TimeSeries.t vazio");
    }
    if (ts.num_colors <= 0) {
        throw std::runtime_error("build_mean_p_series: TimeSeries.num_colors inválido");
    }
    if (static_cast<int>(ts.p_t.size()) != ts.num_colors) {
        throw std::runtime_error("build_mean_p_series: p_t.size() incompatível com num_colors");
    }
    if (static_cast<int>(ts.f_t.size()) != ts.num_colors) {
        throw std::runtime_error("build_mean_p_series: f_t.size() incompatível com num_colors");
    }

    const std::size_t T = ts.t.size();
    std::vector<double> p_mean(T, 0.0);

    for (const auto& row : ts.p_t) {
        if (row.size() != T) {
            throw std::runtime_error("build_mean_p_series: linhas de p_t com tamanhos diferentes de t");
        }
        for (std::size_t i = 0; i < T; ++i) {
            p_mean[i] += row[i];
        }
    }

    for (const auto& row : ts.f_t) {
        if (row.size() != T) {
            throw std::runtime_error("build_mean_p_series: linhas de f_t com tamanhos diferentes de t");
        }
    }

    const double inv = 1.0 / static_cast<double>(ts.p_t.size());
    for (double& v : p_mean) v *= inv;

    return p_mean;
}

inline double estimate_t_eq_from_timeseries(const TimeSeries& ts,
                                            const int window_roll = 15,
                                            const int window_block = 20,
                                            const double threshold = 1.0e-6)
{
    if (ts.t.empty()) {
        throw std::runtime_error("estimate_t_eq_from_timeseries: ts.t vazio");
    }

    const std::vector<double> p_mean = build_mean_p_series(ts);
    const std::vector<double> p_smoothed = centered_moving_average(p_mean, window_roll);

    std::vector<double> t_j;
    std::vector<double> j_w;
    block_mean_regular_time(ts.t, p_smoothed, window_block, t_j, j_w);
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

        if (std::isfinite(sp) && sp < threshold) {
            return t_s[static_cast<std::size_t>(i)];
        }
    }

    return std::numeric_limits<double>::quiet_NaN();
}

} // anonymous namespace


double network::target_fT_create(const int type_f_T,
                                const int t_i,
                                const double f_T,
                                const double a,
                                const double alpha)
{
    if (type_f_T == 0) return f_T;
    if (type_f_T == 1) return a * std::pow(t_i, alpha);
    throw std::invalid_argument("Invalid type_f_T value: " + std::to_string(type_f_T));
}

double network::generate_p(const int type_f_T,
                           const double p_t,
                           const int t_i,
                           const double f_current,
                           const double c,
                           const double f_T,
                           const double a,
                           const double alpha)
{
    const double f_target = target_fT_create(type_f_T, t_i, f_T, a, alpha);
    double p_next = p_t + c * (f_target - f_current);

    if (p_next > 1.0) p_next = 1.0;
    if (p_next < 0.0) p_next = 0.0;

    return p_next;
}

NetworkPattern network::create_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double c_value, const double f_T, const int type_f_T,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng, const bool save_compact)
{
    this->c = c_value;
    this->f_T = f_T;

    const GridRegular grid(dim, lenght_network);
    const std::vector<int> shape = (dim == 2)
        ? std::vector<int>{grid.SX, grid.SY}
        : std::vector<int>{grid.SX, grid.SY, grid.SZ};

    const bool is_node = (type_percolation == "node");
    constexpr bool calculate_detailed_properties = (HEIGHT_STOP_MULTIPLIER == 1);
    const long long base_size = compute_base_size(grid);
    const double norm_factor = static_cast<double>(base_size);

    if (num_colors > 125) {
        throw std::runtime_error(
            "create_network: num_colors exceeds int8_t site-state encoding capacity");
    }

    const int net_seed = rng.get_seed();

    std::vector<std::int8_t> site_state(
        static_cast<std::size_t>(grid.total_size),
        static_cast<std::int8_t>(-1));
    auto get_site = [&](const int idx) -> int {
        return static_cast<int>(site_state[static_cast<std::size_t>(idx)]);
    };
    auto set_site = [&](const int idx, const int value) {
        site_state[static_cast<std::size_t>(idx)] = static_cast<std::int8_t>(value);
    };

    // activation times: UINT32_MAX means never activated
    std::vector<uint32_t> activation_time(
        static_cast<std::size_t>(grid.total_size),
        std::numeric_limits<uint32_t>::max());

    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<double>> f_series(num_colors);
    std::vector<int>                 t_list;
    t_list.reserve(num_of_samples);

    std::vector<double> p_curr = p0;
    std::vector<int>    N_current(num_colors, 0);
    std::vector<double> f_current(num_colors, 0.0);
    std::vector<int>    max_heights(num_colors, 0);
    std::vector<int>    frontier;
    std::vector<int>    next_frontier;
    
    frontier.reserve(static_cast<std::size_t>(base_size));
    next_frontier.reserve(static_cast<std::size_t>(base_size));

    SparseFrontCandidates front_candidates(grid.total_size);
    front_candidates.reserve_frontier(static_cast<std::size_t>(base_size));

    auto commit_step = [&](const int t_k,
                           const std::vector<double>& p_vec,
                           const std::vector<double>& f_vec)
    {
        t_list.push_back(t_k);
        for (int c = 0; c < num_colors; ++c) {
            p_series[c].push_back(p_vec[c]);
            f_series[c].push_back(f_vec[c]);
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
            const int v = get_site(idx);

            if (v == prefer_neg || v == -1) {
                set_site(idx, active_val);
                frontier.push_back(idx);
                ++N_current[c];
                ++activated;
                if (idx >= 0 && idx < grid.total_size) {
                    activation_time[static_cast<std::size_t>(idx)] = 0;
                }
            }
            ++tries;
        }
    }

    // initialize PercolationSeries temporal decomposition containers
    ps_out.sp_lin_preteq.assign(num_colors, -1);
    ps_out.sp_path_lin_preteq.assign(num_colors, std::vector<int>{});
    ps_out.sp_lin_posteq.assign(num_colors, -1);
    ps_out.sp_path_lin_posteq.assign(num_colors, std::vector<int>{});
    ps_out.M_size_preteq.assign(num_colors, -1);
    ps_out.M_size_posteq.assign(num_colors, -1);

    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, std::vector<int>{});
    ps_out.color_percolation.clear();
    ps_out.percolation_order.clear();
    ps_out.M_size_at_perc.clear();

    for (int c = 0; c < num_colors; ++c) {
        f_current[c] = static_cast<double>(N_current[c]) / norm_factor;
    }
    commit_step(0, p_curr, f_current);

    int order_ctr = 0;
    std::vector<bool> percolated(num_colors, false);
    std::vector<bool> died(num_colors, false);
    std::vector<bool> finished(num_colors, false);

    // Guarda a ordem bruta de chegada ao topo, mas só escreve no ps_out no final
    std::vector<int> percolation_rank(num_colors, -1);

    auto finalize_time_series_only = [&]()
    {
        ps_out.color_percolation.clear();
        ps_out.percolation_order.clear();
        ps_out.M_size_at_perc.clear();
        ps_out.sp_len.assign(num_colors, -1);
        ps_out.sp_path_lin.assign(num_colors, std::vector<int>{});

        struct ValidPercolation {
            int rank;
            int color_idx;
        };
        std::vector<ValidPercolation> valid_percs;
        valid_percs.reserve(num_colors);

        for (int c = 0; c < num_colors; ++c) {
            if (!died[c] && percolated[c]) {
                valid_percs.push_back({percolation_rank[c], c});
            }
        }

        std::sort(valid_percs.begin(), valid_percs.end(),
                  [](const ValidPercolation& a, const ValidPercolation& b) {
                      return a.rank < b.rank;
                  });

        int valid_order_ctr = 0;
        for (const auto& vp : valid_percs) {
            ps_out.color_percolation.push_back(vp.color_idx + 1);
            ps_out.percolation_order.push_back(++valid_order_ctr);
        }
    };

    auto largest_component_single_color = [&](const int color_idx) -> int
    {
        const int active_val = color_to_active_value(num_colors, color_idx);

        std::vector<char> visited(grid.total_size, 0);
        std::vector<int> stack;
        int max_component = 0;

        for (int idx = 0; idx < grid.total_size; ++idx) {
            if (visited[idx]) continue;
            if (get_site(idx) != active_val) continue;

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
                    if (get_site(v) != active_val) return;

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

    // Union-find for open bonds. Encoding: 0 = inactive, negative = root
    // storing -component_size, positive = parent_index + 1.
    std::vector<int> dsu(static_cast<std::size_t>(grid.total_size), 0);

    auto dsu_make_active = [&](const int idx)
    {
        if (idx < 0 || idx >= grid.total_size) return;
        if (dsu[static_cast<std::size_t>(idx)] != 0) return;
        dsu[static_cast<std::size_t>(idx)] = -1;
    };

    auto dsu_find = [&](int x)
    {
        int root = x;
        while (dsu[static_cast<std::size_t>(root)] > 0) {
            root = dsu[static_cast<std::size_t>(root)] - 1;
        }
        while (x != root) {
            const int next = dsu[static_cast<std::size_t>(x)] - 1;
            dsu[static_cast<std::size_t>(x)] = root + 1;
            x = next;
        }
        return root;
    };

    auto dsu_union = [&](const int a, const int b)
    {
        if (a < 0 || b < 0 || a >= grid.total_size || b >= grid.total_size) return;
        dsu_make_active(a);
        dsu_make_active(b);

        int ra = dsu_find(a);
        int rb = dsu_find(b);
        if (ra == rb) return;

        if (dsu[static_cast<std::size_t>(ra)] >
            dsu[static_cast<std::size_t>(rb)]) {
            std::swap(ra, rb);
        }

        dsu[static_cast<std::size_t>(ra)] += dsu[static_cast<std::size_t>(rb)];
        dsu[static_cast<std::size_t>(rb)] = ra + 1;
    };

    auto largest_open_component_single_color = [&](const int color_idx) -> int
    {
        const int active_val = color_to_active_value(num_colors, color_idx);
        int max_component = 0;

        for (int idx = 0; idx < grid.total_size; ++idx) {
            if (get_site(idx) != active_val) continue;
            if (dsu[static_cast<std::size_t>(idx)] == 0) continue;
            const int root = dsu_find(idx);
            max_component = std::max(
                max_component,
                -dsu[static_cast<std::size_t>(root)]
            );
        }

        return max_component;
    };

    // To store chosen edge pairs during the simulation.
    std::vector<std::pair<uint32_t,uint32_t>> edge_pairs;
    TestedBondBitmap tested_bonds(grid);
    std::unique_ptr<TestedBondBitmap> open_bonds;
    if constexpr (calculate_detailed_properties) {
        if (!is_node) {
            open_bonds = std::make_unique<TestedBondBitmap>(grid);
        }
    }
    auto open_bond = [&](const int u,
                         const int v,
                         const std::uint64_t edge_bit =
                             std::numeric_limits<std::uint64_t>::max())
    {
        if constexpr (calculate_detailed_properties) {
            dsu_union(u, v);
            if (open_bonds) {
                if (edge_bit == std::numeric_limits<std::uint64_t>::max()) {
                    open_bonds->mark_open(u, v);
                } else {
                    open_bonds->mark_open(edge_bit);
                }
            }
            if (save_compact || is_node) {
                edge_pairs.emplace_back(static_cast<uint32_t>(u),
                                        static_cast<uint32_t>(v));
                edge_pairs.emplace_back(static_cast<uint32_t>(v),
                                        static_cast<uint32_t>(u));
            }
        }
    };

    for (int idx = 0; idx < grid.total_size; ++idx) {
        if (get_site(idx) > 0) {
            dsu_make_active(idx);
        }
    }

    // Independent RNG for purely topological active-active bonds.
    // These bonds must not consume the dynamical RNG used for active-inactive
    // activation attempts; otherwise p(t) changes even though N(t) does not.
    const std::uint64_t topology_seed =
        static_cast<std::uint64_t>(rng.get_seed()) ^
        0x9E3779B97F4A7C15ULL ^
        (static_cast<std::uint64_t>(lenght_network) << 32) ^
        (static_cast<std::uint64_t>(num_colors) << 48);
    std::mt19937_64 topology_rng(topology_seed);

    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        std::fill(f_current.begin(), f_current.end(), 0.0);
        next_frontier.clear();
        front_candidates.clear();

        if (is_node) {
            // Site percolation (Leath/SOP): each perimeter target exposed by
            // the current growth front is considered once. If the selected
            // attempt fails, the target is blocked forever.
            int neigh[6];

            for (const int idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = collect_neighbors(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const int viz_idx = neigh[ni];

                    const int vv = get_site(viz_idx);
                    if (vv >= 0) continue; // already active/blocked

                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx});
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const int viz = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int pick = rng.uniform_int(0, n_cand - 1);
                const FrontCandidate chosen = front_candidates.candidate(slot, pick);
                const int cor_idx = chosen.color_idx;

                if (rng.uniform_real(0.0, 1.0) >= p_curr[cor_idx]) {
                    set_site(viz, 0);
                    continue;
                }

                const int new_val = color_to_active_value(num_colors, cor_idx);
                set_site(viz, new_val);
                next_frontier.push_back(viz);
                ++N_current[cor_idx];
                activation_time[static_cast<std::size_t>(viz)] = static_cast<uint32_t>(t);

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (!percolated[cor_idx] && h == grid.grow_top_coord()) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                }

                open_bond(chosen.activator, viz);
            }
        } else {
            // Bond percolation (SOP): test every allowed bond from the current
            // growth front. A target activates if at least one incident allowed
            // bond opens in this time step.
            NeighborBond neigh[6];

            for (const int idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = collect_neighbor_bonds(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const int viz_idx = neigh[ni].idx;
                    const std::uint64_t edge_bit = neigh[ni].edge_bit;

                    const int vv = get_site(viz_idx);

                    if (vv == 0) {
                        continue; // blocked site
                    }

                    if (vv > 0) {
                        // Purely topological bond: active frontier site to an
                        // already-active neighbor of the same species. This
                        // never changes the state, N(t), f(t), frontier, parent
                        // or activation_time. It also uses topology_rng, not the
                        // dynamical rng, so p(t) is not indirectly changed.
                        const int neigh_color_idx = value_to_color_index(num_colors, vv);
                        if (neigh_color_idx != cor_idx) continue;

                        if (!tested_bonds.mark_if_new(edge_bit)) continue;

                        if (topology_uniform01(topology_rng) < p_curr[cor_idx]) {
                            open_bond(idx, viz_idx, edge_bit);
                        }
                        continue;
                    }

                    // Dynamical bond: active frontier site to an inactive
                    // compatible neighbor. Only this case can activate sites
                    // and therefore contribute to N(t), f(t), and p(t).
                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    if (!tested_bonds.mark_if_new(edge_bit)) continue;

                    if (rng.uniform_real(0.0, 1.0) < p_curr[cor_idx]) {
                        front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx, edge_bit});
                    }
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const int viz = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int target_before = get_site(viz);
                const bool target_without_species = (num_colors > 1 && target_before == -1);

                int cor_idx = -1;
                int chosen_u = -1;

                if (target_without_species) {
                    // For a species-free target, the accepted bond is picked
                    // uniformly, preserving the weight of species with more
                    // accepted links. Once that species wins, all accepted
                    // links from that same species become effective.
                    const int pick = rng.uniform_int(0, n_cand - 1);
                    const FrontCandidate chosen = front_candidates.candidate(slot, pick);
                    cor_idx = chosen.color_idx;
                    chosen_u = chosen.activator;
                } else {
                    // For a target that already belongs to a species, all
                    // accepted bonds from that same species are kept.
                    cor_idx = (num_colors == 1)
                        ? 0
                        : value_to_color_index(num_colors, target_before);

                    if (cor_idx < 0 || cor_idx >= num_colors) {
                        continue;
                    }

                    int same_color_count = 0;
                    for (int i = 0; i < n_cand; ++i) {
                        if (front_candidates.candidate(slot, i).color_idx == cor_idx) {
                            ++same_color_count;
                        }
                    }
                    if (same_color_count <= 0) continue;

                    const int parent_pick = rng.uniform_int(0, same_color_count - 1);
                    int seen = 0;
                    for (int i = 0; i < n_cand; ++i) {
                        const FrontCandidate fc = front_candidates.candidate(slot, i);
                        if (fc.color_idx != cor_idx) continue;
                        if (seen == parent_pick) {
                            chosen_u = fc.activator;
                            break;
                        }
                        ++seen;
                    }
                }

                if (chosen_u < 0 || cor_idx < 0 || cor_idx >= num_colors) {
                    continue;
                }

                const int new_val = color_to_active_value(num_colors, cor_idx);
                set_site(viz, new_val);
                dsu_make_active(viz);
                next_frontier.push_back(viz);
                ++N_current[cor_idx];
                activation_time[static_cast<std::size_t>(viz)] = static_cast<uint32_t>(t);

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (!percolated[cor_idx] && h == grid.grow_top_coord()) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                }

                if (target_without_species) {
                    for (int i = 0; i < n_cand; ++i) {
                        const FrontCandidate fc = front_candidates.candidate(slot, i);
                        if (fc.color_idx != cor_idx) continue;
                        open_bond(fc.activator, viz, fc.edge_bit);
                    }
                } else {
                    for (int i = 0; i < n_cand; ++i) {
                        const FrontCandidate fc = front_candidates.candidate(slot, i);
                        if (fc.color_idx != cor_idx) continue;
                        open_bond(fc.activator, viz, fc.edge_bit);
                    }
                }
            }
        }

        for (int c = 0; c < num_colors; ++c) {
            f_current[c] = static_cast<double>(N_current[c]) / norm_factor;
        }

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c) {
                std::cout
                    << ", p" << (c + 1) << "(t)=" << p_curr[c]
                    << ", f" << (c + 1) << "(t)="
                    << f_current[c]
                    << " max" << (c + 1) << "(t)=" << max_heights[c];
            }
            std::cout << '\n';
        }

        // Se a espécie não cresceu neste passo, então f_i(t) = 0.
        // Caso ainda não tenha percolado, ela morreu; caso já tenha percolado,
        // apenas finalizou o crescimento sem ser congelada no instante de percolação.
        for (int c = 0; c < num_colors; ++c) {
            if (f_current[c] <= 0.0) {
                if (!percolated[c]) {
                    died[c] = true;
                }
                finished[c] = true;
            }
        }

        bool all_dead = true;
        bool all_percolated = true;
        bool any_dead = false;
        bool any_percolated = false;
        bool all_terminal = true;

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

            // Estado terminal para a simulação global:
            // - percolated[c] == true: a espécie já atingiu o topo;
            // - died[c] == true: a espécie não tem mais crescimento, f_i(t)=0.
            // Espécies percoladas continuam crescendo enquanto a simulação global
            // não terminou, mas para o critério global elas já contam como terminais.
            if (!percolated[c] && !died[c]) {
                all_terminal = false;
            }
        }

        // Condições de parada:
        // 1) todas morreram sem percolar
        const bool stop_all_dead = all_dead;

        // 2) todas percolaram
        const bool stop_all_percolated = all_percolated;

        // 3) percolação parcial: pelo menos uma espécie percolou e todas as
        // demais espécies que não percolaram já morreram.
        const bool stop_partial_percolation =
            all_terminal && any_percolated && any_dead;

        if (stop_all_dead || stop_all_percolated || stop_partial_percolation) {
            if constexpr (!calculate_detailed_properties) {
                finalize_time_series_only();
                break;
            }

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

                const int active_val = color_to_active_value(num_colors, c);
                std::vector<int> path = is_node
                    ? shortest_site_path_single_color(grid, active_val, get_site)
                    : shortest_open_bond_path_single_color(
                        grid, *open_bonds, active_val, get_site);

                if (path.empty()) {
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
                ps_out.M_size_at_perc.push_back(
                    is_node
                        ? largest_component_single_color(c)
                        : largest_open_component_single_color(c)
                );
            }

            break;
        }


        std::vector<double> p_next(num_colors);
        for (int c = 0; c < num_colors; ++c) {
            p_next[c] = finished[c] ? p_curr[c]
                                    : generate_p(type_f_T, p_curr[c], t, f_current[c], c_value, f_T, a, alpha);
        }

        commit_step(t, p_next, f_current);
        frontier.swap(next_frontier);
        p_curr.swap(p_next);
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.f_t = std::move(f_series);
    ts_out.t   = std::move(t_list);

    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < static_cast<int>(rho.size())) {
            ps_out.rho.push_back(rho[i]);
        }
    }

    ts_out.t_eq = estimate_t_eq_from_timeseries(
        ts_out,
        15,      // window_roll
        20,      // window_block
        1.0e-6   // s_prime threshold
    );
    ps_out.t_eq = ts_out.t_eq;

    if constexpr (calculate_detailed_properties) {
        if (!std::isfinite(ps_out.t_eq)) {
            for (int c = 0; c < num_colors; ++c) {
                ps_out.sp_lin_preteq[c] = -1;
                ps_out.sp_path_lin_preteq[c].clear();
                ps_out.sp_lin_posteq[c] = -1;
                ps_out.sp_path_lin_posteq[c].clear();
                ps_out.M_size_preteq[c] = -1;
                ps_out.M_size_posteq[c] = -1;
            }
        } else {
        for (int c = 0; c < num_colors; ++c) {
            if (ps_out.sp_len.size() <= static_cast<size_t>(c)) continue;
            if (ps_out.sp_len[c] < 0) {
                ps_out.sp_lin_preteq[c] = -1;
                ps_out.sp_path_lin_preteq[c].clear();
                ps_out.sp_lin_posteq[c] = -1;
                ps_out.sp_path_lin_posteq[c].clear();
                ps_out.M_size_preteq[c] = -1;
                ps_out.M_size_posteq[c] = -1;
                continue;
            }

            const int active_val = color_to_active_value(num_colors, c);
            const std::vector<int>& path = ps_out.sp_path_lin[c];
            ps_out.sp_path_lin_preteq[c].clear();
            ps_out.sp_path_lin_posteq[c].clear();
            if (!path.empty()) {
                ps_out.sp_path_lin_preteq[c].push_back(path.front());
                ps_out.sp_path_lin_posteq[c].push_back(path.front());
            }

            int sp_pre = 0;
            int sp_post = 0;
            for (std::size_t k = 1; k < path.size(); ++k) {
                const int idx = path[k];
                if (idx < 0 || idx >= grid.total_size ||
                    get_site(idx) != active_val ||
                    activation_time[static_cast<std::size_t>(idx)] ==
                        std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error(
                        "create_network: shortest path contains invalid activation time");
                }

                if (static_cast<int>(activation_time[static_cast<std::size_t>(idx)]) <= ps_out.t_eq) {
                    ++sp_pre;
                    ps_out.sp_path_lin_preteq[c].push_back(idx);
                } else {
                    ++sp_post;
                    ps_out.sp_path_lin_posteq[c].push_back(idx);
                }
            }
            ps_out.sp_lin_preteq[c] = sp_pre;
            ps_out.sp_lin_posteq[c] = sp_post;

            int m_pre = 0;
            int m_post = 0;

            if (is_node) {
                std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
                std::vector<int> stack;
                std::vector<int> comp_nodes;
                std::vector<int> best_nodes;

                for (int idx = 0; idx < grid.total_size; ++idx) {
                    if (visited[static_cast<std::size_t>(idx)]) continue;
                    if (get_site(idx) != active_val) continue;

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
                            if (get_site(v) != active_val) return;

                            visited[static_cast<std::size_t>(v)] = 1;
                            stack.push_back(v);
                        });
                    }

                    if (comp_nodes.size() > best_nodes.size()) {
                        best_nodes = comp_nodes;
                    }
                }

                for (const int idx : best_nodes) {
                    if (activation_time[static_cast<std::size_t>(idx)] ==
                        std::numeric_limits<uint32_t>::max()) {
                        throw std::runtime_error(
                            "create_network: largest component contains invalid activation time");
                    }
                    if (static_cast<int>(activation_time[static_cast<std::size_t>(idx)]) <= ps_out.t_eq) ++m_pre;
                    else ++m_post;
                }
            } else {
                int best_root = -1;
                int best_size = 0;

                for (int idx = 0; idx < grid.total_size; ++idx) {
                    if (get_site(idx) != active_val) continue;
                    if (dsu[static_cast<std::size_t>(idx)] == 0) continue;
                    const int root = dsu_find(idx);
                    const int size = -dsu[static_cast<std::size_t>(root)];
                    if (size > best_size) {
                        best_size = size;
                        best_root = root;
                    }
                }

                if (best_root >= 0) {
                    for (int idx = 0; idx < grid.total_size; ++idx) {
                        if (get_site(idx) != active_val) continue;
                        if (dsu[static_cast<std::size_t>(idx)] == 0) continue;
                        if (dsu_find(idx) != best_root) continue;
                        if (activation_time[static_cast<std::size_t>(idx)] ==
                            std::numeric_limits<uint32_t>::max()) {
                            throw std::runtime_error(
                                "create_network: largest bond component contains invalid activation time");
                        }
                        if (static_cast<int>(activation_time[static_cast<std::size_t>(idx)]) <= ps_out.t_eq) ++m_pre;
                        else ++m_post;
                    }
                }
            }

            ps_out.M_size_preteq[c] = m_pre;
            ps_out.M_size_posteq[c] = m_post;
        }
        }
    }

    if (save_compact) {
        NetworkCompact netc;
        netc.N = static_cast<NetworkCompact::index_t>(grid.total_size);
        netc.pos_flat.resize(netc.N);
        for (NetworkCompact::index_t i = 0; i < netc.N; ++i) netc.pos_flat[i] = i;

        netc.species.resize(netc.N);
        for (NetworkCompact::index_t i = 0; i < netc.N; ++i) {
            const int v = get_site(static_cast<int>(i));
            if (v <= 0) {
                netc.species[i] = 0;
            } else {
                const int color_idx = value_to_color_index(num_colors, v);
                netc.species[i] = static_cast<uint8_t>(color_idx + 1);
            }
        }

        netc.activation_time.resize(netc.N);
        for (NetworkCompact::index_t i = 0; i < netc.N; ++i) netc.activation_time[i] = activation_time[i];

        if (!edge_pairs.empty()) {
            netc.build_csr_from_edge_pairs(edge_pairs);
        } else {
            netc.edge_offsets.assign(netc.N + 1, 0);
            netc.edges.clear();
        }

        if (!ps_out.color_percolation.empty()) {
            try {
                std::ostringstream fn;
                fn << "results/network_compact_seed_" << net_seed
                   << "_L_" << lenght_network << "_T_" << num_of_samples << ".bin";
                save_data sd;
                sd.save_network_compact_bin(netc, fn.str());
            } catch (const std::exception &e) {
                std::cerr << "Warning: failed to write compact network: " << e.what() << '\n';
            }
        } else {
            std::cout << "[create_network] no species percolated: not saving compact network." << std::endl;
        }
    }

    std::vector<uint32_t>().swap(activation_time);
    std::vector<int>().swap(dsu);
    std::vector<std::pair<uint32_t,uint32_t>>().swap(edge_pairs);
    SparseFrontCandidates empty_front_candidates;
    std::swap(front_candidates, empty_front_candidates);

    NetworkPattern net(dim, shape, num_colors, rho, save_compact);
    net.seed = net_seed;
    if (save_compact) {
        for (int idx = 0; idx < grid.total_size; ++idx) {
            net.set(idx, get_site(idx));
        }
    }
    return net;
}

NetworkPattern network::animate_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double c_value, const double f_T, const int type_f_T,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng)
{
    this->c = c_value;
    this->f_T = f_T;

    const GridRegular grid(dim, lenght_network);
    const std::vector<int> shape = (dim == 2)
        ? std::vector<int>{grid.SX, grid.SY}
        : std::vector<int>{grid.SX, grid.SY, grid.SZ};

    const bool is_node = (type_percolation == "node");
    constexpr bool calculate_detailed_properties = (HEIGHT_STOP_MULTIPLIER == 1);
    const long long base_size = compute_base_size(grid);
    const double norm_factor = static_cast<double>(base_size);

    if (num_colors > 125) {
        throw std::runtime_error(
            "animate_network: num_colors exceeds int8_t site-state encoding capacity");
    }

    std::vector<std::int8_t> site_state(
        static_cast<std::size_t>(grid.total_size),
        static_cast<std::int8_t>(-1));
    auto get_site = [&](const int idx) -> int {
        return static_cast<int>(site_state[static_cast<std::size_t>(idx)]);
    };
    auto set_site = [&](const int idx, const int value) {
        site_state[static_cast<std::size_t>(idx)] = static_cast<std::int8_t>(value);
    };

    // Codificação:
    //   -1 : nunca ativado
    //    0 : bloqueado no caso node
    // NUMBER*(c+1) + t : espécie c ativada no tempo t
    const int SPECIES_FACTOR = ANIMATION_SPECIES_FACTOR;

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
    std::vector<std::vector<double>> f_series(num_colors);
    std::vector<int>                 t_list;
    t_list.reserve(num_of_samples);

    std::vector<double> p_curr = p0;
    std::vector<int>    N_current(num_colors, 0);
    std::vector<double> f_current(num_colors, 0.0);
    std::vector<int>    max_heights(num_colors, 0);
    std::vector<int>    frontier;
    std::vector<int>    next_frontier;

    frontier.reserve(static_cast<std::size_t>(base_size));
    next_frontier.reserve(static_cast<std::size_t>(base_size));

    SparseFrontCandidates front_candidates(grid.total_size);
    front_candidates.reserve_frontier(static_cast<std::size_t>(base_size));

    auto commit_step = [&](const int t_k,
                           const std::vector<double>& p_vec,
                           const std::vector<double>& f_vec)
    {
        t_list.push_back(t_k);
        for (int c = 0; c < num_colors; ++c) {
            p_series[c].push_back(p_vec[c]);
            f_series[c].push_back(f_vec[c]);
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
    ps_out.t_eq = std::numeric_limits<double>::quiet_NaN();
    ps_out.sp_lin_preteq.assign(num_colors, -1);
    ps_out.sp_path_lin_preteq.assign(num_colors, std::vector<int>{});
    ps_out.sp_lin_posteq.assign(num_colors, -1);
    ps_out.sp_path_lin_posteq.assign(num_colors, std::vector<int>{});
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
            const int v = get_site(idx);

            if (v == prefer_neg || v == -1) {
                set_site(idx, active_val);
                activation_code[static_cast<std::size_t>(idx)] =
                    static_cast<NetworkPattern::state_t>(color_mul[c]);
                frontier.push_back(idx);
                ++N_current[c];
                ++activated;
            }
            ++tries;
        }
    }

    for (int c = 0; c < num_colors; ++c) {
        f_current[c] = static_cast<double>(N_current[c]) / norm_factor;
    }
    commit_step(0, p_curr, f_current);

    int order_ctr = 0;
    std::vector<bool> percolated(num_colors, false);
    std::vector<bool> died(num_colors, false);
    std::vector<bool> finished(num_colors, false);
    std::vector<int>  percolation_rank(num_colors, -1);

    // cache leve do maior componente global por cor: semente + tamanho.
    std::vector<ComponentSummary> largest_comp_cache(num_colors);
    bool global_metrics_finalized = false;
    std::vector<std::pair<uint32_t,uint32_t>> edge_pairs;

    auto largest_component_single_color_summary = [&](const int color_idx) -> ComponentSummary
    {
        const int active_val = color_to_active_value(num_colors, color_idx);

        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<int> stack;
        ComponentSummary best;

        for (int idx = 0; idx < grid.total_size; ++idx) {
            if (visited[static_cast<std::size_t>(idx)]) continue;
            if (get_site(idx) != active_val) continue;

            int comp_size = 0;
            stack.clear();
            stack.push_back(idx);
            visited[static_cast<std::size_t>(idx)] = 1;

            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                ++comp_size;

                grid.for_each_neighbor(u, [&](const int v) {
                    if (v < 0) return;
                    if (visited[static_cast<std::size_t>(v)]) return;
                    if (get_site(v) != active_val) return;

                    visited[static_cast<std::size_t>(v)] = 1;
                    stack.push_back(v);
                });
            }

            if (comp_size > best.size) {
                best.seed = idx;
                best.size = comp_size;
            }
        }

        return best;
    };

    auto largest_open_component_single_color_summary =
        [&](const int color_idx, const OpenBondGraph& graph) -> ComponentSummary
    {
        const int active_val = color_to_active_value(num_colors, color_idx);

        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<int> stack;
        ComponentSummary best;

        for (int idx = 0; idx < grid.total_size; ++idx) {
            if (visited[static_cast<std::size_t>(idx)]) continue;
            if (get_site(idx) != active_val) continue;

            int comp_size = 0;
            stack.clear();
            stack.push_back(idx);
            visited[static_cast<std::size_t>(idx)] = 1;

            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                ++comp_size;

                const uint32_t begin = graph.offsets[static_cast<std::size_t>(u)];
                const uint32_t end = graph.offsets[static_cast<std::size_t>(u) + 1u];
                for (uint32_t k = begin; k < end; ++k) {
                    const int v = static_cast<int>(graph.neighbors[static_cast<std::size_t>(k)]);
                    if (visited[static_cast<std::size_t>(v)]) continue;
                    if (get_site(v) != active_val) continue;
                    visited[static_cast<std::size_t>(v)] = 1;
                    stack.push_back(v);
                }
            }

            if (comp_size > best.size) {
                best.seed = idx;
                best.size = comp_size;
            }
        }

        return best;
    };

    auto count_component_time_split_single_color =
        [&](const int seed, const int color_idx) -> TimeSplit
    {
        TimeSplit split;
        if (seed < 0) return split;

        const int active_val = color_to_active_value(num_colors, color_idx);
        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<int> stack;
        stack.push_back(seed);
        visited[static_cast<std::size_t>(seed)] = 1;

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();

            int t_site = -1;
            const bool ok = is_active_color_site_encoded(
                activation_code, u, color_idx, SPECIES_FACTOR, &t_site);
            if (!ok) {
                throw std::runtime_error(
                    "animate_network: maior componente contém nó inválido no encoded");
            }
            if (t_site <= ps_out.t_eq) ++split.pre;
            else ++split.post;

            grid.for_each_neighbor(u, [&](const int v) {
                if (v < 0) return;
                if (visited[static_cast<std::size_t>(v)]) return;
                if (get_site(v) != active_val) return;
                visited[static_cast<std::size_t>(v)] = 1;
                stack.push_back(v);
            });
        }

        return split;
    };

    auto count_open_component_time_split_single_color =
        [&](const int seed, const int color_idx, const OpenBondGraph& graph) -> TimeSplit
    {
        TimeSplit split;
        if (seed < 0) return split;

        const int active_val = color_to_active_value(num_colors, color_idx);
        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<int> stack;
        stack.push_back(seed);
        visited[static_cast<std::size_t>(seed)] = 1;

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();

            int t_site = -1;
            const bool ok = is_active_color_site_encoded(
                activation_code, u, color_idx, SPECIES_FACTOR, &t_site);
            if (!ok) {
                throw std::runtime_error(
                    "animate_network: maior componente contém nó inválido no encoded");
            }
            if (t_site <= ps_out.t_eq) ++split.pre;
            else ++split.post;

            const uint32_t begin = graph.offsets[static_cast<std::size_t>(u)];
            const uint32_t end = graph.offsets[static_cast<std::size_t>(u) + 1u];
            for (uint32_t k = begin; k < end; ++k) {
                const int v = static_cast<int>(graph.neighbors[static_cast<std::size_t>(k)]);
                if (visited[static_cast<std::size_t>(v)]) continue;
                if (get_site(v) != active_val) continue;
                visited[static_cast<std::size_t>(v)] = 1;
                stack.push_back(v);
            }
        }

        return split;
    };

    auto finalize_global_metrics = [&]()
    {
        ps_out.color_percolation.clear();
        ps_out.percolation_order.clear();
        ps_out.M_size_at_perc.clear();

        ps_out.sp_len.assign(num_colors, -1);
        ps_out.sp_path_lin.assign(num_colors, std::vector<int>{});

        std::fill(largest_comp_cache.begin(), largest_comp_cache.end(), ComponentSummary{});
        const OpenBondGraph open_bond_graph(grid.total_size, edge_pairs);

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

            const int active_val = color_to_active_value(num_colors, c);
            std::vector<int> path = is_node
                ? shortest_site_path_single_color(grid, active_val, get_site)
                : shortest_open_bond_path_single_color(
                    grid, open_bond_graph, active_val, get_site);

            if (path.empty()) {
                ps_out.sp_len[c] = -1;
                ps_out.sp_path_lin[c].clear();
                continue;
            }

            ps_out.sp_path_lin[c] = std::move(path);
            ps_out.sp_len[c] = static_cast<int>(ps_out.sp_path_lin[c].size()) - 1;

            largest_comp_cache[c] = is_node
                ? largest_component_single_color_summary(c)
                : largest_open_component_single_color_summary(c, open_bond_graph);

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
            ps_out.M_size_at_perc.push_back(largest_comp_cache[c].size);
        }

        global_metrics_finalized = true;
    };

    auto finalize_time_series_only = [&]()
    {
        ps_out.color_percolation.clear();
        ps_out.percolation_order.clear();
        ps_out.M_size_at_perc.clear();
        ps_out.sp_len.assign(num_colors, -1);
        ps_out.sp_path_lin.assign(num_colors, std::vector<int>{});

        struct ValidPercolation {
            int rank;
            int color_idx;
        };
        std::vector<ValidPercolation> valid_percs;
        valid_percs.reserve(static_cast<std::size_t>(num_colors));

        for (int c = 0; c < num_colors; ++c) {
            if (!died[c] && percolated[c]) {
                valid_percs.push_back({percolation_rank[c], c});
            }
        }

        std::sort(valid_percs.begin(), valid_percs.end(),
                  [](const ValidPercolation& a, const ValidPercolation& b) {
                      return a.rank < b.rank;
                  });

        int valid_order_ctr = 0;
        for (const auto& vp : valid_percs) {
            ps_out.color_percolation.push_back(vp.color_idx + 1);
            ps_out.percolation_order.push_back(++valid_order_ctr);
        }

        global_metrics_finalized = true;
    };

    TestedBondBitmap tested_bonds(grid);
    const std::uint64_t topology_seed =
        static_cast<std::uint64_t>(rng.get_seed()) ^
        0x9E3779B97F4A7C15ULL ^
        (static_cast<std::uint64_t>(lenght_network) << 32) ^
        (static_cast<std::uint64_t>(num_colors) << 48);
    std::mt19937_64 topology_rng(topology_seed);

    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        std::fill(f_current.begin(), f_current.end(), 0.0);
        next_frontier.clear();
        front_candidates.clear();

        if (is_node) {
            // Same activation logic used by create_network for site/SOP.
            int neigh[6];

            for (const int idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = collect_neighbors(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const int viz_idx = neigh[ni];

                    const int vv = get_site(viz_idx);
                    if (vv >= 0) continue;

                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx});
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const int viz = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int pick = rng.uniform_int(0, n_cand - 1);
                const FrontCandidate chosen = front_candidates.candidate(slot, pick);
                const int cor_idx = chosen.color_idx;

                if (rng.uniform_real(0.0, 1.0) >= p_curr[cor_idx]) {
                    set_site(viz, 0);
                    activation_code[static_cast<std::size_t>(viz)] =
                        static_cast<NetworkPattern::state_t>(0);
                    continue;
                }

                const int new_val = color_to_active_value(num_colors, cor_idx);
                set_site(viz, new_val);
                activation_code[static_cast<std::size_t>(viz)] =
                    static_cast<NetworkPattern::state_t>(color_mul[cor_idx] + t);
                next_frontier.push_back(viz);
                ++N_current[cor_idx];

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (!percolated[cor_idx] && h == grid.grow_top_coord()) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                }

                if constexpr (calculate_detailed_properties) {
                    edge_pairs.emplace_back(static_cast<uint32_t>(chosen.activator),
                                            static_cast<uint32_t>(viz));
                    edge_pairs.emplace_back(static_cast<uint32_t>(viz),
                                            static_cast<uint32_t>(chosen.activator));
                }
            }
        } else {
            // Same activation logic used by create_network for bond/SOP.
            // All accepted bonds from the current front are collected before
            // any target site is activated, removing for-loop order bias.
            NeighborBond neigh[6];

            for (const int idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = collect_neighbor_bonds(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const int viz_idx = neigh[ni].idx;
                    const std::uint64_t edge_bit = neigh[ni].edge_bit;

                    const int vv = get_site(viz_idx);

                    if (vv == 0) {
                        continue;
                    }

                    if (vv > 0) {
                        const int neigh_color_idx = value_to_color_index(num_colors, vv);
                        if (neigh_color_idx != cor_idx) continue;

                        if (!tested_bonds.mark_if_new(edge_bit)) continue;

                        if (topology_uniform01(topology_rng) < p_curr[cor_idx]) {
                            if constexpr (calculate_detailed_properties) {
                                edge_pairs.emplace_back(static_cast<uint32_t>(idx),
                                                        static_cast<uint32_t>(viz_idx));
                                edge_pairs.emplace_back(static_cast<uint32_t>(viz_idx),
                                                        static_cast<uint32_t>(idx));
                            }
                        }
                        continue;
                    }

                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    if (!tested_bonds.mark_if_new(edge_bit)) continue;

                    if (rng.uniform_real(0.0, 1.0) < p_curr[cor_idx]) {
                        front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx, edge_bit});
                    }
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const int viz = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int target_before = get_site(viz);
                const bool target_without_species = (num_colors > 1 && target_before == -1);

                int cor_idx = -1;
                int chosen_u = -1;

                if (target_without_species) {
                    const int pick = rng.uniform_int(0, n_cand - 1);
                    const FrontCandidate chosen = front_candidates.candidate(slot, pick);
                    cor_idx = chosen.color_idx;
                    chosen_u = chosen.activator;
                } else {
                    cor_idx = (num_colors == 1)
                        ? 0
                        : value_to_color_index(num_colors, target_before);

                    if (cor_idx < 0 || cor_idx >= num_colors) {
                        continue;
                    }

                    int same_color_count = 0;
                    for (int i = 0; i < n_cand; ++i) {
                        if (front_candidates.candidate(slot, i).color_idx == cor_idx) {
                            ++same_color_count;
                        }
                    }
                    if (same_color_count <= 0) continue;

                    const int parent_pick = rng.uniform_int(0, same_color_count - 1);
                    int seen = 0;
                    for (int i = 0; i < n_cand; ++i) {
                        const FrontCandidate fc = front_candidates.candidate(slot, i);
                        if (fc.color_idx != cor_idx) continue;
                        if (seen == parent_pick) {
                            chosen_u = fc.activator;
                            break;
                        }
                        ++seen;
                    }
                }

                if (chosen_u < 0 || cor_idx < 0 || cor_idx >= num_colors) {
                    continue;
                }

                const int new_val = color_to_active_value(num_colors, cor_idx);
                set_site(viz, new_val);
                activation_code[static_cast<std::size_t>(viz)] =
                    static_cast<NetworkPattern::state_t>(color_mul[cor_idx] + t);
                next_frontier.push_back(viz);
                ++N_current[cor_idx];

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (!percolated[cor_idx] && h == grid.grow_top_coord()) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                }

                if (target_without_species) {
                    for (int i = 0; i < n_cand; ++i) {
                        const FrontCandidate fc = front_candidates.candidate(slot, i);
                        if (fc.color_idx != cor_idx) continue;
                        if constexpr (calculate_detailed_properties) {
                            edge_pairs.emplace_back(static_cast<uint32_t>(fc.activator),
                                                    static_cast<uint32_t>(viz));
                            edge_pairs.emplace_back(static_cast<uint32_t>(viz),
                                                    static_cast<uint32_t>(fc.activator));
                        }
                    }
                } else {
                    for (int i = 0; i < n_cand; ++i) {
                        const FrontCandidate fc = front_candidates.candidate(slot, i);
                        if (fc.color_idx != cor_idx) continue;
                        if constexpr (calculate_detailed_properties) {
                            edge_pairs.emplace_back(static_cast<uint32_t>(fc.activator),
                                                    static_cast<uint32_t>(viz));
                            edge_pairs.emplace_back(static_cast<uint32_t>(viz),
                                                    static_cast<uint32_t>(fc.activator));
                        }
                    }
                }
            }
        }

        for (int c = 0; c < num_colors; ++c) {
            f_current[c] = static_cast<double>(N_current[c]) / norm_factor;
        }

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c) {
                std::cout
                    << ", p" << (c + 1) << "(t)=" << p_curr[c]
                    << ", f" << (c + 1) << "(t)="
                    << f_current[c]
                    << " max" << (c + 1) << "(t)=" << max_heights[c];
            }
            std::cout << '\n';
        }

        for (int c = 0; c < num_colors; ++c) {
            if (f_current[c] <= 0.0) {
                if (!percolated[c]) {
                    died[c] = true;
                }
                finished[c] = true;
            }
        }

        bool all_dead = true;
        bool all_percolated = true;
        bool any_dead = false;
        bool any_percolated = false;
        bool all_terminal = true;

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

            if (!percolated[c] && !died[c]) {
                all_terminal = false;
            }
        }

        const bool stop_all_dead = all_dead;
        const bool stop_all_percolated = all_percolated;
        const bool stop_partial_percolation =
            all_terminal && any_percolated && any_dead;

        if (stop_all_dead || stop_all_percolated || stop_partial_percolation) {
            if constexpr (calculate_detailed_properties) {
                finalize_global_metrics();
            } else {
                finalize_time_series_only();
            }
            break;
        }


        std::vector<double> p_next(num_colors);
        for (int c = 0; c < num_colors; ++c) {
            p_next[c] = finished[c]
                ? p_curr[c]
                : generate_p(type_f_T, p_curr[c], t, f_current[c], c_value, f_T, a, alpha);
        }

        commit_step(t, p_next, f_current);
        frontier.swap(next_frontier);
        p_curr.swap(p_next);
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.f_t = std::move(f_series);
    ts_out.t   = std::move(t_list);

    if (!global_metrics_finalized) {
        if constexpr (calculate_detailed_properties) {
            finalize_global_metrics();
        } else {
            finalize_time_series_only();
        }
    }

    ts_out.t_eq = estimate_t_eq_from_timeseries(
        ts_out,
        15,      // window_roll
        20,      // window_block
        1.0e-6   // s_prime threshold
    );
    ps_out.t_eq = ts_out.t_eq;

    if constexpr (calculate_detailed_properties) {
        if (!std::isfinite(ps_out.t_eq)) {
            for (int c = 0; c < num_colors; ++c) {
                ps_out.sp_lin_preteq[c] = -1;
                ps_out.sp_path_lin_preteq[c].clear();
                ps_out.sp_lin_posteq[c] = -1;
                ps_out.sp_path_lin_posteq[c].clear();
                ps_out.M_size_preteq[c] = 0;
                ps_out.M_size_posteq[c] = 0;
            }
        } else {
        OpenBondGraph open_bond_graph_for_splits;
        if (!is_node) {
            open_bond_graph_for_splits.build(grid.total_size, edge_pairs);
        }

        for (int c = 0; c < num_colors; ++c) {
            if (ps_out.sp_len[c] < 0 || ps_out.sp_path_lin[c].empty()) {
                ps_out.sp_lin_preteq[c] = -1;
                ps_out.sp_path_lin_preteq[c].clear();
                ps_out.sp_lin_posteq[c] = -1;
                ps_out.sp_path_lin_posteq[c].clear();
                ps_out.M_size_preteq[c] = 0;
                ps_out.M_size_posteq[c] = 0;
                continue;
            }

            // ------------------------------------------------------------
            // Decomposição temporal do shortest path global
            // ------------------------------------------------------------
            int sp_pre = 0;
            int sp_post = 0;

            ps_out.sp_path_lin_preteq[c].clear();
            ps_out.sp_path_lin_posteq[c].clear();

            const std::vector<int>& path = ps_out.sp_path_lin[c];
            if (!path.empty()) {
                // mantém o nó inicial da base em ambos os caminhos, se você quiser
                // preservar a referência completa do caminho original
                ps_out.sp_path_lin_preteq[c].push_back(path.front());
                ps_out.sp_path_lin_posteq[c].push_back(path.front());
            }

            for (std::size_t k = 1; k < path.size(); ++k) {
                const int idx = path[k];

                int t_site = -1;
                const bool ok = is_active_color_site_encoded(
                    activation_code, idx, c, SPECIES_FACTOR, &t_site);

                if (!ok) {
                    throw std::runtime_error(
                        "animate_network: shortest path contém nó inválido no encoded");
                }

                if (t_site <= ps_out.t_eq) {
                    ++sp_pre;
                    ps_out.sp_path_lin_preteq[c].push_back(idx);
                } else {
                    ++sp_post;
                    ps_out.sp_path_lin_posteq[c].push_back(idx);
                }
            }

            ps_out.sp_lin_preteq[c] = sp_pre;
            ps_out.sp_lin_posteq[c] = sp_post;

            // ------------------------------------------------------------
            // Decomposição temporal do maior componente global
            // ------------------------------------------------------------
            const TimeSplit comp_split = is_node
                ? count_component_time_split_single_color(largest_comp_cache[c].seed, c)
                : count_open_component_time_split_single_color(
                    largest_comp_cache[c].seed, c, open_bond_graph_for_splits);

            ps_out.M_size_preteq[c] = comp_split.pre;
            ps_out.M_size_posteq[c] = comp_split.post;
        }
        }
    }

    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < static_cast<int>(rho.size())) {
            ps_out.rho.push_back(rho[i]);
        }
    }

    std::vector<std::int8_t>().swap(site_state);
    std::vector<int>().swap(frontier);
    std::vector<int>().swap(next_frontier);
    std::vector<ComponentSummary>().swap(largest_comp_cache);
    SparseFrontCandidates empty_front_candidates;
    std::swap(front_candidates, empty_front_candidates);

    NetworkPattern net_animation(dim, shape, num_colors, rho, false);
    net_animation.data = std::move(activation_code);
    net_animation.edge_pairs = std::move(edge_pairs);
    return net_animation;
}  

NetworkPattern network::filter_percolating_clusters_from_encoded(
    const NetworkPattern& encoded_net) const
{
    if (encoded_net.dim != 2 && encoded_net.dim != 3) {
        throw std::invalid_argument(
            "[filter_percolating_clusters_from_encoded] dim deve ser 2 ou 3");
    }

    if (static_cast<int>(encoded_net.shape.size()) != encoded_net.dim) {
        throw std::invalid_argument(
            "[filter_percolating_clusters_from_encoded] shape incompatível com dim");
    }

    const int L = encoded_net.shape[0];
    if (L <= 0) {
        throw std::invalid_argument(
            "[filter_percolating_clusters_from_encoded] L inválido");
    }

    if (encoded_net.dim == 2) {
        if (encoded_net.shape[1] != L) {
            throw std::invalid_argument(
                "[filter_percolating_clusters_from_encoded] rede 2D deve ter shape {L,L}");
        }
    } else {
        if (encoded_net.shape[1] != L || encoded_net.shape[2] != L) {
            throw std::invalid_argument(
                "[filter_percolating_clusters_from_encoded] rede 3D deve ter shape {L,L,L}");
        }
    }

    const GridRegular grid(encoded_net.dim, L);

    if (static_cast<int>(encoded_net.data.size()) != grid.total_size) {
        throw std::runtime_error(
            "[filter_percolating_clusters_from_encoded] tamanho de data inconsistente com shape");
    }

    NetworkPattern out(encoded_net.dim, encoded_net.shape,
                       encoded_net.num_colors, encoded_net.rho);
    out.seed = encoded_net.seed;
    std::fill(out.data.begin(), out.data.end(), static_cast<NetworkPattern::state_t>(-1));

    const int num_colors = encoded_net.num_colors;
    const int top_coord = (encoded_net.dim == 2) ? (grid.SY - 1) : (grid.SZ - 1);

    auto color_idx_from_encoded = [&](const int idx) -> int
    {
        const long long code =
            static_cast<long long>(encoded_net.data[static_cast<std::size_t>(idx)]);

        const DecodedValue dv = decode_animation_value(code, ANIMATION_SPECIES_FACTOR);
        if (dv.never_activated || dv.blocked) return -1;
        if (dv.color_idx < 0 || dv.color_idx >= num_colors) return -1;
        return dv.color_idx;
    };

    std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
    std::vector<int> stack;
    std::vector<int> component;

    for (int start = 0; start < grid.total_size; ++start) {
        if (visited[static_cast<std::size_t>(start)]) continue;

        const int color_idx = color_idx_from_encoded(start);
        if (color_idx < 0) continue;

        bool touches_base = false;
        bool touches_top = false;

        stack.clear();
        component.clear();

        stack.push_back(start);
        visited[static_cast<std::size_t>(start)] = 1;

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();

            component.push_back(u);

            const int h = grid.grow_coord(u);
            if (h == 0) touches_base = true;
            if (h == top_coord) touches_top = true;

            grid.for_each_neighbor(u, [&](const int v) {
                if (v < 0) return;
                if (visited[static_cast<std::size_t>(v)]) return;
                if (color_idx_from_encoded(v) != color_idx) return;

                visited[static_cast<std::size_t>(v)] = 1;
                stack.push_back(v);
            });
        }

        if (!touches_base || !touches_top) {
            continue;
        }

        for (const int idx : component) {
            out.data[static_cast<std::size_t>(idx)] =
                encoded_net.data[static_cast<std::size_t>(idx)];
        }
    }

    out.edge_pairs.reserve(encoded_net.edge_pairs.size());
    for (const auto& edge : encoded_net.edge_pairs) {
        const std::size_t u = static_cast<std::size_t>(edge.first);
        const std::size_t v = static_cast<std::size_t>(edge.second);
        if (u >= out.data.size() || v >= out.data.size()) continue;
        if (out.data[u] <= 0 || out.data[v] <= 0) continue;
        out.edge_pairs.push_back(edge);
    }

    return out;
}

// NetworkPattern network::create_shortest_paths_map(const NetworkPattern& net,
//                                                   const PercolationSeries& ps_out)
// {
//     NetworkPattern sp_net(net.dim, net.shape, net.num_colors, net.rho);

//     if (sp_net.data.size() != net.data.size()) {
//         throw std::runtime_error("[create_shortest_paths_map] tamanho de data inconsistente");
//     }

//     std::fill(sp_net.data.begin(), sp_net.data.end(), 0);

//     const int num_colors = net.num_colors;
//     if (static_cast<int>(ps_out.sp_path_lin.size()) < num_colors ||
//         static_cast<int>(ps_out.sp_len.size())      < num_colors) {
//         throw std::runtime_error("[create_shortest_paths_map] PercolationSeries inconsistente");
//     }

//     const std::size_t N = sp_net.data.size();

//     for (int c = 0; c < num_colors; ++c) {
//         if (ps_out.sp_len[c] <= 0) continue;

//         const std::vector<int>& path = ps_out.sp_path_lin[c];
//         if (path.empty()) continue;

//         const int color_label = (num_colors == 1 ? 1 : (c + 2));

//         for (int idx : path) {
//             if (idx < 0 || static_cast<std::size_t>(idx) >= N) {
//                 continue;
//             }
//             sp_net.data[idx] = color_label;
//         }
//     }

//     return sp_net;
// }
