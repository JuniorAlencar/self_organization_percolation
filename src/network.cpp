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
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <deque>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <climits>
#if defined(__linux__) && defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {

constexpr int ANIMATION_SPECIES_FACTOR = 10000000;

struct GridRegular {
    int dim;
    int SX;
    int SY;
    int SZ;
    int grow_axis;
    std::int64_t stride_y;
    std::int64_t stride_z;
    std::int64_t total_size;

    GridRegular(const int dim_,
                const int L,
                const int height_multiplier_ = HEIGHT_STOP_MULTIPLIER,
                const int height_extra_layers_ = 0)
        : dim(dim_),
          SX(L),
          SY(dim_ >= 2
                 ? (dim_ == 2
                        ? std::max(1, height_multiplier_) * L + std::max(0, height_extra_layers_)
                        : L)
                 : 1),
          SZ(dim_ == 3
                 ? std::max(1, height_multiplier_) * L + std::max(0, height_extra_layers_)
                 : 1),
          grow_axis(dim_ - 1),
          stride_y(SX),
          stride_z(static_cast<std::int64_t>(SX) * SY),
          total_size(static_cast<std::int64_t>(SX) * SY * SZ) {}

    inline int grow_height() const {
        return (dim == 2) ? SY : SZ;
    }

    void resize_growth_height(const int new_height) {
        const int h = std::max(1, new_height);
        if (dim == 2) {
            SY = h;
            stride_z = static_cast<std::int64_t>(SX) * SY;
            total_size = static_cast<std::int64_t>(SX) * SY;
        } else {
            SZ = h;
            total_size = static_cast<std::int64_t>(SX) * SY * SZ;
        }
    }

    inline std::int64_t lin_index(const int x, const int y = 0, const int z = 0) const {
        return static_cast<std::int64_t>(x) + static_cast<std::int64_t>(SX) * (static_cast<std::int64_t>(y) + static_cast<std::int64_t>(SY) * z);
    }

    inline int x_of(const std::int64_t idx) const {
        return static_cast<int>(idx % SX);
    }

    inline int y_of(const std::int64_t idx) const {
        return static_cast<int>((idx / SX) % SY);
    }

    inline int z_of(const std::int64_t idx) const {
        return static_cast<int>(idx / stride_z);
    }

    inline int grow_coord(const std::int64_t idx) const {
        return (dim == 2) ? y_of(idx) : z_of(idx);
    }

    inline int grow_top_coord() const {
        return (dim == 2) ? (SY - 1) : (SZ - 1);
    }

    inline void coords_of(const std::int64_t idx, int& x, int& y, int& z) const {
        x = x_of(idx);
        y = (dim >= 2 ? y_of(idx) : 0);
        z = (dim == 3 ? z_of(idx) : 0);
    }

    inline std::int64_t neighbor_xm(const std::int64_t idx) const {
        const int x = static_cast<int>(idx % SX);
        if (grow_axis == 0) {
            return (x == 0) ? -1 : (idx - 1);
        }
        return (x == 0) ? (idx + (SX - 1)) : (idx - 1);
    }

    inline std::int64_t neighbor_xp(const std::int64_t idx) const {
        const int x = static_cast<int>(idx % SX);
        if (grow_axis == 0) {
            return (x == SX - 1) ? -1 : (idx + 1);
        }
        return (x == SX - 1) ? (idx - (SX - 1)) : (idx + 1);
    }

    inline std::int64_t neighbor_ym(const std::int64_t idx) const {
        const int y = static_cast<int>((idx / SX) % SY);
        if (grow_axis == 1) {
            return (y == 0) ? -1 : (idx - SX);
        }
        return (y == 0) ? (idx + static_cast<std::int64_t>(SX) * (SY - 1)) : (idx - SX);
    }

    inline std::int64_t neighbor_yp(const std::int64_t idx) const {
        const int y = static_cast<int>((idx / SX) % SY);
        if (grow_axis == 1) {
            return (y == SY - 1) ? -1 : (idx + SX);
        }
        return (y == SY - 1) ? (idx - static_cast<std::int64_t>(SX) * (SY - 1)) : (idx + SX);
    }

    inline std::int64_t neighbor_zm(const std::int64_t idx) const {
        if (dim != 3) return -1;
        const int z = static_cast<int>(idx / stride_z);
        return (z == 0) ? -1 : (idx - stride_z);
    }

    inline std::int64_t neighbor_zp(const std::int64_t idx) const {
        if (dim != 3) return -1;
        const int z = static_cast<int>(idx / stride_z);
        return (z == SZ - 1) ? -1 : (idx + stride_z);
    }

    inline void for_each_neighbor(const std::int64_t idx, const std::function<void(std::int64_t)>& fn) const {
        const std::int64_t xm = neighbor_xm(idx);
        const std::int64_t xp = neighbor_xp(idx);
        if (xm >= 0) fn(xm);
        if (xp >= 0) fn(xp);

        if (dim >= 2) {
            const std::int64_t ym = neighbor_ym(idx);
            const std::int64_t yp = neighbor_yp(idx);
            if (ym >= 0) fn(ym);
            if (yp >= 0) fn(yp);
        }

        if (dim == 3) {
            const std::int64_t zm = neighbor_zm(idx);
            const std::int64_t zp = neighbor_zp(idx);
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

std::vector<std::int64_t> base_candidates_for_color(const GridRegular& grid,
                                                    const int num_colors,
                                                    const int color_idx,
                                                    const InitialBaseLayout layout,
                                                    const int quota)
{
    std::vector<std::int64_t> out;
    if (color_idx < 0 || color_idx >= num_colors) return out;

    const long long base_size = compute_base_size(grid);
    const bool has_gray_sites =
        static_cast<long long>(quota) * static_cast<long long>(num_colors) < base_size;

    // For two colors with unoccupied base sites, make the prescribed patterns
    // directly instead of spreading each color over its old candidate sector.
    // The ordering for dim=3 is x-major (a full y column before moving in x),
    // so "left / center / right" is visually a left-to-right base pattern.
    auto site_from_base_order = [&](const long long order) -> std::int64_t {
        if (grid.dim == 2) return grid.lin_index(static_cast<int>(order), 0, 0);
        const int x = static_cast<int>(order / grid.SY);
        const int y = static_cast<int>(order % grid.SY);
        return grid.lin_index(x, y, 0);
    };
    if (num_colors == 2 && has_gray_sites && quota > 0) {
        if (layout == InitialBaseLayout::Blocks) {
            out.reserve(static_cast<std::size_t>(quota));
            for (int k = 0; k < quota; ++k) {
                const long long order = (color_idx == 0) ? k : (base_size - 1 - k);
                out.push_back(site_from_base_order(order));
            }
            return out;
        }
        if (layout == InitialBaseLayout::Alternating && grid.dim == 2) {
            const long long gray_count = base_size - 2LL * quota;
            out.reserve(static_cast<std::size_t>(quota));
            // Spread gray sites as evenly as possible through the n+1 gaps
            // around the ordered (s1,s2) pairs.  For L=8, quota=2 and four
            // gray sites this yields: gray, s1, s2, gray, s1, s2, gray, gray.
            long long position = 0;
            for (long long pair = 0; pair <= quota; ++pair) {
                const long long gray_before =
                    ((pair + 1LL) * gray_count) / (quota + 1LL)
                    - (pair * gray_count) / (quota + 1LL);
                position += gray_before;
                if (pair < quota) {
                    out.push_back(site_from_base_order(position + color_idx));
                    position += 2LL;
                }
            }
            return out;
        }
    }

    if (layout == InitialBaseLayout::Blocks) {
        if (grid.dim == 2) {
            const int x0 = (color_idx * grid.SX) / num_colors;
            const int x1 = ((color_idx + 1) * grid.SX) / num_colors;
            out.reserve(static_cast<std::size_t>(std::max(0, x1 - x0)));
            for (int x = x0; x < x1; ++x) {
                out.push_back(grid.lin_index(x, 0, 0));
            }
            return out;
        }

        int nx = static_cast<int>(std::ceil(std::sqrt(
            static_cast<double>(num_colors))));
        while (nx < num_colors && (num_colors % nx) != 0) {
            ++nx;
        }
        const int ny = (num_colors + nx - 1) / nx;
        const int tx = color_idx % nx;
        const int ty = color_idx / nx;
        const int x0 = (tx * grid.SX) / nx;
        const int x1 = ((tx + 1) * grid.SX) / nx;
        const int y0 = (ty * grid.SY) / ny;
        const int y1 = ((ty + 1) * grid.SY) / ny;

        out.reserve(static_cast<std::size_t>(
            std::max(0, x1 - x0) * std::max(0, y1 - y0)));
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                out.push_back(grid.lin_index(x, y, 0));
            }
        }
        return out;
    }

    if (layout == InitialBaseLayout::Alternating) {
        if (grid.dim == 2) {
            out.reserve(static_cast<std::size_t>(
                (grid.SX + num_colors - 1) / num_colors));
            for (int x = 0; x < grid.SX; ++x) {
                if ((x % num_colors) == color_idx) {
                    out.push_back(grid.lin_index(x, 0, 0));
                }
            }
            return out;
        }

        out.reserve(static_cast<std::size_t>(
            (static_cast<std::int64_t>(grid.SX) * grid.SY + num_colors - 1) / num_colors));
        for (int y = 0; y < grid.SY; ++y) {
            for (int x = 0; x < grid.SX; ++x) {
                // Spatial parity gives a checkerboard in dim=3.  The former
                // linear-index rule (x + SX*y) degenerates into stripes when
                // SX is divisible by num_colors.
                if (((x + y) % num_colors) == color_idx) {
                    out.push_back(grid.lin_index(x, y, 0));
                }
            }
        }
    }

    return out;
}

template <typename ActivateSite>
int activate_deterministic_base_sites(const GridRegular& grid,
                                      const int num_colors,
                                      const int color_idx,
                                      const int quota,
                                      const InitialBaseLayout layout,
                                      ActivateSite activate_site)
{
    if (quota <= 0) return 0;

    const std::vector<std::int64_t> candidates =
        base_candidates_for_color(grid, num_colors, color_idx, layout, quota);
    if (candidates.empty()) return 0;

    const int n_activate = std::min<int>(quota, candidates.size());
    int activated = 0;
    int last_pos = -1;
    for (int k = 0; k < n_activate; ++k) {
        int pos = static_cast<int>(
            ((static_cast<long long>(2 * k + 1) *
              static_cast<long long>(candidates.size())) /
             static_cast<long long>(2 * n_activate)));
        pos = std::min<int>(pos, static_cast<int>(candidates.size()) - 1);
        if (pos <= last_pos) {
            pos = std::min<int>(
                last_pos + 1, static_cast<int>(candidates.size()) - 1);
        }
        last_pos = pos;

        if (activate_site(candidates[static_cast<std::size_t>(pos)])) {
            ++activated;
        }
    }
    return activated;
}

template <typename ActivateSite>
int activate_clustered_base_sites(const GridRegular& grid,
                                  const int num_colors,
                                  const int color_idx,
                                  const int quota,
                                  all_random& rng,
                                  ActivateSite activate_site)
{
    if (quota <= 0) return 0;
    int activated = 0;

    if (grid.dim == 2) {
        int x_center = 0;
        if (num_colors == 1) {
            x_center = rng.uniform_int(0, grid.SX - 1);
        } else {
            const int sector_w = grid.SX / num_colors;
            const int sector_start = color_idx * sector_w;
            x_center = sector_start + sector_w / 2;
        }

        int dx = 0;
        int step = 0;
        int tries = 0;
        const int max_tries = grid.SX * 2;
        while (activated < quota && tries < max_tries) {
            int x = 0;
            if (step == 0) {
                x = x_center;
            } else if (step % 2 == 1) {
                dx = (step + 1) / 2;
                x = (x_center + dx) % grid.SX;
                if (x < 0) x += grid.SX;
            } else {
                dx = step / 2;
                x = (x_center - dx) % grid.SX;
                if (x < 0) x += grid.SX;
            }
            ++step;
            ++tries;
            const std::int64_t idx = grid.lin_index(x, 0, 0);
            if (activate_site(idx)) {
                ++activated;
            }
        }
    } else {
        int x_center = 0;
        int y_center = 0;
        if (num_colors == 1) {
            x_center = rng.uniform_int(0, grid.SX - 1);
            y_center = rng.uniform_int(0, grid.SY - 1);
        } else {
            int nx = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_colors))));
            while (nx < num_colors && (num_colors % nx) != 0) ++nx;
            const int ny = (num_colors + nx - 1) / nx;
            const int tx = color_idx % nx;
            const int ty = color_idx / nx;
            x_center = (tx * grid.SX) / nx + grid.SX / (2 * nx);
            y_center = (ty * grid.SY) / ny + grid.SY / (2 * ny);
        }

        std::vector<char> visited(static_cast<std::size_t>(static_cast<std::int64_t>(grid.SX) * grid.SY), 0);
        std::deque<std::pair<int, int>> queue;
        queue.push_back({x_center, y_center});
        visited[static_cast<std::size_t>(x_center + static_cast<std::int64_t>(grid.SX) * y_center)] = 1;

        int tries = 0;
        const int max_tries = static_cast<int>(std::min<std::int64_t>(
            std::numeric_limits<int>::max(), static_cast<std::int64_t>(grid.SX) * grid.SY * 2));
        while (!queue.empty() && activated < quota && tries < max_tries) {
            const auto [cx, cy] = queue.front();
            queue.pop_front();
            ++tries;

            const std::int64_t idx = grid.lin_index(cx, cy, 0);
            if (activate_site(idx)) {
                ++activated;
            }

            const int dxs[4] = {1, -1, 0, 0};
            const int dys[4] = {0, 0, 1, -1};
            for (int dir = 0; dir < 4; ++dir) {
                int nx = (cx + dxs[dir]) % grid.SX;
                if (nx < 0) nx += grid.SX;
                int ny = (cy + dys[dir]) % grid.SY;
                if (ny < 0) ny += grid.SY;
                const std::size_t v_idx = static_cast<std::size_t>(nx + static_cast<std::int64_t>(grid.SX) * ny);
                if (!visited[v_idx]) {
                    visited[v_idx] = 1;
                    queue.push_back({nx, ny});
                }
            }
        }
    }
    return activated;
}

const char* initial_base_layout_name(const InitialBaseLayout layout)
{
    switch (layout) {
        case InitialBaseLayout::Clustered:
            return "clustered";
        case InitialBaseLayout::Blocks:
            return "blocks";
        case InitialBaseLayout::Alternating:
            return "alternating";
        case InitialBaseLayout::Random:
        default:
            return "random";
    }
}

struct DecodedValue {
    bool never_activated = false;
    bool blocked = false;
    int color_1b = -1;
    int color_idx = -1;
    int time = -1;
};


struct FrontCandidate {
    std::uint64_t activator = std::numeric_limits<std::uint64_t>::max();
    std::uint8_t color_idx = std::numeric_limits<std::uint8_t>::max();
    std::uint64_t edge_bit = std::numeric_limits<std::uint64_t>::max();

    FrontCandidate() = default;

    FrontCandidate(const std::int64_t activator_, const int color_idx_)
        : activator(static_cast<std::uint64_t>(activator_)),
          color_idx(static_cast<std::uint8_t>(color_idx_)) {}

    FrontCandidate(const std::int64_t activator_, const int color_idx_, const std::uint64_t edge_bit_)
        : activator(static_cast<std::uint64_t>(activator_)),
          color_idx(static_cast<std::uint8_t>(color_idx_)),
          edge_bit(edge_bit_) {}
};

struct NeighborBond {
    std::int64_t idx = -1;
    std::uint64_t edge_bit = std::numeric_limits<std::uint64_t>::max();
};

inline int collect_neighbors(const GridRegular& grid, const std::int64_t idx, std::int64_t out[6])
{
    int n = 0;

    const std::int64_t xm = grid.neighbor_xm(idx);
    const std::int64_t xp = grid.neighbor_xp(idx);
    if (xm >= 0) out[n++] = xm;
    if (xp >= 0) out[n++] = xp;

    if (grid.dim >= 2) {
        const std::int64_t ym = grid.neighbor_ym(idx);
        const std::int64_t yp = grid.neighbor_yp(idx);
        if (ym >= 0) out[n++] = ym;
        if (yp >= 0) out[n++] = yp;
    }

    if (grid.dim == 3) {
        const std::int64_t zm = grid.neighbor_zm(idx);
        const std::int64_t zp = grid.neighbor_zp(idx);
        if (zm >= 0) out[n++] = zm;
        if (zp >= 0) out[n++] = zp;
    }

    return n;
}

inline int collect_neighbor_bonds(const GridRegular& grid, const std::int64_t idx, NeighborBond out[6])
{
    int n = 0;

    const std::int64_t xm = grid.neighbor_xm(idx);
    const std::int64_t xp = grid.neighbor_xp(idx);
    if (xm >= 0) out[n++] = NeighborBond{xm, static_cast<std::uint64_t>(xm)};
    if (xp >= 0) out[n++] = NeighborBond{xp, static_cast<std::uint64_t>(idx)};

    if (grid.dim >= 2) {
        const std::uint64_t y_offset = static_cast<std::uint64_t>(grid.total_size);
        const std::int64_t ym = grid.neighbor_ym(idx);
        const std::int64_t yp = grid.neighbor_yp(idx);
        if (ym >= 0) out[n++] = NeighborBond{ym, y_offset + static_cast<std::uint64_t>(ym)};
        if (yp >= 0) out[n++] = NeighborBond{yp, y_offset + static_cast<std::uint64_t>(idx)};
    }

    if (grid.dim == 3) {
        const std::uint64_t z_offset = 2ull * static_cast<std::uint64_t>(grid.total_size);
        const std::int64_t zm = grid.neighbor_zm(idx);
        const std::int64_t zp = grid.neighbor_zp(idx);
        if (zm >= 0) out[n++] = NeighborBond{zm, z_offset + static_cast<std::uint64_t>(zm)};
        if (zp >= 0) out[n++] = NeighborBond{zp, z_offset + static_cast<std::uint64_t>(idx)};
    }

    return n;
}

inline std::uint64_t dynamic_bond_key(const int direction, const std::int64_t lower_idx)
{
    return (static_cast<std::uint64_t>(direction) << 62) |
           (static_cast<std::uint64_t>(lower_idx) & ((1ull << 62) - 1ull));
}

struct DynamicTestedBondBitmap {
    std::array<std::vector<std::uint64_t>, 3> words_by_direction;

    void ensure_capacity(const std::size_t total_sites)
    {
        const std::size_t needed = (total_sites + 63u) >> 6;
        for (auto& words : words_by_direction) {
            if (needed > words.size()) {
                if (needed > words.capacity()) {
                    words.reserve(needed + (needed >> 2));
                }
                words.resize(needed, 0ull);
            }
        }
    }

    inline bool mark_if_new(const std::uint64_t key)
    {
        const int direction = static_cast<int>(key >> 62);
        const std::uint64_t idx = key & ((1ull << 62) - 1ull);
        const std::size_t word_idx = static_cast<std::size_t>(idx >> 6);
        const std::uint64_t mask = 1ull << (idx & 63ull);

        auto& words = words_by_direction[static_cast<std::size_t>(direction)];
        if (__builtin_expect(word_idx >= words.size(), 0)) {
            words.resize(word_idx + 1u, 0ull);
        }

        const bool is_new = (words[word_idx] & mask) == 0ull;
        words[word_idx] |= mask;
        return is_new;
    }
};

inline int collect_neighbor_bonds_dynamic(const GridRegular& grid, const std::int64_t idx, NeighborBond out[6])
{
    int n = 0;

    const std::int64_t xm = grid.neighbor_xm(idx);
    const std::int64_t xp = grid.neighbor_xp(idx);
    if (xm >= 0) out[n++] = NeighborBond{xm, dynamic_bond_key(0, xm)};
    if (xp >= 0) out[n++] = NeighborBond{xp, dynamic_bond_key(0, idx)};

    if (grid.dim >= 2) {
        const std::int64_t ym = grid.neighbor_ym(idx);
        const std::int64_t yp = grid.neighbor_yp(idx);
        if (ym >= 0) out[n++] = NeighborBond{ym, dynamic_bond_key(1, ym)};
        if (yp >= 0) out[n++] = NeighborBond{yp, dynamic_bond_key(1, idx)};
    }

    if (grid.dim == 3) {
        const std::int64_t zm = grid.neighbor_zm(idx);
        const std::int64_t zp = grid.neighbor_zp(idx);
        if (zm >= 0) out[n++] = NeighborBond{zm, dynamic_bond_key(2, zm)};
        if (zp >= 0) out[n++] = NeighborBond{zp, dynamic_bond_key(2, idx)};
    }

    return n;
}

struct OpenBondGraph {
    std::vector<std::size_t> offsets;
    std::vector<std::int64_t> neighbors;

    OpenBondGraph() = default;

    OpenBondGraph(const std::int64_t total_size,
                  const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edge_pairs)
    {
        build(total_size, edge_pairs);
    }

    void build(const std::int64_t total_size,
               const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edge_pairs)
    {
        offsets.assign(static_cast<std::size_t>(total_size) + 1u, 0u);

        for (const auto& edge : edge_pairs) {
            if (static_cast<std::int64_t>(edge.first) >= total_size ||
                static_cast<std::int64_t>(edge.second) >= total_size) {
                continue;
            }
            ++offsets[static_cast<std::size_t>(edge.first) + 1u];
        }

        for (std::size_t i = 1; i <= static_cast<std::size_t>(total_size); ++i) {
            offsets[i] += offsets[i - 1u];
        }

        neighbors.assign(edge_pairs.size(), 0);
        std::vector<std::size_t> cursor = offsets;
        for (const auto& edge : edge_pairs) {
            if (static_cast<std::int64_t>(edge.first) >= total_size ||
                static_cast<std::int64_t>(edge.second) >= total_size) {
                continue;
            }
            neighbors[cursor[static_cast<std::size_t>(edge.first)]++] = static_cast<std::int64_t>(edge.second);
        }
    }
};

struct ComponentSummary {
    std::int64_t seed = -1;
    std::int64_t size = 0;
};

struct TimeSplit {
    std::int64_t pre = 0;
    std::int64_t post = 0;
};

std::vector<std::int64_t> shortest_open_bond_path_single_color(
    const GridRegular& grid,
    const OpenBondGraph& graph,
    const int active_val,
    const std::function<int(std::int64_t)>& get_site,
    const int target_height = -1)
{
    const int top_coord = (target_height >= 0)
        ? std::min(target_height, grid.grow_top_coord())
        : grid.grow_top_coord();
    std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
    std::vector<std::int64_t> parent(static_cast<std::size_t>(grid.total_size), -2);
    std::vector<std::int64_t> queue;
    queue.reserve(static_cast<std::size_t>(compute_base_size(grid)));

    for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
        if (get_site(idx) != active_val) continue;
        if (grid.grow_coord(idx) != 0) continue;
        visited[static_cast<std::size_t>(idx)] = 1;
        parent[static_cast<std::size_t>(idx)] = -1;
        queue.push_back(idx);
    }

    std::int64_t target = -1;
    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
        const std::int64_t u = queue[qi];
        if (grid.grow_coord(u) == top_coord) {
            target = u;
            break;
        }

        const std::size_t begin = graph.offsets[static_cast<std::size_t>(u)];
        const std::size_t end = graph.offsets[static_cast<std::size_t>(u) + 1u];
        for (std::size_t k = begin; k < end; ++k) {
            const std::int64_t v = static_cast<std::int64_t>(graph.neighbors[k]);
            if (visited[static_cast<std::size_t>(v)]) continue;
            if (get_site(v) != active_val) continue;
            visited[static_cast<std::size_t>(v)] = 1;
            parent[static_cast<std::size_t>(v)] = u;
            queue.push_back(v);
        }
    }

    std::vector<std::int64_t> path;
    if (target < 0) return path;

    for (std::int64_t cur = target; cur >= 0; cur = parent[static_cast<std::size_t>(cur)]) {
        path.push_back(cur);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<std::int64_t> shortest_site_path_single_color(
    const GridRegular& grid,
    const int active_val,
    const std::function<int(std::int64_t)>& get_site,
    const int target_height = -1)
{
    const int top_coord = (target_height >= 0)
        ? std::min(target_height, grid.grow_top_coord())
        : grid.grow_top_coord();
    std::vector<std::int64_t> parent(static_cast<std::size_t>(grid.total_size), -2);
    std::vector<std::int64_t> current;
    std::vector<std::int64_t> next;
    current.reserve(static_cast<std::size_t>(compute_base_size(grid)));
    next.reserve(static_cast<std::size_t>(compute_base_size(grid)));

    for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
        if (grid.grow_coord(idx) != 0) continue;
        if (get_site(idx) != active_val) continue;
        parent[static_cast<std::size_t>(idx)] = -1;
        current.push_back(idx);
    }

    std::int64_t reached_top = -1;
    std::int64_t neigh[6];

    while (!current.empty() && reached_top < 0) {
        next.clear();

        for (const std::int64_t u : current) {
            if (grid.grow_coord(u) == top_coord) {
                reached_top = u;
                break;
            }

            const int nneigh = collect_neighbors(grid, u, neigh);
            for (int ni = 0; ni < nneigh; ++ni) {
                const std::int64_t v = neigh[ni];
                if (parent[static_cast<std::size_t>(v)] != -2) continue;
                if (get_site(v) != active_val) continue;

                parent[static_cast<std::size_t>(v)] = u;
                next.push_back(v);
            }
        }

        current.swap(next);
    }

    if (reached_top < 0) return {};

    std::vector<std::int64_t> path;
    for (std::int64_t at = reached_top; at != -1; at = parent[static_cast<std::size_t>(at)]) {
        path.push_back(at);
    }
    std::reverse(path.begin(), path.end());
    return path;
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

    std::uint64_t edge_index(const std::int64_t u, const std::int64_t v) const {
        const std::int64_t lo = std::min(u, v);
        const std::int64_t hi = std::max(u, v);
        const std::int64_t diff = hi - lo;

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
                    diff == static_cast<std::int64_t>(grid.SX) * (grid.SY - 1) &&
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

    bool mark_if_new(const std::int64_t u, const std::int64_t v) {
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

    void mark_open(const std::int64_t u, const std::int64_t v) {
        const std::uint64_t bit = edge_index(u, v);
        mark_open(bit);
    }

    void mark_open(const std::uint64_t bit) {
        words[static_cast<std::size_t>(bit >> 6u)] |= (1ull << (bit & 63u));
    }

    bool contains(const std::int64_t u, const std::int64_t v) const {
        const std::uint64_t bit = edge_index(u, v);
        return contains(bit);
    }

    bool contains(const std::uint64_t bit) const {
        const std::uint64_t word = words[static_cast<std::size_t>(bit >> 6u)];
        return (word & (1ull << (bit & 63u))) != 0u;
    }
};

inline bool mark_bond_if_new(TestedBondBitmap& tested_bonds,
                             const std::int64_t u,
                             const std::int64_t v)
{
    return tested_bonds.mark_if_new(u, v);
}

std::vector<std::int64_t> shortest_open_bond_path_single_color(
    const GridRegular& grid,
    const TestedBondBitmap& open_bonds,
    const int active_val,
    const std::function<int(std::int64_t)>& get_site,
    const int target_height = -1)
{
    const int top_coord = (target_height >= 0)
        ? std::min(target_height, grid.grow_top_coord())
        : grid.grow_top_coord();
    std::vector<std::int64_t> parent(static_cast<std::size_t>(grid.total_size), -2);
    std::vector<std::int64_t> current;
    std::vector<std::int64_t> next;
    current.reserve(static_cast<std::size_t>(compute_base_size(grid)));
    next.reserve(static_cast<std::size_t>(compute_base_size(grid)));

    for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
        if (grid.grow_coord(idx) != 0) continue;
        if (get_site(idx) != active_val) continue;
        parent[static_cast<std::size_t>(idx)] = -1;
        current.push_back(idx);
    }

    std::int64_t reached_top = -1;
    std::int64_t neigh[6];

    while (!current.empty() && reached_top < 0) {
        next.clear();

        for (const std::int64_t u : current) {
            if (grid.grow_coord(u) == top_coord) {
                reached_top = u;
                break;
            }

            const int nneigh = collect_neighbors(grid, u, neigh);
            for (int ni = 0; ni < nneigh; ++ni) {
                const std::int64_t v = neigh[ni];
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

    std::vector<std::int64_t> path;
    for (std::int64_t at = reached_top; at != -1; at = parent[static_cast<std::size_t>(at)]) {
        path.push_back(at);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

inline double topology_uniform01(std::mt19937_64& rng)
{
    return std::generate_canonical<double, 53>(rng);
}

struct FastSlotMap {
    struct Entry {
        std::int64_t key = -1;
        int val = -1;
    };
    std::vector<Entry> table;
    std::vector<uint32_t> touched_slots;
    uint32_t mask = 0;

    void ensure_capacity(const std::size_t expected) {
        std::size_t cap = 256;
        while (cap < expected * 2) cap <<= 1;
        if (cap > table.size()) {
            table.assign(cap, {-1, -1});
            mask = static_cast<uint32_t>(cap - 1);
            touched_slots.reserve(expected);
        }
    }

    inline int find(const std::int64_t key) const {
        if (mask == 0) return -1;
        uint32_t h = static_cast<uint32_t>((static_cast<std::uint64_t>(key) * 11400714819323198485ull) >> 32) & mask;
        while (table[h].key != -1) {
            if (table[h].key == key) return table[h].val;
            h = (h + 1) & mask;
        }
        return -1;
    }

    inline void insert(const std::int64_t key, const int val) {
        if (mask == 0 || (touched_slots.size() * 2 >= mask)) {
            rehash(table.empty() ? 512 : table.size() * 2);
        }
        uint32_t h = static_cast<uint32_t>((static_cast<std::uint64_t>(key) * 11400714819323198485ull) >> 32) & mask;
        while (table[h].key != -1 && table[h].key != key) {
            h = (h + 1) & mask;
        }
        if (table[h].key == -1) {
            table[h].key = key;
            touched_slots.push_back(h);
        }
        table[h].val = val;
    }

    void rehash(const std::size_t new_cap) {
        std::vector<Entry> old = std::move(table);
        std::vector<uint32_t> old_touched = std::move(touched_slots);
        table.assign(new_cap, {-1, -1});
        mask = static_cast<uint32_t>(new_cap - 1);
        touched_slots.reserve(old_touched.size() * 2);
        for (const uint32_t slot : old_touched) {
            if (old[slot].key != -1) {
                insert(old[slot].key, old[slot].val);
            }
        }
    }

    inline void clear() {
        for (const uint32_t h : touched_slots) {
            table[h].key = -1;
            table[h].val = -1;
        }
        touched_slots.clear();
    }
};

struct SparseFrontCandidates {
    // A site on the regular lattice has at most one candidate per incident
    // bond: four in 2D and six in 3D.  Keeping room for sixteen candidates
    // used considerably more memory at the growth front without admitting a
    // state the model can produce.
    static constexpr int max_candidates_per_target = 6;

    std::vector<std::int64_t> touched_targets;
    std::vector<int> slot_of_target;
    FastSlotMap sparse_slot_of_target;
    std::vector<std::array<std::uint64_t, max_candidates_per_target>> activators;
    std::vector<std::array<std::uint8_t, max_candidates_per_target>> colors;
    std::vector<std::array<std::uint64_t, max_candidates_per_target>> edge_bits;
    std::vector<unsigned char> counts;
    bool sparse_lookup = false;

    explicit SparseFrontCandidates(const std::int64_t nsites = 0,
                                   const bool sparse_lookup_ = false)
        : slot_of_target(sparse_lookup_
                             ? 0u
                             : static_cast<std::size_t>(nsites),
                         -1),
          sparse_lookup(sparse_lookup_)
    {
    }

    void resize_site_count(const std::int64_t nsites) {
        if (sparse_lookup) return;
        slot_of_target.resize(static_cast<std::size_t>(nsites), -1);
    }

    void reserve_frontier(const std::size_t expected_targets) {
        touched_targets.reserve(expected_targets);
        activators.reserve(expected_targets);
        colors.reserve(expected_targets);
        edge_bits.reserve(expected_targets);
        counts.reserve(expected_targets);
        if (sparse_lookup) {
            sparse_slot_of_target.ensure_capacity(expected_targets);
        }
    }

    void clear() {
        if (sparse_lookup) {
            sparse_slot_of_target.clear();
        } else {
            for (const std::int64_t target : touched_targets) {
                slot_of_target[static_cast<std::size_t>(target)] = -1;
            }
        }
        touched_targets.clear();
        activators.clear();
        colors.clear();
        edge_bits.clear();
        counts.clear();
    }

    void add(const std::int64_t target, const FrontCandidate candidate) {
        int slot = -1;
        if (sparse_lookup) {
            slot = sparse_slot_of_target.find(target);
        } else {
            slot = slot_of_target[static_cast<std::size_t>(target)];
        }

        if (slot < 0) {
            slot = static_cast<int>(counts.size());
            if (sparse_lookup) {
                sparse_slot_of_target.insert(target, slot);
            } else {
                slot_of_target[static_cast<std::size_t>(target)] = slot;
            }
            touched_targets.push_back(target);
            activators.emplace_back();
            colors.emplace_back();
            edge_bits.emplace_back();
            counts.push_back(0);
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
            static_cast<std::int64_t>(activators[slot][static_cast<std::size_t>(idx)]),
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
                                         const std::int64_t idx,
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

inline std::vector<double> centered_moving_average_tail(const std::vector<double>& x,
                                                       const int window,
                                                       const std::size_t start_idx)
{
    if (start_idx >= x.size()) return {};
    if (window <= 1) {
        return std::vector<double>(
            x.begin() + static_cast<std::ptrdiff_t>(start_idx),
            x.end());
    }

    const int n = static_cast<int>(x.size() - start_idx);
    const int half_left = window / 2;
    const int half_right = window - half_left - 1;

    std::vector<double> prefix(static_cast<std::size_t>(n + 1), 0.0);
    for (int i = 0; i < n; ++i) {
        prefix[static_cast<std::size_t>(i + 1)] =
            prefix[static_cast<std::size_t>(i)] +
            x[start_idx + static_cast<std::size_t>(i)];
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
                                    std::vector<double>& j_w,
                                    const std::size_t t_start_idx = 0)
{
    t_center.clear();
    j_w.clear();

    if (window_block < 1) {
        throw std::runtime_error("block_mean_regular_time: window_block deve ser >= 1");
    }

    if (t_start_idx >= t.size()) return;

    const int n = std::min(
        static_cast<int>(t.size() - t_start_idx),
        static_cast<int>(y.size()));
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
            st += static_cast<double>(
                t[t_start_idx + static_cast<std::size_t>(i)]);
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

inline double estimate_t_eq_from_series(const std::vector<int>& t,
                                        const std::vector<double>& p,
                                        const int window_roll = 15,
                                        const int window_block = 10,
                                        const int min_stable_steps = 15,
                                        const double rel_tol = 2.5e-2,
                                        const double abs_tol = 1.0e-6,
                                        const double s_prime_threshold = 1.0e-5,
                                        const bool require_target = false,
                                        const double target = 0.0,
                                        const int validation_window_steps = 0,
                                        const std::size_t start_idx = 0)
{
    if (t.empty()) {
        throw std::runtime_error("estimate_t_eq_from_series: t vazio");
    }
    if (p.size() != t.size()) {
        throw std::runtime_error("estimate_t_eq_from_series: p.size != t.size");
    }
    if (start_idx >= t.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const std::vector<double> p_smoothed =
        centered_moving_average_tail(p, window_roll, start_idx);

    std::vector<double> t_j;
    std::vector<double> j_w;
    block_mean_regular_time(t, p_smoothed, window_block, t_j, j_w, start_idx);
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
            static_cast<double>(min_stable_steps) /
            static_cast<double>(std::max(1, window_block)))));
    const int validation_links_required =
        validation_window_steps > 0
            ? std::max(1, static_cast<int>(std::ceil(
                static_cast<double>(validation_window_steps) /
                static_cast<double>(std::max(1, window_block)))))
            : 0;

    auto stationary_validation_ok = [&](const int first_point,
                                        const int last_point) -> bool {
        const int n_points = last_point - first_point + 1;
        if (n_points < 3) return false;

        const int edge_count = std::max(1, n_points / 3);
        double first_mean = 0.0;
        double last_mean = 0.0;
        for (int k = 0; k < edge_count; ++k) {
            first_mean += j_w[static_cast<std::size_t>(first_point + k)];
            last_mean += j_w[static_cast<std::size_t>(last_point - k)];
        }
        first_mean /= static_cast<double>(edge_count);
        last_mean /= static_cast<double>(edge_count);

        const double drift_scale =
            std::max(std::max(std::abs(first_mean), std::abs(last_mean)),
                     1.0e-12);
        const double drift_threshold = std::max(abs_tol, rel_tol * drift_scale);
        if (std::abs(last_mean - first_mean) > drift_threshold) {
            return false;
        }

        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_xx = 0.0;
        double sum_xy = 0.0;
        for (int k = first_point; k <= last_point; ++k) {
            const double x =
                t_j[static_cast<std::size_t>(k)] -
                t_j[static_cast<std::size_t>(first_point)];
            const double y = j_w[static_cast<std::size_t>(k)];
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        const double n = static_cast<double>(n_points);
        const double denom = n * sum_xx - sum_x * sum_x;
        if (denom > 0.0) {
            const double slope = (n * sum_xy - sum_x * sum_y) / denom;
            const double projected_drift =
                std::abs(slope) *
                (t_j[static_cast<std::size_t>(last_point)] -
                 t_j[static_cast<std::size_t>(first_point)]);
            if (projected_drift > drift_threshold) {
                return false;
            }
        }

        if (require_target) {
            const double target_threshold =
                std::max(abs_tol,
                         rel_tol * std::max(std::abs(target), 1.0e-12));
            const double window_mean = sum_y / n;
            if (std::abs(window_mean - target) > target_threshold) {
                return false;
            }
        }

        return true;
    };

    if (validation_links_required > 0) {
        for (int first_point = 0;
             first_point + validation_links_required < static_cast<int>(j_w.size());
             ++first_point) {
            const int last_point = first_point + validation_links_required;
            const int tail_last_point = static_cast<int>(j_w.size()) - 1;
            if (stationary_validation_ok(first_point, last_point) &&
                stationary_validation_ok(first_point, tail_last_point)) {
                return t_j[static_cast<std::size_t>(first_point)];
            }
        }
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

        const double p_scale = std::max(
            std::abs(j_w[static_cast<std::size_t>(i)]),
            std::abs(j_w[static_cast<std::size_t>(i + 1)]));
        const double s_threshold = std::max(abs_tol, rel_tol * p_scale);
        const double target_threshold = std::max(
            abs_tol,
            rel_tol * std::max(std::abs(target), 1.0e-12));
        const bool target_ok =
            !require_target ||
            (std::abs(j_w[static_cast<std::size_t>(i)] - target) <= target_threshold &&
             std::abs(j_w[static_cast<std::size_t>(i + 1)] - target) <= target_threshold);
        const bool stable =
            std::isfinite(sp) &&
            std::abs(sp) < s_prime_threshold &&
            s[static_cast<std::size_t>(i)] <= s_threshold &&
            target_ok;

        if (stable) {
            ++stable_run;
            if (stable_run >= stable_links_required) {
                const int first_stable = i - stable_run + 1;
                if (validation_links_required > 0) {
                    const int validation_last =
                        first_stable + validation_links_required - 1;
                    if (validation_last >= ns) {
                        continue;
                    }

                    const int first_point = first_stable;
                    const int last_point = validation_last + 1;
                    const int n_points = last_point - first_point + 1;
                    const int edge_count = std::max(1, n_points / 3);
                    double first_mean = 0.0;
                    double last_mean = 0.0;
                    for (int k = 0; k < edge_count; ++k) {
                        first_mean +=
                            j_w[static_cast<std::size_t>(first_point + k)];
                        last_mean +=
                            j_w[static_cast<std::size_t>(last_point - k)];
                    }
                    first_mean /= static_cast<double>(edge_count);
                    last_mean /= static_cast<double>(edge_count);

                    const double start = first_mean;
                    const double end = last_mean;
                    const double drift_scale =
                        std::max(std::max(std::abs(start), std::abs(end)),
                                 1.0e-12);
                    const double drift_threshold =
                        std::max(abs_tol, rel_tol * drift_scale);
                    if (std::abs(end - start) > drift_threshold) {
                        continue;
                    }

                    double sum_x = 0.0;
                    double sum_y = 0.0;
                    double sum_xx = 0.0;
                    double sum_xy = 0.0;
                    for (int k = first_point; k <= last_point; ++k) {
                        const double x =
                            t_j[static_cast<std::size_t>(k)] -
                            t_j[static_cast<std::size_t>(first_point)];
                        const double y = j_w[static_cast<std::size_t>(k)];
                        sum_x += x;
                        sum_y += y;
                        sum_xx += x * x;
                        sum_xy += x * y;
                    }
                    const double n = static_cast<double>(n_points);
                    const double denom = n * sum_xx - sum_x * sum_x;
                    if (denom > 0.0) {
                        const double slope =
                            (n * sum_xy - sum_x * sum_y) / denom;
                        const double projected_drift =
                            std::abs(slope) *
                            (t_j[static_cast<std::size_t>(last_point)] -
                             t_j[static_cast<std::size_t>(first_point)]);
                        if (projected_drift > drift_threshold) {
                            continue;
                        }
                    }
                }
                return t_s[static_cast<std::size_t>(first_stable)];
            }
        } else {
            stable_run = 0;
        }
    }

    return std::numeric_limits<double>::quiet_NaN();
}

inline double growth_test_effective_rel_tol(const double base_rel_tol,
                                            const int num_colors)
{
    (void) num_colors;
    return base_rel_tol;
}

inline int growth_test_global_min_stable_steps(const int configured_steps,
                                               const int window_block)
{
    return std::max(configured_steps, 10 * std::max(1, window_block));
}

inline int growth_test_global_validation_window_steps(const int extra_steps,
                                                      const int window_block)
{
    return std::max(std::max(0, extra_steps), 20 * std::max(1, window_block));
}

double compute_upper_surface_fraction(
    const GridRegular& grid,
    const int num_colors,
    const int color_idx,
    const std::vector<std::int8_t>& site_state,
    const int upper_height,
    const int lower_height)
{
    if (num_colors <= 0 || color_idx < 0 || color_idx >= num_colors ||
        upper_height < 0) {
        return 0.0;
    }

    const int active_val = color_to_active_value(num_colors, color_idx);
    const int top = std::min(upper_height, grid.grow_top_coord());
    const int bottom = std::max(0, std::min(lower_height, top));
    const std::int64_t n_sites = std::min(
        grid.total_size,
        static_cast<std::int64_t>(site_state.size()));

    std::vector<char> exterior_air(static_cast<std::size_t>(grid.total_size), 0);
    std::deque<std::int64_t> queue;

    for (std::int64_t idx = 0; idx < n_sites; ++idx) {
        if (grid.grow_coord(idx) != top) continue;
        if (static_cast<int>(site_state[static_cast<std::size_t>(idx)]) > 0) {
            continue;
        }
        exterior_air[static_cast<std::size_t>(idx)] = 1;
        queue.push_back(idx);
    }

    std::int64_t neigh[6];
    while (!queue.empty()) {
        const std::int64_t u = queue.front();
        queue.pop_front();

        const int nneigh = collect_neighbors(grid, u, neigh);
        for (int ni = 0; ni < nneigh; ++ni) {
            const std::int64_t v = neigh[ni];
            if (v < 0 || v >= n_sites) continue;
            const int hv = grid.grow_coord(v);
            if (hv > top || hv < bottom) continue;
            if (exterior_air[static_cast<std::size_t>(v)]) continue;
            if (static_cast<int>(site_state[static_cast<std::size_t>(v)]) > 0) {
                continue;
            }
            exterior_air[static_cast<std::size_t>(v)] = 1;
            queue.push_back(v);
        }
    }

    std::uint32_t count = 0u;
    for (std::int64_t idx = 0; idx < n_sites; ++idx) {
        const int h = grid.grow_coord(idx);
        if (h < bottom || h > top) continue;
        if (static_cast<int>(site_state[static_cast<std::size_t>(idx)]) == active_val) {
            bool exposed_to_upper_air = (h == top);
            const int nneigh = collect_neighbors(grid, idx, neigh);
            for (int ni = 0; ni < nneigh && !exposed_to_upper_air; ++ni) {
                const std::int64_t v = neigh[ni];
                if (v < 0 || v >= n_sites) continue;
                if (grid.grow_coord(v) > top ||
                    exterior_air[static_cast<std::size_t>(v)]) {
                    exposed_to_upper_air = true;
                }
            }
            if (exposed_to_upper_air) ++count;
        }
    }

    const double layer_area = (grid.dim == 2)
        ? static_cast<double>(grid.SX)
        : static_cast<double>(grid.SX) * static_cast<double>(grid.SY);
    return layer_area > 0.0 ? static_cast<double>(count) / layer_area : 0.0;
}

std::vector<int> compute_upper_surface_heights(
    const GridRegular& grid,
    const int num_colors,
    const int color_idx,
    const std::vector<std::int8_t>& site_state,
    const int lower_height,
    const int upper_height)
{
    if (num_colors <= 0 || color_idx < 0 || color_idx >= num_colors ||
        upper_height < lower_height) {
        return {};
    }

    const int active_val = color_to_active_value(num_colors, color_idx);
    const int bottom = std::max(0, lower_height);
    const int top = std::min(upper_height, grid.grow_top_coord());
    if (top < bottom) return {};
    const std::int64_t lateral_size = (grid.dim == 2) ? grid.SX : static_cast<std::int64_t>(grid.SX) * grid.SY;
    std::vector<int> heights(static_cast<std::size_t>(lateral_size), -1);

    const std::int64_t n_sites = std::min(
        grid.total_size,
        static_cast<std::int64_t>(site_state.size()));
    for (std::int64_t idx = 0; idx < n_sites; ++idx) {
        if (static_cast<int>(site_state[static_cast<std::size_t>(idx)]) != active_val) {
            continue;
        }

        const int h = grid.grow_coord(idx);
        if (h < bottom || h > top) continue;

        const std::int64_t lateral_idx = (grid.dim == 2)
            ? grid.x_of(idx)
            : static_cast<std::int64_t>(grid.x_of(idx)) + static_cast<std::int64_t>(grid.SX) * grid.y_of(idx);
        heights[static_cast<std::size_t>(lateral_idx)] =
            std::max(heights[static_cast<std::size_t>(lateral_idx)], h);
    }

    return heights;
}

double compute_surface_width(const std::vector<int>& heights)
{
    double sum = 0.0;
    double sum_sq = 0.0;
    int n = 0;
    for (const int h : heights) {
        if (h < 0) continue;
        const double value = static_cast<double>(h);
        sum += value;
        sum_sq += value * value;
        ++n;
    }
    if (n <= 0) return 0.0;
    const double mean = sum / static_cast<double>(n);
    const double mean_sq = sum_sq / static_cast<double>(n);
    return std::sqrt(std::max(0.0, mean_sq - mean * mean));
}

double compute_surface_gradient(
    const GridRegular& grid,
    const std::vector<int>& heights)
{
    if (heights.empty()) return 0.0;

    double sum = 0.0;
    int n = 0;

    if (grid.dim == 2) {
        for (int x = 0; x < grid.SX; ++x) {
            const int h0 = heights[static_cast<std::size_t>(x)];
            const int h1 = heights[static_cast<std::size_t>((x + 1) % grid.SX)];
            if (h0 < 0 || h1 < 0) continue;
            sum += std::abs(h1 - h0);
            ++n;
        }
    } else {
        for (int y = 0; y < grid.SY; ++y) {
            for (int x = 0; x < grid.SX; ++x) {
                const std::int64_t idx = x + static_cast<std::int64_t>(grid.SX) * y;
                const std::int64_t hx = ((x + 1) % grid.SX) + static_cast<std::int64_t>(grid.SX) * y;
                const std::int64_t hy = x + static_cast<std::int64_t>(grid.SX) * ((y + 1) % grid.SY);
                const int h0 = heights[static_cast<std::size_t>(idx)];
                const int h_x = heights[static_cast<std::size_t>(hx)];
                const int h_y = heights[static_cast<std::size_t>(hy)];
                if (h0 >= 0 && h_x >= 0) {
                    sum += std::abs(h_x - h0);
                    ++n;
                }
                if (h0 >= 0 && h_y >= 0) {
                    sum += std::abs(h_y - h0);
                    ++n;
                }
            }
        }
    }

    return n > 0 ? sum / static_cast<double>(n) : 0.0;
}

double compute_lateral_area(const GridRegular& grid)
{
    return (grid.dim == 2)
        ? static_cast<double>(grid.SX)
        : static_cast<double>(grid.SX) * static_cast<double>(grid.SY);
}

struct SlabClusterMassResult {
    double largest_spanning_mass = 0.0;
    std::vector<std::int64_t> cluster_sizes;
};

SlabClusterMassResult compute_slab_cluster_masses(
    const GridRegular& grid,
    const int num_colors,
    const int color_idx,
    const std::vector<std::int8_t>& site_state,
    const int lower_height,
    const int upper_height)
{
    if (num_colors <= 0 || color_idx < 0 || color_idx >= num_colors ||
        upper_height < lower_height) {
        return {};
    }

    const int active_val = color_to_active_value(num_colors, color_idx);
    const int bottom = std::max(0, lower_height);
    const int top = std::min(upper_height, grid.grow_top_coord());
    if (top < bottom) return {};

    const std::int64_t n_sites = std::min(
        grid.total_size,
        static_cast<std::int64_t>(site_state.size()));
    std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
    std::deque<std::int64_t> queue;
    std::int64_t neigh[6];
    std::uint64_t largest_spanning_mass = 0u;
    std::vector<std::int64_t> cluster_sizes;

    for (std::int64_t idx = 0; idx < n_sites; ++idx) {
        if (visited[static_cast<std::size_t>(idx)]) continue;
        if (static_cast<int>(site_state[static_cast<std::size_t>(idx)]) != active_val) {
            continue;
        }

        const int h = grid.grow_coord(idx);
        if (h < bottom || h > top) {
            continue;
        }

        visited[static_cast<std::size_t>(idx)] = 1;
        queue.push_back(idx);
        std::uint64_t component_mass = 0u;
        bool touches_bottom = false;
        bool touches_top = false;

        while (!queue.empty()) {
            const std::int64_t u = queue.front();
            queue.pop_front();
            ++component_mass;
            const int hu = grid.grow_coord(u);
            touches_bottom = touches_bottom || (hu == bottom);
            touches_top = touches_top || (hu == top);

            const int nneigh = collect_neighbors(grid, u, neigh);
            for (int ni = 0; ni < nneigh; ++ni) {
                const std::int64_t v = neigh[ni];
                if (v < 0 || v >= n_sites) continue;
                if (visited[static_cast<std::size_t>(v)]) continue;
                if (static_cast<int>(site_state[static_cast<std::size_t>(v)]) != active_val) {
                    continue;
                }

                const int hv = grid.grow_coord(v);
                if (hv < bottom || hv > top) continue;

                visited[static_cast<std::size_t>(v)] = 1;
                queue.push_back(v);
            }
        }

        cluster_sizes.push_back(static_cast<std::int64_t>(component_mass));
        if (touches_bottom && touches_top) {
            largest_spanning_mass =
                std::max(largest_spanning_mass, component_mass);
        }
    }

    std::sort(cluster_sizes.begin(), cluster_sizes.end(), std::greater<std::int64_t>());
    return {
        static_cast<double>(largest_spanning_mass),
        std::move(cluster_sizes)
    };
}

double compute_volume_fraction_between_upper_surfaces(
    const GridRegular& grid,
    const int num_colors,
    const int color_idx,
    const std::vector<std::int8_t>& site_state,
    const std::vector<int>& previous_heights,
    const std::vector<int>& current_heights,
    const int lower_height,
    const int upper_height)
{
    if (num_colors <= 0 || color_idx < 0 || color_idx >= num_colors) return 0.0;
    if (previous_heights.size() != current_heights.size()) return 0.0;

    const int active_val = color_to_active_value(num_colors, color_idx);
    const int bottom = std::max(0, lower_height);
    const int top = std::min(upper_height, grid.grow_top_coord());
    if (top < bottom) return 0.0;

    long long occupied_between_surfaces = 0;
    const std::int64_t n_sites = std::min(
        grid.total_size,
        static_cast<std::int64_t>(site_state.size()));
    for (std::int64_t idx = 0; idx < n_sites; ++idx) {
        if (static_cast<int>(site_state[static_cast<std::size_t>(idx)]) != active_val) {
            continue;
        }

        const std::int64_t lateral_idx = (grid.dim == 2)
            ? grid.x_of(idx)
            : static_cast<std::int64_t>(grid.x_of(idx)) + static_cast<std::int64_t>(grid.SX) * grid.y_of(idx);
        if (lateral_idx < 0 ||
            lateral_idx >= static_cast<std::int64_t>(current_heights.size())) {
            continue;
        }

        const int h = grid.grow_coord(idx);
        if (h < bottom || h > top) continue;
        if (h > previous_heights[static_cast<std::size_t>(lateral_idx)] &&
            h <= current_heights[static_cast<std::size_t>(lateral_idx)]) {
            ++occupied_between_surfaces;
        }
    }

    const double lateral_area = (grid.dim == 2)
        ? static_cast<double>(grid.SX)
        : static_cast<double>(grid.SX) * static_cast<double>(grid.SY);
    const double max_volume =
        lateral_area * static_cast<double>(std::max(1, top - bottom + 1));
    return max_volume > 0.0
        ? static_cast<double>(occupied_between_surfaces) / max_volume
        : 0.0;
}

inline double estimate_t_eq_from_timeseries(const TimeSeries& ts,
                                            const int window_roll = 15,
                                            const int window_block = 10,
                                            const int min_stable_steps = 15,
                                            const double rel_tol = 2.5e-2,
                                            const double abs_tol = 1.0e-6,
                                            const double s_prime_threshold = 1.0e-5)
{
    if (ts.t.empty()) {
        throw std::runtime_error("estimate_t_eq_from_timeseries: ts.t vazio");
    }

    const std::vector<double> p_mean = build_mean_p_series(ts);
    return estimate_t_eq_from_series(
        ts.t,
        p_mean,
        window_roll,
        window_block,
        min_stable_steps,
        rel_tol,
        abs_tol,
        s_prime_threshold);
}

inline std::vector<double> estimate_t_eq_by_species_from_timeseries(
    const TimeSeries& ts,
    const int window_roll = 15,
    const int window_block = 10,
    const int min_stable_steps = 15,
    const double rel_tol = 2.5e-2,
    const double abs_tol = 1.0e-6,
    const double s_prime_threshold = 1.0e-5,
    const bool use_front_fraction = false,
    const bool require_target = false,
    const double target = 0.0,
    const int validation_window_steps = 0)
{
    if (ts.t.empty()) {
        throw std::runtime_error("estimate_t_eq_by_species_from_timeseries: ts.t vazio");
    }
    if (ts.num_colors <= 0 ||
        static_cast<int>(ts.p_t.size()) != ts.num_colors ||
        static_cast<int>(ts.f_t.size()) != ts.num_colors) {
        throw std::runtime_error(
            "estimate_t_eq_by_species_from_timeseries: p_t/f_t incompatível");
    }

    std::vector<double> out(static_cast<std::size_t>(ts.num_colors),
                            std::numeric_limits<double>::quiet_NaN());
    for (int c = 0; c < ts.num_colors; ++c) {
        const auto& series = use_front_fraction
            ? ts.f_t[static_cast<std::size_t>(c)]
            : ts.p_t[static_cast<std::size_t>(c)];
        out[static_cast<std::size_t>(c)] = estimate_t_eq_from_series(
            ts.t,
            series,
            window_roll,
            window_block,
            min_stable_steps,
            rel_tol,
            abs_tol,
            s_prime_threshold,
            require_target,
            target,
            validation_window_steps);
    }
    return out;
}

inline double mean_finite_values(const std::vector<double>& values)
{
    double sum = 0.0;
    int count = 0;
    for (const double v : values) {
        if (!std::isfinite(v)) continue;
        sum += v;
        ++count;
    }
    if (count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(count);
}

inline std::uint64_t splitmix64_hash(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct AbsSiteKeyHash {
    std::size_t operator()(const std::uint64_t key) const noexcept {
        return static_cast<std::size_t>(splitmix64_hash(key));
    }
};

inline std::uint64_t abs_site_key(const int x, const int y, const int z)
{
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(z)) << 32) |
           (static_cast<std::uint64_t>(static_cast<std::uint16_t>(y)) << 16) |
           static_cast<std::uint64_t>(static_cast<std::uint16_t>(x));
}

inline void decode_abs_site_key(const std::uint64_t key, int& x, int& y, int& z)
{
    x = static_cast<int>(key & 0xffffu);
    y = static_cast<int>((key >> 16) & 0xffffu);
    z = static_cast<int>((key >> 32) & 0xffffffffu);
}

inline int abs_grow_coord_from_key(const std::uint64_t key, const int dim)
{
    int x = 0, y = 0, z = 0;
    decode_abs_site_key(key, x, y, z);
    return (dim == 2) ? y : z;
}

inline std::uint64_t abs_bond_key(std::uint64_t a, std::uint64_t b)
{
    if (b < a) std::swap(a, b);
    return splitmix64_hash(a) ^ (splitmix64_hash(b) + 0x9e3779b97f4a7c15ULL +
                                 (splitmix64_hash(a) << 6) + (splitmix64_hash(a) >> 2));
}

struct AbsBondKey {
    std::uint64_t a = 0;
    std::uint64_t b = 0;

    bool operator==(const AbsBondKey& other) const noexcept {
        return a == other.a && b == other.b;
    }
};

struct AbsBondKeyHash {
    std::size_t operator()(const AbsBondKey& key) const noexcept {
        return static_cast<std::size_t>(
            splitmix64_hash(key.a) ^
            (splitmix64_hash(key.b) + 0x9e3779b97f4a7c15ULL +
             (splitmix64_hash(key.a) << 6) + (splitmix64_hash(key.a) >> 2)));
    }
};

inline AbsBondKey make_abs_bond_key(std::uint64_t a, std::uint64_t b)
{
    if (b < a) std::swap(a, b);
    return {a, b};
}

inline void collect_abs_neighbors(const int dim,
                                  const int L,
                                  const std::uint64_t key,
                                  std::uint64_t out[6],
                                  int& count)
{
    int x = 0, y = 0, z = 0;
    decode_abs_site_key(key, x, y, z);
    count = 0;

    out[count++] = abs_site_key((x + L - 1) % L, y, z);
    out[count++] = abs_site_key((x + 1) % L, y, z);

    if (dim == 2) {
        if (y > 0) out[count++] = abs_site_key(x, y - 1, 0);
        out[count++] = abs_site_key(x, y + 1, 0);
        return;
    }

    out[count++] = abs_site_key(x, (y + L - 1) % L, z);
    out[count++] = abs_site_key(x, (y + 1) % L, z);
    if (z > 0) out[count++] = abs_site_key(x, y, z - 1);
    out[count++] = abs_site_key(x, y, z + 1);
}

struct PackedBitset {
    std::vector<std::uint64_t> words;

    explicit PackedBitset(const std::uint64_t nbits = 0)
        : words(static_cast<std::size_t>((nbits + 63ULL) >> 6), 0ULL) {}

    inline void clear() {
        std::fill(words.begin(), words.end(), 0ULL);
    }

    inline bool test(const std::uint64_t idx) const {
        return ((words[static_cast<std::size_t>(idx >> 6)] >> (idx & 63ULL)) & 1ULL) != 0ULL;
    }

    inline void set(const std::uint64_t idx) {
        words[static_cast<std::size_t>(idx >> 6)] |= (1ULL << (idx & 63ULL));
    }

    inline bool test_and_set(const std::uint64_t idx) {
        std::uint64_t& w = words[static_cast<std::size_t>(idx >> 6)];
        const std::uint64_t m = (1ULL << (idx & 63ULL));
        if ((w & m) != 0ULL) return false;
        w |= m;
        return true;
    }
};

struct SlidingSlabGrid {
    int dim = 2;
    int L = 1;
    int SY_lat = 1;
    bool is_node = false;
    std::size_t layer_sites = 1;
    int min_retained_h = 0;

    struct Layer {
        std::vector<std::int8_t> state;
        std::vector<std::uint8_t> bonds;
    };
    std::deque<Layer> layers;

    SlidingSlabGrid(const int dim_, const int L_, const bool is_node_)
        : dim(dim_),
          L(L_),
          SY_lat(dim_ == 2 ? 1 : L_),
          is_node(is_node_),
          layer_sites(static_cast<std::size_t>(L_) *
                      static_cast<std::size_t>(dim_ == 2 ? 1 : L_)),
          min_retained_h(0) {}

    inline std::size_t lat_index(const int x, const int y_lat = 0) const {
        return static_cast<std::size_t>(y_lat) * static_cast<std::size_t>(L) +
               static_cast<std::size_t>(x);
    }

    inline void decode_lat(const std::size_t lat, int& x, int& y_lat) const {
        const std::size_t l_sz = static_cast<std::size_t>(L);
        x = static_cast<int>(lat % l_sz);
        y_lat = (dim == 3) ? static_cast<int>(lat / l_sz) : 0;
    }

    inline std::int64_t encode_key(const int h, const std::size_t lat) const {
        return static_cast<std::int64_t>(h) * static_cast<std::int64_t>(layer_sites) +
               static_cast<std::int64_t>(lat);
    }

    inline std::int64_t encode_coords(const int x, const int y_lat, const int h) const {
        return encode_key(h, lat_index(x, y_lat));
    }

    inline void decode_key(const std::int64_t key, int& h, std::size_t& lat) const {
        const std::int64_t ls = static_cast<std::int64_t>(layer_sites);
        h = static_cast<int>(key / ls);
        lat = static_cast<std::size_t>(key % ls);
    }

    inline int grow_coord(const std::int64_t key) const {
        return static_cast<int>(key / static_cast<std::int64_t>(layer_sites));
    }

    static inline std::uint64_t pack_bond_ref(const int owner_h,
                                              const std::size_t owner_lat,
                                              const int dir) {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(owner_h)) << 34) |
               ((static_cast<std::uint64_t>(owner_lat) & 0xffffffffULL) << 2) |
               static_cast<std::uint64_t>(dir & 3);
    }

    static inline void unpack_bond_ref(const std::uint64_t ref,
                                       int& owner_h,
                                       std::size_t& owner_lat,
                                       int& dir) {
        dir = static_cast<int>(ref & 3ULL);
        owner_lat = static_cast<std::size_t>((ref >> 2) & 0xffffffffULL);
        owner_h = static_cast<int>(ref >> 34);
    }

    inline int get_site(const int h, const std::size_t lat) const {
        if (h < min_retained_h) return 0; // discarded history or h < 0 acts as blocked
        const std::size_t rel = static_cast<std::size_t>(h - min_retained_h);
        if (rel >= layers.size()) return -1;
        const Layer& lyr = layers[rel];
        if (lyr.state.empty()) return -1;
        return static_cast<int>(lyr.state[lat]);
    }

    inline int get_site_key(const std::int64_t key) const {
        if (key < 0) return 0;
        int h = 0;
        std::size_t lat = 0;
        decode_key(key, h, lat);
        return get_site(h, lat);
    }

    inline void set_site(const int h, const std::size_t lat, const int value) {
        if (h < min_retained_h) return;
        const std::size_t rel = static_cast<std::size_t>(h - min_retained_h);
        if (rel >= layers.size()) {
            layers.resize(rel + 1u);
        }
        Layer& lyr = layers[rel];
        if (lyr.state.empty()) {
            lyr.state.assign(layer_sites, static_cast<std::int8_t>(-1));
        }
        lyr.state[lat] = static_cast<std::int8_t>(value);
    }

    inline void set_site_key(const std::int64_t key, const int value) {
        if (key < 0) return;
        int h = 0;
        std::size_t lat = 0;
        decode_key(key, h, lat);
        set_site(h, lat, value);
    }

    inline bool mark_tested_bond_packed(const std::uint64_t ref) {
        if (is_node) return true;
        int owner_h = 0, dir = 0;
        std::size_t owner_lat = 0;
        unpack_bond_ref(ref, owner_h, owner_lat, dir);
        if (owner_h < min_retained_h) return false;
        const std::size_t rel = static_cast<std::size_t>(owner_h - min_retained_h);
        if (rel >= layers.size()) {
            layers.resize(rel + 1u);
        }
        Layer& lyr = layers[rel];
        if (lyr.bonds.empty()) {
            lyr.bonds.assign(layer_sites, 0u);
        }
        const std::uint8_t bit = static_cast<std::uint8_t>(1u << dir);
        std::uint8_t& cell = lyr.bonds[owner_lat];
        if ((cell & bit) != 0u) return false;
        cell |= bit;
        return true;
    }

    inline void set_open_bond_packed(const std::uint64_t ref) {
        if (is_node) return;
        int owner_h = 0, dir = 0;
        std::size_t owner_lat = 0;
        unpack_bond_ref(ref, owner_h, owner_lat, dir);
        if (owner_h < min_retained_h) return;
        const std::size_t rel = static_cast<std::size_t>(owner_h - min_retained_h);
        if (rel >= layers.size()) {
            layers.resize(rel + 1u);
        }
        Layer& lyr = layers[rel];
        if (lyr.bonds.empty()) {
            lyr.bonds.assign(layer_sites, 0u);
        }
        lyr.bonds[owner_lat] |= static_cast<std::uint8_t>(1u << (dir + 3));
    }

    inline void set_open_bond_dir(const int owner_h,
                                  const std::size_t owner_lat,
                                  const int dir) {
        set_open_bond_packed(pack_bond_ref(owner_h, owner_lat, dir));
    }

    inline bool is_open_bond(const int owner_h,
                             const std::size_t owner_lat,
                             const int dir) const {
        if (is_node || owner_h < min_retained_h) return false;
        const std::size_t rel = static_cast<std::size_t>(owner_h - min_retained_h);
        if (rel >= layers.size()) return false;
        const Layer& lyr = layers[rel];
        if (lyr.bonds.empty()) return false;
        return (lyr.bonds[owner_lat] & static_cast<std::uint8_t>(1u << (dir + 3))) != 0u;
    }

    void discard_below(const int target_min_h, const bool trim_heap = false) {
        if (target_min_h <= min_retained_h) return;
        bool freed_any = false;
        while (min_retained_h < target_min_h && !layers.empty()) {
            layers.pop_front();
            ++min_retained_h;
            freed_any = true;
        }
        if (layers.empty() && min_retained_h < target_min_h) {
            min_retained_h = target_min_h;
        }
#if defined(__linux__) && defined(__GLIBC__)
        if (trim_heap && freed_any) {
            malloc_trim(0);
        }
#else
        (void)trim_heap;
        (void)freed_any;
#endif
    }
};

struct SlabFractionResult {
    double p_node = 0.0;
    double p_bond = 0.0;
    std::vector<double> p_node_by_species;
    std::vector<double> p_bond_by_species;
    int num_spanning_clusters = 0;
    long long largest_spanning_sites = 0;
    long long largest_spanning_bonds = 0;
    int largest_spanning_sp = -1;
    long long largest_spanning_hull = 0;
    long long largest_spanning_full_boundary = 0;
    int largest_spanning_species = 0;
    long long largest_spanning_interspecies_bonds = 0;
    std::vector<long long> other_spanning_sites;
    std::vector<long long> other_spanning_bonds;
    std::vector<int> other_spanning_sp;
    std::vector<long long> other_spanning_hull;
    std::vector<int> other_spanning_species;
    std::vector<long long> other_spanning_interspecies_bonds;
    long long interspecies_contact_bonds = 0;
    long long interspecies_contact_sites = 0;
    long long spanning_interspecies_bonds = 0;
    long long spanning_interspecies_sites = 0;
    FractalCountsSample fractal_counts;
};

void sort_spanning_clusters(FractalCountsSample& sample)
{
    std::sort(sample.spanning_clusters.begin(), sample.spanning_clusters.end(),
              [](const FractalClusterCounts& a, const FractalClusterCounts& b) {
        if (a.largest_component_sites != b.largest_component_sites)
            return a.largest_component_sites > b.largest_component_sites;
        if (a.largest_component_bonds != b.largest_component_bonds)
            return a.largest_component_bonds > b.largest_component_bonds;
        return a.largest_component_seed < b.largest_component_seed;
    });
}

// Computes stabilized slab metrics on [bottom, top] (where top = bottom + L - 1):
// - Box counting, yardstick sphere covering, and chemical distance origins are OFF for all sizes.
// - Monochromatic spanning clusters (touching both bottom and top, strictly separated by species)
//   are detected for all lattice sizes L in 2D and 3D using 1-bit-per-site packed bitsets.
// - Computes: largest spanning cluster size, other spanning cluster sizes (if any),
//   accessible exterior perimeter (Hull total), and shortest path of the spanning cluster(s).
SlabFractionResult compute_slab_stabilized(
    const SlidingSlabGrid& grid,
    const int num_colors,
    const int bottom,
    const int top,
    const int sample_index,
    const int t_stab)
{
    SlabFractionResult res;
    res.p_node_by_species.assign(static_cast<std::size_t>(std::max(1, num_colors)), 0.0);
    res.p_bond_by_species.assign(static_cast<std::size_t>(std::max(1, num_colors)), 0.0);
    if (top < bottom || grid.L <= 0) {
        return res;
    }

    const int dim = grid.dim;
    const int L = grid.L;
    const bool is_node = grid.is_node;
    const int h_count = top - bottom + 1;
    const std::size_t layer_sites = grid.layer_sites;
    const std::uint64_t slab_sites =
        static_cast<std::uint64_t>(h_count) * static_cast<std::uint64_t>(layer_sites);

    const long long total_sites = static_cast<long long>(slab_sites);
    const long long lateral_area = static_cast<long long>(layer_sites);
    const long long total_bonds = (dim == 2)
        ? (lateral_area * static_cast<long long>(h_count) +
           lateral_area * static_cast<long long>(std::max(0, h_count - 1)))
        : (2LL * lateral_area * static_cast<long long>(h_count) +
           lateral_area * static_cast<long long>(std::max(0, h_count - 1)));

    long long occupied_sites = 0;
    long long open_bonds_in_slab = 0;
    long long interspecies_contact_bonds = 0;
    long long interspecies_contact_sites = 0;
    std::vector<long long> occ_by_sp(static_cast<std::size_t>(num_colors), 0);
    std::vector<long long> bonds_by_sp(static_cast<std::size_t>(num_colors), 0);

    for (int h = bottom; h <= top; ++h) {
        for (std::size_t lat = 0; lat < layer_sites; ++lat) {
            const int val = grid.get_site(h, lat);
            if (val <= 0) continue;
            ++occupied_sites;
            const int c = value_to_color_index(num_colors, val);
            if (c >= 0 && c < num_colors) {
                ++occ_by_sp[static_cast<std::size_t>(c)];
            }
            if (!is_node || num_colors > 1) {
                int x = 0, y_lat = 0;
                grid.decode_lat(lat, x, y_lat);
                const std::size_t lat_xp = grid.lat_index((x + 1 == L) ? 0 : (x + 1), y_lat);
                const int v_xp = grid.get_site(h, lat_xp);
                if (!is_node && grid.is_open_bond(h, lat, 0) && v_xp == val) {
                    ++open_bonds_in_slab;
                    if (c >= 0 && c < num_colors) ++bonds_by_sp[static_cast<std::size_t>(c)];
                }
                int v_yp = -1;
                if (dim == 3) {
                    const std::size_t lat_yp = grid.lat_index(x, (y_lat + 1 == L) ? 0 : (y_lat + 1));
                    v_yp = grid.get_site(h, lat_yp);
                    if (!is_node && grid.is_open_bond(h, lat, 1) && v_yp == val) {
                        ++open_bonds_in_slab;
                        if (c >= 0 && c < num_colors) ++bonds_by_sp[static_cast<std::size_t>(c)];
                    }
                }
                const int v_hp = (h < top) ? grid.get_site(h + 1, lat) : -1;
                if (!is_node && h < top && grid.is_open_bond(h, lat, 2) && v_hp == val) {
                    ++open_bonds_in_slab;
                    if (c >= 0 && c < num_colors) ++bonds_by_sp[static_cast<std::size_t>(c)];
                }

                if (num_colors > 1) {
                    bool touches_other_sp = false;
                    if (L > 1) {
                        if (v_xp > 0 && v_xp != val) {
                            ++interspecies_contact_bonds;
                            touches_other_sp = true;
                        }
                        const int v_xm = grid.get_site(h, grid.lat_index((x == 0) ? (L - 1) : (x - 1), y_lat));
                        if (v_xm > 0 && v_xm != val) touches_other_sp = true;
                        if (dim == 3) {
                            if (v_yp > 0 && v_yp != val) {
                                ++interspecies_contact_bonds;
                                touches_other_sp = true;
                            }
                            const int v_ym = grid.get_site(h, grid.lat_index(x, (y_lat == 0) ? (L - 1) : (y_lat - 1)));
                            if (v_ym > 0 && v_ym != val) touches_other_sp = true;
                        }
                    }
                    if (h < top && v_hp > 0 && v_hp != val) {
                        ++interspecies_contact_bonds;
                        touches_other_sp = true;
                    }
                    if (h > bottom) {
                        const int v_hm = grid.get_site(h - 1, lat);
                        if (v_hm > 0 && v_hm != val) touches_other_sp = true;
                    }
                    if (touches_other_sp) {
                        ++interspecies_contact_sites;
                    }
                }
            }
        }
    }

    res.interspecies_contact_bonds = interspecies_contact_bonds;
    res.interspecies_contact_sites = interspecies_contact_sites;

    res.p_node = total_sites > 0
        ? static_cast<double>(occupied_sites) / static_cast<double>(total_sites)
        : 0.0;
    res.p_bond = total_bonds > 0
        ? static_cast<double>(open_bonds_in_slab) / static_cast<double>(total_bonds)
        : 0.0;
    for (int c = 0; c < num_colors; ++c) {
        res.p_node_by_species[static_cast<std::size_t>(c)] = total_sites > 0
            ? static_cast<double>(occ_by_sp[static_cast<std::size_t>(c)]) /
              static_cast<double>(total_sites)
            : 0.0;
        res.p_bond_by_species[static_cast<std::size_t>(c)] = total_bonds > 0
            ? static_cast<double>(bonds_by_sp[static_cast<std::size_t>(c)]) /
              static_cast<double>(total_bonds)
            : 0.0;
    }

    auto for_each_connected_neighbor = [&](const std::uint64_t u_local,
                                           const int target_val,
                                           auto&& fn) {
        const int h_local = static_cast<int>(u_local / layer_sites);
        const std::size_t lat = static_cast<std::size_t>(u_local % layer_sites);
        const int h = bottom + h_local;
        int x = 0, y_lat = 0;
        grid.decode_lat(lat, x, y_lat);
        const std::uint64_t layer_base = static_cast<std::uint64_t>(h_local) * layer_sites;

        // -x
        {
            const int nx = (x == 0) ? (L - 1) : (x - 1);
            const std::size_t nlat = grid.lat_index(nx, y_lat);
            if (grid.get_site(h, nlat) == target_val &&
                (is_node || grid.is_open_bond(h, nlat, 0))) {
                fn(layer_base + nlat, h_local);
            }
        }
        // +x
        if (L > 1) {
            const int nx = (x + 1 == L) ? 0 : (x + 1);
            const std::size_t nlat = grid.lat_index(nx, y_lat);
            if (grid.get_site(h, nlat) == target_val &&
                (is_node || grid.is_open_bond(h, lat, 0))) {
                fn(layer_base + nlat, h_local);
            }
        }
        if (dim == 3) {
            // -y
            {
                const int ny = (y_lat == 0) ? (L - 1) : (y_lat - 1);
                const std::size_t nlat = grid.lat_index(x, ny);
                if (grid.get_site(h, nlat) == target_val &&
                    (is_node || grid.is_open_bond(h, nlat, 1))) {
                    fn(layer_base + nlat, h_local);
                }
            }
            // +y
            if (L > 1) {
                const int ny = (y_lat + 1 == L) ? 0 : (y_lat + 1);
                const std::size_t nlat = grid.lat_index(x, ny);
                if (grid.get_site(h, nlat) == target_val &&
                    (is_node || grid.is_open_bond(h, lat, 1))) {
                    fn(layer_base + nlat, h_local);
                }
            }
        }
        // -h
        if (h_local > 0) {
            if (grid.get_site(h - 1, lat) == target_val &&
                (is_node || grid.is_open_bond(h - 1, lat, 2))) {
                fn(u_local - layer_sites, h_local - 1);
            }
        }
        // +h
        if (h_local + 1 < h_count) {
            if (grid.get_site(h + 1, lat) == target_val &&
                (is_node || grid.is_open_bond(h, lat, 2))) {
                fn(u_local + layer_sites, h_local + 1);
            }
        }
    };

    struct SpanningCandidate {
        std::uint64_t seed_local = 0;
        int active_val = 0;
        int species = 1;
        long long num_sites = 0;
        long long num_bonds = 0;
    };

    PackedBitset bits_a(slab_sites);
    std::vector<std::uint64_t> curr_wave;
    std::vector<std::uint64_t> next_wave;
    curr_wave.reserve(std::min<std::size_t>(layer_sites, 1024u * 1024u));
    next_wave.reserve(std::min<std::size_t>(layer_sites, 1024u * 1024u));

    std::vector<SpanningCandidate> spanning_candidates;

    // Every spanning cluster must touch z = bottom (h_local = 0).
    // Seeding exclusively on z = bottom discovers 100% of spanning clusters
    // across all species while skipping non-spanning interior components.
    for (std::size_t lat = 0; lat < layer_sites; ++lat) {
        const int seed_val = grid.get_site(bottom, lat);
        if (seed_val <= 0) continue;
        const std::uint64_t seed_local = static_cast<std::uint64_t>(lat);
        if (!bits_a.test_and_set(seed_local)) continue;

        const int c_idx = value_to_color_index(num_colors, seed_val);
        if (c_idx < 0 || c_idx >= num_colors) continue;

        long long comp_sites = 0;
        long long comp_edge_visits = 0;
        bool touches_top = (h_count == 1);

        curr_wave.clear();
        curr_wave.push_back(seed_local);
        while (!curr_wave.empty()) {
            next_wave.clear();
            for (const std::uint64_t u : curr_wave) {
                ++comp_sites;
                const int uh = static_cast<int>(u / layer_sites);
                if (uh == h_count - 1) touches_top = true;

                for_each_connected_neighbor(
                    u, seed_val, [&](const std::uint64_t v, const int vh) {
                        ++comp_edge_visits;
                        if (vh == h_count - 1) touches_top = true;
                        if (bits_a.test_and_set(v)) {
                            next_wave.push_back(v);
                        }
                    });
            }
            curr_wave.swap(next_wave);
        }

        if (touches_top) {
            spanning_candidates.push_back({
                seed_local,
                seed_val,
                c_idx + 1,
                comp_sites,
                comp_edge_visits / 2
            });
        }
    }

    std::sort(spanning_candidates.begin(), spanning_candidates.end(),
              [](const SpanningCandidate& a, const SpanningCandidate& b) {
                  if (a.num_sites != b.num_sites) return a.num_sites > b.num_sites;
                  if (a.num_bonds != b.num_bonds) return a.num_bonds > b.num_bonds;
                  return a.seed_local < b.seed_local;
              });

    res.fractal_counts.sample_index = sample_index;
    res.fractal_counts.anchor_z = bottom;
    res.fractal_counts.slab_top_z = top;
    res.fractal_counts.t_stab = t_stab;
    res.fractal_counts.interspecies_contact_bonds = interspecies_contact_bonds;
    res.fractal_counts.interspecies_contact_sites = interspecies_contact_sites;
    res.num_spanning_clusters = static_cast<int>(spanning_candidates.size());

    if (spanning_candidates.empty()) {
        return res;
    }

    PackedBitset bits_b(slab_sites);
    std::vector<std::uint64_t> bottom_sources;
    bottom_sources.reserve(layer_sites);

    for (std::size_t ci = 0; ci < spanning_candidates.size(); ++ci) {
        const SpanningCandidate& cand = spanning_candidates[ci];

        // 1. Mark cluster membership in bits_a and collect all bottom-plane sources.
        bits_a.clear();
        bottom_sources.clear();
        bits_a.set(cand.seed_local);
        curr_wave.clear();
        curr_wave.push_back(cand.seed_local);
        while (!curr_wave.empty()) {
            next_wave.clear();
            for (const std::uint64_t u : curr_wave) {
                if (u < layer_sites) {
                    bottom_sources.push_back(u);
                }
                for_each_connected_neighbor(
                    u, cand.active_val, [&](const std::uint64_t v, const int /*vh*/) {
                        if (bits_a.test_and_set(v)) {
                            next_wave.push_back(v);
                        }
                    });
            }
            curr_wave.swap(next_wave);
        }

        // 2. Shortest path (in edges) from z = bottom to z = top inside this spanning cluster.
        int sp_edges = -1;
        std::uint64_t top_reached_local = cand.seed_local;
        if (!bottom_sources.empty()) {
            bits_b.clear();
            curr_wave.clear();
            for (const std::uint64_t src : bottom_sources) {
                if (bits_b.test_and_set(src)) {
                    curr_wave.push_back(src);
                }
            }
            int dist = 0;
            while (!curr_wave.empty() && sp_edges < 0) {
                next_wave.clear();
                for (const std::uint64_t u : curr_wave) {
                    const int uh = static_cast<int>(u / layer_sites);
                    if (uh == h_count - 1) {
                        sp_edges = dist;
                        top_reached_local = u;
                        break;
                    }
                    for_each_connected_neighbor(
                        u, cand.active_val, [&](const std::uint64_t v, const int /*vh*/) {
                            if (bits_a.test(v) && bits_b.test_and_set(v)) {
                                next_wave.push_back(v);
                            }
                        });
                }
                curr_wave.swap(next_wave);
                ++dist;
            }
        }

        // 3. Accessible exterior perimeter (Hull total) via exterior complement flood-fill.
        bits_b.clear();
        curr_wave.clear();
        const std::uint64_t top_layer_offset =
            static_cast<std::uint64_t>(h_count - 1) * static_cast<std::uint64_t>(layer_sites);
        for (std::size_t lat = 0; lat < layer_sites; ++lat) {
            const std::uint64_t b_idx = static_cast<std::uint64_t>(lat);
            if (!bits_a.test(b_idx) && bits_b.test_and_set(b_idx)) {
                curr_wave.push_back(b_idx);
            }
            if (h_count > 1) {
                const std::uint64_t t_idx = top_layer_offset + static_cast<std::uint64_t>(lat);
                if (!bits_a.test(t_idx) && bits_b.test_and_set(t_idx)) {
                    curr_wave.push_back(t_idx);
                }
            }
        }

        while (!curr_wave.empty()) {
            next_wave.clear();
            for (const std::uint64_t u : curr_wave) {
                const int h_local = static_cast<int>(u / layer_sites);
                const std::size_t lat = static_cast<std::size_t>(u % layer_sites);
                int x = 0, y_lat = 0;
                grid.decode_lat(lat, x, y_lat);
                const std::uint64_t lbase = static_cast<std::uint64_t>(h_local) * layer_sites;

                const std::uint64_t v_xm = lbase + grid.lat_index((x == 0) ? (L - 1) : (x - 1), y_lat);
                if (!bits_a.test(v_xm) && bits_b.test_and_set(v_xm)) next_wave.push_back(v_xm);

                const std::uint64_t v_xp = lbase + grid.lat_index((x + 1 == L) ? 0 : (x + 1), y_lat);
                if (!bits_a.test(v_xp) && bits_b.test_and_set(v_xp)) next_wave.push_back(v_xp);

                if (dim == 3) {
                    const std::uint64_t v_ym = lbase + grid.lat_index(x, (y_lat == 0) ? (L - 1) : (y_lat - 1));
                    if (!bits_a.test(v_ym) && bits_b.test_and_set(v_ym)) next_wave.push_back(v_ym);

                    const std::uint64_t v_yp = lbase + grid.lat_index(x, (y_lat + 1 == L) ? 0 : (y_lat + 1));
                    if (!bits_a.test(v_yp) && bits_b.test_and_set(v_yp)) next_wave.push_back(v_yp);
                }

                if (h_local > 0) {
                    const std::uint64_t v_hm = u - layer_sites;
                    if (!bits_a.test(v_hm) && bits_b.test_and_set(v_hm)) next_wave.push_back(v_hm);
                }
                if (h_local + 1 < h_count) {
                    const std::uint64_t v_hp = u + layer_sites;
                    if (!bits_a.test(v_hp) && bits_b.test_and_set(v_hp)) next_wave.push_back(v_hp);
                }
            }
            curr_wave.swap(next_wave);
        }

        long long cluster_hull = 0;
        long long cluster_full_boundary = 0;
        long long cluster_interspecies_boundary = 0;
        for (std::size_t wi = 0; wi < bits_a.words.size(); ++wi) {
            std::uint64_t word = bits_a.words[wi];
            if (word == 0ULL) continue;
            const std::uint64_t base_idx = static_cast<std::uint64_t>(wi) << 6;
            while (word != 0ULL) {
                const int b = __builtin_ctzll(word);
                word &= (word - 1ULL);
                const std::uint64_t u = base_idx + static_cast<std::uint64_t>(b);
                if (u >= slab_sites) break;

                const int h_local = static_cast<int>(u / layer_sites);
                const std::size_t lat = static_cast<std::size_t>(u % layer_sites);
                const int h = bottom + h_local;
                int x = 0, y_lat = 0;
                grid.decode_lat(lat, x, y_lat);
                const std::uint64_t lbase = static_cast<std::uint64_t>(h_local) * layer_sites;

                auto check_neighbor_face = [&](const std::uint64_t v,
                                               const int nh,
                                               const std::size_t nlat) {
                    if (!bits_a.test(v)) {
                        ++cluster_full_boundary;
                        if (bits_b.test(v)) {
                            ++cluster_hull;
                        }
                        if (num_colors > 1) {
                            const int nv = grid.get_site(nh, nlat);
                            if (nv > 0 && nv != cand.active_val) {
                                ++cluster_interspecies_boundary;
                            }
                        }
                    }
                };

                {
                    const std::size_t nlat = grid.lat_index((x == 0) ? (L - 1) : (x - 1), y_lat);
                    check_neighbor_face(lbase + nlat, h, nlat);
                }
                {
                    const std::size_t nlat = grid.lat_index((x + 1 == L) ? 0 : (x + 1), y_lat);
                    check_neighbor_face(lbase + nlat, h, nlat);
                }
                if (dim == 3) {
                    {
                        const std::size_t nlat = grid.lat_index(x, (y_lat == 0) ? (L - 1) : (y_lat - 1));
                        check_neighbor_face(lbase + nlat, h, nlat);
                    }
                    {
                        const std::size_t nlat = grid.lat_index(x, (y_lat + 1 == L) ? 0 : (y_lat + 1));
                        check_neighbor_face(lbase + nlat, h, nlat);
                    }
                }

                if (h_local == 0) {
                    ++cluster_full_boundary;
                    ++cluster_hull;
                } else {
                    check_neighbor_face(u - layer_sites, h - 1, lat);
                }

                if (h_local == h_count - 1) {
                    ++cluster_full_boundary;
                    ++cluster_hull;
                } else {
                    check_neighbor_face(u + layer_sites, h + 1, lat);
                }
            }
        }

        FractalClusterCounts fc;
        fc.eligible = true;
        fc.sample_index = sample_index;
        fc.species = cand.species;
        fc.dim = dim;
        fc.L = L;
        fc.anchor_z = bottom;
        fc.slab_top_z = top;
        fc.t_stab = t_stab;
        fc.type_percolation = is_node ? "node" : "bond";
        fc.boundary_conditions = (dim == 2)
            ? std::vector<std::string>{"x:periodic", "y:open"}
            : std::vector<std::string>{"x:periodic", "y:periodic", "z:open"};
        fc.total_occupied_sites = occupied_sites;
        fc.total_active_bonds = open_bonds_in_slab;
        fc.largest_component_sites = cand.num_sites;
        fc.largest_component_bonds = cand.num_bonds;
        fc.hull_length = cluster_hull;
        fc.full_boundary_length = cluster_full_boundary;
        fc.interspecies_boundary_length = cluster_interspecies_boundary;
        fc.shortest_path_edges = sp_edges;
        fc.largest_component_seed = cand.seed_local;

        int bx = 0, by = 0, tx = 0, ty = 0;
        grid.decode_lat(static_cast<std::size_t>(cand.seed_local % layer_sites), bx, by);
        grid.decode_lat(static_cast<std::size_t>(top_reached_local % layer_sites), tx, ty);
        fc.minimum_path_yardstick.defined = (sp_edges >= 0);
        fc.minimum_path_yardstick.reason_if_undefined =
            (sp_edges >= 0) ? "" : "no_spanning_path";
        fc.minimum_path_yardstick.path_length = sp_edges;
        fc.minimum_path_yardstick.base_site = cand.seed_local;
        fc.minimum_path_yardstick.top_site = top_reached_local;
        fc.minimum_path_yardstick.base_coordinates = (dim == 2)
            ? std::vector<int>{bx, bottom}
            : std::vector<int>{bx, by, bottom};
        fc.minimum_path_yardstick.top_coordinates = (dim == 2)
            ? std::vector<int>{tx, top}
            : std::vector<int>{tx, ty, top};

        res.fractal_counts.spanning_clusters.push_back(std::move(fc));

        if (ci == 0) {
            res.largest_spanning_sites = cand.num_sites;
            res.largest_spanning_bonds = cand.num_bonds;
            res.largest_spanning_sp = sp_edges;
            res.largest_spanning_hull = cluster_hull;
            res.largest_spanning_full_boundary = cluster_full_boundary;
            res.largest_spanning_species = cand.species;
            res.largest_spanning_interspecies_bonds = cluster_interspecies_boundary;
        } else {
            res.other_spanning_sites.push_back(cand.num_sites);
            res.other_spanning_bonds.push_back(cand.num_bonds);
            res.other_spanning_sp.push_back(sp_edges);
            res.other_spanning_hull.push_back(cluster_hull);
            res.other_spanning_species.push_back(cand.species);
            res.other_spanning_interspecies_bonds.push_back(cluster_interspecies_boundary);
        }
    }

    // 4. Direct intersection between spanning clusters of different species (0 extra memory).
    if (num_colors > 1 && spanning_candidates.size() >= 2) {
        bool has_multiple_spanning_species = false;
        const int first_sp = spanning_candidates[0].species;
        for (std::size_t ci = 1; ci < spanning_candidates.size(); ++ci) {
            if (spanning_candidates[ci].species != first_sp) {
                has_multiple_spanning_species = true;
                break;
            }
        }
        if (has_multiple_spanning_species) {
            // bits_a currently holds the last spanning cluster; add all remaining spanning clusters.
            for (const auto& cand : spanning_candidates) {
                if (!bits_a.test_and_set(cand.seed_local)) continue;
                curr_wave.clear();
                curr_wave.push_back(cand.seed_local);
                while (!curr_wave.empty()) {
                    next_wave.clear();
                    for (const std::uint64_t u : curr_wave) {
                        for_each_connected_neighbor(
                            u, cand.active_val, [&](const std::uint64_t v, const int /*vh*/) {
                                if (bits_a.test_and_set(v)) {
                                    next_wave.push_back(v);
                                }
                            });
                    }
                    curr_wave.swap(next_wave);
                }
            }

            long long span_inter_bonds = 0;
            long long span_inter_sites = 0;
            for (std::size_t wi = 0; wi < bits_a.words.size(); ++wi) {
                std::uint64_t word = bits_a.words[wi];
                if (word == 0ULL) continue;
                const std::uint64_t base_idx = static_cast<std::uint64_t>(wi) << 6;
                while (word != 0ULL) {
                    const int b = __builtin_ctzll(word);
                    word &= (word - 1ULL);
                    const std::uint64_t u = base_idx + static_cast<std::uint64_t>(b);
                    if (u >= slab_sites) break;

                    const int h_local = static_cast<int>(u / layer_sites);
                    const std::size_t lat = static_cast<std::size_t>(u % layer_sites);
                    const int h = bottom + h_local;
                    const int val_u = grid.get_site(h, lat);
                    int x = 0, y_lat = 0;
                    grid.decode_lat(lat, x, y_lat);
                    const std::uint64_t lbase = static_cast<std::uint64_t>(h_local) * layer_sites;

                    bool touches_other_span = false;
                    if (L > 1) {
                        const std::size_t nlat_xp = grid.lat_index((x + 1 == L) ? 0 : (x + 1), y_lat);
                        if (bits_a.test(lbase + nlat_xp) && grid.get_site(h, nlat_xp) != val_u) {
                            ++span_inter_bonds;
                            touches_other_span = true;
                        }
                        const std::size_t nlat_xm = grid.lat_index((x == 0) ? (L - 1) : (x - 1), y_lat);
                        if (bits_a.test(lbase + nlat_xm) && grid.get_site(h, nlat_xm) != val_u) {
                            touches_other_span = true;
                        }
                        if (dim == 3) {
                            const std::size_t nlat_yp = grid.lat_index(x, (y_lat + 1 == L) ? 0 : (y_lat + 1));
                            if (bits_a.test(lbase + nlat_yp) && grid.get_site(h, nlat_yp) != val_u) {
                                ++span_inter_bonds;
                                touches_other_span = true;
                            }
                            const std::size_t nlat_ym = grid.lat_index(x, (y_lat == 0) ? (L - 1) : (y_lat - 1));
                            if (bits_a.test(lbase + nlat_ym) && grid.get_site(h, nlat_ym) != val_u) {
                                touches_other_span = true;
                            }
                        }
                    }
                    if (h_local + 1 < h_count) {
                        if (bits_a.test(u + layer_sites) && grid.get_site(h + 1, lat) != val_u) {
                            ++span_inter_bonds;
                            touches_other_span = true;
                        }
                    }
                    if (h_local > 0) {
                        if (bits_a.test(u - layer_sites) && grid.get_site(h - 1, lat) != val_u) {
                            touches_other_span = true;
                        }
                    }
                    if (touches_other_span) {
                        ++span_inter_sites;
                    }
                }
            }
            res.spanning_interspecies_bonds = span_inter_bonds;
            res.spanning_interspecies_sites = span_inter_sites;
            res.fractal_counts.spanning_interspecies_bonds = span_inter_bonds;
            res.fractal_counts.spanning_interspecies_sites = span_inter_sites;
        }
    }

    return res;
}

inline bool all_values_finite(const std::vector<double>& values)
{
    if (values.empty()) return false;
    for (const double v : values) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

inline bool any_value_finite(const std::vector<double>& values)
{
    for (const double v : values) {
        if (std::isfinite(v)) return true;
    }
    return false;
}

inline void set_species_order_from_t_eq(PercolationSeries& ps,
                                        const std::vector<double>& t_eq_by_species,
                                        const bool include_unequilibrated = false)
{
    ps.color_percolation.clear();
    ps.percolation_order.clear();
    ps.M_size_at_perc.clear();

    if (!include_unequilibrated && !any_value_finite(t_eq_by_species)) {
        return;
    }

    struct SpeciesEq {
        bool is_finite;
        double t_eq;
        int color_idx;
    };

    std::vector<SpeciesEq> ordered;
    ordered.reserve(t_eq_by_species.size());
    for (int c = 0; c < static_cast<int>(t_eq_by_species.size()); ++c) {
        const double val = t_eq_by_species[static_cast<std::size_t>(c)];
        const bool finite = std::isfinite(val);
        if (!include_unequilibrated && !finite) {
            continue;
        }
        ordered.push_back({finite, val, c});
    }

    std::sort(ordered.begin(), ordered.end(),
              [](const SpeciesEq& a, const SpeciesEq& b) {
                  if (a.is_finite != b.is_finite) return a.is_finite > b.is_finite;
                  if (a.is_finite && a.t_eq != b.t_eq) return a.t_eq < b.t_eq;
                  return a.color_idx < b.color_idx;
              });

    int order = 0;
    for (const auto& item : ordered) {
        ps.color_percolation.push_back(item.color_idx + 1);
        ps.percolation_order.push_back(order++);
    }
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
                           const double alpha,
                           const FeedbackControlRule control_rule)
{
    const double f_target = target_fT_create(type_f_T, t_i, f_T, a, alpha);
    double p_step = 0.0;

    switch (control_rule) {
        case FeedbackControlRule::RelativeError: {
            if (f_target > 0.0) {
                p_step = c * ((f_target - f_current) / f_target);
            } else {
                p_step = c * (f_target - f_current);
            }
            // Symmetrical protection against excessive step:
            if (p_step < -c) {
                p_step = -c;
            }
            if (p_step > c) {
                p_step = c;
            }
            break;
        }
        case FeedbackControlRule::Linear:
            p_step = c * (f_target - f_current);
            break;
    }

    double p_next = p_t + p_step;

    if (p_next > 1.0) p_next = 1.0;
    if (p_next < 0.0) p_next = 0.0;

    return p_next;
}

RawFractionsSeries network::create_raw_fractions(
    const int dim,
    const int lenght_network,
    const int fraction_samples,
    const double c_value,
    const double f_T,
    const int type_f_T,
    const std::vector<double> p0,
    const double P0,
    const double a,
    const double alpha,
    const std::string& type_percolation,
    const int& num_colors,
    const std::vector<double>& rho,
    all_random& rng,
    const double sample_gap_over_L,
    const GrowthStopConfig stop_config)
{
    this->c = c_value;
    this->f_T = f_T;

    if (dim != 2 && dim != 3) {
        throw std::runtime_error("create_raw_fractions: dim must be 2 or 3");
    }
    if (lenght_network <= 0) {
        throw std::runtime_error("create_raw_fractions: L must be positive");
    }
    if (num_colors <= 0 || num_colors > 125) {
        throw std::runtime_error("create_raw_fractions: invalid num_colors");
    }

    const bool is_node = (type_percolation == "node");
    const int L = lenght_network;
    const long long base_size = (dim == 2)
        ? static_cast<long long>(L)
        : static_cast<long long>(L) * static_cast<long long>(L);
    const double norm_factor = static_cast<double>(base_size);
    const int requested = std::max(1, fraction_samples);
    const double gap_over_L = std::max(0.0, sample_gap_over_L);
    const int sample_gap_layers = std::max(
        0,
        static_cast<int>(std::llround(gap_over_L * static_cast<double>(L))));
    double stability_over_L = 2.0;
    if (const char* env_stab = std::getenv("SOP_FRACTION_STABILITY_OVER_L")) {
        if (*env_stab != '\0') {
            stability_over_L = std::max(0.0, std::stod(env_stab));
        }
    }
    const int sample_stability_layers = std::max(
        0,
        static_cast<int>(std::llround(stability_over_L * static_cast<double>(L))));
    int hard_max_steps = stop_config.hard_max_steps;
    if (const char* env_steps = std::getenv("SOP_FRACTION_MAX_STEPS")) {
        if (*env_steps != '\0') {
            hard_max_steps = std::max(1, std::stoi(env_steps));
        }
    }
    int progress_interval = std::max(500, L);
    if (const char* env_progress = std::getenv("SOP_FRACTION_PROGRESS_INTERVAL")) {
        if (*env_progress != '\0') {
            progress_interval = std::max(0, std::stoi(env_progress));
        }
    }

    RawFractionsSeries out;
    out.dim = dim;
    out.L = L;
    out.seed = rng.get_seed();
    out.num_colors = num_colors;
    out.requested_samples = requested;
    if (dim == 2) {
        const long long LL = static_cast<long long>(L);
        out.N_total = LL * LL;
        out.E_total = 2LL * LL * LL - LL;
    } else {
        const long long LL = static_cast<long long>(L);
        out.N_total = LL * LL * LL;
        out.E_total = 3LL * LL * LL * LL - LL * LL;
    }
    out.sample_stability_layers = sample_stability_layers;
    out.sample_stability_over_L = stability_over_L;
    out.sample_gap_layers = sample_gap_layers;
    out.sample_gap_over_L = gap_over_L;
    out.type_percolation = type_percolation;
    out.rho = rho;
    out.t_eq_by_species.assign(static_cast<std::size_t>(num_colors),
                               std::numeric_limits<double>::quiet_NaN());
    out.z_stat_by_species.assign(static_cast<std::size_t>(num_colors), -1);
    out.stop_reason = "not_stopped";

    // Optional 2D snapshot environment flags (opt-in only; zero memory cost when unset).
    const char* snapshot_dir_env = std::getenv("SOP_FRACTION_SNAPSHOT_DIR");
    const bool capture_2d_snapshots =
        (dim == 2) && snapshot_dir_env != nullptr && snapshot_dir_env[0] != '\0';
    const char* base_snapshot_env = std::getenv("SOP_FRACTION_BASE_SNAPSHOT");
    const bool need_base_snapshot =
        capture_2d_snapshots && base_snapshot_env != nullptr && base_snapshot_env[0] != '\0';
    const char* gap_snapshot_env = std::getenv("SOP_FRACTION_GAP_SNAPSHOT");
    const bool need_gap_snapshot =
        capture_2d_snapshots && gap_snapshot_env != nullptr && gap_snapshot_env[0] != '\0';

    SlidingSlabGrid grid(dim, L, is_node);
    GridRegular base_grid(dim, L, 0, 1);

    std::vector<double> p_curr = p0;
    if (static_cast<int>(p_curr.size()) < num_colors) {
        p_curr.resize(static_cast<std::size_t>(num_colors),
                      p_curr.empty() ? 0.5 : p_curr.back());
    }
    std::vector<double> p_next(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<long long> N_current(static_cast<std::size_t>(num_colors), 0);
    std::vector<double> f_current(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<int> max_heights(static_cast<std::size_t>(num_colors), 0);

    std::vector<std::int64_t> frontier;
    std::vector<std::int64_t> next_frontier;
    frontier.reserve(static_cast<std::size_t>(base_size));
    next_frontier.reserve(static_cast<std::size_t>(base_size));

    SparseFrontCandidates front_candidates(0, true);
    front_candidates.reserve_frontier(static_cast<std::size_t>(base_size));

    std::vector<std::vector<double>> p_series(static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> f_series(static_cast<std::size_t>(num_colors));
    std::vector<int> t_list;
    t_list.reserve(4096);
    auto commit_equilibrium_step = [&](const int t_step) {
        t_list.push_back(t_step);
        for (int color = 0; color < num_colors; ++color) {
            p_series[static_cast<std::size_t>(color)].push_back(
                p_curr[static_cast<std::size_t>(color)]);
            f_series[static_cast<std::size_t>(color)].push_back(
                f_current[static_cast<std::size_t>(color)]);
        }
    };

    std::vector<int> seeds_quota(static_cast<std::size_t>(num_colors), 0);
    for (int color = 0; color < num_colors; ++color) {
        const double rho_c = (color < static_cast<int>(rho.size()))
            ? rho[static_cast<std::size_t>(color)]
            : (1.0 / static_cast<double>(num_colors));
        long long q = std::llround(P0 * rho_c * static_cast<double>(base_size));
        q = std::max<long long>(0, std::min<long long>(q, base_size));
        seeds_quota[static_cast<std::size_t>(color)] = static_cast<int>(q);
    }

    // Exact same base initialization as create_network / animate_network for all 1..N species
    // and all layouts (Clustered, Random, Blocks, Alternating) in both 2D and 3D.
    for (int color = 0; color < num_colors; ++color) {
        const int prefer_neg = color_to_negative_value(num_colors, color);
        const int active_val = color_to_active_value(num_colors, color);

        auto activate_base_site = [&](const std::int64_t base_idx) -> bool {
            const std::size_t lat = static_cast<std::size_t>(base_idx);
            const int v = grid.get_site(0, lat);
            if (v != prefer_neg && v != -1) return false;
            grid.set_site(0, lat, active_val);
            frontier.push_back(grid.encode_key(0, lat));
            ++N_current[static_cast<std::size_t>(color)];
            return true;
        };

        if (stop_config.initial_base_layout == InitialBaseLayout::Clustered) {
            activate_clustered_base_sites(
                base_grid,
                num_colors,
                color,
                seeds_quota[static_cast<std::size_t>(color)],
                rng,
                activate_base_site);
        } else if (stop_config.initial_base_layout == InitialBaseLayout::Random) {
            int activated = 0;
            int tries = 0;
            const int max_tries = static_cast<int>(
                std::min<long long>(static_cast<long long>(std::numeric_limits<int>::max()),
                                    base_size * 20LL));
            while (activated < seeds_quota[static_cast<std::size_t>(color)] &&
                   tries < max_tries) {
                const int x = rng.uniform_int(0, L - 1);
                const int y_lat = (dim == 3) ? rng.uniform_int(0, L - 1) : 0;
                if (activate_base_site(static_cast<std::int64_t>(grid.lat_index(x, y_lat)))) {
                    ++activated;
                }
                ++tries;
            }
        } else {
            activate_deterministic_base_sites(
                base_grid,
                num_colors,
                color,
                seeds_quota[static_cast<std::size_t>(color)],
                stop_config.initial_base_layout,
                activate_base_site);
        }
    }

    for (int color = 0; color < num_colors; ++color) {
        f_current[static_cast<std::size_t>(color)] =
            static_cast<double>(N_current[static_cast<std::size_t>(color)]) / norm_factor;
    }
    commit_equilibrium_step(0);

    std::vector<bool> died(static_cast<std::size_t>(num_colors), false);
    std::vector<bool> species_equilibrated(static_cast<std::size_t>(num_colors), false);
    bool all_equilibrated = false;
    int anchor = -1;
    std::deque<int> pending_anchors;

    auto compute_alive_height_bounds = [&](int& min_alive_max_h, int& max_alive_max_h) -> bool {
        bool any_alive = false;
        min_alive_max_h = std::numeric_limits<int>::max();
        max_alive_max_h = 0;
        for (int c = 0; c < num_colors; ++c) {
            if (died[static_cast<std::size_t>(c)]) continue;
            any_alive = true;
            min_alive_max_h = std::min(min_alive_max_h, max_heights[static_cast<std::size_t>(c)]);
            max_alive_max_h = std::max(max_alive_max_h, max_heights[static_cast<std::size_t>(c)]);
        }
        if (!any_alive) {
            min_alive_max_h = *std::max_element(max_heights.begin(), max_heights.end());
            max_alive_max_h = min_alive_max_h;
        }
        return any_alive;
    };

    auto write_optional_2d_snapshots = [&](const int bottom,
                                           const int top,
                                           const int sample_idx) {
        if (!capture_2d_snapshots) return;
        const int h_count = top - bottom + 1;
        const std::size_t slab_sites =
            static_cast<std::size_t>(h_count) * static_cast<std::size_t>(L);
        std::vector<std::uint32_t> component_labels(slab_sites, 0u);
        std::vector<std::uint32_t> queue;
        queue.reserve(static_cast<std::size_t>(L));
        std::uint32_t next_label = 0u;
        std::uint32_t largest_spanning_label = 0u;
        long long best_spanning_sites = 0;

        for (int y = bottom; y <= top; ++y) {
            for (int x = 0; x < L; ++x) {
                const std::size_t local =
                    static_cast<std::size_t>(y - bottom) * static_cast<std::size_t>(L) +
                    static_cast<std::size_t>(x);
                const int seed_val = grid.get_site(y, static_cast<std::size_t>(x));
                if (seed_val <= 0 || component_labels[local] != 0u) continue;
                const std::uint32_t label = ++next_label;
                bool touches_b = false, touches_t = false;
                long long comp_sz = 0;
                queue.clear();
                queue.push_back(static_cast<std::uint32_t>(local));
                component_labels[local] = label;
                for (std::size_t head = 0; head < queue.size(); ++head) {
                    const std::uint32_t cur = queue[head];
                    const int cy = bottom + static_cast<int>(cur / static_cast<std::uint32_t>(L));
                    const int cx = static_cast<int>(cur % static_cast<std::uint32_t>(L));
                    ++comp_sz;
                    touches_b |= (cy == bottom);
                    touches_t |= (cy == top);
                    const int nx_arr[4] = {(cx == 0 ? L - 1 : cx - 1), (cx + 1 == L ? 0 : cx + 1), cx, cx};
                    const int ny_arr[4] = {cy, cy, cy - 1, cy + 1};
                    for (int ni = 0; ni < 4; ++ni) {
                        const int nx = nx_arr[ni];
                        const int ny = ny_arr[ni];
                        if (ny < bottom || ny > top) continue;
                        if (grid.get_site(ny, static_cast<std::size_t>(nx)) != seed_val) continue;
                        if (!is_node) {
                            const bool ok = (ni == 0) ? grid.is_open_bond(cy, static_cast<std::size_t>(nx), 0)
                                          : (ni == 1) ? grid.is_open_bond(cy, static_cast<std::size_t>(cx), 0)
                                          : (ni == 2) ? grid.is_open_bond(ny, static_cast<std::size_t>(cx), 2)
                                                      : grid.is_open_bond(cy, static_cast<std::size_t>(cx), 2);
                            if (!ok) continue;
                        }
                        const std::size_t nloc =
                            static_cast<std::size_t>(ny - bottom) * static_cast<std::size_t>(L) +
                            static_cast<std::size_t>(nx);
                        if (component_labels[nloc] != 0u) continue;
                        component_labels[nloc] = label;
                        queue.push_back(static_cast<std::uint32_t>(nloc));
                    }
                }
                if (touches_b && touches_t && comp_sz > best_spanning_sites) {
                    best_spanning_sites = comp_sz;
                    largest_spanning_label = label;
                }
            }
        }

        const std::filesystem::path snapshot_dir(snapshot_dir_env);
        std::filesystem::create_directories(snapshot_dir);
        const auto snapshot_path =
            snapshot_dir / ("sample_" + std::to_string(sample_idx + 1) + ".components");
        std::ofstream snapshot(snapshot_path, std::ios::binary | std::ios::trunc);
        if (!snapshot) {
            throw std::runtime_error("Could not open fraction snapshot: " + snapshot_path.string());
        }
        snapshot.write("SOPCOMP1", 8);
        const std::uint32_t header[] = {
            static_cast<std::uint32_t>(L),
            static_cast<std::uint32_t>(h_count),
            largest_spanning_label
        };
        snapshot.write(reinterpret_cast<const char*>(header), sizeof(header));
        std::vector<std::uint32_t> row(static_cast<std::size_t>(L), 0u);
        for (int y = top; y >= bottom; --y) {
            const std::size_t row_off =
                static_cast<std::size_t>(y - bottom) * static_cast<std::size_t>(L);
            for (int x = 0; x < L; ++x) {
                row[static_cast<std::size_t>(x)] =
                    grid.get_site(y, static_cast<std::size_t>(x)) <= 0
                        ? 0u
                        : component_labels[row_off + static_cast<std::size_t>(x)];
            }
            snapshot.write(reinterpret_cast<const char*>(row.data()),
                           static_cast<std::streamsize>(row.size() * sizeof(row[0])));
        }

        if (sample_idx == 0 && need_base_snapshot) {
            const std::filesystem::path base_path(base_snapshot_env);
            if (!base_path.parent_path().empty()) {
                std::filesystem::create_directories(base_path.parent_path());
            }
            std::ofstream base_snap(base_path, std::ios::binary | std::ios::trunc);
            if (base_snap) {
                base_snap.write("SOPBASE1", 8);
                const std::uint32_t bheader[] = {
                    static_cast<std::uint32_t>(L),
                    static_cast<std::uint32_t>(bottom)
                };
                base_snap.write(reinterpret_cast<const char*>(bheader), sizeof(bheader));
                std::vector<unsigned char> brow(static_cast<std::size_t>(L), 0u);
                for (int y = bottom - 1; y >= 0; --y) {
                    for (int x = 0; x < L; ++x) {
                        brow[static_cast<std::size_t>(x)] =
                            grid.get_site(y, static_cast<std::size_t>(x)) > 0 ? 1u : 0u;
                    }
                    base_snap.write(reinterpret_cast<const char*>(brow.data()),
                                    static_cast<std::streamsize>(brow.size()));
                }
            }
        }

        if (sample_idx == 1 && need_gap_snapshot) {
            const std::filesystem::path gap_path(gap_snapshot_env);
            if (!gap_path.parent_path().empty()) {
                std::filesystem::create_directories(gap_path.parent_path());
            }
            std::ofstream gap_snap(gap_path, std::ios::binary | std::ios::trunc);
            if (gap_snap) {
                gap_snap.write("SOPGAP1", 8);
                const int actual_gap_h = sample_gap_layers > 0 ? sample_gap_layers : L;
                const std::uint32_t gheader[] = {
                    static_cast<std::uint32_t>(L),
                    static_cast<std::uint32_t>(actual_gap_h)
                };
                gap_snap.write(reinterpret_cast<const char*>(gheader), sizeof(gheader));
                std::vector<unsigned char> grow(static_cast<std::size_t>(L), 0u);
                for (int y = bottom - 1; y >= bottom - actual_gap_h; --y) {
                    for (int x = 0; x < L; ++x) {
                        grow[static_cast<std::size_t>(x)] =
                            grid.get_site(y, static_cast<std::size_t>(x)) > 0 ? 1u : 0u;
                    }
                    gap_snap.write(reinterpret_cast<const char*>(grow.data()),
                                   static_cast<std::streamsize>(grow.size()));
                }
            }
        }

        if (sample_idx == requested - 1) {
            const char* tail_snapshot_env = std::getenv("SOP_FRACTION_TAIL_SNAPSHOT");
            if (tail_snapshot_env != nullptr && tail_snapshot_env[0] != '\0') {
                const std::filesystem::path tail_path(tail_snapshot_env);
                if (!tail_path.parent_path().empty()) {
                    std::filesystem::create_directories(tail_path.parent_path());
                }
                std::ofstream tail_snap(tail_path, std::ios::binary | std::ios::trunc);
                if (tail_snap) {
                    tail_snap.write("SOPTAIL1", 8);
                    double tail_over_L = 0.75;
                    if (const char* tail_env = std::getenv("SOP_FRACTION_TAIL_OVER_L")) {
                        tail_over_L = std::max(0.0, std::stod(tail_env));
                    }
                    const int cur_z_max = *std::max_element(max_heights.begin(), max_heights.end());
                    const int desired_tail_h = static_cast<int>(
                        std::llround(tail_over_L * static_cast<double>(L)));
                    const int tail_bottom = top + 1;
                    const int avail_h = std::max(0, cur_z_max - tail_bottom + 1);
                    const int actual_tail_h =
                        desired_tail_h > 0 ? std::min(desired_tail_h, avail_h) : avail_h;
                    const std::uint32_t theader[] = {
                        static_cast<std::uint32_t>(L),
                        static_cast<std::uint32_t>(actual_tail_h)
                    };
                    tail_snap.write(reinterpret_cast<const char*>(theader), sizeof(theader));
                    std::vector<unsigned char> trow(static_cast<std::size_t>(L), 0u);
                    for (int y = tail_bottom + actual_tail_h - 1; y >= tail_bottom; --y) {
                        for (int x = 0; x < L; ++x) {
                            trow[static_cast<std::size_t>(x)] =
                                grid.get_site(y, static_cast<std::size_t>(x)) > 0 ? 1u : 0u;
                        }
                        tail_snap.write(reinterpret_cast<const char*>(trow.data()),
                                        static_cast<std::streamsize>(trow.size()));
                    }
                }
            }
        }
    };

    for (int t = 1; hard_max_steps <= 0 || t <= hard_max_steps; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        std::fill(f_current.begin(), f_current.end(), 0.0);
        next_frontier.clear();
        front_candidates.clear();

        if (is_node) {
            for (const std::int64_t u_key : frontier) {
                int h = 0;
                std::size_t lat = 0;
                grid.decode_key(u_key, h, lat);
                const int a_val = grid.get_site(h, lat);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors || died[static_cast<std::size_t>(cor_idx)]) {
                    continue;
                }

                int x = 0, y_lat = 0;
                grid.decode_lat(lat, x, y_lat);

                auto try_add_node_target = [&](const int nh, const std::size_t nlat) {
                    if (nh < grid.min_retained_h) return;
                    const int vv = grid.get_site(nh, nlat);
                    if (vv >= 0) return;
                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color = (vv == -1);
                    if (!same_color && !no_color) return;
                    front_candidates.add(grid.encode_key(nh, nlat), FrontCandidate{u_key, cor_idx});
                };

                try_add_node_target(h, grid.lat_index((x == 0) ? (L - 1) : (x - 1), y_lat));
                try_add_node_target(h, grid.lat_index((x + 1 == L) ? 0 : (x + 1), y_lat));
                if (dim == 3) {
                    try_add_node_target(h, grid.lat_index(x, (y_lat == 0) ? (L - 1) : (y_lat - 1)));
                    try_add_node_target(h, grid.lat_index(x, (y_lat + 1 == L) ? 0 : (y_lat + 1)));
                }
                if (h > 0) try_add_node_target(h - 1, lat);
                try_add_node_target(h + 1, lat);
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const std::int64_t viz_key = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int pick = rng.uniform_int(0, n_cand - 1);
                const FrontCandidate chosen = front_candidates.candidate(slot, pick);
                const int cor_idx = static_cast<int>(chosen.color_idx);

                if (rng.uniform_real(0.0, 1.0) >= p_curr[static_cast<std::size_t>(cor_idx)]) {
                    grid.set_site_key(viz_key, 0);
                    continue;
                }

                grid.set_site_key(viz_key, color_to_active_value(num_colors, cor_idx));
                next_frontier.push_back(viz_key);
                ++N_current[static_cast<std::size_t>(cor_idx)];
                const int h_viz = grid.grow_coord(viz_key);
                if (h_viz > max_heights[static_cast<std::size_t>(cor_idx)]) {
                    max_heights[static_cast<std::size_t>(cor_idx)] = h_viz;
                }
            }
        } else {
            for (const std::int64_t u_key : frontier) {
                int h = 0;
                std::size_t lat = 0;
                grid.decode_key(u_key, h, lat);
                const int a_val = grid.get_site(h, lat);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors || died[static_cast<std::size_t>(cor_idx)]) {
                    continue;
                }

                int x = 0, y_lat = 0;
                grid.decode_lat(lat, x, y_lat);

                auto try_bond_neighbor = [&](const int nh,
                                             const std::size_t nlat,
                                             const std::uint64_t bond_ref) {
                    if (nh < grid.min_retained_h) return;
                    const int vv = grid.get_site(nh, nlat);
                    if (vv == 0) return;

                    if (vv > 0) {
                        const int neigh_color = value_to_color_index(num_colors, vv);
                        if (neigh_color != cor_idx) return;
                        if (!grid.mark_tested_bond_packed(bond_ref)) return;
                        if (rng.uniform_real(0.0, 1.0) < p_curr[static_cast<std::size_t>(cor_idx)]) {
                            grid.set_open_bond_packed(bond_ref);
                        }
                        return;
                    }

                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color = (vv == -1);
                    if (!same_color && !no_color) return;

                    if (!grid.mark_tested_bond_packed(bond_ref)) return;
                    if (rng.uniform_real(0.0, 1.0) < p_curr[static_cast<std::size_t>(cor_idx)]) {
                        front_candidates.add(grid.encode_key(nh, nlat),
                                             FrontCandidate{u_key, cor_idx, bond_ref});
                    }
                };

                {
                    const std::size_t nlat = grid.lat_index((x == 0) ? (L - 1) : (x - 1), y_lat);
                    try_bond_neighbor(h, nlat, SlidingSlabGrid::pack_bond_ref(h, nlat, 0));
                }
                {
                    const std::size_t nlat = grid.lat_index((x + 1 == L) ? 0 : (x + 1), y_lat);
                    try_bond_neighbor(h, nlat, SlidingSlabGrid::pack_bond_ref(h, lat, 0));
                }
                if (dim == 3) {
                    {
                        const std::size_t nlat = grid.lat_index(x, (y_lat == 0) ? (L - 1) : (y_lat - 1));
                        try_bond_neighbor(h, nlat, SlidingSlabGrid::pack_bond_ref(h, nlat, 1));
                    }
                    {
                        const std::size_t nlat = grid.lat_index(x, (y_lat + 1 == L) ? 0 : (y_lat + 1));
                        try_bond_neighbor(h, nlat, SlidingSlabGrid::pack_bond_ref(h, lat, 1));
                    }
                }
                if (h > 0) {
                    try_bond_neighbor(h - 1, lat, SlidingSlabGrid::pack_bond_ref(h - 1, lat, 2));
                }
                try_bond_neighbor(h + 1, lat, SlidingSlabGrid::pack_bond_ref(h, lat, 2));
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const std::int64_t viz_key = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int target_before = grid.get_site_key(viz_key);
                const bool target_without_species = (num_colors > 1 && target_before == -1);

                int winning_color = -1;
                bool valid = false;
                if (target_without_species) {
                    const int pick = rng.uniform_int(0, n_cand - 1);
                    winning_color = static_cast<int>(front_candidates.candidate(slot, pick).color_idx);
                    valid = (winning_color >= 0 && winning_color < num_colors);
                } else {
                    winning_color = (num_colors == 1)
                        ? 0
                        : value_to_color_index(num_colors, target_before);
                    if (winning_color >= 0 && winning_color < num_colors) {
                        for (int i = 0; i < n_cand; ++i) {
                            if (static_cast<int>(front_candidates.candidate(slot, i).color_idx) == winning_color) {
                                valid = true;
                                break;
                            }
                        }
                    }
                }
                if (!valid) continue;

                grid.set_site_key(viz_key, color_to_active_value(num_colors, winning_color));
                for (int i = 0; i < n_cand; ++i) {
                    const FrontCandidate fc = front_candidates.candidate(slot, i);
                    if (static_cast<int>(fc.color_idx) != winning_color) continue;
                    grid.set_open_bond_packed(fc.edge_bit);
                }

                next_frontier.push_back(viz_key);
                ++N_current[static_cast<std::size_t>(winning_color)];
                const int h_viz = grid.grow_coord(viz_key);
                if (h_viz > max_heights[static_cast<std::size_t>(winning_color)]) {
                    max_heights[static_cast<std::size_t>(winning_color)] = h_viz;
                }
            }
        }

        for (int color = 0; color < num_colors; ++color) {
            f_current[static_cast<std::size_t>(color)] =
                static_cast<double>(N_current[static_cast<std::size_t>(color)]) / norm_factor;
            if (!died[static_cast<std::size_t>(color)] &&
                N_current[static_cast<std::size_t>(color)] == 0) {
                died[static_cast<std::size_t>(color)] = true;
            }
        }

        int min_frontier_h = std::numeric_limits<int>::max();
        for (const std::int64_t k : next_frontier) {
            min_frontier_h = std::min(min_frontier_h, grid.grow_coord(k));
        }

        if (!all_equilibrated) {
            commit_equilibrium_step(t);
            constexpr int teq_window_block = 10;
            const double teq_rel_tol = growth_test_effective_rel_tol(
                stop_config.equilibrium_rel_tol, num_colors);
            const int teq_min_stable_steps =
                growth_test_global_min_stable_steps(
                    stop_config.equilibrium_consecutive_steps,
                    teq_window_block);
            const int teq_validation_window_steps =
                growth_test_global_validation_window_steps(
                    stop_config.post_equilibrium_extra_steps,
                    teq_window_block);

            if (t % teq_window_block == 0) {
                for (int color = 0; color < num_colors; ++color) {
                    if (died[static_cast<std::size_t>(color)] ||
                        species_equilibrated[static_cast<std::size_t>(color)]) {
                        continue;
                    }
                    const bool use_front_fraction_teq = (std::abs(c_value) <= 1e-12);
                    if (!use_front_fraction_teq &&
                        max_heights[static_cast<std::size_t>(color)] < L) {
                        continue;
                    }
                    const auto& series_to_check = use_front_fraction_teq
                        ? f_series[static_cast<std::size_t>(color)]
                        : p_series[static_cast<std::size_t>(color)];
                    const double t_eq = estimate_t_eq_from_series(
                        t_list,
                        series_to_check,
                        15,
                        teq_window_block,
                        teq_min_stable_steps,
                        teq_rel_tol,
                        stop_config.equilibrium_abs_tol,
                        1.0e-5,
                        false,
                        0.0,
                        teq_validation_window_steps,
                        0u);
                    if (std::isfinite(t_eq)) {
                        species_equilibrated[static_cast<std::size_t>(color)] = true;
                        out.t_eq_by_species[static_cast<std::size_t>(color)] = t_eq;
                        out.z_stat_by_species[static_cast<std::size_t>(color)] =
                            max_heights[static_cast<std::size_t>(color)];
                    }
                }
            }

            bool any_alive = false;
            all_equilibrated = true;
            for (int color = 0; color < num_colors; ++color) {
                if (died[static_cast<std::size_t>(color)]) continue;
                any_alive = true;
                if (!species_equilibrated[static_cast<std::size_t>(color)]) {
                    all_equilibrated = false;
                }
            }
            if (!any_alive) {
                out.stop_reason = "all_dead";
                out.stop_time = t;
                break;
            }

            // Keep memory bounded during long equilibration transients.
            if (!need_base_snapshot && !next_frontier.empty()) {
                int min_alive_h = 0, max_alive_h = 0;
                compute_alive_height_bounds(min_alive_h, max_alive_h);
                int safe_keep = std::min(min_frontier_h - 2,
                                         min_alive_h - std::max(L, sample_stability_layers));
                for (int c = 0; c < num_colors; ++c) {
                    if (species_equilibrated[static_cast<std::size_t>(c)] &&
                        out.z_stat_by_species[static_cast<std::size_t>(c)] >= 0) {
                        safe_keep = std::min(safe_keep,
                                             out.z_stat_by_species[static_cast<std::size_t>(c)] - 2);
                    }
                }
                if (safe_keep > grid.min_retained_h) {
                    grid.discard_below(safe_keep, false);
                }
            }

            if (all_equilibrated) {
                anchor = *std::max_element(out.z_stat_by_species.begin(),
                                           out.z_stat_by_species.end());
                out.z_stab = anchor;
                std::cout << "[fractions] stabilized"
                          << " t=" << t
                          << " z_stab=" << out.z_stab
                          << " next_stab_z=" << (anchor + L + sample_stability_layers)
                          << " stability_layers=" << sample_stability_layers
                          << " gap_layers=" << sample_gap_layers
                          << " requested=" << requested
                          << std::endl;
                p_series.clear();
                f_series.clear();
                t_list.clear();
                if (!need_base_snapshot && !next_frontier.empty()) {
                    const int safe_keep = std::min(anchor, min_frontier_h - 2);
                    if (safe_keep > grid.min_retained_h) {
                        grid.discard_below(safe_keep, true);
                    }
                }
            }
        } else {
            int z_ready = 0, z_max = 0;
            const bool any_alive = compute_alive_height_bounds(z_ready, z_max);
            while (out.collected_samples + static_cast<int>(pending_anchors.size()) < requested &&
                   z_ready >= anchor + L) {
                pending_anchors.push_back(anchor);
                anchor += L + sample_gap_layers;
            }

            while (!pending_anchors.empty() &&
                   z_ready >= pending_anchors.front() + L + sample_stability_layers) {
                const int sample_anchor = pending_anchors.front();
                pending_anchors.pop_front();
                const int sample_top = sample_anchor + L - 1;
                const int sample_idx = out.collected_samples;

                // Free layers strictly below sample_anchor before allocating temporary bitsets,
                // unless opt-in 2D base/gap snapshots still need them.
                if (!need_base_snapshot && !(need_gap_snapshot && sample_idx <= 1)) {
                    const int pre_keep = next_frontier.empty()
                        ? sample_anchor
                        : std::min(sample_anchor, min_frontier_h - 2);
                    grid.discard_below(pre_keep, true);
                }

                const SlabFractionResult stab = compute_slab_stabilized(
                    grid, num_colors, sample_anchor, sample_top, sample_idx, t);

                if (capture_2d_snapshots) {
                    write_optional_2d_snapshots(sample_anchor, sample_top, sample_idx);
                }

                out.anchor_z.push_back(sample_anchor);
                out.slab_top_z.push_back(sample_top);
                out.t_stab.push_back(t);
                out.p_stab_bond.push_back(stab.p_bond);
                out.p_stab_node.push_back(stab.p_node);
                out.p_stab_bond_by_species.push_back(stab.p_bond_by_species);
                out.p_stab_node_by_species.push_back(stab.p_node_by_species);
                out.num_spanning_clusters.push_back(stab.num_spanning_clusters);
                out.S_stab.push_back(stab.largest_spanning_sites);
                out.E_stab.push_back(stab.largest_spanning_bonds);
                out.SP_stab.push_back(stab.largest_spanning_sp);
                out.hull_length.push_back(stab.largest_spanning_hull);
                out.full_boundary_length.push_back(stab.largest_spanning_full_boundary);
                out.largest_spanning_species.push_back(stab.largest_spanning_species);
                out.largest_spanning_interspecies_bonds.push_back(
                    stab.largest_spanning_interspecies_bonds);
                out.other_spanning_sites.push_back(stab.other_spanning_sites);
                out.other_spanning_bonds.push_back(stab.other_spanning_bonds);
                out.other_spanning_sp.push_back(stab.other_spanning_sp);
                out.other_spanning_hull.push_back(stab.other_spanning_hull);
                out.other_spanning_species.push_back(stab.other_spanning_species);
                out.other_spanning_interspecies_bonds.push_back(
                    stab.other_spanning_interspecies_bonds);
                out.interspecies_contact_bonds.push_back(stab.interspecies_contact_bonds);
                out.interspecies_contact_sites.push_back(stab.interspecies_contact_sites);
                out.spanning_interspecies_bonds.push_back(stab.spanning_interspecies_bonds);
                out.spanning_interspecies_sites.push_back(stab.spanning_interspecies_sites);
                out.fractal_counts.push_back(stab.fractal_counts);
                out.collected_samples = static_cast<int>(out.S_stab.size());

                std::cout << "[fractions] sample_collected"
                          << " sample=" << out.collected_samples << "/" << requested
                          << " t=" << t
                          << " window=[" << sample_anchor << "," << sample_top << "]"
                          << " spanning_clusters=" << stab.num_spanning_clusters
                          << " S=" << stab.largest_spanning_sites
                          << " hull=" << stab.largest_spanning_hull
                          << " SP=" << stab.largest_spanning_sp;
                if (num_colors > 1) {
                    std::cout << " inter_span=" << stab.spanning_interspecies_bonds
                              << " inter_total=" << stab.interspecies_contact_bonds;
                }
                std::cout << std::endl;

                // Immediately discard the completed slab below the next oldest anchor.
                const int next_oldest = pending_anchors.empty() ? anchor : pending_anchors.front();
                if (!(need_gap_snapshot && out.collected_samples == 1)) {
                    const int post_keep = next_frontier.empty()
                        ? next_oldest
                        : std::min(next_oldest, min_frontier_h - 2);
                    grid.discard_below(post_keep, true);
                }

                if (out.collected_samples >= requested) {
                    out.stop_reason = "requested_samples_collected";
                    out.stop_time = t;
                    break;
                }
            }

            if (out.collected_samples >= requested) {
                break;
            }
            if (!any_alive) {
                out.stop_reason = "all_dead";
                out.stop_time = t;
                break;
            }
        }

        if (progress_interval > 0 && (t % progress_interval) == 0) {
            int min_alive_h = 0, max_alive_h = 0;
            compute_alive_height_bounds(min_alive_h, max_alive_h);
            const int next_target = !all_equilibrated
                ? L
                : ((pending_anchors.empty() ? anchor : pending_anchors.front()) +
                   L + sample_stability_layers);
            std::cout << "[fractions] phase=" << (!all_equilibrated ? "equilibrating" : "waiting_stab")
                      << " t=" << t
                      << " z_ready=" << min_alive_h
                      << " z_max=" << max_alive_h
                      << " next_stab_z=" << next_target
                      << " collected=" << out.collected_samples << "/" << requested
                      << " retained_layers=" << grid.layers.size()
                      << " anchor=" << (pending_anchors.empty() ? anchor : pending_anchors.front());
            for (int c = 0; c < num_colors; ++c) {
                std::cout << " p" << (c + 1) << "=" << p_curr[static_cast<std::size_t>(c)];
            }
            std::cout << std::endl;
        }

        if (std::all_of(died.begin(), died.end(), [](const bool v) { return v; })) {
            out.stop_reason = "all_dead";
            out.stop_time = t;
            break;
        }

        for (int color = 0; color < num_colors; ++color) {
            p_next[static_cast<std::size_t>(color)] =
                died[static_cast<std::size_t>(color)]
                    ? p_curr[static_cast<std::size_t>(color)]
                    : generate_p(type_f_T,
                                 p_curr[static_cast<std::size_t>(color)],
                                 t,
                                 f_current[static_cast<std::size_t>(color)],
                                 c_value,
                                 f_T,
                                 a,
                                 alpha,
                                 stop_config.feedback_control_rule);
        }

        frontier.swap(next_frontier);
        p_curr.swap(p_next);
    }

    if (out.stop_time < 0 && hard_max_steps > 0) {
        out.stop_time = hard_max_steps;
        out.stop_reason = "hard_max_steps";
    }

    return out;
}

NetworkPattern network::create_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double c_value, const double f_T, const int type_f_T,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng,
    const bool save_compact, const bool calculate_detailed_properties,
    const bool calculate_height_timeseries,
    const GrowthStopConfig stop_config)
{
    this->c = c_value;
    this->f_T = f_T;

    GridRegular grid(
        dim,
        lenght_network,
        stop_config.height_multiplier,
        stop_config.height_extra_layers);
    const bool is_node = (type_percolation == "node");
    const long long base_size = compute_base_size(grid);
    const double norm_factor = static_cast<double>(base_size);
    const int percolation_height = lenght_network - 1;
    const int hard_max_steps = stop_config.hard_max_steps > 0
        ? std::min(num_of_samples - 1, stop_config.hard_max_steps)
        : (num_of_samples - 1);
    const int dynamic_min_stop_height = -1;
    const int dynamic_max_stop_height = -1;

    if (num_colors > 125) {
        throw std::runtime_error(
            "create_network: num_colors exceeds int8_t site-state encoding capacity");
    }

    const int net_seed = rng.get_seed();

    std::vector<std::int8_t> site_state(
        static_cast<std::size_t>(grid.total_size),
        static_cast<std::int8_t>(-1));
    auto get_site = [&](const std::int64_t idx) -> int {
        return static_cast<int>(site_state[static_cast<std::size_t>(idx)]);
    };
    auto set_site = [&](const std::int64_t idx, const int value) {
        site_state[static_cast<std::size_t>(idx)] = static_cast<std::int8_t>(value);
    };

    const bool save_lateral_observables = stop_config.save_lateral_observables;

    // Activation times are only required by consumers that retain/revisit the
    // complete network.  The time-series-only path never reads them: keeping
    // an uint32_t for every visited site was its largest avoidable allocation.
    const bool needs_activation_time =
        save_compact || calculate_detailed_properties || save_lateral_observables;

    // activation times: UINT32_MAX means never activated
    std::vector<uint32_t> activation_time;
    if (needs_activation_time) {
        activation_time.assign(
            static_cast<std::size_t>(grid.total_size),
            std::numeric_limits<uint32_t>::max());
    }
    std::vector<std::int64_t> activated_indices_for_lateral;
    if (save_lateral_observables) {
        activated_indices_for_lateral.reserve(static_cast<std::size_t>(base_size));
    }
    auto set_activation_time = [&](const std::int64_t idx, const uint32_t t_value)
    {
        if (!needs_activation_time) return;
        activation_time[static_cast<std::size_t>(idx)] = t_value;
        if (save_lateral_observables) {
            activated_indices_for_lateral.push_back(idx);
        }
    };

    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<double>> f_series(num_colors);
    std::vector<std::vector<double>> y_mean_series(num_colors);
    std::vector<std::vector<double>> y_width_series(num_colors);
    std::vector<std::vector<int>> y_max_series(num_colors);
    std::vector<std::vector<double>> y_front_mean_series(num_colors);
    std::vector<std::vector<double>> y_front_width_series(num_colors);
    std::vector<std::vector<uint32_t>> y_front_count_series(num_colors);
    std::vector<int>                 t_list;
    t_list.reserve(num_of_samples);

    std::vector<double> p_curr = p0;
    std::vector<double> p_next(num_colors, 0.0);
    std::vector<int>    N_current(num_colors, 0);
    std::vector<double> f_current(num_colors, 0.0);
    std::vector<int>    max_heights(num_colors, 0);
    std::vector<int>    z_max_at_perc(num_colors, -1);
    std::vector<std::int64_t> frontier;
    std::vector<std::int64_t> next_frontier;

    const std::size_t lateral_size = static_cast<std::size_t>(base_size);
    std::vector<std::vector<int>> surface_height;
    if (calculate_height_timeseries) {
        surface_height.assign(
            static_cast<std::size_t>(num_colors),
            std::vector<int>(lateral_size, -1));
    }
    std::vector<double> surface_sum_height(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<double> surface_sum_height_sq(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<uint32_t> surface_active_columns(static_cast<std::size_t>(num_colors), 0);
    std::vector<int> surface_max_height(static_cast<std::size_t>(num_colors), -1);
    std::vector<double> front_sum_height(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<double> front_sum_height_sq(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<uint32_t> front_count(static_cast<std::size_t>(num_colors), 0);

    auto lateral_index_of = [&](const std::int64_t idx) -> std::size_t {
        if (dim == 2) {
            return static_cast<std::size_t>(grid.x_of(idx));
        }
        return static_cast<std::size_t>(
            grid.x_of(idx) + static_cast<std::int64_t>(grid.SX) * grid.y_of(idx));
    };

    auto reset_front_height_accumulators = [&]() {
        if (!calculate_height_timeseries) return;
        std::fill(front_sum_height.begin(), front_sum_height.end(), 0.0);
        std::fill(front_sum_height_sq.begin(), front_sum_height_sq.end(), 0.0);
        std::fill(front_count.begin(), front_count.end(), 0u);
    };

    auto record_height_activation = [&](const int color_idx, const std::int64_t idx) {
        if (!calculate_height_timeseries) return;
        if (color_idx < 0 || color_idx >= num_colors) return;
        const std::size_t c = static_cast<std::size_t>(color_idx);
        const int h = grid.grow_coord(idx);
        front_sum_height[c] += static_cast<double>(h);
        front_sum_height_sq[c] += static_cast<double>(h) * static_cast<double>(h);
        ++front_count[c];

        const std::size_t lateral_idx = lateral_index_of(idx);
        int& old_h = surface_height[c][lateral_idx];
        if (h <= old_h) return;

        if (old_h < 0) {
            ++surface_active_columns[c];
            surface_sum_height[c] += static_cast<double>(h);
            surface_sum_height_sq[c] += static_cast<double>(h) * static_cast<double>(h);
        } else {
            surface_sum_height[c] += static_cast<double>(h - old_h);
            surface_sum_height_sq[c] +=
                static_cast<double>(h) * static_cast<double>(h) -
                static_cast<double>(old_h) * static_cast<double>(old_h);
        }
        old_h = h;
        if (h > surface_max_height[c]) {
            surface_max_height[c] = h;
        }
    };
    
    frontier.reserve(static_cast<std::size_t>(base_size));
    next_frontier.reserve(static_cast<std::size_t>(base_size));

    SparseFrontCandidates front_candidates(
        stop_config.dynamic_height ? 0 : grid.total_size,
        stop_config.dynamic_height);
    front_candidates.reserve_frontier(static_cast<std::size_t>(base_size));

    auto commit_step = [&](const int t_k,
                           const std::vector<double>& p_vec,
                           const std::vector<double>& f_vec)
    {
        t_list.push_back(t_k);
        for (int c = 0; c < num_colors; ++c) {
            p_series[c].push_back(p_vec[c]);
            f_series[c].push_back(f_vec[c]);
            if (!calculate_height_timeseries) continue;

            const std::size_t row = static_cast<std::size_t>(c);
            const uint32_t ncols = surface_active_columns[row];
            if (ncols > 0) {
                const double mean =
                    surface_sum_height[row] / static_cast<double>(ncols);
                const double mean_sq =
                    surface_sum_height_sq[row] / static_cast<double>(ncols);
                y_mean_series[row].push_back(mean);
                y_width_series[row].push_back(
                    std::sqrt(std::max(0.0, mean_sq - mean * mean)));
            } else {
                y_mean_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
                y_width_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
            }
            y_max_series[row].push_back(surface_max_height[row]);

            const uint32_t nf = front_count[row];
            if (nf > 0) {
                const double mean =
                    front_sum_height[row] / static_cast<double>(nf);
                const double mean_sq =
                    front_sum_height_sq[row] / static_cast<double>(nf);
                y_front_mean_series[row].push_back(mean);
                y_front_width_series[row].push_back(
                    std::sqrt(std::max(0.0, mean_sq - mean * mean)));
            } else {
                y_front_mean_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
                y_front_width_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
            }
            y_front_count_series[row].push_back(nf);
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
        const int prefer_neg = color_to_negative_value(num_colors, c);
        const int active_val = color_to_active_value(num_colors, c);

        auto activate_base_site = [&](const std::int64_t idx) -> bool {
            const int v = get_site(idx);
            if (v != prefer_neg && v != -1) return false;

            set_site(idx, active_val);
            frontier.push_back(idx);
            ++N_current[c];
            set_activation_time(idx, 0u);
            record_height_activation(c, idx);
            return true;
        };

        if (stop_config.initial_base_layout == InitialBaseLayout::Clustered) {
            activate_clustered_base_sites(
                grid,
                num_colors,
                c,
                seeds_quota[static_cast<std::size_t>(c)],
                rng,
                activate_base_site);
        } else if (stop_config.initial_base_layout == InitialBaseLayout::Random) {
            int activated = 0;
            int tries = 0;
            const int max_tries = static_cast<int>(std::min<long long>(static_cast<long long>(std::numeric_limits<int>::max()), base_size * 20));

            while (activated < seeds_quota[c] && tries < max_tries) {
                const int x = rng.uniform_int(0, grid.SX - 1);
                const int y = (dim == 3 ? rng.uniform_int(0, grid.SY - 1) : 0);
                const int z = 0;
                const std::int64_t idx = grid.lin_index(x, y, z);

                if (activate_base_site(idx)) {
                    ++activated;
                }
                ++tries;
            }
        } else {
            activate_deterministic_base_sites(
                grid,
                num_colors,
                c,
                seeds_quota[static_cast<std::size_t>(c)],
                stop_config.initial_base_layout,
                activate_base_site);
        }
    }

    // initialize PercolationSeries temporal decomposition containers
    ps_out.initial_base_layout =
        initial_base_layout_name(stop_config.initial_base_layout);
    ps_out.sp_lin_preteq.assign(num_colors, -1);
    ps_out.sp_path_lin_preteq.assign(num_colors, std::vector<std::int64_t>{});
    ps_out.sp_lin_posteq.assign(num_colors, -1);
    ps_out.sp_path_lin_posteq.assign(num_colors, std::vector<std::int64_t>{});
    ps_out.M_size_preteq.assign(num_colors, -1);
    ps_out.M_size_posteq.assign(num_colors, -1);

    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, std::vector<std::int64_t>{});
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

    std::string growth_test_stop_reason = "not_stopped";
    int growth_test_stop_time = -1;

    auto finalize_time_series_only = [&]()
    {
        ps_out.color_percolation.clear();
        ps_out.percolation_order.clear();
        ps_out.M_size_at_perc.clear();
        ps_out.sp_len.assign(num_colors, -1);
        ps_out.sp_path_lin.assign(num_colors, std::vector<std::int64_t>{});

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
        std::vector<char> included(static_cast<std::size_t>(num_colors), 0);
        for (const auto& vp : valid_percs) {
            ps_out.color_percolation.push_back(vp.color_idx + 1);
            ps_out.percolation_order.push_back(++valid_order_ctr);
            included[static_cast<std::size_t>(vp.color_idx)] = 1;
        }
        if (growth_test_stop_reason == "hard_max_steps") {
            for (int c = 0; c < num_colors; ++c) {
                if (included[static_cast<std::size_t>(c)]) continue;
                ps_out.color_percolation.push_back(c + 1);
                ps_out.percolation_order.push_back(++valid_order_ctr);
            }
        }
    };

    auto largest_component_single_color = [&](const int color_idx) -> std::int64_t
    {
        const int active_val = color_to_active_value(num_colors, color_idx);

        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<std::int64_t> stack;
        std::int64_t max_component = 0;

        for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
            if (visited[static_cast<std::size_t>(idx)]) continue;
            if (get_site(idx) != active_val) continue;

            std::int64_t comp_size = 0;
            stack.clear();
            stack.push_back(idx);
            visited[static_cast<std::size_t>(idx)] = 1;

            while (!stack.empty()) {
                const std::int64_t u = stack.back();
                stack.pop_back();
                ++comp_size;

                grid.for_each_neighbor(u, [&](const std::int64_t v) {
                    if (v < 0) return;
                    if (visited[static_cast<std::size_t>(v)]) return;
                    if (get_site(v) != active_val) return;

                    visited[static_cast<std::size_t>(v)] = 1;
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
    std::vector<std::int64_t> dsu;
    if (calculate_detailed_properties) {
        dsu.assign(static_cast<std::size_t>(grid.total_size), 0);
    }

    auto dsu_make_active = [&](const std::int64_t idx)
    {
        if (!calculate_detailed_properties) return;
        if (idx < 0 || idx >= grid.total_size) return;
        if (dsu[static_cast<std::size_t>(idx)] != 0) return;
        dsu[static_cast<std::size_t>(idx)] = -1;
    };

    auto dsu_find = [&](std::int64_t x)
    {
        std::int64_t root = x;
        while (dsu[static_cast<std::size_t>(root)] > 0) {
            root = dsu[static_cast<std::size_t>(root)] - 1;
        }
        while (x != root) {
            const std::int64_t next = dsu[static_cast<std::size_t>(x)] - 1;
            dsu[static_cast<std::size_t>(x)] = root + 1;
            x = next;
        }
        return root;
    };

    auto dsu_union = [&](const std::int64_t a, const std::int64_t b)
    {
        if (!calculate_detailed_properties) return;
        if (a < 0 || b < 0 || a >= grid.total_size || b >= grid.total_size) return;
        dsu_make_active(a);
        dsu_make_active(b);

        std::int64_t ra = dsu_find(a);
        std::int64_t rb = dsu_find(b);
        if (ra == rb) return;

        if (dsu[static_cast<std::size_t>(ra)] >
            dsu[static_cast<std::size_t>(rb)]) {
            std::swap(ra, rb);
        }

        dsu[static_cast<std::size_t>(ra)] += dsu[static_cast<std::size_t>(rb)];
        dsu[static_cast<std::size_t>(rb)] = ra + 1;
    };

    auto largest_open_component_single_color = [&](const int color_idx) -> std::int64_t
    {
        const int active_val = color_to_active_value(num_colors, color_idx);
        std::int64_t max_component = 0;

        for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
            if (get_site(idx) != active_val) continue;
            if (dsu[static_cast<std::size_t>(idx)] == 0) continue;
            const std::int64_t root = dsu_find(idx);
            max_component = std::max(
                max_component,
                -dsu[static_cast<std::size_t>(root)]
            );
        }

        return max_component;
    };

    // To store chosen edge pairs during the simulation.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> edge_pairs;
    std::unique_ptr<TestedBondBitmap> tested_bonds_bitmap;
    DynamicTestedBondBitmap tested_bonds_dynamic;
    if (stop_config.dynamic_height) {
        tested_bonds_dynamic.ensure_capacity(static_cast<std::size_t>(grid.total_size));
    } else {
        tested_bonds_bitmap = std::make_unique<TestedBondBitmap>(grid);
    }
    std::unique_ptr<TestedBondBitmap> open_bonds;
    if (calculate_detailed_properties) {
        if (stop_config.dynamic_height) {
            throw std::runtime_error(
                "growth_test dynamic_height ainda nao suporta Properties=true.");
        }
        if (!is_node) {
            open_bonds = std::make_unique<TestedBondBitmap>(grid);
        }
    }
    auto mark_tested_bond = [&](const std::uint64_t edge_bit) -> bool
    {
        if (stop_config.dynamic_height) {
            return tested_bonds_dynamic.mark_if_new(edge_bit);
        }
        return tested_bonds_bitmap->mark_if_new(edge_bit);
    };
    auto open_bond = [&](const std::int64_t u,
                         const std::int64_t v,
                         const std::uint64_t edge_bit =
                             std::numeric_limits<std::uint64_t>::max())
    {
        if (calculate_detailed_properties) {
            dsu_union(u, v);
            if (open_bonds) {
                if (edge_bit == std::numeric_limits<std::uint64_t>::max()) {
                    open_bonds->mark_open(u, v);
                } else {
                    open_bonds->mark_open(edge_bit);
                }
            }
            if (save_compact || is_node) {
                edge_pairs.emplace_back(static_cast<std::uint64_t>(u),
                                        static_cast<std::uint64_t>(v));
                edge_pairs.emplace_back(static_cast<std::uint64_t>(v),
                                        static_cast<std::uint64_t>(u));
            }
        }
    };

    auto ensure_growth_capacity = [&](const int min_grow_coord)
    {
        if (!stop_config.dynamic_height) return;
        if (min_grow_coord <= grid.grow_top_coord()) return;

        const std::int64_t old_total = grid.total_size;
        grid.resize_growth_height(min_grow_coord + 1);
        const std::int64_t new_total = grid.total_size;
        if (new_total <= old_total) return;

        if (static_cast<std::size_t>(new_total) > site_state.capacity()) {
            site_state.reserve(static_cast<std::size_t>(new_total) +
                               (static_cast<std::size_t>(new_total) >> 2));
        }
        site_state.resize(
            static_cast<std::size_t>(new_total),
            static_cast<std::int8_t>(-1));

        tested_bonds_dynamic.ensure_capacity(static_cast<std::size_t>(new_total));

        if (needs_activation_time) {
            if (static_cast<std::size_t>(new_total) > activation_time.capacity()) {
                activation_time.reserve(static_cast<std::size_t>(new_total) +
                                       (static_cast<std::size_t>(new_total) >> 2));
            }
            activation_time.resize(
                static_cast<std::size_t>(new_total),
                std::numeric_limits<uint32_t>::max());
        }
        if (calculate_detailed_properties) {
            dsu.resize(static_cast<std::size_t>(new_total), 0);
        }
        front_candidates.resize_site_count(new_total);
    };

    if (calculate_detailed_properties) {
        for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
            if (get_site(idx) > 0) {
                dsu_make_active(idx);
            }
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

    constexpr int progress_interval = 200;
    std::vector<int> equilibrium_stable_run_by_species(
        static_cast<std::size_t>(num_colors), 0);
    std::vector<bool> species_equilibrated(
        static_cast<std::size_t>(num_colors), false);
    std::vector<int> z_stat_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<double> t_eq_online_by_species(
        static_cast<std::size_t>(num_colors),
        std::numeric_limits<double>::quiet_NaN());
    constexpr int surface_deltaT = 15;
    const int post_equilibrium_sampling_height_increment =
        std::max(1, lenght_network);
    const int post_equilibrium_height_increment =
        (5 * std::max(1, lenght_network) + 1) / 2;
    std::vector<int> equilibrium_detection_time_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<int> post_equilibrium_sampling_stop_height_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<int> post_equilibrium_stop_height_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<int> next_surface_sample_time_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<bool> surface_sampling_finished_by_species(
        static_cast<std::size_t>(num_colors), false);
    std::vector<std::vector<double>> f_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<int>> t_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> volume_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<int>> t_volume_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<std::vector<int>>> h_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> w_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> grad_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> S_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<int>> previous_surface_heights_by_species(
        static_cast<std::size_t>(num_colors));
    std::deque<double> error_window;
    std::deque<double> derivative_window;
    std::vector<std::deque<double>> control_derivative_windows(
        static_cast<std::size_t>(num_colors));
    std::vector<double> previous_p_for_dynamics = p_curr;
    double previous_error = std::numeric_limits<double>::quiet_NaN();

    for (int t = 1; t <= hard_max_steps; ++t) {
        if (stop_config.dynamic_height && !frontier.empty()) {
            int max_frontier_h = 0;
            for (const std::int64_t idx : frontier) {
                max_frontier_h = std::max(max_frontier_h, grid.grow_coord(idx));
            }
            ensure_growth_capacity(max_frontier_h + 1);
        }

        std::fill(N_current.begin(), N_current.end(), 0);
        std::fill(f_current.begin(), f_current.end(), 0.0);
        reset_front_height_accumulators();
        next_frontier.clear();
        front_candidates.clear();

        if (is_node) {
            // Site percolation (Leath/SOP): each perimeter target exposed by
            // the current growth front is considered once. If the selected
            // attempt fails, the target is blocked forever.
            std::int64_t neigh[6];

            for (const std::int64_t idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = collect_neighbors(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const std::int64_t viz_idx = neigh[ni];

                    const int vv = get_site(viz_idx);
                    if (vv >= 0) continue; // already active/blocked

                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx});
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const std::int64_t viz = front_candidates.touched_targets[slot];
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
                set_activation_time(viz, static_cast<uint32_t>(t));
                record_height_activation(cor_idx, viz);

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (stop_config.stop_at_percolation &&
                    !percolated[cor_idx] && h >= percolation_height) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                    z_max_at_perc[cor_idx] = max_heights[cor_idx];
                }

                open_bond(chosen.activator, viz);
            }
        } else {
            // Bond percolation (SOP): test every allowed bond from the current
            // growth front. A target activates if at least one incident allowed
            // bond opens in this time step.
            NeighborBond neigh[6];

            for (const std::int64_t idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = stop_config.dynamic_height
                    ? collect_neighbor_bonds_dynamic(grid, idx, neigh)
                    : collect_neighbor_bonds(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const std::int64_t viz_idx = neigh[ni].idx;
                    const std::uint64_t edge_bit = neigh[ni].edge_bit;

                    const int vv = get_site(viz_idx);

                    if (vv == 0) {
                        continue; // blocked site
                    }

                    if (vv > 0) {
                        const int neigh_color_idx = value_to_color_index(num_colors, vv);
                        if (neigh_color_idx != cor_idx) continue;
                        if (!mark_tested_bond(edge_bit)) continue;

                        if (rng.uniform_real(0.0, 1.0) < p_curr[cor_idx]) {
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

                    if (!mark_tested_bond(edge_bit)) continue;

                    if (rng.uniform_real(0.0, 1.0) < p_curr[cor_idx]) {
                        front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx, edge_bit});
                    }
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const std::int64_t viz = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int target_before = get_site(viz);
                const bool target_without_species = (num_colors > 1 && target_before == -1);

                int cor_idx = -1;
                std::int64_t chosen_u = -1;

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
                set_activation_time(viz, static_cast<uint32_t>(t));
                record_height_activation(cor_idx, viz);

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (stop_config.stop_at_percolation &&
                    !percolated[cor_idx] && h >= percolation_height) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                    z_max_at_perc[cor_idx] = max_heights[cor_idx];
                }

                for (int i = 0; i < n_cand; ++i) {
                    const FrontCandidate fc = front_candidates.candidate(slot, i);
                    if (fc.color_idx != cor_idx) continue;
                    open_bond(fc.activator, viz, fc.edge_bit);
                }
            }
        }

        for (int c = 0; c < num_colors; ++c) {
            f_current[c] = static_cast<double>(N_current[c]) / norm_factor;
        }

        if (t < 10 || t % progress_interval == 0) {
            std::cout << "[" << type_percolation << "] t=" << t
                      << ", frontier=" << next_frontier.size();
            for (int c = 0; c < num_colors; ++c) {
                std::cout
                    << ", p" << (c + 1) << "=" << p_curr[c]
                    << ", f" << (c + 1) << "=" << f_current[c]
                    << ", h" << (c + 1) << "=" << max_heights[c]
                    << "/" << grid.grow_top_coord();
            }
            std::cout << '\n';
        }

        commit_step(t, p_curr, f_current);

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
        bool all_finished = true;

        for (int c = 0; c < num_colors; ++c) {
            if (!finished[c]) {
                all_finished = false;
            }

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
            if (!(stop_config.stop_at_percolation && percolated[c]) && !died[c]) {
                all_terminal = false;
            }
        }

        // Condições de parada:
        // 1) todas morreram sem percolar
        const bool stop_all_dead = all_dead || all_finished;

        // 2) todas percolaram
        const bool stop_all_percolated =
            stop_config.stop_at_percolation && all_percolated;

        // 3) percolação parcial: pelo menos uma espécie percolou e todas as
        // demais espécies que não percolaram já morreram.
        const bool stop_partial_percolation =
            stop_config.stop_at_percolation &&
            all_terminal && any_percolated && any_dead;

        bool stop_equilibrated = false;
        if (stop_config.stop_at_equilibrium) {
            bool any_running = false;
            bool all_running_equilibrated = true;
            constexpr int teq_window_block = 10;
            const double teq_rel_tol = growth_test_effective_rel_tol(
                stop_config.equilibrium_rel_tol, num_colors);
            const int teq_min_stable_steps =
                growth_test_global_min_stable_steps(
                    stop_config.equilibrium_consecutive_steps,
                    teq_window_block);
            const int teq_validation_window_steps =
                growth_test_global_validation_window_steps(
                    stop_config.post_equilibrium_extra_steps,
                    teq_window_block);
            for (int c = 0; c < num_colors; ++c) {
                if (finished[c]) continue;
                any_running = true;

                if (!species_equilibrated[static_cast<std::size_t>(c)] &&
                    t % teq_window_block == 0) {
                    const bool use_front_fraction_teq = (std::abs(c_value) <= 1e-12);
                    const auto& series_to_check = use_front_fraction_teq
                        ? f_series[static_cast<std::size_t>(c)]
                        : p_series[static_cast<std::size_t>(c)];
                    const bool reached_global_height = use_front_fraction_teq
                        ? true
                        : (!stop_config.dynamic_height ||
                           max_heights[static_cast<std::size_t>(c)] >= lenght_network);
                    const double t_eq_candidate = reached_global_height
                        ? estimate_t_eq_from_series(
                            t_list,
                            series_to_check,
                            15,
                            teq_window_block,
                            teq_min_stable_steps,
                            teq_rel_tol,
                            stop_config.equilibrium_abs_tol,
                            1.0e-5,
                            false,
                            0.0,
                            teq_validation_window_steps,
                            0u)
                        : std::numeric_limits<double>::quiet_NaN();
                    if (std::isfinite(t_eq_candidate)) {
                        species_equilibrated[static_cast<std::size_t>(c)] = true;
                        t_eq_online_by_species[static_cast<std::size_t>(c)] =
                            t_eq_candidate;
                        z_stat_by_species[static_cast<std::size_t>(c)] =
                            max_heights[static_cast<std::size_t>(c)];
                        equilibrium_detection_time_by_species[static_cast<std::size_t>(c)] = t;
                        post_equilibrium_sampling_stop_height_by_species[static_cast<std::size_t>(c)] =
                            z_stat_by_species[static_cast<std::size_t>(c)] +
                            post_equilibrium_sampling_height_increment;
                        post_equilibrium_stop_height_by_species[static_cast<std::size_t>(c)] =
                            z_stat_by_species[static_cast<std::size_t>(c)] +
                            post_equilibrium_height_increment;
                        next_surface_sample_time_by_species[static_cast<std::size_t>(c)] =
                            t + surface_deltaT;
                    }
                }

                if (!species_equilibrated[static_cast<std::size_t>(c)]) {
                    all_running_equilibrated = false;
                    continue;
                }

                const int next_surface_t =
                    next_surface_sample_time_by_species[static_cast<std::size_t>(c)];
                const int sampling_stop_height =
                    post_equilibrium_sampling_stop_height_by_species[static_cast<std::size_t>(c)];
                const bool reached_sampling_height =
                    sampling_stop_height >= 0 &&
                    max_heights[static_cast<std::size_t>(c)] >= sampling_stop_height;
                const bool should_sample_surface =
                    stop_config.save_surface_observables &&
                    !surface_sampling_finished_by_species[static_cast<std::size_t>(c)] &&
                    next_surface_t >= 0 &&
                    (t >= next_surface_t || reached_sampling_height);
                if (should_sample_surface) {
                    const int analysis_bottom =
                        z_stat_by_species[static_cast<std::size_t>(c)];
                    const int analysis_top = sampling_stop_height;
                    const int sample_top = std::min(
                        max_heights[static_cast<std::size_t>(c)],
                        analysis_top);
                    std::vector<int> current_surface_heights =
                        compute_upper_surface_heights(
                            grid,
                            num_colors,
                            c,
                            site_state,
                            analysis_bottom,
                            sample_top);
                    std::vector<int>& previous_surface_heights =
                        previous_surface_heights_by_species[static_cast<std::size_t>(c)];
                    if (!previous_surface_heights.empty()) {
                        volume_sur_by_species[static_cast<std::size_t>(c)].push_back(
                            compute_volume_fraction_between_upper_surfaces(
                                grid,
                                num_colors,
                                c,
                                site_state,
                                previous_surface_heights,
                                current_surface_heights,
                                analysis_bottom,
                                analysis_top));
                        t_volume_sur_by_species[static_cast<std::size_t>(c)].push_back(
                            static_cast<int>(
                                t_volume_sur_by_species[static_cast<std::size_t>(c)].size()) + 1);
                    }

                    t_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        static_cast<int>(
                            t_sur_by_species[static_cast<std::size_t>(c)].size()) + 1);
                    w_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        compute_surface_width(current_surface_heights));
                    grad_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        compute_surface_gradient(grid, current_surface_heights));
                    h_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        current_surface_heights);
                    const double f_sur_value =
                        compute_upper_surface_fraction(
                            grid,
                            num_colors,
                            c,
                            site_state,
                            sample_top,
                            analysis_bottom);
                    f_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        f_sur_value);
                    S_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        f_sur_value * compute_lateral_area(grid));
                    if (reached_sampling_height) {
                        surface_sampling_finished_by_species[static_cast<std::size_t>(c)] = true;
                        next_surface_sample_time_by_species[static_cast<std::size_t>(c)] = -1;
                    } else {
                        next_surface_sample_time_by_species[static_cast<std::size_t>(c)] =
                            t + surface_deltaT;
                    }
                    previous_surface_heights = std::move(current_surface_heights);
                }
            }
            if (any_running && all_running_equilibrated) {
                bool all_reached_post_equilibrium_height = true;
                for (int c = 0; c < num_colors; ++c) {
                    if (finished[c]) continue;
                    const int target_height =
                        post_equilibrium_stop_height_by_species[static_cast<std::size_t>(c)];
                    if (target_height < 0 ||
                        max_heights[static_cast<std::size_t>(c)] < target_height) {
                        all_reached_post_equilibrium_height = false;
                        break;
                    }
                }
                stop_equilibrated =
                    all_reached_post_equilibrium_height &&
                    t % teq_window_block == 0;
            }
        }

        const bool stop_hard_limit =
            (stop_config.stop_at_equilibrium || stop_config.hard_max_steps > 0) &&
            (t >= hard_max_steps);
        bool stop_stationary_dynamics = false;

        if (stop_all_dead || stop_all_percolated ||
            stop_partial_percolation || stop_equilibrated ||
            stop_stationary_dynamics || stop_hard_limit) {
            growth_test_stop_time = t;
            if (stop_all_dead) {
                growth_test_stop_reason = all_dead ? "all_dead" : "all_finished";
            } else if (stop_all_percolated) {
                growth_test_stop_reason = "all_percolated";
            } else if (stop_partial_percolation) {
                growth_test_stop_reason = "partial_percolation";
            } else if (stop_equilibrated) {
                growth_test_stop_reason = "post_equilibrium_height_reached";
            } else if (stop_stationary_dynamics) {
                growth_test_stop_reason = "stationary_dynamics";
            } else if (stop_hard_limit) {
                growth_test_stop_reason = "hard_max_steps";
            }
            if (!calculate_detailed_properties) {
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
                std::vector<std::int64_t> path = is_node
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

            if (growth_test_stop_reason == "hard_max_steps") {
                std::vector<char> included(static_cast<std::size_t>(num_colors), 0);
                for (const auto& vp : valid_percs) {
                    included[static_cast<std::size_t>(vp.color_idx)] = 1;
                }
                for (int c = 0; c < num_colors; ++c) {
                    if (included[static_cast<std::size_t>(c)]) continue;
                    ps_out.color_percolation.push_back(c + 1);
                    ps_out.percolation_order.push_back(++valid_order_ctr);
                    ps_out.M_size_at_perc.push_back(
                        is_node
                            ? largest_component_single_color(c)
                            : largest_open_component_single_color(c)
                    );
                }
            }

            break;
        }

        std::fill(p_next.begin(), p_next.end(), 0.0);
        for (int c = 0; c < num_colors; ++c) {
            p_next[c] = finished[c] ? p_curr[c]
                                    : generate_p(
                                        type_f_T, p_curr[c], t, f_current[c],
                                        c_value, f_T, a, alpha,
                                        stop_config.feedback_control_rule);
        }

        frontier.swap(next_frontier);
        p_curr.swap(p_next);
    }

    for (int c = 0; c < num_colors; ++c) {
        if (!died[static_cast<std::size_t>(c)] &&
            !species_equilibrated[static_cast<std::size_t>(c)]) {
            const bool use_front_fraction_teq = (std::abs(c_value) <= 1e-12);
            const auto& series_to_check = use_front_fraction_teq
                ? f_series[static_cast<std::size_t>(c)]
                : p_series[static_cast<std::size_t>(c)];
            const bool reached_global_height = use_front_fraction_teq
                ? true
                : (!stop_config.dynamic_height ||
                   max_heights[static_cast<std::size_t>(c)] >= lenght_network);
            if (reached_global_height) {
                constexpr int teq_window_block = 10;
                const double teq_rel_tol = growth_test_effective_rel_tol(
                    stop_config.equilibrium_rel_tol, num_colors);
                const int teq_min_stable_steps =
                    growth_test_global_min_stable_steps(
                        stop_config.equilibrium_consecutive_steps,
                        teq_window_block);
                const int teq_validation_window_steps =
                    growth_test_global_validation_window_steps(
                        stop_config.post_equilibrium_extra_steps,
                        teq_window_block);
                const double t_eq_candidate = estimate_t_eq_from_series(
                    t_list,
                    series_to_check,
                    15,
                    teq_window_block,
                    teq_min_stable_steps,
                    teq_rel_tol,
                    stop_config.equilibrium_abs_tol,
                    1.0e-5,
                    false,
                    0.0,
                    teq_validation_window_steps,
                    0u);
                if (std::isfinite(t_eq_candidate)) {
                    species_equilibrated[static_cast<std::size_t>(c)] = true;
                    t_eq_online_by_species[static_cast<std::size_t>(c)] = t_eq_candidate;
                    z_stat_by_species[static_cast<std::size_t>(c)] =
                        max_heights[static_cast<std::size_t>(c)];
                }
            }
        }
    }

    if (save_lateral_observables) {
        ts_out.lateral_observables = compute_lateral_observables(
            dim,
            lenght_network,
            grid.SX,
            grid.SY,
            grid.SZ,
            activation_time,
            t_list,
            "periodic",
            false,
            &activated_indices_for_lateral);
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.f_t = std::move(f_series);
    ts_out.y_mean_t = std::move(y_mean_series);
    ts_out.y_width_t = std::move(y_width_series);
    ts_out.y_max_t = std::move(y_max_series);
    ts_out.y_front_mean_t = std::move(y_front_mean_series);
    ts_out.y_front_width_t = std::move(y_front_width_series);
    ts_out.y_front_count_t = std::move(y_front_count_series);
    ts_out.t   = std::move(t_list);

    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < static_cast<int>(rho.size())) {
            ps_out.rho.push_back(rho[i]);
        }
    }
    ps_out.feedback_control_param = c_value;

    if (stop_config.stop_at_equilibrium) {
        if (stop_config.stop_at_percolation) {
            ps_out.z_max_at_perc = std::move(z_max_at_perc);
        } else {
            ps_out.z_max_at_perc.clear();
        }
        ps_out.z_max_final = max_heights;
        ps_out.z_stat_by_species = z_stat_by_species;
        ps_out.species_final_status.assign(
            static_cast<std::size_t>(num_colors), 0);
        for (int c = 0; c < num_colors; ++c) {
            if (species_equilibrated[static_cast<std::size_t>(c)]) {
                ps_out.species_final_status[static_cast<std::size_t>(c)] = 1;
            } else if (died[static_cast<std::size_t>(c)]) {
                ps_out.species_final_status[static_cast<std::size_t>(c)] = -1;
            }
        }
        ps_out.growth_test_stop_reason = growth_test_stop_reason;
        ps_out.growth_test_stop_time = growth_test_stop_time;
        ps_out.equilibrium_consecutive_steps =
            stop_config.equilibrium_consecutive_steps;
        ps_out.dynamics_window_steps = stop_config.dynamics_window_steps;
        ps_out.post_equilibrium_extra_steps =
            stop_config.post_equilibrium_extra_steps;
        ps_out.dynamic_min_stop_height = dynamic_min_stop_height;
        ps_out.dynamic_max_stop_height = dynamic_max_stop_height;
        ps_out.equilibrium_rel_tol = stop_config.equilibrium_rel_tol;
        ps_out.equilibrium_abs_tol = stop_config.equilibrium_abs_tol;
    } else {
        ps_out.z_max_at_perc.clear();
        ps_out.z_max_final.clear();
        ps_out.z_stat_by_species.clear();
        ps_out.species_final_status.clear();
        ps_out.growth_test_stop_reason.clear();
        ps_out.growth_test_stop_time = -1;
        ps_out.equilibrium_consecutive_steps = -1;
        ps_out.dynamics_window_steps = -1;
        ps_out.post_equilibrium_extra_steps = -1;
        ps_out.dynamic_min_stop_height = -1;
        ps_out.dynamic_max_stop_height = -1;
        ps_out.equilibrium_rel_tol = std::numeric_limits<double>::quiet_NaN();
        ps_out.equilibrium_abs_tol = std::numeric_limits<double>::quiet_NaN();
    }

    if (stop_config.stop_at_equilibrium && !stop_config.stop_at_percolation) {
        ps_out.t_eq_by_species = t_eq_online_by_species;
        for (int c = 0; c < num_colors; ++c) {
            if (died[static_cast<std::size_t>(c)] ||
                !species_equilibrated[static_cast<std::size_t>(c)]) {
                ps_out.t_eq_by_species[static_cast<std::size_t>(c)] =
                    std::numeric_limits<double>::quiet_NaN();
                ps_out.z_stat_by_species[static_cast<std::size_t>(c)] = -1;
            }
        }
        ts_out.t_eq = any_value_finite(ps_out.t_eq_by_species)
            ? mean_finite_values(ps_out.t_eq_by_species)
            : std::numeric_limits<double>::quiet_NaN();
    } else {
        ps_out.t_eq_by_species.clear();
        ts_out.t_eq = estimate_t_eq_from_timeseries(
            ts_out,
            15,      // window_roll
            10,      // window_block
            15,      // min stable steps
            2.5e-2,  // relative block-change tolerance
            1.0e-6,  // absolute block-change tolerance
            1.0e-5   // |s_prime| threshold
        );
    }
    ps_out.t_eq = ts_out.t_eq;

    if (stop_config.stop_at_equilibrium && !stop_config.stop_at_percolation) {
        set_species_order_from_t_eq(
            ps_out,
            ps_out.t_eq_by_species,
            growth_test_stop_reason == "hard_max_steps");
    }

    if (stop_config.stop_at_equilibrium && stop_config.save_surface_observables) {
        ps_out.f_sur_by_species = std::move(f_sur_by_species);
        ps_out.t_sur_by_species = std::move(t_sur_by_species);
        ps_out.volume_sur_by_species = std::move(volume_sur_by_species);
        ps_out.t_volume_sur_by_species = std::move(t_volume_sur_by_species);
        ps_out.h_sur_by_species = std::move(h_sur_by_species);
        ps_out.w_sur_by_species = std::move(w_sur_by_species);
        ps_out.grad_sur_by_species = std::move(grad_sur_by_species);
        ps_out.M_L_by_species.assign(
            static_cast<std::size_t>(num_colors),
            std::numeric_limits<double>::quiet_NaN());
        ps_out.M_cluster_sizes_by_species.assign(
            static_cast<std::size_t>(num_colors),
            std::vector<std::int64_t>{});
        for (int c = 0; c < num_colors; ++c) {
            const int z_stat = z_stat_by_species[static_cast<std::size_t>(c)];
            if (z_stat >= 0) {
                const int analysis_top = z_stat + post_equilibrium_sampling_height_increment;
                const int analysis_bottom = z_stat;
                SlabClusterMassResult cluster_masses =
                    compute_slab_cluster_masses(
                        grid,
                        num_colors,
                        c,
                        site_state,
                        analysis_bottom,
                        analysis_top);
                ps_out.M_L_by_species[static_cast<std::size_t>(c)] =
                    cluster_masses.largest_spanning_mass;
                ps_out.M_cluster_sizes_by_species[static_cast<std::size_t>(c)] =
                    std::move(cluster_masses.cluster_sizes);
            }
        }
        ps_out.S_sur_by_species = std::move(S_sur_by_species);
        ps_out.surface_deltaT = surface_deltaT;
        ps_out.surface_observables_enabled = true;
    } else {
        ps_out.f_sur_by_species.clear();
        ps_out.t_sur_by_species.clear();
        ps_out.volume_sur_by_species.clear();
        ps_out.t_volume_sur_by_species.clear();
        ps_out.h_sur_by_species.clear();
        ps_out.w_sur_by_species.clear();
        ps_out.grad_sur_by_species.clear();
        ps_out.S_sur_by_species.clear();
        ps_out.M_L_by_species.clear();
        ps_out.M_cluster_sizes_by_species.clear();
        ps_out.surface_deltaT = -1;
        ps_out.surface_observables_enabled = false;
    }

    if (calculate_detailed_properties) {
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
            const std::vector<std::int64_t>& path = ps_out.sp_path_lin[c];
            ps_out.sp_path_lin_preteq[c].clear();
            ps_out.sp_path_lin_posteq[c].clear();
            if (!path.empty()) {
                ps_out.sp_path_lin_preteq[c].push_back(path.front());
                ps_out.sp_path_lin_posteq[c].push_back(path.front());
            }

            int sp_pre = 0;
            int sp_post = 0;
            for (std::size_t k = 1; k < path.size(); ++k) {
                const std::int64_t idx = path[k];
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

            std::int64_t m_pre = 0;
            std::int64_t m_post = 0;

            if (is_node) {
                std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
                std::vector<std::int64_t> stack;
                std::vector<std::int64_t> comp_nodes;
                std::vector<std::int64_t> best_nodes;

                for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
                    if (visited[static_cast<std::size_t>(idx)]) continue;
                    if (get_site(idx) != active_val) continue;

                    stack.clear();
                    comp_nodes.clear();

                    stack.push_back(idx);
                    visited[static_cast<std::size_t>(idx)] = 1;

                    while (!stack.empty()) {
                        const std::int64_t u = stack.back();
                        stack.pop_back();
                        comp_nodes.push_back(u);

                        grid.for_each_neighbor(u, [&](const std::int64_t v) {
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

                for (const std::int64_t idx : best_nodes) {
                    if (activation_time[static_cast<std::size_t>(idx)] ==
                        std::numeric_limits<uint32_t>::max()) {
                        throw std::runtime_error(
                            "create_network: largest component contains invalid activation time");
                    }
                    if (static_cast<int>(activation_time[static_cast<std::size_t>(idx)]) <= ps_out.t_eq) ++m_pre;
                    else ++m_post;
                }
            } else {
                std::int64_t best_root = -1;
                std::int64_t best_size = 0;

                for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
                    if (get_site(idx) != active_val) continue;
                    if (dsu[static_cast<std::size_t>(idx)] == 0) continue;
                    const std::int64_t root = dsu_find(idx);
                    const std::int64_t size = -dsu[static_cast<std::size_t>(root)];
                    if (size > best_size) {
                        best_size = size;
                        best_root = root;
                    }
                }

                if (best_root >= 0) {
                    for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
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
            const int v = get_site(static_cast<std::int64_t>(i));
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
    std::vector<std::int64_t>().swap(dsu);
    std::vector<std::pair<std::uint64_t,std::uint64_t>>().swap(edge_pairs);
    SparseFrontCandidates empty_front_candidates;
    std::swap(front_candidates, empty_front_candidates);

    // grid may have grown beyond the initial shape in dynamic-height runs.
    // Allocate the returned pattern with its final extent before copying it.
    const std::vector<int> final_shape = (dim == 2)
        ? std::vector<int>{grid.SX, grid.SY}
        : std::vector<int>{grid.SX, grid.SY, grid.SZ};
    NetworkPattern net(dim, final_shape, num_colors, rho, save_compact);
    net.seed = net_seed;
    if (save_compact) {
        for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
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
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng,
    const bool calculate_detailed_properties,
    const bool calculate_height_timeseries,
    const GrowthStopConfig stop_config)
{
    this->c = c_value;
    this->f_T = f_T;

    GridRegular grid(
        dim,
        lenght_network,
        stop_config.height_multiplier,
        stop_config.height_extra_layers);
    const std::vector<int> shape = (dim == 2)
        ? std::vector<int>{grid.SX, grid.SY}
        : std::vector<int>{grid.SX, grid.SY, grid.SZ};

    const bool is_node = (type_percolation == "node");
    const long long base_size = compute_base_size(grid);
    const double norm_factor = static_cast<double>(base_size);
    const int percolation_height = lenght_network - 1;
    const int hard_max_steps = stop_config.hard_max_steps > 0
        ? std::min(num_of_samples - 1, stop_config.hard_max_steps)
        : (num_of_samples - 1);
    const int dynamic_min_stop_height = -1;
    const int dynamic_max_stop_height = -1;

    if (num_colors > 125) {
        throw std::runtime_error(
            "animate_network: num_colors exceeds int8_t site-state encoding capacity");
    }

    std::vector<std::int8_t> site_state(
        static_cast<std::size_t>(grid.total_size),
        static_cast<std::int8_t>(-1));
    auto get_site = [&](const std::int64_t idx) -> int {
        return static_cast<int>(site_state[static_cast<std::size_t>(idx)]);
    };
    auto set_site = [&](const std::int64_t idx, const int value) {
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
    std::vector<std::vector<double>> y_mean_series(num_colors);
    std::vector<std::vector<double>> y_width_series(num_colors);
    std::vector<std::vector<int>> y_max_series(num_colors);
    std::vector<std::vector<double>> y_front_mean_series(num_colors);
    std::vector<std::vector<double>> y_front_width_series(num_colors);
    std::vector<std::vector<uint32_t>> y_front_count_series(num_colors);
    std::vector<int>                 t_list;
    t_list.reserve(num_of_samples);

    std::vector<double> p_curr = p0;
    std::vector<double> p_next(num_colors, 0.0);
    std::vector<int>    N_current(num_colors, 0);
    std::vector<double> f_current(num_colors, 0.0);
    std::vector<int>    max_heights(num_colors, 0);
    std::vector<int>    z_max_at_perc(num_colors, -1);
    std::vector<std::int64_t> frontier;
    std::vector<std::int64_t> next_frontier;

    const std::size_t lateral_size = static_cast<std::size_t>(base_size);
    std::vector<std::vector<int>> surface_height;
    if (calculate_height_timeseries) {
        surface_height.assign(
            static_cast<std::size_t>(num_colors),
            std::vector<int>(lateral_size, -1));
    }
    std::vector<double> surface_sum_height(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<double> surface_sum_height_sq(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<uint32_t> surface_active_columns(static_cast<std::size_t>(num_colors), 0);
    std::vector<int> surface_max_height(static_cast<std::size_t>(num_colors), -1);
    std::vector<double> front_sum_height(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<double> front_sum_height_sq(static_cast<std::size_t>(num_colors), 0.0);
    std::vector<uint32_t> front_count(static_cast<std::size_t>(num_colors), 0);

    auto lateral_index_of = [&](const std::int64_t idx) -> std::size_t {
        if (dim == 2) {
            return static_cast<std::size_t>(grid.x_of(idx));
        }
        return static_cast<std::size_t>(
            grid.x_of(idx) + static_cast<std::int64_t>(grid.SX) * grid.y_of(idx));
    };

    auto reset_front_height_accumulators = [&]() {
        if (!calculate_height_timeseries) return;
        std::fill(front_sum_height.begin(), front_sum_height.end(), 0.0);
        std::fill(front_sum_height_sq.begin(), front_sum_height_sq.end(), 0.0);
        std::fill(front_count.begin(), front_count.end(), 0u);
    };

    auto record_height_activation = [&](const int color_idx, const std::int64_t idx) {
        if (!calculate_height_timeseries) return;
        if (color_idx < 0 || color_idx >= num_colors) return;
        const std::size_t c = static_cast<std::size_t>(color_idx);
        const int h = grid.grow_coord(idx);
        front_sum_height[c] += static_cast<double>(h);
        front_sum_height_sq[c] += static_cast<double>(h) * static_cast<double>(h);
        ++front_count[c];

        const std::size_t lateral_idx = lateral_index_of(idx);
        int& old_h = surface_height[c][lateral_idx];
        if (h <= old_h) return;

        if (old_h < 0) {
            ++surface_active_columns[c];
            surface_sum_height[c] += static_cast<double>(h);
            surface_sum_height_sq[c] += static_cast<double>(h) * static_cast<double>(h);
        } else {
            surface_sum_height[c] += static_cast<double>(h - old_h);
            surface_sum_height_sq[c] +=
                static_cast<double>(h) * static_cast<double>(h) -
                static_cast<double>(old_h) * static_cast<double>(old_h);
        }
        old_h = h;
        if (h > surface_max_height[c]) {
            surface_max_height[c] = h;
        }
    };

    frontier.reserve(static_cast<std::size_t>(base_size));
    next_frontier.reserve(static_cast<std::size_t>(base_size));

    SparseFrontCandidates front_candidates(
        stop_config.dynamic_height ? 0 : grid.total_size,
        stop_config.dynamic_height);
    front_candidates.reserve_frontier(static_cast<std::size_t>(base_size));

    auto commit_step = [&](const int t_k,
                           const std::vector<double>& p_vec,
                           const std::vector<double>& f_vec)
    {
        t_list.push_back(t_k);
        for (int c = 0; c < num_colors; ++c) {
            p_series[c].push_back(p_vec[c]);
            f_series[c].push_back(f_vec[c]);
            if (!calculate_height_timeseries) continue;

            const std::size_t row = static_cast<std::size_t>(c);
            const uint32_t ncols = surface_active_columns[row];
            if (ncols > 0) {
                const double mean =
                    surface_sum_height[row] / static_cast<double>(ncols);
                const double mean_sq =
                    surface_sum_height_sq[row] / static_cast<double>(ncols);
                y_mean_series[row].push_back(mean);
                y_width_series[row].push_back(
                    std::sqrt(std::max(0.0, mean_sq - mean * mean)));
            } else {
                y_mean_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
                y_width_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
            }
            y_max_series[row].push_back(surface_max_height[row]);

            const uint32_t nf = front_count[row];
            if (nf > 0) {
                const double mean =
                    front_sum_height[row] / static_cast<double>(nf);
                const double mean_sq =
                    front_sum_height_sq[row] / static_cast<double>(nf);
                y_front_mean_series[row].push_back(mean);
                y_front_width_series[row].push_back(
                    std::sqrt(std::max(0.0, mean_sq - mean * mean)));
            } else {
                y_front_mean_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
                y_front_width_series[row].push_back(
                    std::numeric_limits<double>::quiet_NaN());
            }
            y_front_count_series[row].push_back(nf);
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
    ps_out.initial_base_layout =
        initial_base_layout_name(stop_config.initial_base_layout);
    ps_out.color_percolation.clear();
    ps_out.percolation_order.clear();
    ps_out.M_size_at_perc.clear();
    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, std::vector<std::int64_t>{});

    // novas métricas temporais
    ps_out.t_eq = std::numeric_limits<double>::quiet_NaN();
    ps_out.sp_lin_preteq.assign(num_colors, -1);
    ps_out.sp_path_lin_preteq.assign(num_colors, std::vector<std::int64_t>{});
    ps_out.sp_lin_posteq.assign(num_colors, -1);
    ps_out.sp_path_lin_posteq.assign(num_colors, std::vector<std::int64_t>{});
    ps_out.M_size_preteq.assign(num_colors, 0);
    ps_out.M_size_posteq.assign(num_colors, 0);

    // Ativação inicial na base
    for (int c = 0; c < num_colors; ++c) {
        const int prefer_neg = color_to_negative_value(num_colors, c);
        const int active_val = color_to_active_value(num_colors, c);

        auto activate_base_site = [&](const std::int64_t idx) -> bool {
            const int v = get_site(idx);
            if (v != prefer_neg && v != -1) return false;

            set_site(idx, active_val);
            activation_code[static_cast<std::size_t>(idx)] =
                static_cast<NetworkPattern::state_t>(color_mul[c]);
            frontier.push_back(idx);
            ++N_current[c];
            record_height_activation(c, idx);
            return true;
        };

        if (stop_config.initial_base_layout == InitialBaseLayout::Clustered) {
            activate_clustered_base_sites(
                grid,
                num_colors,
                c,
                seeds_quota[static_cast<std::size_t>(c)],
                rng,
                activate_base_site);
        } else if (stop_config.initial_base_layout == InitialBaseLayout::Random) {
            int activated = 0;
            int tries = 0;
            const int max_tries = static_cast<int>(std::min<long long>(static_cast<long long>(std::numeric_limits<int>::max()), base_size * 20));

            while (activated < seeds_quota[c] && tries < max_tries) {
                const int x = rng.uniform_int(0, grid.SX - 1);
                const int y = (dim == 3 ? rng.uniform_int(0, grid.SY - 1) : 0);
                const int z = 0;
                const std::int64_t idx = grid.lin_index(x, y, z);

                if (activate_base_site(idx)) {
                    ++activated;
                }
                ++tries;
            }
        } else {
            activate_deterministic_base_sites(
                grid,
                num_colors,
                c,
                seeds_quota[static_cast<std::size_t>(c)],
                stop_config.initial_base_layout,
                activate_base_site);
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
    std::string growth_test_stop_reason = "not_stopped";
    int growth_test_stop_time = -1;
    std::vector<int> equilibrium_stable_run_by_species(
        static_cast<std::size_t>(num_colors), 0);
    std::vector<bool> species_equilibrated(
        static_cast<std::size_t>(num_colors), false);
    std::vector<int> z_stat_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<double> t_eq_online_by_species(
        static_cast<std::size_t>(num_colors),
        std::numeric_limits<double>::quiet_NaN());
    constexpr int surface_deltaT = 15;
    const int post_equilibrium_sampling_height_increment =
        std::max(1, lenght_network);
    const int post_equilibrium_height_increment =
        (5 * std::max(1, lenght_network) + 1) / 2;
    std::vector<int> equilibrium_detection_time_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<int> post_equilibrium_sampling_stop_height_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<int> post_equilibrium_stop_height_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<int> next_surface_sample_time_by_species(
        static_cast<std::size_t>(num_colors), -1);
    std::vector<bool> surface_sampling_finished_by_species(
        static_cast<std::size_t>(num_colors), false);
    std::vector<std::vector<double>> f_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<int>> t_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> volume_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<int>> t_volume_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<std::vector<int>>> h_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> w_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> grad_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<double>> S_sur_by_species(
        static_cast<std::size_t>(num_colors));
    std::vector<std::vector<int>> previous_surface_heights_by_species(
        static_cast<std::size_t>(num_colors));
    std::deque<double> error_window;
    std::deque<double> derivative_window;
    std::vector<std::deque<double>> control_derivative_windows(
        static_cast<std::size_t>(num_colors));
    std::vector<double> previous_p_for_dynamics = p_curr;
    double previous_error = std::numeric_limits<double>::quiet_NaN();

    // cache leve do maior componente global por cor: semente + tamanho.
    std::vector<ComponentSummary> largest_comp_cache(num_colors);
    bool global_metrics_finalized = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> edge_pairs;

    auto largest_component_single_color_summary = [&](const int color_idx) -> ComponentSummary
    {
        const int active_val = color_to_active_value(num_colors, color_idx);

        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<std::int64_t> stack;
        ComponentSummary best;

        for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
            if (visited[static_cast<std::size_t>(idx)]) continue;
            if (get_site(idx) != active_val) continue;

            std::int64_t comp_size = 0;
            stack.clear();
            stack.push_back(idx);
            visited[static_cast<std::size_t>(idx)] = 1;

            while (!stack.empty()) {
                const std::int64_t u = stack.back();
                stack.pop_back();
                ++comp_size;

                grid.for_each_neighbor(u, [&](const std::int64_t v) {
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
        std::vector<std::int64_t> stack;
        ComponentSummary best;

        for (std::int64_t idx = 0; idx < grid.total_size; ++idx) {
            if (visited[static_cast<std::size_t>(idx)]) continue;
            if (get_site(idx) != active_val) continue;

            std::int64_t comp_size = 0;
            stack.clear();
            stack.push_back(idx);
            visited[static_cast<std::size_t>(idx)] = 1;

            while (!stack.empty()) {
                const std::int64_t u = stack.back();
                stack.pop_back();
                ++comp_size;

                const std::uint64_t begin = graph.offsets[static_cast<std::size_t>(u)];
                const std::uint64_t end = graph.offsets[static_cast<std::size_t>(u) + 1u];
                for (std::uint64_t k = begin; k < end; ++k) {
                    const std::int64_t v = graph.neighbors[static_cast<std::size_t>(k)];
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
        [&](const std::int64_t seed, const int color_idx) -> TimeSplit
    {
        TimeSplit split;
        if (seed < 0) return split;

        const int active_val = color_to_active_value(num_colors, color_idx);
        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<std::int64_t> stack;
        stack.push_back(seed);
        visited[static_cast<std::size_t>(seed)] = 1;

        while (!stack.empty()) {
            const std::int64_t u = stack.back();
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

            grid.for_each_neighbor(u, [&](const std::int64_t v) {
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
        [&](const std::int64_t seed, const int color_idx, const OpenBondGraph& graph) -> TimeSplit
    {
        TimeSplit split;
        if (seed < 0) return split;

        const int active_val = color_to_active_value(num_colors, color_idx);
        std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
        std::vector<std::int64_t> stack;
        stack.push_back(seed);
        visited[static_cast<std::size_t>(seed)] = 1;

        while (!stack.empty()) {
            const std::int64_t u = stack.back();
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

            const std::uint64_t begin = graph.offsets[static_cast<std::size_t>(u)];
            const std::uint64_t end = graph.offsets[static_cast<std::size_t>(u) + 1u];
            for (std::uint64_t k = begin; k < end; ++k) {
                const std::int64_t v = graph.neighbors[static_cast<std::size_t>(k)];
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
        ps_out.sp_path_lin.assign(num_colors, std::vector<std::int64_t>{});

        std::fill(largest_comp_cache.begin(), largest_comp_cache.end(), ComponentSummary{});
        const OpenBondGraph open_bond_graph(grid.total_size, edge_pairs);

        struct ValidPercolation {
            int rank;
            int color_idx;
        };

        std::vector<ValidPercolation> valid_percs;
        valid_percs.reserve(static_cast<std::size_t>(num_colors));
        const bool dynamic_equilibrium_mode =
            stop_config.dynamic_height &&
            stop_config.stop_at_equilibrium &&
            !stop_config.stop_at_percolation;

        for (int c = 0; c < num_colors; ++c) {
            if (died[c]) {
                ps_out.sp_len[c] = -1;
                ps_out.sp_path_lin[c].clear();
                continue;
            }

            const bool valid_species = dynamic_equilibrium_mode
                ? species_equilibrated[static_cast<std::size_t>(c)]
                : percolated[c];
            if (!valid_species) {
                ps_out.sp_len[c] = -1;
                ps_out.sp_path_lin[c].clear();
                continue;
            }

            const int active_val = color_to_active_value(num_colors, c);
            const int target_height = dynamic_equilibrium_mode
                ? max_heights[static_cast<std::size_t>(c)]
                : grid.grow_top_coord();
            std::vector<std::int64_t> path = is_node
                ? shortest_site_path_single_color(
                    grid, active_val, get_site, target_height)
                : shortest_open_bond_path_single_color(
                    grid, open_bond_graph, active_val, get_site, target_height);

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

            valid_percs.push_back({
                dynamic_equilibrium_mode ? c : percolation_rank[c],
                c
            });
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

        if (growth_test_stop_reason == "hard_max_steps") {
            std::vector<char> included(static_cast<std::size_t>(num_colors), 0);
            for (const auto& vp : valid_percs) {
                included[static_cast<std::size_t>(vp.color_idx)] = 1;
            }
            for (int c = 0; c < num_colors; ++c) {
                if (included[static_cast<std::size_t>(c)]) continue;
                ps_out.color_percolation.push_back(c + 1);
                ps_out.percolation_order.push_back(++valid_order_ctr);
                ps_out.M_size_at_perc.push_back(largest_comp_cache[c].size);
            }
        }

        global_metrics_finalized = true;
    };

    auto finalize_time_series_only = [&]()
    {
        ps_out.color_percolation.clear();
        ps_out.percolation_order.clear();
        ps_out.M_size_at_perc.clear();
        ps_out.sp_len.assign(num_colors, -1);
        ps_out.sp_path_lin.assign(num_colors, std::vector<std::int64_t>{});

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
        std::vector<char> included(static_cast<std::size_t>(num_colors), 0);
        for (const auto& vp : valid_percs) {
            ps_out.color_percolation.push_back(vp.color_idx + 1);
            ps_out.percolation_order.push_back(++valid_order_ctr);
            included[static_cast<std::size_t>(vp.color_idx)] = 1;
        }
        if (growth_test_stop_reason == "hard_max_steps") {
            for (int c = 0; c < num_colors; ++c) {
                if (included[static_cast<std::size_t>(c)]) continue;
                ps_out.color_percolation.push_back(c + 1);
                ps_out.percolation_order.push_back(++valid_order_ctr);
            }
        }

        global_metrics_finalized = true;
    };

    std::unique_ptr<TestedBondBitmap> tested_bonds_bitmap;
    DynamicTestedBondBitmap tested_bonds_dynamic;
    if (!stop_config.dynamic_height) {
        tested_bonds_bitmap = std::make_unique<TestedBondBitmap>(grid);
    }
    auto mark_tested_bond = [&](const std::uint64_t edge_bit) -> bool
    {
        if (stop_config.dynamic_height) {
            return tested_bonds_dynamic.mark_if_new(edge_bit);
        }
        return tested_bonds_bitmap->mark_if_new(edge_bit);
    };

    auto ensure_growth_capacity = [&](const int min_grow_coord)
    {
        if (!stop_config.dynamic_height) return;
        if (min_grow_coord <= grid.grow_top_coord()) return;

        const std::int64_t old_total = grid.total_size;
        grid.resize_growth_height(min_grow_coord + 1);
        const std::int64_t new_total = grid.total_size;
        if (new_total <= old_total) return;

        site_state.resize(
            static_cast<std::size_t>(new_total),
            static_cast<std::int8_t>(-1));
        activation_code.resize(
            static_cast<std::size_t>(new_total),
            static_cast<NetworkPattern::state_t>(-1));
        front_candidates.resize_site_count(new_total);
    };
    const std::uint64_t topology_seed =
        static_cast<std::uint64_t>(rng.get_seed()) ^
        0x9E3779B97F4A7C15ULL ^
        (static_cast<std::uint64_t>(lenght_network) << 32) ^
        (static_cast<std::uint64_t>(num_colors) << 48);
    std::mt19937_64 topology_rng(topology_seed);

    constexpr int progress_interval = 200;

    for (int t = 1; t <= hard_max_steps; ++t) {
        if (stop_config.dynamic_height && !frontier.empty()) {
            int max_frontier_h = 0;
            for (const std::int64_t idx : frontier) {
                max_frontier_h = std::max(max_frontier_h, grid.grow_coord(idx));
            }
            ensure_growth_capacity(max_frontier_h + 1);
        }

        std::fill(N_current.begin(), N_current.end(), 0);
        std::fill(f_current.begin(), f_current.end(), 0.0);
        reset_front_height_accumulators();
        next_frontier.clear();
        front_candidates.clear();

        if (is_node) {
            // Same activation logic used by create_network for site/SOP.
            std::int64_t neigh[6];

            for (const std::int64_t idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = collect_neighbors(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const std::int64_t viz_idx = neigh[ni];

                    const int vv = get_site(viz_idx);
                    if (vv >= 0) continue;

                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx});
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const std::int64_t viz = front_candidates.touched_targets[slot];
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
                record_height_activation(cor_idx, viz);

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (stop_config.stop_at_percolation &&
                    !percolated[cor_idx] && h >= percolation_height) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                    z_max_at_perc[cor_idx] = max_heights[cor_idx];
                }

                edge_pairs.emplace_back(chosen.activator,
                                        static_cast<std::uint64_t>(viz));
                edge_pairs.emplace_back(static_cast<std::uint64_t>(viz),
                                        chosen.activator);
            }
        } else {
            // Same activation logic used by create_network for bond/SOP.
            // All accepted bonds from the current front are collected before
            // any target site is activated, removing for-loop order bias.
            NeighborBond neigh[6];

            for (const std::int64_t idx : frontier) {
                const int a_val = get_site(idx);
                if (a_val <= 0) continue;

                const int cor_idx = value_to_color_index(num_colors, a_val);
                if (cor_idx < 0 || cor_idx >= num_colors) continue;
                if (finished[cor_idx]) continue;

                const int nneigh = stop_config.dynamic_height
                    ? collect_neighbor_bonds_dynamic(grid, idx, neigh)
                    : collect_neighbor_bonds(grid, idx, neigh);
                for (int ni = 0; ni < nneigh; ++ni) {
                    const std::int64_t viz_idx = neigh[ni].idx;
                    const std::uint64_t edge_bit = neigh[ni].edge_bit;

                    const int vv = get_site(viz_idx);

                    if (vv == 0) {
                        continue;
                    }

                    if (vv > 0) {
                        const int neigh_color_idx = value_to_color_index(num_colors, vv);
                        if (neigh_color_idx != cor_idx) continue;

                        if (!mark_tested_bond(edge_bit)) continue;

                        if (rng.uniform_real(0.0, 1.0) < p_curr[cor_idx]) {
                            edge_pairs.emplace_back(static_cast<std::uint64_t>(idx),
                                                    static_cast<std::uint64_t>(viz_idx));
                            edge_pairs.emplace_back(static_cast<std::uint64_t>(viz_idx),
                                                    static_cast<std::uint64_t>(idx));
                        }
                        continue;
                    }

                    const bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    if (!mark_tested_bond(edge_bit)) continue;

                    if (rng.uniform_real(0.0, 1.0) < p_curr[cor_idx]) {
                        front_candidates.add(viz_idx, FrontCandidate{idx, cor_idx, edge_bit});
                    }
                }
            }

            for (std::size_t slot = 0; slot < front_candidates.touched_targets.size(); ++slot) {
                const std::int64_t viz = front_candidates.touched_targets[slot];
                const int n_cand = static_cast<int>(front_candidates.counts[slot]);
                if (n_cand <= 0) continue;

                const int target_before = get_site(viz);
                const bool target_without_species = (num_colors > 1 && target_before == -1);

                int cor_idx = -1;
                std::int64_t chosen_u = -1;

                if (target_without_species) {
                    const int pick = rng.uniform_int(0, n_cand - 1);
                    const FrontCandidate chosen = front_candidates.candidate(slot, pick);
                    cor_idx = chosen.color_idx;
                    chosen_u = static_cast<std::int64_t>(chosen.activator);
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
                            chosen_u = static_cast<std::int64_t>(fc.activator);
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
                record_height_activation(cor_idx, viz);

                const int h = grid.grow_coord(viz);
                if (h > max_heights[cor_idx]) {
                    max_heights[cor_idx] = h;
                }

                if (stop_config.stop_at_percolation &&
                    !percolated[cor_idx] && h >= percolation_height) {
                    percolated[cor_idx] = true;
                    percolation_rank[cor_idx] = ++order_ctr;
                    z_max_at_perc[cor_idx] = max_heights[cor_idx];
                }

                for (int i = 0; i < n_cand; ++i) {
                    const FrontCandidate fc = front_candidates.candidate(slot, i);
                    if (fc.color_idx != cor_idx) continue;
                    edge_pairs.emplace_back(fc.activator,
                                            static_cast<std::uint64_t>(viz));
                    edge_pairs.emplace_back(static_cast<std::uint64_t>(viz),
                                            fc.activator);
                }
            }
        }

        for (int c = 0; c < num_colors; ++c) {
            f_current[c] = static_cast<double>(N_current[c]) / norm_factor;
        }

        if (t < 10 || t % progress_interval == 0) {
            std::cout << "[" << type_percolation << "] t=" << t
                      << ", frontier=" << next_frontier.size();
            for (int c = 0; c < num_colors; ++c) {
                std::cout
                    << ", p" << (c + 1) << "=" << p_curr[c]
                    << ", f" << (c + 1) << "=" << f_current[c]
                    << ", h" << (c + 1) << "=" << max_heights[c]
                    << "/" << grid.grow_top_coord();
            }
            std::cout << '\n';
        }

        commit_step(t, p_curr, f_current);

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
        bool all_finished = true;

        for (int c = 0; c < num_colors; ++c) {
            if (!finished[c]) {
                all_finished = false;
            }

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

            if (!(stop_config.stop_at_percolation && percolated[c]) && !died[c]) {
                all_terminal = false;
            }
        }

        const bool stop_all_dead = all_dead || all_finished;
        const bool stop_all_percolated =
            stop_config.stop_at_percolation && all_percolated;
        const bool stop_partial_percolation =
            stop_config.stop_at_percolation &&
            all_terminal && any_percolated && any_dead;

        bool stop_equilibrated = false;
        if (stop_config.stop_at_equilibrium) {
            bool any_running = false;
            bool all_running_equilibrated = true;
            constexpr int teq_window_block = 10;
            const double teq_rel_tol = growth_test_effective_rel_tol(
                stop_config.equilibrium_rel_tol, num_colors);
            const int teq_min_stable_steps =
                growth_test_global_min_stable_steps(
                    stop_config.equilibrium_consecutive_steps,
                    teq_window_block);
            const int teq_validation_window_steps =
                growth_test_global_validation_window_steps(
                    stop_config.post_equilibrium_extra_steps,
                    teq_window_block);
            for (int c = 0; c < num_colors; ++c) {
                if (finished[c]) continue;
                any_running = true;

                if (!species_equilibrated[static_cast<std::size_t>(c)] &&
                    t % teq_window_block == 0) {
                    const bool use_front_fraction_teq = (std::abs(c_value) <= 1e-12);
                    const auto& series_to_check = use_front_fraction_teq
                        ? f_series[static_cast<std::size_t>(c)]
                        : p_series[static_cast<std::size_t>(c)];
                    const bool reached_global_height = use_front_fraction_teq
                        ? true
                        : (!stop_config.dynamic_height ||
                           max_heights[static_cast<std::size_t>(c)] >= lenght_network);
                    const double t_eq_candidate = reached_global_height
                        ? estimate_t_eq_from_series(
                            t_list,
                            series_to_check,
                            15,
                            teq_window_block,
                            teq_min_stable_steps,
                            teq_rel_tol,
                            stop_config.equilibrium_abs_tol,
                            1.0e-5,
                            false,
                            0.0,
                            teq_validation_window_steps,
                            0u)
                        : std::numeric_limits<double>::quiet_NaN();
                    if (std::isfinite(t_eq_candidate)) {
                        species_equilibrated[static_cast<std::size_t>(c)] = true;
                        t_eq_online_by_species[static_cast<std::size_t>(c)] =
                            t_eq_candidate;
                        z_stat_by_species[static_cast<std::size_t>(c)] =
                            max_heights[static_cast<std::size_t>(c)];
                        equilibrium_detection_time_by_species[static_cast<std::size_t>(c)] = t;
                        post_equilibrium_sampling_stop_height_by_species[static_cast<std::size_t>(c)] =
                            z_stat_by_species[static_cast<std::size_t>(c)] +
                            post_equilibrium_sampling_height_increment;
                        post_equilibrium_stop_height_by_species[static_cast<std::size_t>(c)] =
                            z_stat_by_species[static_cast<std::size_t>(c)] +
                            post_equilibrium_height_increment;
                        next_surface_sample_time_by_species[static_cast<std::size_t>(c)] =
                            t + surface_deltaT;
                    }
                }

                if (!species_equilibrated[static_cast<std::size_t>(c)]) {
                    all_running_equilibrated = false;
                    continue;
                }

                const int next_surface_t =
                    next_surface_sample_time_by_species[static_cast<std::size_t>(c)];
                const int sampling_stop_height =
                    post_equilibrium_sampling_stop_height_by_species[static_cast<std::size_t>(c)];
                const bool reached_sampling_height =
                    sampling_stop_height >= 0 &&
                    max_heights[static_cast<std::size_t>(c)] >= sampling_stop_height;
                const bool should_sample_surface =
                    stop_config.save_surface_observables &&
                    !surface_sampling_finished_by_species[static_cast<std::size_t>(c)] &&
                    next_surface_t >= 0 &&
                    (t >= next_surface_t || reached_sampling_height);
                if (should_sample_surface) {
                    const int analysis_bottom =
                        z_stat_by_species[static_cast<std::size_t>(c)];
                    const int analysis_top = sampling_stop_height;
                    const int sample_top = std::min(
                        max_heights[static_cast<std::size_t>(c)],
                        analysis_top);
                    std::vector<int> current_surface_heights =
                        compute_upper_surface_heights(
                            grid,
                            num_colors,
                            c,
                            site_state,
                            analysis_bottom,
                            sample_top);
                    std::vector<int>& previous_surface_heights =
                        previous_surface_heights_by_species[static_cast<std::size_t>(c)];
                    if (!previous_surface_heights.empty()) {
                        volume_sur_by_species[static_cast<std::size_t>(c)].push_back(
                            compute_volume_fraction_between_upper_surfaces(
                                grid,
                                num_colors,
                                c,
                                site_state,
                                previous_surface_heights,
                                current_surface_heights,
                                analysis_bottom,
                                analysis_top));
                        t_volume_sur_by_species[static_cast<std::size_t>(c)].push_back(
                            static_cast<int>(
                                t_volume_sur_by_species[static_cast<std::size_t>(c)].size()) + 1);
                    }

                    t_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        static_cast<int>(
                            t_sur_by_species[static_cast<std::size_t>(c)].size()) + 1);
                    w_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        compute_surface_width(current_surface_heights));
                    grad_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        compute_surface_gradient(grid, current_surface_heights));
                    h_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        current_surface_heights);
                    const double f_sur_value =
                        compute_upper_surface_fraction(
                            grid,
                            num_colors,
                            c,
                            site_state,
                            sample_top,
                            analysis_bottom);
                    f_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        f_sur_value);
                    S_sur_by_species[static_cast<std::size_t>(c)].push_back(
                        f_sur_value * compute_lateral_area(grid));
                    if (reached_sampling_height) {
                        surface_sampling_finished_by_species[static_cast<std::size_t>(c)] = true;
                        next_surface_sample_time_by_species[static_cast<std::size_t>(c)] = -1;
                    } else {
                        next_surface_sample_time_by_species[static_cast<std::size_t>(c)] =
                            t + surface_deltaT;
                    }
                    previous_surface_heights = std::move(current_surface_heights);
                }
            }
            if (any_running && all_running_equilibrated) {
                bool all_reached_post_equilibrium_height = true;
                for (int c = 0; c < num_colors; ++c) {
                    if (finished[c]) continue;
                    const int target_height =
                        post_equilibrium_stop_height_by_species[static_cast<std::size_t>(c)];
                    if (target_height < 0 ||
                        max_heights[static_cast<std::size_t>(c)] < target_height) {
                        all_reached_post_equilibrium_height = false;
                        break;
                    }
                }
                stop_equilibrated =
                    all_reached_post_equilibrium_height &&
                    t % teq_window_block == 0;
            }
        }

        const bool stop_hard_limit =
            (stop_config.stop_at_equilibrium || stop_config.hard_max_steps > 0) &&
            (t >= hard_max_steps);
        bool stop_stationary_dynamics = false;

        if (stop_all_dead || stop_all_percolated ||
            stop_partial_percolation || stop_equilibrated ||
            stop_stationary_dynamics || stop_hard_limit) {
            growth_test_stop_time = t;
            if (stop_all_dead) {
                growth_test_stop_reason = all_dead ? "all_dead" : "all_finished";
            } else if (stop_all_percolated) {
                growth_test_stop_reason = "all_percolated";
            } else if (stop_partial_percolation) {
                growth_test_stop_reason = "partial_percolation";
            } else if (stop_equilibrated) {
                growth_test_stop_reason = "post_equilibrium_height_reached";
            } else if (stop_stationary_dynamics) {
                growth_test_stop_reason = "stationary_dynamics";
            } else if (stop_hard_limit) {
                growth_test_stop_reason = "hard_max_steps";
            }
            if (calculate_detailed_properties) {
                finalize_global_metrics();
            } else {
                finalize_time_series_only();
            }
            break;
        }

        std::fill(p_next.begin(), p_next.end(), 0.0);
        for (int c = 0; c < num_colors; ++c) {
            p_next[c] = finished[c]
                ? p_curr[c]
                : generate_p(
                    type_f_T, p_curr[c], t, f_current[c],
                    c_value, f_T, a, alpha,
                    stop_config.feedback_control_rule);
        }

        frontier.swap(next_frontier);
        p_curr.swap(p_next);
    }

    for (int c = 0; c < num_colors; ++c) {
        if (!died[static_cast<std::size_t>(c)] &&
            !species_equilibrated[static_cast<std::size_t>(c)]) {
            const bool use_front_fraction_teq = (std::abs(c_value) <= 1e-12);
            const auto& series_to_check = use_front_fraction_teq
                ? f_series[static_cast<std::size_t>(c)]
                : p_series[static_cast<std::size_t>(c)];
            const bool reached_global_height = use_front_fraction_teq
                ? true
                : (!stop_config.dynamic_height ||
                   max_heights[static_cast<std::size_t>(c)] >= lenght_network);
            if (reached_global_height) {
                constexpr int teq_window_block = 10;
                const double teq_rel_tol = growth_test_effective_rel_tol(
                    stop_config.equilibrium_rel_tol, num_colors);
                const int teq_min_stable_steps =
                    growth_test_global_min_stable_steps(
                        stop_config.equilibrium_consecutive_steps,
                        teq_window_block);
                const int teq_validation_window_steps =
                    growth_test_global_validation_window_steps(
                        stop_config.post_equilibrium_extra_steps,
                        teq_window_block);
                const double t_eq_candidate = estimate_t_eq_from_series(
                    t_list,
                    series_to_check,
                    15,
                    teq_window_block,
                    teq_min_stable_steps,
                    teq_rel_tol,
                    stop_config.equilibrium_abs_tol,
                    1.0e-5,
                    false,
                    0.0,
                    teq_validation_window_steps,
                    0u);
                if (std::isfinite(t_eq_candidate)) {
                    species_equilibrated[static_cast<std::size_t>(c)] = true;
                    t_eq_online_by_species[static_cast<std::size_t>(c)] = t_eq_candidate;
                    z_stat_by_species[static_cast<std::size_t>(c)] =
                        max_heights[static_cast<std::size_t>(c)];
                }
            }
        }
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.f_t = std::move(f_series);
    ts_out.y_mean_t = std::move(y_mean_series);
    ts_out.y_width_t = std::move(y_width_series);
    ts_out.y_max_t = std::move(y_max_series);
    ts_out.y_front_mean_t = std::move(y_front_mean_series);
    ts_out.y_front_width_t = std::move(y_front_width_series);
    ts_out.y_front_count_t = std::move(y_front_count_series);
    ts_out.t   = std::move(t_list);

    if (!global_metrics_finalized) {
        if (calculate_detailed_properties) {
            finalize_global_metrics();
        } else {
            finalize_time_series_only();
        }
    }

    if (stop_config.stop_at_equilibrium && !stop_config.stop_at_percolation) {
        ps_out.t_eq_by_species = t_eq_online_by_species;
        for (int c = 0; c < num_colors; ++c) {
            if (died[static_cast<std::size_t>(c)] ||
                !species_equilibrated[static_cast<std::size_t>(c)]) {
                ps_out.t_eq_by_species[static_cast<std::size_t>(c)] =
                    std::numeric_limits<double>::quiet_NaN();
                ps_out.z_stat_by_species[static_cast<std::size_t>(c)] = -1;
            }
        }
        ts_out.t_eq = any_value_finite(ps_out.t_eq_by_species)
            ? mean_finite_values(ps_out.t_eq_by_species)
            : std::numeric_limits<double>::quiet_NaN();
    } else {
        ps_out.t_eq_by_species.clear();
        ts_out.t_eq = estimate_t_eq_from_timeseries(
            ts_out,
            15,      // window_roll
            10,      // window_block
            15,      // min stable steps
            2.5e-2,  // relative block-change tolerance
            1.0e-6,  // absolute block-change tolerance
            1.0e-5   // |s_prime| threshold
        );
    }
    ps_out.t_eq = ts_out.t_eq;

    if (stop_config.stop_at_equilibrium && !stop_config.stop_at_percolation) {
        set_species_order_from_t_eq(
            ps_out,
            ps_out.t_eq_by_species,
            growth_test_stop_reason == "hard_max_steps");
    }

    if (stop_config.stop_at_equilibrium && stop_config.save_surface_observables) {
        ps_out.f_sur_by_species = std::move(f_sur_by_species);
        ps_out.t_sur_by_species = std::move(t_sur_by_species);
        ps_out.volume_sur_by_species = std::move(volume_sur_by_species);
        ps_out.t_volume_sur_by_species = std::move(t_volume_sur_by_species);
        ps_out.h_sur_by_species = std::move(h_sur_by_species);
        ps_out.w_sur_by_species = std::move(w_sur_by_species);
        ps_out.grad_sur_by_species = std::move(grad_sur_by_species);
        ps_out.M_L_by_species.assign(
            static_cast<std::size_t>(num_colors),
            std::numeric_limits<double>::quiet_NaN());
        ps_out.M_cluster_sizes_by_species.assign(
            static_cast<std::size_t>(num_colors),
            std::vector<std::int64_t>{});
        for (int c = 0; c < num_colors; ++c) {
            const int z_stat = z_stat_by_species[static_cast<std::size_t>(c)];
            if (z_stat >= 0) {
                const int analysis_top = z_stat + post_equilibrium_sampling_height_increment;
                const int analysis_bottom = z_stat;
                SlabClusterMassResult cluster_masses =
                    compute_slab_cluster_masses(
                        grid,
                        num_colors,
                        c,
                        site_state,
                        analysis_bottom,
                        analysis_top);
                ps_out.M_L_by_species[static_cast<std::size_t>(c)] =
                    cluster_masses.largest_spanning_mass;
                ps_out.M_cluster_sizes_by_species[static_cast<std::size_t>(c)] =
                    std::move(cluster_masses.cluster_sizes);
            }
        }
        ps_out.S_sur_by_species = std::move(S_sur_by_species);
        ps_out.surface_deltaT = surface_deltaT;
        ps_out.surface_observables_enabled = true;
    } else {
        ps_out.f_sur_by_species.clear();
        ps_out.t_sur_by_species.clear();
        ps_out.volume_sur_by_species.clear();
        ps_out.t_volume_sur_by_species.clear();
        ps_out.h_sur_by_species.clear();
        ps_out.w_sur_by_species.clear();
        ps_out.grad_sur_by_species.clear();
        ps_out.S_sur_by_species.clear();
        ps_out.M_L_by_species.clear();
        ps_out.M_cluster_sizes_by_species.clear();
        ps_out.surface_deltaT = -1;
        ps_out.surface_observables_enabled = false;
    }

    if (calculate_detailed_properties) {
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
        if (stop_config.stop_at_equilibrium && !stop_config.stop_at_percolation) {
            ps_out.M_size_at_perc.clear();
            for (const int color_1b : ps_out.color_percolation) {
                const int c = color_1b - 1;
                if (c < 0 || c >= static_cast<int>(largest_comp_cache.size())) {
                    ps_out.M_size_at_perc.push_back(-1);
                    continue;
                }
                ps_out.M_size_at_perc.push_back(largest_comp_cache[c].size);
            }
        }

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

            const std::vector<std::int64_t>& path = ps_out.sp_path_lin[c];
            if (!path.empty()) {
                // mantém o nó inicial da base em ambos os caminhos, se você quiser
                // preservar a referência completa do caminho original
                ps_out.sp_path_lin_preteq[c].push_back(path.front());
                ps_out.sp_path_lin_posteq[c].push_back(path.front());
            }

            for (std::size_t k = 1; k < path.size(); ++k) {
                const std::int64_t idx = path[k];

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
    ps_out.feedback_control_param = c_value;

    if (stop_config.stop_at_equilibrium) {
        if (stop_config.stop_at_percolation) {
            ps_out.z_max_at_perc = std::move(z_max_at_perc);
        } else {
            ps_out.z_max_at_perc.clear();
        }
        ps_out.z_max_final = max_heights;
        ps_out.z_stat_by_species = z_stat_by_species;
        ps_out.species_final_status.assign(
            static_cast<std::size_t>(num_colors), 0);
        for (int c = 0; c < num_colors; ++c) {
            if (species_equilibrated[static_cast<std::size_t>(c)]) {
                ps_out.species_final_status[static_cast<std::size_t>(c)] = 1;
            } else if (died[static_cast<std::size_t>(c)]) {
                ps_out.species_final_status[static_cast<std::size_t>(c)] = -1;
            }
        }
        ps_out.growth_test_stop_reason = growth_test_stop_reason;
        ps_out.growth_test_stop_time = growth_test_stop_time;
        ps_out.equilibrium_consecutive_steps =
            stop_config.equilibrium_consecutive_steps;
        ps_out.dynamics_window_steps = stop_config.dynamics_window_steps;
        ps_out.post_equilibrium_extra_steps =
            stop_config.post_equilibrium_extra_steps;
        ps_out.dynamic_min_stop_height = dynamic_min_stop_height;
        ps_out.dynamic_max_stop_height = dynamic_max_stop_height;
        ps_out.equilibrium_rel_tol = stop_config.equilibrium_rel_tol;
        ps_out.equilibrium_abs_tol = stop_config.equilibrium_abs_tol;
    } else {
        ps_out.z_max_at_perc.clear();
        ps_out.z_max_final.clear();
        ps_out.z_stat_by_species.clear();
        ps_out.species_final_status.clear();
        ps_out.growth_test_stop_reason.clear();
        ps_out.growth_test_stop_time = -1;
        ps_out.equilibrium_consecutive_steps = -1;
        ps_out.dynamics_window_steps = -1;
        ps_out.post_equilibrium_extra_steps = -1;
        ps_out.dynamic_min_stop_height = -1;
        ps_out.dynamic_max_stop_height = -1;
        ps_out.equilibrium_rel_tol = std::numeric_limits<double>::quiet_NaN();
        ps_out.equilibrium_abs_tol = std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<std::int8_t>().swap(site_state);
    std::vector<std::int64_t>().swap(frontier);
    std::vector<std::int64_t>().swap(next_frontier);
    std::vector<ComponentSummary>().swap(largest_comp_cache);
    SparseFrontCandidates empty_front_candidates;
    std::swap(front_candidates, empty_front_candidates);

    const std::vector<int> final_shape = (dim == 2)
        ? std::vector<int>{grid.SX, grid.SY}
        : std::vector<int>{grid.SX, grid.SY, grid.SZ};
    NetworkPattern net_animation(dim, final_shape, num_colors, rho, false);
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

    if (static_cast<std::int64_t>(encoded_net.data.size()) != grid.total_size) {
        throw std::runtime_error(
            "[filter_percolating_clusters_from_encoded] tamanho de data inconsistente com shape");
    }

    NetworkPattern out(encoded_net.dim, encoded_net.shape,
                       encoded_net.num_colors, encoded_net.rho);
    out.seed = encoded_net.seed;
    std::fill(out.data.begin(), out.data.end(), static_cast<NetworkPattern::state_t>(-1));

    const int num_colors = encoded_net.num_colors;
    const int top_coord = (encoded_net.dim == 2) ? (grid.SY - 1) : (grid.SZ - 1);

    auto color_idx_from_encoded = [&](const std::int64_t idx) -> int
    {
        const long long code =
            static_cast<long long>(encoded_net.data[static_cast<std::size_t>(idx)]);

        const DecodedValue dv = decode_animation_value(code, ANIMATION_SPECIES_FACTOR);
        if (dv.never_activated || dv.blocked) return -1;
        if (dv.color_idx < 0 || dv.color_idx >= num_colors) return -1;
        return dv.color_idx;
    };

    std::vector<char> visited(static_cast<std::size_t>(grid.total_size), 0);
    std::vector<std::int64_t> stack;
    std::vector<std::int64_t> component;

    for (std::int64_t start = 0; start < grid.total_size; ++start) {
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
            const std::int64_t u = stack.back();
            stack.pop_back();

            component.push_back(u);

            const int h = grid.grow_coord(u);
            if (h == 0) touches_base = true;
            if (h == top_coord) touches_top = true;

            grid.for_each_neighbor(u, [&](const std::int64_t v) {
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

        for (const std::int64_t idx : component) {
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
