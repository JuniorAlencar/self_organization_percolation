#include "LargestComponentBFS.hpp"

// ===== Helpers =====
static inline int sx(const std::vector<int>& shape) { return shape[0]; }
static inline int sy(const std::vector<int>& shape, int dim) { return (dim >= 2 ? shape[1] : 1); }
static inline int sz(const std::vector<int>& shape, int dim) { return (dim == 3 ? shape[2] : 1); }

static inline int lin_index(const std::vector<int>& shape, int dim, int x, int y, int z) {
    const int SX = sx(shape);
    if (dim == 2) return x + SX * y;
    const int SY = sy(shape, dim);
    return x + SX * (y + SY * z);
}

static inline void unravel(const std::vector<int>& shape, int dim, int lin, int& x, int& y, int& z) {
    const int SX = sx(shape);
    const int SY = sy(shape, dim);
    if (dim == 2) {
        z = 0;
        y = lin / SX;
        x = lin % SX;
    } else {
        const int plane = SX * SY;
        z = lin / plane;
        int rem = lin % plane;
        y = rem / SX;
        x = rem % SX;
    }
}

static inline int color_index_from_val(int v, int num_colors) {
    if (v <= 0) return -1;
    if (num_colors == 1) return 0;
    return std::abs(v) - 2;
}

static inline int level_of_lin(const std::vector<int>& shape, int dim, int grow_axis, int lin) {
    int x, y, z;
    unravel(shape, dim, lin, x, y, z);
    return (grow_axis == 0 ? x : (grow_axis == 1 ? y : z));
}

static inline void append_neighbors_lin(
    int lin,
    int dim,
    const std::vector<int>& shape,
    int grow_axis,
    std::vector<int>& out
) {
    out.clear();

    const int SX = sx(shape);
    const int SY = sy(shape, dim);
    const int SZ = sz(shape, dim);
    const int plane = SX * SY;

    int x, y, z;
    unravel(shape, dim, lin, x, y, z);

    // eixo x
    if (dim >= 1) {
        if (grow_axis == 0) {
            if (x > 0)      out.push_back(lin_index(shape, dim, x - 1, y, z));
            if (x + 1 < SX) out.push_back(lin_index(shape, dim, x + 1, y, z));
        } else {
            const int xm = (x == 0 ? SX - 1 : x - 1);
            const int xp = (x + 1 == SX ? 0 : x + 1);
            out.push_back(lin_index(shape, dim, xm, y, z));
            out.push_back(lin_index(shape, dim, xp, y, z));
        }
    }

    // eixo y
    if (dim >= 2) {
        if (grow_axis == 1) {
            if (y > 0)      out.push_back(lin_index(shape, dim, x, y - 1, z));
            if (y + 1 < SY) out.push_back(lin_index(shape, dim, x, y + 1, z));
        } else {
            const int ym = (y == 0 ? SY - 1 : y - 1);
            const int yp = (y + 1 == SY ? 0 : y + 1);
            out.push_back(lin_index(shape, dim, x, ym, z));
            out.push_back(lin_index(shape, dim, x, yp, z));
        }
    }

    // eixo z
    if (dim == 3) {
        if (grow_axis == 2) {
            if (z > 0)      out.push_back(lin_index(shape, dim, x, y, z - 1));
            if (z + 1 < SZ) out.push_back(lin_index(shape, dim, x, y, z + 1));
        } else {
            const int zm = (z == 0 ? SZ - 1 : z - 1);
            const int zp = (z + 1 == SZ ? 0 : z + 1);
            out.push_back(lin_index(shape, dim, x, y, zm));
            out.push_back(lin_index(shape, dim, x, y, zp));
        }
    }
}

// ===== Shortest path base->topo =====
void BiggestComponent::compute_shortest_paths_to_base(
    const NetworkPattern&               net,
    int                                 dim,
    const std::vector<int>&             shape,
    int                                 grow_axis,
    int                                 num_colors,
    const std::vector<int>&             parent,
    PercolationSeries&                  ps_out
) {
    const int SX = sx(shape);
    const int SY = sy(shape, dim);
    const int SZ = sz(shape, dim);
    const int GRID_N = SX * SY * SZ;

    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, {});

    if ((int)parent.size() != GRID_N) {
        return;
    }

    const int top_level = shape[grow_axis] - 1;

    for (int c = 0; c < num_colors; ++c) {
        int top_lin = -1;

        if (dim == 2) {
            for (int y = 0; y < SY && top_lin == -1; ++y) {
                for (int x = 0; x < SX; ++x) {
                    const int xx = (grow_axis == 0 ? top_level : x);
                    const int yy = (grow_axis == 1 ? top_level : y);
                    const int lin = lin_index(shape, dim, xx, yy, 0);
                    if (color_index_from_val(net.get(lin), num_colors) == c) {
                        top_lin = lin;
                        break;
                    }
                }
            }
        } else {
            for (int z = 0; z < SZ && top_lin == -1; ++z) {
                for (int y = 0; y < SY && top_lin == -1; ++y) {
                    for (int x = 0; x < SX; ++x) {
                        const int xx = (grow_axis == 0 ? top_level : x);
                        const int yy = (grow_axis == 1 ? top_level : y);
                        const int zz = (grow_axis == 2 ? top_level : z);
                        const int lin = lin_index(shape, dim, xx, yy, zz);
                        if (color_index_from_val(net.get(lin), num_colors) == c) {
                            top_lin = lin;
                            break;
                        }
                    }
                }
            }
        }

        if (top_lin == -1 || top_lin < 0 || top_lin >= GRID_N) {
            continue;
        }

        std::vector<int> path;
        path.reserve(shape[grow_axis]);

        int cur = top_lin;
        while (true) {
            if (cur < 0 || cur >= GRID_N) {
                path.clear();
                break;
            }

            if (color_index_from_val(net.get(cur), num_colors) != c) {
                path.clear();
                break;
            }

            path.push_back(cur);

            const int par = parent[cur];
            if (par == -1) {
                break;
            }
            if (par < 0 || par >= GRID_N) {
                path.clear();
                break;
            }

            cur = par;
        }

        if (path.empty()) {
            continue;
        }

        std::reverse(path.begin(), path.end());
        ps_out.sp_len[c] = static_cast<int>(path.size());
        ps_out.sp_path_lin[c] = std::move(path);
    }
}

// ===== Maior cluster QUE PERCOLA por cor =====
std::vector<int> BiggestComponent::largest_cluster_sizes(
    const NetworkPattern&   net,
    int                     dim,
    const std::vector<int>& shape,
    int                     grow_axis,
    int                     num_colors
) {
    const int SX = sx(shape);
    const int SY = sy(shape, dim);
    const int SZ = sz(shape, dim);
    const int GRID_N = SX * SY * SZ;
    const int L = shape[grow_axis];

    std::vector<int> best(num_colors, 0);
    std::vector<unsigned char> seen((size_t)GRID_N, 0);
    std::vector<int> q;
    std::vector<int> neigh;

    q.reserve(4096);
    neigh.reserve(2 * dim);

    for (int c = 0; c < num_colors; ++c) {
        std::fill(seen.begin(), seen.end(), 0);
        int best_c = 0;

        for (int lin = 0; lin < GRID_N; ++lin) {
            if (seen[lin]) continue;
            if (color_index_from_val(net.get(lin), num_colors) != c) continue;

            q.clear();
            q.push_back(lin);
            seen[lin] = 1;

            int comp_size = 0;
            bool touches_base = false;
            bool touches_top = false;

            while (!q.empty()) {
                const int u = q.back();
                q.pop_back();
                ++comp_size;

                const int h = level_of_lin(shape, dim, grow_axis, u);
                if (h == 0) touches_base = true;
                if (h == L - 1) touches_top = true;

                append_neighbors_lin(u, dim, shape, grow_axis, neigh);
                for (int w : neigh) {
                    if (seen[w]) continue;
                    if (color_index_from_val(net.get(w), num_colors) != c) continue;
                    seen[w] = 1;
                    q.push_back(w);
                }
            }

            if (touches_base && touches_top) {
                best_c = std::max(best_c, comp_size);
            }
        }

        best[c] = best_c;
    }

    return best;
}
