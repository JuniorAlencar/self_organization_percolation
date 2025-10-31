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
    } else { // dim == 3
        const int plane = SX * SY;
        z = lin / plane;
        int rem = lin % plane;
        y = rem / SX;
        x = rem % SX;
    }
}

static inline bool apply_bc_and_validate(std::vector<int>& v, int dim, const std::vector<int>& shape, int grow_axis) {
    for (int j = 0; j < dim; ++j) {
        if (j == grow_axis) {
            if (v[j] < 0 || v[j] >= shape[j]) return false; // aberto no grow_axis
        } else {
            if (v[j] < 0) v[j] = shape[j] - 1;
            else if (v[j] >= shape[j]) v[j] = 0;            // periódico nos demais
        }
    }
    return true;
}

static inline int color_index_from_val(int v, int num_colors) {
    if (v <= 0) return -1;               // inativo/checado
    if (num_colors == 1) return 0;       // ativo (1)
    return std::abs(v) - 2;              // ativos {+2,+3,...} -> {0,1,...}
}


// ===== Shortest path base->topo =====
void BiggestComponent::compute_shortest_paths_to_base(
    const NetworkPattern&                           net,
    int                                             dim,
    const std::vector<int>&                         shape,
    int                                             grow_axis,
    int                                             num_colors,
    const std::vector<std::vector<int>>&            parent,
    PercolationSeries&                              ps_out
) {
    const int SX = sx(shape);
    const int SY = sy(shape, dim);
    const int SZ = sz(shape, dim);
    const int GRID_N = SX * SY * SZ;

    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, {});

    const int top_level = shape[grow_axis] - 1;

    for (int c = 0; c < num_colors; ++c) {
        int top_lin = -1;

        // Varre camada do topo (grow_axis = L-1) procurando um nó ativo da cor c
        if (dim == 2) {
            for (int y = 0; y < SY; ++y)
                for (int x = 0; x < SX; ++x) {
                    int xx = x, yy = y;
                    if (grow_axis == 0) xx = top_level;
                    else                yy = top_level;
                    int lin = lin_index(shape, dim, xx, yy, 0);
                    if (color_index_from_val(net.get(lin), num_colors) == c) { top_lin = lin; break; }
                }
        } else {
            for (int z = 0; z < SZ; ++z)
                for (int y = 0; y < SY; ++y)
                    for (int x = 0; x < SX; ++x) {
                        int xx = x, yy = y, zz = z;
                        if      (grow_axis == 0) xx = top_level;
                        else if (grow_axis == 1) yy = top_level;
                        else                     zz = top_level;
                        int lin = lin_index(shape, dim, xx, yy, zz);
                        if (color_index_from_val(net.get(lin), num_colors) == c) { top_lin = lin; goto found_top; }
                    }
        }
found_top:

        if (top_lin == -1 || parent[c].empty()) { ps_out.sp_len[c] = -1; continue; }

        // Backtracking topo -> base via parent[c]
        std::vector<int> path;
        int cur = top_lin;
        const auto& P = parent[c];

        if (cur < 0 || cur >= GRID_N) { ps_out.sp_len[c] = -1; continue; }

        path.push_back(cur);
        while (true) {
            int par = P[cur];
            if (par == -1) break;          // semente/base
            if (par < 0)  { path.clear(); break; } // inconsistente
            path.push_back(par);
            cur = par;
        }
        if (path.empty()) { ps_out.sp_len[c] = -1; continue; }
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

    auto level_of = [&](int lin)->int{
        int x,y,z; unravel(shape, dim, lin, x, y, z);
        return (grow_axis == 0 ? x : (grow_axis == 1 ? y : z));
    };

    auto neighbors = [&](int lin, std::vector<int>& out){
        out.clear();
        int x,y,z; unravel(shape, dim, lin, x, y, z);
        for (int ax = 0; ax < dim; ++ax) {
            for (int delta : {-1, 1}) {
                std::vector<int> v = {x, y, z};
                v[ax] += delta;
                if (!apply_bc_and_validate(v, dim, shape, grow_axis)) continue;
                out.push_back(lin_index(shape, dim, v[0], v[1], v[2]));
            }
        }
    };

    std::vector<unsigned char> seen((size_t)GRID_N, 0);
    std::vector<int> q; q.reserve(1024);
    std::vector<int> neigh; neigh.reserve(2 * dim);

    for (int c = 0; c < num_colors; ++c) {
        std::fill(seen.begin(), seen.end(), 0);
        int best_c = 0;

        for (int lin = 0; lin < GRID_N; ++lin) {
            if (seen[lin]) continue;
            int v = net.get(lin);
            if (color_index_from_val(v, num_colors) != c) continue;

            // BFS da componente desta cor
            q.clear();
            q.push_back(lin);
            seen[lin] = 1;

            int comp_size = 0;
            bool touches_base = false;
            bool touches_top  = false;

            while (!q.empty()) {
                int u = q.back(); q.pop_back();
                ++comp_size;

                int h = level_of(u);
                if (h == 0)     touches_base = true;
                if (h == L - 1) touches_top  = true;

                neighbors(u, neigh);
                for (int w : neigh) {
                    if (seen[w]) continue;
                    int vw = net.get(w);
                    if (color_index_from_val(vw, num_colors) != c) continue;
                    seen[w] = 1;
                    q.push_back(w);
                }
            }

            // Só considera se a componente realmente percola (base & topo)
            if (touches_base && touches_top)
                best_c = std::max(best_c, comp_size);
        }

        best[c] = best_c; // 0 se nenhuma componente percolar
    }

    return best;
}

