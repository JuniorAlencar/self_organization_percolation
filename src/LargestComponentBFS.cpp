#include "LargestComponentBFS.hpp"
#include <utility>
#include <cstddef>

int BiggestComponent::lin_index(int dim, int SX, int SY, int /*SZ*/,
                                int x, int y, int z) {
    if (dim == 2) return y * SX + x;
    return (z * SY + y) * SX + x;
}

void BiggestComponent::unlin(int dim, int SX, int SY, int /*SZ*/,
                             int id, int& x, int& y, int& z) {
    if (dim == 2) {
        x = id % SX; y = id / SX; z = 0;
    } else {
        x = id % SX; int t = id / SX; y = t % SY; z = t / SY;
    }
}

bool BiggestComponent::is_active_same_color(
    int /*dim*/, int /*SX*/, int /*SY*/, int /*SZ*/,
    int x, int y, int z,
    const std::function<int(int,int,int)>& get_val,
    int active_val
) {
    return get_val(x,y,z) == active_val;
}

BFSResult BiggestComponent::bfs_component_from_seed(
    int dim, int SX, int SY, int SZ,
    int seed_id, int active_val,
    const std::function<int(int,int,int)>& get_val,
    const std::function<bool(int&,int&,int&)>& valid_coord,  // <<<
    std::vector<char>& visited
) {
    BFSResult res;
    const int N = SX * SY * (dim == 3 ? SZ : 1);
    if (seed_id < 0 || seed_id >= N) return res;

    int sx, sy, sz;
    unlin(dim, SX, SY, SZ, seed_id, sx, sy, sz);
    if (!is_active_same_color(dim, SX, SY, SZ, sx, sy, sz, get_val, active_val))
        return res;

    std::vector<int> q; q.reserve(1024);
    visited[seed_id] = 1;
    q.push_back(seed_id);

    const int L = (dim == 3 ? SZ : SY);

    auto try_push = [&](int nx, int ny, int nz) {
        // wrap + checagem via valid_coord (pode MODIFICAR nx,ny,nz)
        if (!valid_coord(nx,ny,nz)) return;
        // Agora nx,ny,nz já estão dentro dos limites
        int nid = lin_index(dim, SX, SY, SZ, nx, ny, nz);
        if (nid < 0 || nid >= N) return;
        if (visited[nid]) return;
        if (!is_active_same_color(dim, SX, SY, SZ, nx, ny, nz, get_val, active_val)) return;

        visited[nid] = 1;
        q.push_back(nid);
    };

    for (size_t head = 0; head < q.size(); ++head) {
        int u = q[head];
        res.nodes.push_back(u);
        ++res.size;

        int ux, uy, uz;
        unlin(dim, SX, SY, SZ, u, ux, uy, uz);

        int gc = (dim == 3 ? uz : uy);
        if (gc < res.zmin) res.zmin = gc;
        if (gc > res.zmax) res.zmax = gc;
        if (gc == 0)     res.touches_base = true;
        if (gc == L - 1) res.touches_top  = true;

        // Vizinhos cartesianos (±1)
        try_push(ux - 1, uy, uz);
        try_push(ux + 1, uy, uz);

        if (dim >= 2) {
            try_push(ux, uy - 1, uz);
            try_push(ux, uy + 1, uz);
        }
        if (dim == 3) {
            try_push(ux, uy, uz - 1);
            try_push(ux, uy, uz + 1);
        }
    }
    return res;
}

BFSResult BiggestComponent::find_largest_component_for_color(
    int dim, int SX, int SY, int SZ,
    const std::function<int(int,int,int)>& get_val,
    const std::function<bool(int&,int&,int&)>& valid_coord,  // <<<
    int active_val
) {
    BFSResult best;
    const int N = SX * SY * (dim == 3 ? SZ : 1);
    std::vector<char> visited((size_t)N, 0);

    for (int id = 0; id < N; ++id) {
        if (visited[id]) continue;

        int x, y, z;
        unlin(dim, SX, SY, SZ, id, x, y, z);
        if (!is_active_same_color(dim, SX, SY, SZ, x, y, z, get_val, active_val))
            continue;

        BFSResult cur = bfs_component_from_seed(
            dim, SX, SY, SZ, id, active_val, get_val, valid_coord, visited
        );
        if (cur.size > best.size) best = std::move(cur);
    }
    return best;
}
