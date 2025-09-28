#include "DSU.hpp"

DSU::DSU(int dim_, int Lx_, int Ly_, int Lz_, int grow_axis_)
: dim(dim_), Lx(Lx_), Ly(Ly_>0?Ly_:1), Lz(Lz_>0?Lz_:1), grow_axis(grow_axis_) {
    TOT = static_cast<std::int64_t>(Lx) * static_cast<std::int64_t>(Ly) * static_cast<std::int64_t>(Lz);
    parent.assign(TOT, -1);
    sz.assign(TOT, 0);
    active.assign(TOT, 0);
    touch_base.assign(TOT, 0);
    touch_top.assign(TOT, 0);
    open_edges.reserve(static_cast<size_t>(TOT)); // heurística
}

inline int DSU::id(int x,int y,int z) const {
    return x + Lx * (y + Ly * z);
}

inline void DSU::unid(int idv, int& x, int& y, int& z) const {
    z = (dim==3 ? idv / (Lx*Ly) : 0);
    int r = (dim==3 ? idv - z*(Lx*Ly) : idv);
    y = (dim>=2 ? r / Lx : 0);
    x = (dim>=1 ? r - y*Lx : 0);
}

int DSU::find(int a){
    if (a < 0 || a >= (int)parent.size()) return a;
    return (parent[a]==a ? a : parent[a] = find(parent[a]));
}

void DSU::make_active(int a, int coord_grow, int L){
    if (a < 0 || a >= (int)active.size()) return;
    if (active[a]) return;
    active[a] = 1;
    parent[a] = a;
    sz[a]     = 1;
    touch_base[a] = (coord_grow == 0);
    touch_top[a]  = (coord_grow == L-1);
}

int DSU::unite(int a, int b){
    a = find(a); b = find(b);
    if (a == b) return a;
    if (sz[a] < sz[b]) std::swap(a,b);
    parent[b] = a;
    sz[a] += sz[b];
    touch_base[a] = (touch_base[a] || touch_base[b]);
    touch_top[a]  = (touch_top[a]  || touch_top[b]);
    return a;
}

bool DSU::wrap_and_validate(std::vector<int>& v) const {
    // aberto no grow_axis; periódico nos demais
    for (int j=0;j<dim;++j){
        int Lax = (j==0?Lx:(j==1?Ly:Lz));
        if (j == grow_axis) {
            if (v[j] < 0 || v[j] >= Lax) return false;
        } else {
            if (v[j] < 0) v[j] = Lax-1;
            else if (v[j] >= Lax) v[j] = 0;
        }
    }
    return true;
}

std::uint64_t DSU::edge_key(int a, int b){
    if (a > b) std::swap(a,b);
    // chave 64-bit combinando 2 inteiros 32-bit
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32)
         ^  static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
}

void DSU::open_bond(int a, int b){
    if (a<0 || b<0) return;
    if (!is_active(a) || !is_active(b)) return;
    open_edges.insert(edge_key(a,b));
    // ao abrir a aresta, os componentes passam a estar conectados
    unite(a,b);
}

bool DSU::is_bond_open(int a, int b) const {
    if (a<0 || b<0) return false;
    auto key = edge_key(a,b);
    return (open_edges.find(key) != open_edges.end());
}

void DSU::connect_if_site_adjacent(int a, int b){
    if (is_active(a) && is_active(b)) unite(a,b);
}

void DSU::neighbors(int id0, std::vector<int>& out_ids) const {
    out_ids.clear();
    out_ids.reserve(2*dim);
    int x,y,z;
    unid(id0,x,y,z);
    for (int ax=0; ax<dim; ++ax){
        for (int d : {-1,1}){
            std::vector<int> v = {x, (dim>=2?y:0), (dim==3?z:0)};
            v[ax] += d;
            if (!wrap_and_validate(v)) continue;
            int nid = id(v[0], (dim>=2?v[1]:0), (dim==3?v[2]:0));
            out_ids.push_back(nid);
        }
    }
}

std::vector<int> DSU::shortest_path_base_to_top(int root, PercolationMode mode) const {
    std::vector<int> path;
    if (root < 0 || root >= (int)parent.size()) return path;

    // Coletar fontes na BASE (coord_grow == 0) pertencentes ao componente 'root'
    std::vector<int> sources;
    sources.reserve((Lx*Ly + Lx*Lz + Ly*Lz)); // heurística

    auto belongs = [&](int idn)->bool{
        if (!is_active(idn)) return false;
        return (const_cast<DSU*>(this)->find(idn) == const_cast<DSU*>(this)->find(root));
    };

    auto is_top = [&](int idn)->bool{
        int x,y,z; unid(idn,x,y,z);
        int cg = (grow_axis==0? x : (grow_axis==1? y : z));
        return (cg == ((grow_axis==0?Lx:(grow_axis==1?Ly:Lz)) - 1));
    };

    // fontes na base
    if (grow_axis == 0){
        for (int z=0; z<(dim==3?Lz:1); ++z)
            for (int y=0; y<(dim>=2?Ly:1); ++y){
                int id0 = id(0,y,z);
                if (belongs(id0)) sources.push_back(id0);
            }
    } else if (grow_axis == 1){
        for (int z=0; z<(dim==3?Lz:1); ++z)
            for (int x=0; x<Lx; ++x){
                int id0 = id(x,0,z);
                if (belongs(id0)) sources.push_back(id0);
            }
    } else { // grow_axis == 2
        for (int y=0; y<Ly; ++y)
            for (int x=0; x<Lx; ++x){
                int id0 = id(x,y,0);
                if (belongs(id0)) sources.push_back(id0);
            }
    }

    if (sources.empty()) return path;

    // BFS multi-origem
    const int N = static_cast<int>(TOT);
    std::vector<char> vis(N, 0);
    std::vector<int>  prev(N, -1);
    std::queue<int> q;
    for (int s : sources) { vis[s]=1; q.push(s); }

    std::vector<int> neigh;
    neigh.reserve(2*dim);

    int target = -1;

    while(!q.empty()){
        int u = q.front(); q.pop();
        if (is_top(u)) { target = u; break; }

        neighbors(u, neigh);
        for (int v : neigh){
            if (!belongs(v)) continue;

            if (mode == PercolationMode::Bond){
                // exige que a aresta (u,v) esteja aberta
                if (!is_bond_open(u,v)) continue;
            } else {
                // Site: apenas requer ambos ativos (belongs já garante ativo+mesmo componente)
            }

            if (vis[v]) continue;
            vis[v] = 1; prev[v] = u; q.push(v);
        }
    }

    if (target == -1) return path; // não achou caminho até o topo (inconsistente se spans==true)

    for (int cur = target; cur != -1; cur = prev[cur]) path.push_back(cur);
    std::reverse(path.begin(), path.end());
    return path;
}
