#include "DSU.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

DSU::DSU(int dim_, int Lx_, int Ly_, int Lz_, int grow_axis_)
: dim(dim_), Lx(Lx_), Ly(Ly_>0?Ly_:1), Lz(Lz_>0?Lz_:1), grow_axis(grow_axis_)
{
    TOT = static_cast<std::int64_t>(Lx) * static_cast<std::int64_t>(Ly) * static_cast<std::int64_t>(Lz);
    parent.assign(TOT, 0);            // válido quando active=1
    active.assign(TOT, 0);
    touch_base.assign(TOT, 0);
    touch_top.assign(TOT, 0);
    bond_flags.assign(TOT, std::uint8_t(0));

    // ligar view
    sz.d = this;
}

// ---------- Index helpers ----------
int DSU::id(int x,int y,int z) const {
    return x + Lx * (y + Ly * z);
}

void DSU::unid(int id0, int& x, int& y, int& z) const {
    if (dim == 2) {
        x = id0 % Lx;
        y = id0 / Lx;
        z = 0;
    } else {
        x = id0 % Lx;
        int t = id0 / Lx;
        y = t % Ly;
        z = t / Ly;
    }
}

// ---------- UF básico ----------
int DSU::find(int a){
    if (a < 0 || a >= (int)parent.size()) return a;
    if (!is_active(a)) return a; // inativo → devolve ele mesmo
    int r = a;
    while (parent[r] >= 0) r = parent[r];
    // compressão de caminho
    while (a != r) {
        int p = parent[a];
        parent[a] = r;
        a = p;
    }
    return r;
}

// versão const: não faz compressão (para uso em métodos const)
int DSU::find(int a) const {
    if (a < 0 || a >= (int)parent.size()) return a;
    if (!is_active(a)) return a;
    int r = a;
    while (parent[r] >= 0) r = parent[r];
    return r;
}

void DSU::make_active(int a, int coord_grow, int L){
    if (a < 0 || a >= (int)active.size()) return;
    if (active[a]) return;
    active[a] = 1;
    parent[a] = -1; // tamanho 1
    touch_base[a] = (coord_grow == 0);
    touch_top[a]  = (coord_grow == L-1);
}

int DSU::unite(int a, int b){
    a = find(a); b = find(b);
    if (a == b) return a;
    // union-by-size
    if (-parent[a] < -parent[b]) std::swap(a,b);
    parent[a] += parent[b];   // soma tamanhos (negativos)
    parent[b]  = a;           // b aponta para a

    // mescla flags
    touch_base[a] = (touch_base[a] || touch_base[b]);
    touch_top[a]  = (touch_top[a]  || touch_top[b]);
    return a;
}

// ---------- Contorno ----------
bool DSU::wrap_and_validate(std::vector<int>& v) const {
    for (int j=0;j<dim;++j){
        int Lax = (j==0?Lx:(j==1?Ly:Lz));
        if (j == grow_axis) {
            if (v[j] < 0 || v[j] >= Lax) return false; // aberto
        } else {
            // periódico
            if      (v[j] < 0)     v[j] += Lax;
            else if (v[j] >= Lax)  v[j] -= Lax;
        }
    }
    return true;
}

// ---------- Bonds ----------
bool DSU::bond_adjacent(int a, int b, int& pivot, int& dir_bit) const {
    if (a < 0 || b < 0 || a == b) return false;
    int xa,ya,za, xb,yb,zb;
    unid(a, xa,ya,za);
    unid(b, xb,yb,zb);

    if (ya==yb && za==zb && std::abs(xa-xb)==1) {
        dir_bit = 0; // X
        pivot = (xa < xb) ? a : b; // aresta positiva no menor x
        return true;
    }
    if (xa==xb && za==zb && std::abs(ya-yb)==1) {
        dir_bit = 1; // Y
        pivot = (ya < yb) ? a : b;
        return true;
    }
    if (dim==3 && xa==xb && ya==yb && std::abs(za-zb)==1) {
        dir_bit = 2; // Z
        pivot = (za < zb) ? a : b;
        return true;
    }
    return false;
}

void DSU::open_bond(int a, int b){
    if (a<0 || b<0) return;
    if (!is_active(a) || !is_active(b)) return;

    int pivot=-1, dir=-1;
    if (!bond_adjacent(a,b,pivot,dir)) return;

    bond_flags[pivot] = static_cast<std::uint8_t>(bond_flags[pivot] | (1u << dir));
    unite(a,b);
}

bool DSU::is_bond_open(int a, int b) const {
    if (a<0 || b<0) return false;
    int pivot=-1, dir=-1;
    if (!bond_adjacent(a,b,pivot,dir)) return false;
    return (bond_flags[pivot] & (1u << dir)) != 0;
}

// ---------- Adjacência de sítio ----------
void DSU::connect_if_site_adjacent(int a, int b){
    if (is_active(a) && is_active(b)) unite(a,b);
}

// ---------- Vizinhança cartesiana ----------
void DSU::neighbors(int id0, std::vector<int>& out_ids) const {
    out_ids.clear();
    out_ids.reserve(2*dim);
    int x,y,z;
    unid(id0,x,y,z);
    for (int ax=0; ax<dim; ++ax){
        for (int sgn=-1; sgn<=1; sgn+=2){
            std::vector<int> v(3);
            v[0]=x; v[1]=y; v[2]=z;
            v[ax] += sgn;
            if (!wrap_and_validate(v)) continue;
            int nid = id(v[0],v[1],v[2]);
            out_ids.push_back(nid);
        }
    }
}

// ---------- Base/Topo ----------
bool DSU::is_base(int id0) const {
    int x,y,z; unid(id0,x,y,z);
    if (dim==2) return (y==0);
    return (z==0);
}
bool DSU::is_top(int id0) const {
    int x,y,z; unid(id0,x,y,z);
    if (dim==2) return (y==Ly-1);
    return (z==Lz-1);
}

// ---------- Pertence ao root ----------
bool DSU::belongs_to_root(int id0, int root) const {
    if (!is_active(id0)) return false;
    return find(id0) == root;   // usa find const
}

// ---------- Spanning ----------
bool DSU::spans(int root) const {
    root = find(root); // seguro mesmo se não for root
    if (root < 0 || root >= (int)parent.size()) return false;
    return (touch_base[root] && touch_top[root]);
}

// ---------- Caminho mínimo base->topo ----------
std::vector<int> DSU::shortest_path_base_to_top(int root, PercolationMode mode) const {
    std::vector<int> path;
    root = find(root); // versão const

    if (root < 0 || root >= (int)TOT) return path;

    auto belongs = [&](int u)->bool{ return belongs_to_root(u, root); };

    // fontes: nós do root na base
    std::vector<int> sources;
    if (grow_axis == 1 && dim == 2) {
        for (int x=0; x<Lx; ++x) {
            int id0 = id(x,0,0);
            if (belongs(id0)) sources.push_back(id0);
        }
    } else if (grow_axis == 2 && dim == 3) {
        for (int x=0; x<Lx; ++x)
            for (int y=0; y<Ly; ++y) {
                int id0 = id(x,y,0);
                if (belongs(id0)) sources.push_back(id0);
            }
    } else {
        for (int u=0; u<(int)TOT; ++u) {
            if (belongs(u) && is_base(u)) sources.push_back(u);
        }
    }
    if (sources.empty()) return path;

    const int N = static_cast<int>(TOT);
    std::vector<unsigned char> vis(N, 0);
    std::vector<int> prev(N, -1);
    std::queue<int> q;
    for (int s : sources) { vis[s]=1; q.push(s); }

    std::vector<int> neigh; neigh.reserve(2*dim);
    int target = -1;

    while(!q.empty()){
        int u = q.front(); q.pop();
        if (is_top(u)) { target = u; break; }

        neighbors(u, neigh);
        for (int v : neigh){
            if (!belongs(v)) continue;

            if (mode == PercolationMode::Bond){
                if (!is_bond_open(u,v)) continue;
            }
            if (!vis[v]) {
                vis[v] = 1;
                prev[v] = u;
                q.push(v);
            }
        }
    }

    if (target < 0) return path;
    for (int cur = target; cur >= 0; cur = prev[cur]) {
        path.push_back(cur);
        if (prev[cur] < 0) break;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// ---------- Estatísticas ----------
DSU::StatsSnapshot DSU::compute_snapshot_stats() const {
    StatsSnapshot out;
    long long sum_sizes = 0;
    long long sum_sizes_sq = 0;
    int Smax = 0;

    const int N = static_cast<int>(TOT);
    for (int i=0;i<N;++i){
        if (!is_active(i)) continue;
        if (parent[i] < 0) {          // raiz
            int sz_i = -parent[i];
            if (sz_i <= 0) continue;
            sum_sizes += sz_i;
            sum_sizes_sq += 1LL * sz_i * sz_i;
            if (sz_i > Smax) Smax = sz_i;
        }
    }
    out.Smax = Smax;
    out.Ntot = static_cast<int>(sum_sizes);
    if (sum_sizes > 0) {
        long long numer = sum_sizes_sq - 1LL * Smax * Smax;
        out.chi = static_cast<double>(numer) / static_cast<double>(sum_sizes);
        if (out.chi < 0.0) out.chi = 0.0;
    } else {
        out.chi = 0.0;
    }
    return out;
}

void DSU::append_stats_row(std::vector<int>& Smax_series,
                           std::vector<int>& Ni_series,
                           std::vector<double>& chi_series) const
{
    StatsSnapshot st = compute_snapshot_stats();
    if (st.Smax > st.Ntot) {
        std::cerr << "[WARN] Smax > Ntot: Smax=" << st.Smax << " Ntot=" << st.Ntot << "\n";
    }
    Smax_series.push_back(st.Smax);
    Ni_series.push_back(st.Ntot);
    chi_series.push_back(st.chi);
}
