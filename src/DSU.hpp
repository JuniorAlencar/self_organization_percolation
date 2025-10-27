#ifndef DSU_HPP
#define DSU_HPP

#include <vector>
#include <cstdint>
#include <iostream>
#include <queue>

// Modo de adjacência para rotas/caminhos
enum class PercolationMode {
    Site,
    Bond
};

class DSU {
public:
    // ---------- Parâmetros geométricos ----------
    int dim {2};
    int Lx {1}, Ly {1}, Lz {1};
    int grow_axis {1};              // eixo "vertical": y(=1) em 2D, z(=2) em 3D
    std::int64_t TOT {0};           // número total de sítios

    // ---------- Estruturas compactas ----------
    // parent[i] < 0 => raiz; tamanho = -parent[i]
    // parent[i] >= 0 => pai
    std::vector<int>        parent;
    std::vector<unsigned char> active;     // 0/1 por sítio
    std::vector<unsigned char> touch_base; // 0/1 por índice (válido no root)
    std::vector<unsigned char> touch_top;  // 0/1 por índice (válido no root)

    // 3 bits por nó (X+, Y+, Z+) para percolação por ligação
    std::vector<std::uint8_t> bond_flags;

    // ---------- “View” compatível com o antigo dsu.sz[ root ] ----------
    struct SizeView {
        const DSU* d {nullptr};
        // devolve tamanho do componente cujo representante é r
        int operator[](int r) const {
            if (!d || r < 0 || r >= (int)d->parent.size()) return 0;
            if (d->parent[r] >= 0) {
                // se não for raiz, interpreta como 0 (rede antiga sempre passava root)
                return 0;
            }
            return -d->parent[r];
        }
    } sz;

public:
    // ---------- Ctor ----------
    DSU(int dim_, int Lx_, int Ly_, int Lz_, int grow_axis_);

    // ---------- Index helpers ----------
    int  id(int x, int y, int z) const;
    void unid(int id0, int& x, int& y, int& z) const;

    // ---------- UF básico ----------
    int  find(int a);             // com path compression
    int  find(int a) const;       // versão const (sem compressão)
    void make_active(int a, int coord_grow, int L);
    int  unite(int a, int b);     // union by size, mescla flags

    bool is_active(int a) const { return (a>=0 && a<(int)active.size() && active[a]); }

    // ---------- Contorno ----------
    // aberto no grow_axis; periódico nos demais
    bool wrap_and_validate(std::vector<int>& v) const;

    // ---------- Bonds ----------
    void open_bond(int a, int b);
    bool is_bond_open(int a, int b) const;

    // a e b são vizinhos cartesianos imediatos? se sim, retorna pivot+dir_bit
    bool bond_adjacent(int a, int b, int& pivot, int& dir_bit) const;

    // ---------- Adjacência de sítio ----------
    void connect_if_site_adjacent(int a, int b);

    // ---------- Vizinhança cartesiana (±1 em cada eixo) ----------
    void neighbors(int id0, std::vector<int>& out_ids) const;

    // ---------- Checagens base/topo ----------
    bool is_base(int id0) const;
    bool is_top (int id0) const;

    // ---------- Pertence ao root ----------
    bool belongs_to_root(int id0, int root) const;

    // ---------- Spanning (compatível com network.spans(root)) ----------
    bool spans(int root) const;   // toca base e topo?

    // ---------- Caminho mínimo base->topo dentro do componente ----------
    std::vector<int> shortest_path_base_to_top(int root, PercolationMode mode) const;

    // ---------- Estatísticas ----------
    struct StatsSnapshot {
        int    Smax = 0;
        int    Ntot = 0;
        double chi  = 0.0; // (sum s^2 - Smax^2)/Ntot
    };
    StatsSnapshot compute_snapshot_stats() const;

    void append_stats_row(std::vector<int>&   Smax_series,
                          std::vector<int>&   Ni_series,
                          std::vector<double>& chi_series) const;
    
    static PercolationMode percolation_mode_from_string(const std::string& s);
    static const char*     to_string(PercolationMode m) noexcept;
    int component_size_from_safe(int seed, PercolationMode mode, std::vector<int>* out_nodes = nullptr) const;
};

#endif // DSU_HPP
