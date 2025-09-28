#ifndef DSU_hpp
#define DSU_hpp

#pragma once
#include <vector>
#include <queue>
#include <unordered_set>
#include <cstdint>
#include <algorithm>

enum class PercolationMode {
    Site,  // adjacência por vizinhos ativos
    Bond   // adjacência restrita a arestas abertas
};

struct DSU {
    // ---- parâmetros geométricos / contorno ----
    int dim{2}, Lx{1}, Ly{1}, Lz{1}, grow_axis{1};
    std::int64_t TOT{0};

    // ---- Union-Find ----
    std::vector<int>  parent;      // pai (índice linear) ou -1 se inativo
    std::vector<int>  sz;          // tamanho do componente (válido na raiz)
    std::vector<char> active;      // 1 se o nó está ativo
    std::vector<char> touch_base;  // (por raiz) componente toca a base
    std::vector<char> touch_top;   // (por raiz) componente toca o topo

    // ---- Arestas abertas (bond) ----
    // Representamos cada aresta não-direcionada por uma chave uint64_t (min,max).
    std::unordered_set<std::uint64_t> open_edges;

    // ---- construção ----
    DSU(int dim_, int Lx_, int Ly_, int Lz_, int grow_axis_);

    // ---- helpers de indexação ----
    inline int id(int x,int y,int z) const;
    inline void unid(int id, int& x, int& y, int& z) const;

    // ---- operações básicas UF ----
    int  find(int a);
    void make_active(int a, int coord_grow, int L);
    int  unite(int a, int b);

    inline bool is_active(int a) const { return (a>=0 && a < (int)active.size() && active[a]); }

    // ---- contorno (aberto no grow_axis; periódico nas demais) ----
    bool wrap_and_validate(std::vector<int>& v) const;

    // ---- bond handling ----
    // abre uma aresta (não-direcionada) entre 'a' e 'b' e faz unite, se ambos ativos
    void open_bond(int a, int b);
    bool is_bond_open(int a, int b) const;

    // ---- site adjacency (para percolação por sítio) ----
    // une se ambos ativos (não verifica cor; isso é responsabilidade de nível superior)
    void connect_if_site_adjacent(int a, int b);

    // ---- vizinhos de um id (±1 em cada eixo) com contorno apropriado ----
    void neighbors(int id0, std::vector<int>& out_ids) const;

    // ---- shortest path (base -> topo) dentro do componente 'root' ----
    // Retorna ids lineares do caminho mínimo. Vazio se não encontrar.
    std::vector<int> shortest_path_base_to_top(int root, PercolationMode mode) const;

    // ---- flags base/top para uma raiz ----
    inline bool spans(int root) {
        int r = find(root);
        return (touch_base[r] && touch_top[r]);
    }

private:
    // chave de aresta não-direcionada (min,max) em 64 bits
    static inline std::uint64_t edge_key(int a, int b);
};



#endif // DSU_hpp