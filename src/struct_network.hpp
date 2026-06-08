#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "rand_utils.hpp"
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <cstddef>

struct NetworkPattern {
    using state_t = int32_t;

    int dim;                        // Dimensão da rede (2D ou 3D)
    int num_colors;                 // Número de cores
    int seed;                       // Semente do gerador aleatório
    std::vector<int> shape;         // Forma da rede, ex: {L, L} ou {L, L, L}
    std::vector<state_t> data;      // Estado por sítio
    std::vector<double> rho;        // Fração global por cor (aplicada na rede toda)
    std::vector<std::pair<uint32_t, uint32_t>> edge_pairs; // Arestas dirigidas u -> v

    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_,
                   const std::vector<double>& rho_, bool initialize_data = true)
        : dim(dim_),
          num_colors(num_colors_),
          seed(-1),
          shape(shape_),
          rho(rho_)
    {
        if (dim != 2 && dim != 3) {
            throw std::invalid_argument("NetworkPattern: dim must be 2 or 3.");
        }
        if (static_cast<int>(shape.size()) != dim) {
            throw std::invalid_argument("NetworkPattern: shape size must match dim.");
        }

        const std::size_t total_sites = static_cast<std::size_t>(std::accumulate(
            shape.begin(), shape.end(), 1LL, std::multiplies<long long>()));

        if (initialize_data) {
            // Inicializa a rede com todos os valores como -1 (sem cor / inativo sem cor)
            data.assign(total_sites, static_cast<state_t>(-1));
        }
    }

    inline int get(int idx) const {
        return static_cast<int>(data[static_cast<std::size_t>(idx)]);
    }

    inline void set(int idx, int value) {
        data[static_cast<std::size_t>(idx)] = static_cast<state_t>(value);
    }

    inline void clear() {
        std::fill(data.begin(), data.end(), static_cast<state_t>(-1));
    }

    void print() const {
        if (dim == 2) {
            int idx = 0;
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) {
                    std::cout << get(idx++) << ' ';
                }
                std::cout << '\n';
            }
            return;
        }

        const int plane = shape[0] * shape[1];
        for (int k = 0; k < shape[2]; ++k) {
            std::cout << "z = " << k << '\n';
            int base = k * plane;
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) {
                    std::cout << get(base + i * shape[1] + j) << ' ';
                }
                std::cout << '\n';
            }
            std::cout << '\n';
        }
    }

    inline int size() const {
        return static_cast<int>(data.size());
    }
};

struct TimeSeries {
    int num_colors = 0;
    double t_eq = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> p_t;   // [cor][t]
    std::vector<std::vector<double>> f_t;   // [cor][t], f_i(t) = N_i(t) / L^(dim-1)
    std::vector<int> t;                     // [t]
};

struct PercolationSeries {
    std::vector<double> rho;

    std::vector<int> color_percolation;
    std::vector<int> percolation_order;

    std::vector<int> M_size_at_perc;
    std::vector<int> sp_len;
    std::vector<std::vector<int>> sp_path_lin;

    // novos campos
    double t_eq = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> t_eq_by_species;
    std::vector<int> sp_lin_preteq;
    std::vector<std::vector<int>> sp_path_lin_preteq;
    std::vector<int> sp_lin_posteq;
    std::vector<std::vector<int>> sp_path_lin_posteq;
    std::vector<int> M_size_preteq;
    std::vector<int> M_size_posteq;
    std::vector<int> z_max_at_perc;
    std::vector<int> z_max_final;
    std::vector<int> z_stat_by_species;
    int equilibrium_consecutive_steps = -1;
    int dynamics_window_steps = -1;
    int dynamic_min_stop_height = -1;
    int dynamic_max_stop_height = -1;
    double equilibrium_rel_tol = std::numeric_limits<double>::quiet_NaN();
    double equilibrium_abs_tol = std::numeric_limits<double>::quiet_NaN();
    std::string initial_base_layout;
    
};

// Compact network representation: Structure of Arrays (SoA) + CSR edges
struct NetworkCompact {
    using index_t = uint32_t;

    index_t N = 0;                         // número de sítios
    std::vector<index_t> pos_flat;        // posição flattened por sítio (N)
    std::vector<uint8_t> species;         // espécie por sítio (N)
    std::vector<uint32_t> activation_time;// instante de ativação (N)

    // CSR representation for undirected/directed links
    // edge_offsets size = N+1, edges size = N_edges
    std::vector<index_t> edge_offsets;    
    std::vector<index_t> edges;

    inline index_t neighbors_start(index_t v) const { return edge_offsets[v]; }
    inline index_t neighbors_end(index_t v) const { return edge_offsets[v+1]; }
    inline std::size_t num_edges() const { return edges.size(); }

    // Build CSR from a list of edge pairs (u,v). Assumes 0 <= u,v < N.
    void build_csr_from_edge_pairs(const std::vector<std::pair<index_t,index_t>> &pairs) {
        // Count degree
        edge_offsets.assign(N + 1, 0);
        for (const auto &e : pairs) {
            if (e.first < N) ++edge_offsets[e.first + 1];
        }
        // Prefix sum
        for (index_t i = 1; i <= N; ++i) edge_offsets[i] += edge_offsets[i-1];

        edges.assign(pairs.size(), 0);
        // temporary cursor copy
        std::vector<index_t> cursor(edge_offsets.begin(), edge_offsets.end());
        for (const auto &e : pairs) {
            index_t u = e.first;
            index_t v = e.second;
            if (u >= N) continue;
            edges[cursor[u]++] = v;
        }
    }

    // Append a single edge pair to an edge-pairs buffer (helper - not part of CSR)
    static void append_edge_pair(std::vector<std::pair<index_t,index_t>> &buf, index_t u, index_t v) {
        buf.emplace_back(u, v);
    }

    // Write compact binary format
    bool write_binary(const std::string &path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) return false;
        uint32_t magic = 0x4E455447; // 'NETG'
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&N), sizeof(N));
        uint64_t E = static_cast<uint64_t>(edges.size());
        out.write(reinterpret_cast<const char*>(&E), sizeof(E));

        out.write(reinterpret_cast<const char*>(pos_flat.data()), N * sizeof(index_t));
        // Debug: print first species values to stderr
        {
            std::cerr << "[write_binary] species sample:";
            for (index_t i = 0; i < std::min<index_t>(N, 20); ++i) {
                std::cerr << ' ' << static_cast<int>(species[i]);
            }
            std::cerr << '\n';
        }
        // Write species as single bytes to avoid accidental padding issues
        for (index_t i = 0; i < N; ++i) {
            const uint8_t s = species[i];
            out.write(reinterpret_cast<const char*>(&s), sizeof(s));
        }
        out.write(reinterpret_cast<const char*>(activation_time.data()), N * sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(edge_offsets.data()), (N + 1) * sizeof(index_t));
        out.write(reinterpret_cast<const char*>(edges.data()), edges.size() * sizeof(index_t));
        return out.good();
    }

    // Read compact binary format (overwrites this object)
    bool read_binary(const std::string &path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        uint32_t magic = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x4E455447) return false;
        in.read(reinterpret_cast<char*>(&N), sizeof(N));
        uint64_t E = 0;
        in.read(reinterpret_cast<char*>(&E), sizeof(E));

        pos_flat.assign(N, 0);
        species.assign(N, 0);
        activation_time.assign(N, 0);
        edge_offsets.assign(N + 1, 0);
        edges.assign(static_cast<std::size_t>(E), 0);

        in.read(reinterpret_cast<char*>(pos_flat.data()), N * sizeof(index_t));
        in.read(reinterpret_cast<char*>(species.data()), N * sizeof(uint8_t));
        in.read(reinterpret_cast<char*>(activation_time.data()), N * sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(edge_offsets.data()), (N + 1) * sizeof(index_t));
        in.read(reinterpret_cast<char*>(edges.data()), E * sizeof(index_t));
        return in.good();
    }

    // Candidate selection per iteration: given per-target candidate lists, pick one uniformly
    // and append chosen pairs to out_pairs.
    static void select_from_candidates(std::vector<std::vector<index_t>> &candidates,
                                       std::vector<std::pair<index_t,index_t>> &out_pairs,
                                       std::mt19937 &rng) {
        const std::size_t Nloc = candidates.size();
        for (std::size_t v = 0; v < Nloc; ++v) {
            auto &c = candidates[v];
            if (c.empty()) continue;
            std::uniform_int_distribution<std::size_t> dist(0, c.size() - 1);
            std::size_t pick = dist(rng);
            index_t u = c[pick];
            out_pairs.emplace_back(u, static_cast<index_t>(v));
        }
    }

    // Return a filtered NetworkCompact containing only active nodes (species != 0).
    // Remaps indices and rebuilds CSR edges accordingly.
    NetworkCompact filter_active() const {
        NetworkCompact out;
        out.N = 0;
        const index_t Nold = static_cast<index_t>(pos_flat.size());
        if (Nold == 0) return out;

        const index_t invalid = std::numeric_limits<index_t>::max();
        std::vector<index_t> remap(Nold, invalid);

        // count active
        for (index_t i = 0; i < Nold; ++i) {
            if (species[i] != 0) {
                remap[i] = out.N++;
            }
        }

        out.pos_flat.resize(out.N);
        out.species.resize(out.N);
        out.activation_time.resize(out.N);

        for (index_t i = 0; i < Nold; ++i) {
            if (remap[i] == invalid) continue;
            const index_t j = remap[i];
            out.pos_flat[j] = pos_flat[i];
            // Preserve species code (expected to be 0..ns). Clamp to 0..255.
            out.species[j] = static_cast<uint8_t>(species[i]);
            out.activation_time[j] = activation_time[i];
        }

        // rebuild edge list from existing CSR
        std::vector<std::pair<index_t,index_t>> pairs;
        pairs.reserve(edges.size());
        for (index_t u = 0; u < Nold; ++u) {
            const index_t start = edge_offsets[u];
            const index_t end = edge_offsets[u+1];
            if (remap[u] == invalid) continue;
            for (index_t k = start; k < end; ++k) {
                const index_t v = edges[k];
                if (v >= Nold) continue;
                if (remap[v] == invalid) continue;
                pairs.emplace_back(remap[u], remap[v]);
            }
        }

        if (!pairs.empty()) {
            out.build_csr_from_edge_pairs(pairs);
        } else {
            out.edge_offsets.assign(out.N + 1, 0);
            out.edges.clear();
        }

        return out;
    }
};

#endif // STRUCT_NETWORK_HPP
