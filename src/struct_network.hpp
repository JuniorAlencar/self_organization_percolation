#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <functional>
#include "rand_utils.hpp"

using namespace std;



struct NetworkPattern {
    int dim;
    int num_colors;
    int seed;
    std::vector<int> shape;        // (2D) {L, L}  |  (3D) {L, L, L}
    std::vector<uint8_t> data;     // armazenamento compacto (1 byte por sítio)
    std::vector<double> rho;

    // Codificação compacta (dependente de num_colors):
    //   0              -> cinza (era -1)
    //   1              -> checado (era 0)
    //   2..(nc+1)      -> inativo colorido -(c+2), c = val-2
    //   (nc+2)..(2nc+1)-> ativo +(c+2),     c = val-(nc+2)

    static inline uint8_t enc_gray() { return 0u; }
    static inline uint8_t enc_checked() { return 1u; }
    static inline uint8_t enc_inactive(int c) { return uint8_t(2 + c); }
    static inline uint8_t enc_active(int nc, int c) { return uint8_t(nc + 2 + c); }

    static inline bool is_gray(uint8_t v) { return v == 0u; }
    static inline bool is_checked(uint8_t v) { return v == 1u; }
    static inline bool is_inactive_color(uint8_t v, int nc) { return v >= 2u && v <= uint8_t(nc + 1); }
    static inline bool is_active_color(uint8_t v, int nc) { return v >= uint8_t(nc + 2) && v <= uint8_t(2*nc + 1); }
    static inline int color_of(uint8_t v, int nc) {
        if (is_inactive_color(v, nc)) return int(v) - 2;
        if (is_active_color(v, nc))   return int(v) - (nc + 2);
        return -1;
    }

    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_,
                   const std::vector<double>& rho_, class all_random& rng_)
    : dim(dim_), shape(shape_), num_colors(num_colors_), rho(rho_), seed(rng_.get_seed())
    {
        // total de sítios
        const size_t total_sites = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                                   std::multiplies<size_t>());
        data.assign(total_sites, enc_gray()); // tudo cinza por padrão
    }

    size_t to_index(const std::vector<int>& coords) const {
        if ((int)coords.size() != dim) {
            throw std::invalid_argument("Número de coordenadas incompatível com a dimensão");
        }
        size_t idx = 0;
        size_t stride = 1;
        for (int i = dim - 1; i >= 0; --i) {
            if (coords[i] < 0 || coords[i] >= shape[i])
                throw std::out_of_range("Coordenada fora dos limites");
            idx += size_t(coords[i]) * stride;
            stride *= size_t(shape[i]);
        }
        return idx;
    }

    // ----- compatibilidade com API antiga -----
    // get: retorna nos valores antigos: -1, 0, -(c+2), +(c+2)
    int get(const std::vector<int>& coords) const {
        uint8_t v = data[to_index(coords)];
        if (v == enc_gray())    return -1;
        if (v == enc_checked()) return 0;
        if (is_inactive_color(v, num_colors)) return -(color_of(v, num_colors) + 2);
        if (is_active_color(v, num_colors))   return  (color_of(v, num_colors) + 2);
        return -1; // fallback
    }

    // set: aceita valores antigos e converte para o codificado compacto
    void set(const std::vector<int>& coords, int value) {
        uint8_t out = enc_gray();
        if (value == -1) out = enc_gray();
        else if (value == 0) out = enc_checked();
        else if (value < 0) {
            int c = -value - 2; // -(c+2) -> c
            if (c < 0) c = 0;
            out = enc_inactive(c);
        } else { // value > 0
            int c =  value - 2; // +(c+2) -> c
            if (c < 0) {
                // caso 1 cor (+1)
                out = enc_active(num_colors, 0);
            } else {
                out = enc_active(num_colors, c);
            }
        }
        data[to_index(coords)] = out;
    }

    size_t size() const { return data.size(); }
};




// Struct to load (t, pt_i) and (t, Nt_i)
struct TimeSeries {
    int num_colors;
    vector<vector<double>> p_t;   // [cor][t]
    vector<vector<int>>    Nt;    // [cor][t]
    vector<int>            t;     // [t]
};

struct PercolationSeries {
    vector<int> color_percolation;   // 1-based
    vector<int> percolation_order;
    vector<double> rho;              // por cor (0-based)
    vector<int> M_size_at_perc;      // por evento
    // SP armazenado internamente (opcional no writer)
    vector<int>              sp_len;       // [cor] = #nós no caminho (ou -1)
    vector<vector<int>> sp_path_lin;  // [cor] ids lineares (não será escrito)
};

#endif // network_hpp
