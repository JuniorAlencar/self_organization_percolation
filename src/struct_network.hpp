#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "rand_utils.hpp"

struct NetworkPattern {
    using state_t = int16_t;

    int dim;                        // Dimensão da rede (2D ou 3D)
    int num_colors;                 // Número de cores
    int seed;                       // Semente do gerador aleatório
    std::vector<int> shape;         // Forma da rede, ex: {L, L} ou {L, L, L}
    std::vector<state_t> data;      // Estado por sítio
    std::vector<double> rho;        // Fração global por cor (aplicada na rede toda)

    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_, const std::vector<double>& rho_)
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

        // Inicializa a rede com todos os valores como -1 (sem cor / inativo sem cor)
        data.assign(total_sites, static_cast<state_t>(-1));
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
    int num_colors;
    std::vector<std::vector<double>> p_t;   // [cor][t]
    std::vector<std::vector<int>> Nt;       // [cor][t]
    std::vector<int> t;                     // [t]
};

struct PercolationSeries {
    std::vector<int> color_percolation;     // 1-based
    std::vector<int> percolation_order;
    std::vector<double> rho;                // por cor (0-based)
    std::vector<int> M_size_at_perc;        // por evento
    std::vector<int> sp_len;                // [cor] = #nós no caminho (ou -1)
    std::vector<std::vector<int>> sp_path_lin; // [cor] ids lineares
};

#endif // STRUCT_NETWORK_HPP
