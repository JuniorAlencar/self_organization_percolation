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

#include <vector>
#include <numeric>
#include <functional>
#include <random>
#include <iostream>

using namespace std;

struct NetworkPattern {
    int dim;                 // Dimensão da rede (2D ou 3D)
    int num_colors;          // Número de cores
    int seed;                // Semente do gerador aleatório
    std::vector<int> shape;  // Forma da rede, ex: {L, L} ou {L, L, L}
    std::vector<int> data;   // Estado por sítio
    std::vector<double> rho; // Fração global por cor (aplicada na rede toda)

    // Construtores
    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_, const std::vector<double>& rho_)
        : dim(dim_), shape(shape_), num_colors(num_colors_), rho(rho_) {

        // Tamanho total da rede
        int total_sites = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        
        // Inicializa a rede com todos os valores como -1 (sem cor)
        data.resize(total_sites, -1);
    }

    // Método para acessar um site específico (getter)
    int get(int idx) const {
        return data[idx];
    }

    // Método para modificar um site específico (setter)
    void set(int idx, int value) {
        data[idx] = value;
    }

    // Método para limpar a rede, preenchendo tudo com -1 (cinza)
    void clear() {
        std::fill(data.begin(), data.end(), -1);
    }

    // Função para imprimir a rede (para visualização)
    void print() const {
        int idx = 0;
        for (int i = 0; i < shape[0]; ++i) {
            for (int j = 0; j < shape[1]; ++j) {
                std::cout << get(idx++) << " ";
            }
            std::cout << std::endl;
        }
    }

    // Função para acessar o tamanho da rede
    int size() const {
        return data.size();
    }
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
