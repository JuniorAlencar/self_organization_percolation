#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <numeric>
#include <random>
#include <algorithm>
#include "rand_utils.hpp"

using namespace std;

struct NetworkPattern {
    int dim;
    int num_colors;
    int seed;
    std::vector<int> shape;   // (2D) {L, L}  |  (3D) {L, L, L}
    std::vector<int> data;    // estado por sítio
    std::vector<double> rho;  // fração na borda por cor
    // Convenções de estado:
    //   >0 : ativo, +1 (1 cor) ou +(c+2) (multi-cor)
    //   -1 : inativo sem cor (pode virar qualquer cor)
    //   -(c+2) : inativo com rótulo da cor c (c=0..num_colors-1)
    //    0 : checado e não ativado (não conecta mais)

    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_,
                   const std::vector<double>& rho_, all_random& rng_)
    : dim(dim_), shape(shape_), num_colors(num_colors_), rho(rho_), seed(rng_.get_seed())
    {
        // tamanho total
        const size_t total_sites = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                                   std::multiplies<size_t>());
        data.resize(total_sites, -1); // toda a rede começa "sem cor" (−1)

        // eixo de crescimento = última dimensão
        const int Lgrow = shape.back();

        // tamanho da borda (plano/linha onde grow-index = 0)
        int total_base = 1;
        for (int ax = 0; ax < dim - 1; ++ax) total_base *= shape[ax];

        // monta vetor de rótulos (negativos) para a borda de partida
        std::vector<int> labels;
        labels.reserve(total_base);

        if (num_colors_ == 1) {
            // uma única cor: borda toda com −1 (sem-cor) está OK,
            // mas se quiser "pré-rotular" explicitamente, mantenha −1:
            labels.insert(labels.end(), total_base, -1);
        } else {
            // múltiplas cores: preencher quantidades proporcionais a rho na borda
            int filled = 0;
            for (int c = 0; c < num_colors_; ++c) {
                const int want = static_cast<int>(std::round(rho_[c] * total_base));
                labels.insert(labels.end(), want, -(c + 2)); // −2, −3, ...
                filled += want;
            }
            // completa o restante com −1 (sem cor)
            if (filled < total_base) labels.insert(labels.end(), total_base - filled, -1);
            // embaralha
            std::shuffle(labels.begin(), labels.end(), rng_.get_gen());
        }

        // grava os rótulos na borda: índice da última dimensão = 0
        for (int idx = 0; idx < total_base; ++idx) {
            // reconstrói coordenadas da borda a partir de idx linear nas (dim-1) primeiras dims
            std::vector<int> coord(dim, 0);
            int rem = idx;
            for (int ax = dim - 2; ax >= 0; --ax) {
                coord[ax] = rem % shape[ax];
                rem /= shape[ax];
            }
            coord.back() = 0; // grow-axis = 0
            data[to_index(coord)] = labels[idx];
        }
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

    int get(const std::vector<int>& coords) const { return data[to_index(coords)]; }
    void set(const std::vector<int>& coords, int value) { data[to_index(coords)] = value; }
    size_t size() const { return data.size(); }
};



// Struct to load (t, pt_i) and (t, Nt_i)
struct TimeSeries{
    int num_colors;
    vector<vector<double>> p_t;
    vector<int> t;
    vector<vector<int>> Nt;
};

struct PercolationSeries{
    vector<int> color_percolation;
    vector<int> time_percolation;
    vector<int> percolation_order;
};

#endif // network_hpp
