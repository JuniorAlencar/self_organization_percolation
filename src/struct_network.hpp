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
    std::vector<double> rho;  // agora usada apenas para COLORIR A BASE
    // Convenções:
    //   >0 : ativo, +1 (1 cor) ou +(c+2) (multi-cor)
    //   -1 : inativo sem cor (cinza)
    //   -(c+2) : inativo rotulado da cor c (c=0..num_colors-1)
    //    0 : checado e não ativado (não conecta mais)

    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_,
                   const std::vector<double>& rho_, all_random& rng_)
    : dim(dim_), shape(shape_), num_colors(num_colors_), rho(rho_), seed(rng_.get_seed())
    {
        const size_t total_sites =
            std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
        data.assign(total_sites, -1); // TODA a rede cinza
        // (não colorimos nada aqui; a coloração da BASE será feita fora)
    }

    size_t to_index(const std::vector<int>& coords) const {
        if ((int)coords.size() != dim) {
            throw std::invalid_argument("Número de coordenadas incompatível com a dimensão");
        }
        size_t idx = 0, stride = 1;
        for (int i = dim - 1; i >= 0; --i) {
            if (coords[i] < 0 || coords[i] >= shape[i])
                throw std::out_of_range("Coordenada fora dos limites");
            idx += size_t(coords[i]) * stride;
            stride *= size_t(shape[i]);
        }
        return idx;
    }

    int  get(const std::vector<int>& coords) const { return data[to_index(coords)]; }
    void set(const std::vector<int>& coords, int value) { data[to_index(coords)] = value; }
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
