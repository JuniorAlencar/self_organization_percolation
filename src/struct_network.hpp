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
    std::vector<int> shape;
    std::vector<int> data;
    std::vector<double> rho;
    
    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_, const std::vector<double>& rho_, all_random& rng_)
    : dim(dim_), shape(shape_), num_colors(num_colors_), rho(rho_), seed(rng_.get_seed()) {

    const size_t total_sites = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data.resize(total_sites, 0);

    int total_base = (dim == 2) ? shape[1] : shape[1] * shape[2];

    std::vector<int> labels;

    if (num_colors_ == 1) {
        // Caso especial: uma única cor (como no artigo original)
        labels.insert(labels.end(), total_base, -1);  // Tudo começa como -1 (sem cor)
    } else {
        // Múltiplas cores: preencher com base em rho
        for (int i = 0; i < num_colors_; ++i) {
            int label = -(i + 2);  // -2, -3, ...
            int count = static_cast<int>(rho_[i] * total_base);
            labels.insert(labels.end(), count, label);
        }

        int filled = static_cast<int>(labels.size());
        if (filled < total_base) {
            labels.insert(labels.end(), total_base - filled, -1);
        }
        std::shuffle(labels.begin(), labels.end(), rng_.get_gen());;
    }

    for (int t = 0; t < shape[0]; ++t) {
        for (int i = 0; i < total_base; ++i) {
            std::vector<int> coord;
            if (dim == 2)
                coord = {t, i};
            else
                coord = {t, i / shape[2], i % shape[2]};
            data[to_index(coord)] = labels[i];
            }
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
            idx += coords[i] * stride;
            stride *= shape[i];
        }
        return idx;
    }

    int get(const std::vector<int>& coords) const {
        return data[to_index(coords)];
    }

    void set(const std::vector<int>& coords, int value) {
        data[to_index(coords)] = value;
    }

    size_t size() const {
        return data.size();
    }
};

// Struct to load (t, pt_i) and (t, Nt_i)
struct TimeSeries{
    int num_colors;
    vector<vector<double>> p_t;
    vector<int> t;
    vector<vector<int>> Nt;
};

#endif // network_hpp
