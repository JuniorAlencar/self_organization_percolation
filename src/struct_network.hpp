#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <numeric>
#include <random>
#include <algorithm>

using namespace std;

struct NetworkPattern {
    int dim;                    // Dimensão da rede (ex: 2 ou 3)
    std::vector<int> shape;     // Tamanho de cada dimensão, ex: {Lx, Ly, Lz}
    std::vector<int> data;      // -1 = inativo sem cor, -2,-3,... = inativo com cor, 0 = checado sem ativação, +2,+3,... = ativado com cor

    NetworkPattern(int dimension, const std::vector<int>& shape_, int num_colors = 1, const std::vector<double>& rho = {})
        : dim(dimension), shape(shape_)
    {
        if ((int)shape.size() != dim)
            throw std::invalid_argument("Dimensão e tamanho do shape não coincidem");

        size_t total_size = 1;
        for (int s : shape) total_size *= s;

        data.resize(total_size, -1); // Todos começam como inativos sem cor

        if (num_colors > 1 && !rho.empty()) {
            if ((int)rho.size() != num_colors)
                throw std::invalid_argument("rho deve ter o mesmo tamanho de num_colors");

            // Normaliza rho
            double sum_rho = std::accumulate(rho.begin(), rho.end(), 0.0);
            std::vector<double> norm_rho(num_colors);
            for (int i = 0; i < num_colors; ++i)
                norm_rho[i] = rho[i] / sum_rho;

            // Índices embaralhados
            std::vector<size_t> indices(total_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);

            // Atribui as cores
            size_t current = 0;
            for (int c = 0; c < num_colors; ++c) {
                size_t qtd = static_cast<size_t>(norm_rho[c] * total_size);
                for (size_t i = 0; i < qtd && current < total_size; ++i, ++current) {
                    data[indices[current]] = -(c + 2);  // Ex: -2, -3, -4...
                }
            }
            // Demais permanecem como -1
        }
        // Caso num_colors == 1, já está inicializado como -1 em toda a rede
    }

    std::vector<int> get_row(int fixed_dim, int fixed_index, int varying_dim) {
        std::vector<int> row;
        for (int j = 0; j < shape[varying_dim]; ++j) {
            std::vector<int> coords(dim, 0);
            coords[fixed_dim] = fixed_index;
            coords[varying_dim] = j;
            row.push_back(get(coords));
        }
        return row;
    }

    size_t to_index(const std::vector<int>& coords) const {
        if ((int)coords.size() != dim)
            throw std::invalid_argument("Número de coordenadas incompatível com a dimensão");

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
