#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <numeric>
#include <random>
#include <algorithm>

struct NetworkPattern {
    int dim;                    // Dimensão da rede (ex: 2 ou 3)
    std::vector<int> shape;     // Tamanho de cada dimensão, ex: {Lx, Ly, Lz}
    std::vector<int> data;      // -1 = inacessado, 0 = não ativado, 1 = ativado ou valores com cor

    // Constructor
    NetworkPattern(int dimension, const std::vector<int>& shape_, int num_colors = 0, const std::vector<double>& rho = {})
        : dim(dimension), shape(shape_)
    {
        if ((int)shape.size() != dim) {
            throw std::invalid_argument("Dimensão e tamanho do shape não coincidem");
        }

        size_t total_size = 1;
        for (int s : shape) total_size *= s;

        data.resize(total_size, -1); // Inicializa todos como inacessado

        if (num_colors > 1 && !rho.empty()) {
            if ((int)rho.size() != num_colors) {
                throw std::invalid_argument("rho deve ter o mesmo tamanho de num_colors");
            }

            // Normaliza rho para garantir soma = 1.0
            double sum_rho = std::accumulate(rho.begin(), rho.end(), 0.0);
            std::vector<double> norm_rho(num_colors);
            for (int i = 0; i < num_colors; ++i)
                norm_rho[i] = rho[i] / sum_rho;

            // Gera índices embaralhados
            std::vector<size_t> indices(total_size);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            // Preenche as cores
            size_t current = 0;
            for (int c = 0; c < num_colors; ++c) {
                size_t num_sites = static_cast<size_t>(norm_rho[c] * total_size);
                for (size_t i = 0; i < num_sites && current < total_size; ++i, ++current) {
                    data[indices[current]] = -(c + 2); // Inativo com cor (ex: -2, -3...)
                }
            }
            // Restantes permanecem -1 (sem cor)
        }
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

    // Convert coordinates to linear index
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

    // Getter
    int get(const std::vector<int>& coords) const {
        return data[to_index(coords)];
    }

    // Setter
    void set(const std::vector<int>& coords, int value) {
        data[to_index(coords)] = value;
    }

    // Total size
    size_t size() const {
        return data.size();
    }
};

#endif // network_hpp
