#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>

struct NetworkPattern {
    int dim;                    // Dimensão da rede (ex: 2 ou 3)
    std::vector<int> shape;     // Tamanho de cada dimensão, ex: {Lx, Ly, Lz}
    std::vector<int> data;      // -1 = inacessado, 0 = não ativado, 1 = ativado

    // Constructor
    NetworkPattern(int dimension, const std::vector<int>& shape_)
        : dim(dimension), shape(shape_)
    {
        if ((int)shape.size() != dim) {
            throw std::invalid_argument("Dimensão e tamanho do shape não coincidem");
        }

        size_t total_size = 1;
        for (int s : shape) total_size *= s;

        data.resize(total_size, -1); // Initialize with -1 in all positions (not acess)
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