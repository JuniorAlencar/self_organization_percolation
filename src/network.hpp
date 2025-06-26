#ifndef network_hpp
#define network_hpp

#include <vector>
#include <stdexcept>
#include <iostream>

struct NetworkPattern {
    int dim;                // Dimensão da rede (ex: 2 ou 3)
    std::vector<int> shape; // Tamanho de cada dimensão, ex: {Lx, Ly, Lz}
    std::vector<char> data; // 0 = sem sítio, 1 = com sítio

    // Construtor
    NetworkPattern(int dimension, const std::vector<int>& shape_)
        : dim(dimension), shape(shape_) 
    {
        if ((int)shape.size() != dim) {
            throw std::invalid_argument("Dimensão e tamanho do shape não coincidem");
        }

        size_t total_size = 1;
        for (int s : shape) {
            total_size *= s;
        }
        data.resize(total_size, 0); // Inicializa tudo como 0 (sem sítio)
    }

    // Função para converter coordenadas para índice linear
    size_t to_index(const std::vector<int>& coords) const {
        if ((int)coords.size() != dim) {
            throw std::invalid_argument("Número de coordenadas incompatível com dimensão");
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

    // Acessar valor (getter)
    char get(const std::vector<int>& coords) const {
        return data[to_index(coords)];
    }

    // Definir valor (setter)
    void set(const std::vector<int>& coords, char value) {
        data[to_index(coords)] = value;
    }

    // Tamanho total
    size_t size() const {
        return data.size();
    }
};

std::vector<char> get_row(int fixed_dim, int fixed_index, int varying_dim) {
    std::vector<char> row;

    for (int j = 0; j < shape[varying_dim]; ++j) {
        std::vector<int> coords(dim);
        coords[fixed_dim] = fixed_index;
        coords[varying_dim] = j;
        row.push_back(get(coords));
    }

    return row;
}


class network{
    private:
        int lenght_network;     // Length of Network - L
        int num_of_samples;     // Number of samples - t
        double k;               // Kinetic Coefficient
        double N_t;             // Threshold parameters
        int seed;               // Used for random processes
    public:

};


#endif // network_hpp
