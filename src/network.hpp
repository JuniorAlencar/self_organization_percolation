#ifndef network_hpp
#define network_hpp

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>  // std::iota
#include <chrono>   // para fallback de seed, opcional
#include "rand_utils.hpp"

using namespace std;

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
    // Define
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


class network{
    private:
        int lenght_network;     // Length of Network - L
        int num_of_samples;     // Number of samples - t
        double k;               // Kinetic Coefficient
        double N_t = 0;         // Threshold parameters
        int type_N_t;           // type_N_t = 0 => N_t = const || type_N_t = 1 => N_t = at^\alpha
        double a;               // Used to N_t if type = 1(at^\alpha)
        double alpha;           // Used to N_t if type = 1(at^\alpha)
        int dim;                // Dimension of network
        int seed;               // Used for random processes
        double p0;                 // p(t=0) Initial probability of occupation of candidate sites for growth.
        double P0;              // Initial number of sites actives
        std::vector<double> p;  // Allocate p-values
    public:
        const std::vector<double>& get_p() const;
        // Constructor to p
        network(int num_samples)
            : num_of_samples(num_samples), p(num_samples, 0.0) {
        };
        double generate_p(const int type_N_t, const double p0, const int t_i, const int N_current, const double k, const double a, const double alpha);
        double type_Nt_create(const int type_N_t, const int t_i, const double a, const double alpha);
        NetworkPattern create_network(const int dim, const int lenght_network, const int num_of_samples, const double k, const double N_t, 
                                const int seed, const int type_N_t, const double p0, const double P0, const double a, const double alpha);
};


#endif // network_hpp
