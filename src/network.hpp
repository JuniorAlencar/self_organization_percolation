#ifndef network_hpp
#define network_hpp

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <random>
#include <numeric>  // std::iota
#include <chrono>   // para fallback de seed, opcional
#include "rand_utils.hpp"
#include "struct_network.hpp"

using namespace std;

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
        string type_percolation;
        double p0;                 // p(t=0) Initial probability of occupation of candidate sites for growth.
        double P0;              // Initial number of sites actives
        std::vector<double> p;  // Allocate p-values
        std::vector<int> N_t_list;
        std::vector<int> t_list;
    
    public:
        const std::vector<double>& get_p() const;
        const std::vector<int>& get_N_t() const;
        const std::vector<int>& get_t() const;
        // Constructor to p
        network(int num_samples)
            : num_of_samples(num_samples), p(num_samples, 0.0) {
        };
        double generate_p(const int type_N_t, const double p0, const int t_i, const int N_current, const double k, const double a, const double alpha);
        double type_Nt_create(const int type_N_t, const int t_i, const double a, const double alpha);
        NetworkPattern create_network(const int dim, const int lenght_network, const int num_of_samples,
                                       const double k, const double N_t, const int seed, const int type_N_t,
                                       const double p0, const double P0, const double a, const double alpha,
                                       const std::string& type_percolation);
};


#endif // network_hpp
