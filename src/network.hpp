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
#include <queue>
#include <map>
#include <cstddef>
#include <iomanip>

#include <unordered_set>
#include "rand_utils.hpp"
#include "struct_network.hpp"
#include "DSU.hpp"

using namespace std;

// Helpers to decide the activates for new nodes/bonds
inline bool is_active(int v)        { return v > 0; }
inline bool is_gray(int v)          { return v == -1; }
inline bool is_inactive_color(int v, int c) { return v == -(c + 2); }
inline int  color_index_from_val(int v, int num_colors) {
    return (num_colors == 1) ? 0 : (std::abs(v) - 2); // válido para v≠-1,0
}

class network{
    private:
        int lenght_network;         // Length of Network - L
        int num_of_samples;         // Number of samples - t
        double k;                   // Kinetic Coefficient
        double N_t = 0;             // Threshold parameters
        int type_N_t;               // type_N_t = 0 => N_t = const || type_N_t = 1 => N_t = at^\alpha
        double a;                   // Used to N_t if type = 1(at^\alpha)
        double alpha;               // Used to N_t if type = 1(at^\alpha)
        int dim;                    // Dimension of network
        int seed;                   // Used for random processes
        string type_percolation;    // Bond or node
        vector<double> p0;          // p(t=0) Initial probability of occupation of candidate sites for growth.
        double P0;                  // Initial number of sites actives
        int num_colors;             // Number of colors of network
        vector<double> rho;         // Allocate initial density colors of network
        vector<double> p;           // Allocate p(t)-values

    public:
        // Constructor to p
        network(int num_samples, int num_colors_)
            : num_of_samples(num_samples), p(num_samples, 0.0), rho(num_colors_, 0.0), p0(num_colors_,0.0) {
        };
        
        double generate_p(const int type_N_t, const double p0, const int t_i, const int N_current, const double k, const double a, const double alpha);
        
        double type_Nt_create(const int type_N_t, const int t_i, const double a, const double alpha);
        
        NetworkPattern create_network(const int dim, const int lenght_network, const int num_of_samples,
                                       const double k, const double N_t, const int type_N_t,
                                       const std::vector<double> p0, const double P0, const double a, const double alpha,
                                       const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
                                       TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng);
        
        NetworkPattern initialize_network(int dim, int length_network, int num_colors,
                                           double P0,
                                           const std::vector<double>& rho,
                                           const std::vector<double>& p0,
                                           int seed);
        
        void print_initial_site_fractions(const NetworkPattern& net);
        
        // Create a net for animate 
        NetworkPattern animate_network(const int dim, const int lenght_network, const int num_of_samples,
                                       const double k, const double N_t, const int seed, const int type_N_t,
                                       const std::vector<double> p0, const double P0, const double a, const double alpha,
                                       const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
                                       TimeSeries& ts_out);
        
        // Calculate the bigger cluster for each color
        void compute_max_cluster_per_color(
            const NetworkPattern& net,
            int dim,
            const std::vector<int>& shape,
            int grow_axis,
            int num_colors,
            std::vector<int>& M_size_out);
};


#endif // network_hpp
