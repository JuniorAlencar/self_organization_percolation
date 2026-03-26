#ifndef network_hpp
#define network_hpp

#include <string>
#include <vector>

#include "rand_utils.hpp"
#include "LargestComponentBFS.hpp"

class network {
private:
    int lenght_network;
    int num_of_samples;
    double k;
    double N_t = 0.0;
    int type_N_t;
    double a;
    double alpha;
    int dim;
    int seed;
    std::string type_percolation;
    std::vector<double> p0;
    double P0;
    int num_colors;
    std::vector<double> rho;
    std::vector<double> p;
    bool DSU_calculate_;

public:
    network(int num_samples, int num_colors_)
        : num_of_samples(num_samples),
          p(num_colors_, 0.0),
          p0(num_colors_, 0.0),
          rho(num_colors_, 0.0) {}

    double generate_p(int type_N_t, double p0, int t_i, int N_current, double k, double a, double alpha);
    double type_Nt_create(int type_N_t, int t_i, double a, double alpha);

    NetworkPattern create_network(
        int dim,
        int lenght_network,
        int num_of_samples,
        double k,
        double N_t,
        int type_N_t,
        std::vector<double> p0,
        double P0,
        double a,
        double alpha,
        const std::string& type_percolation,
        const int& num_colors,
        const std::vector<double>& rho,
        TimeSeries& ts_out,
        PercolationSeries& ps_out,
        all_random& rng);

    NetworkPattern initialize_network(
        int dim,
        int length_network,
        int num_colors,
        double P0,
        const std::vector<double>& rho,
        const std::vector<double>& p0,
        int seed);

    void print_initial_site_fractions(const NetworkPattern& net);

    NetworkPattern animate_network(
        int dim,
        int lenght_network,
        int num_of_samples,
        double k,
        double N_t,
        int type_N_t,
        std::vector<double> p0,
        double P0,
        double a,
        double alpha,
        const std::string& type_percolation,
        const int& num_colors,
        const std::vector<double>& rho,
        TimeSeries& ts_out,
        PercolationSeries& ps_out,
        all_random& rng);

    NetworkPattern create_shortest_paths_map(
        const NetworkPattern& net,
        const PercolationSeries& ps_out);
};

#endif // network_hpp
