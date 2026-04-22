#ifndef EQUILIBRATION_PARTITION_HPP
#define EQUILIBRATION_PARTITION_HPP

#include "struct_network.hpp"

struct EquilibrationConfig {
    int species_factor = 10000000;
    int smoothing_window = 25;
    int min_stable_steps = 25;
    double rel_tol = 2.0e-2;
    double abs_tol = 1.0e-6;
    double sigma_multiplier = 2.0;
};

int estimate_t_eq(const TimeSeries& ts, const EquilibrationConfig& cfg = {});
void compute_equilibration_partition_metrics(
    const NetworkPattern& encoded_net,
    const TimeSeries& ts,
    PercolationSeries& ps,
    const EquilibrationConfig& cfg = {});


struct EquilibrationCutNetworks {
    NetworkPattern pre_teq;
    NetworkPattern post_teq;

    EquilibrationCutNetworks(const NetworkPattern& pre,
                             const NetworkPattern& post)
        : pre_teq(pre), post_teq(post) {}
};


EquilibrationCutNetworks build_equilibration_cut_networks(
    const NetworkPattern& encoded_net,
    int t_eq,
    int species_factor = 10000000);

#endif // EQUILIBRATION_PARTITION_HPP