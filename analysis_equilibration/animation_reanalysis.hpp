#pragma once

#include <string>
#include <vector>

#include "../src/struct_network.hpp"

struct ReanalysisConfig {
    int species_factor = 10000000;

    int smoothing_window = 7;
    int min_stable_steps = 20;

    double rel_tol = 2.0e-2;
    double abs_tol = 1.0e-6;
    double sigma_multiplier = 2.0;
};

struct SubgraphAnalysis {
    NetworkPattern net;

    // Mantidos por compatibilidade com o write_save,
    // mas aqui não têm semântica de "percolação global".
    std::vector<int> color_percolation;
    std::vector<int> percolation_order;

    std::vector<int> largest_component;
    std::vector<int> sp_len;
    std::vector<std::vector<int>> sp_path_lin;

    SubgraphAnalysis()
        : net(2, std::vector<int>{1, 1}, 1, std::vector<double>{1.0}) {}
};

struct ReanalysisResult {
    int t_eq = 0;

    // sítios ativos com t <= t_eq
    SubgraphAnalysis pre_teq;

    // sítios ativos com t > t_eq
    SubgraphAnalysis post_teq;
};

TimeSeries load_timeseries_from_json(const std::string& json_path);

NetworkPattern load_encoded_network_from_npz(const std::string& npz_path);

int estimate_t_eq(const TimeSeries& ts, const ReanalysisConfig& cfg);
int estimate_t_eq_from_json(const std::string& json_path, const ReanalysisConfig& cfg);

NetworkPattern rebuild_network_from_animation(
    const NetworkPattern& encoded_net,
    int t_eq,
    int species_factor = 10000000);

ReanalysisResult reanalyze_animation(
    const std::string& json_path,
    const std::string& npz_path,
    const ReanalysisConfig& cfg);