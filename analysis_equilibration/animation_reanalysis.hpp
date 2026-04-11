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

struct ReanalysisResult {
    int t_eq = 0;

    NetworkPattern filtered_net;

    std::vector<int> color_percolation;
    std::vector<int> percolation_order;

    // tamanho do maior componente por espécie (índice 0-based)
    // -1 para espécies não percolantes
    std::vector<int> largest_component;

    // shortest path por espécie (índice 0-based)
    // vazio para espécies não percolantes
    std::vector<int> sp_len;
    std::vector<std::vector<int>> sp_path_lin;

    ReanalysisResult()
        : filtered_net(2, std::vector<int>{1, 1}, 1, std::vector<double>{1.0}) {}
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