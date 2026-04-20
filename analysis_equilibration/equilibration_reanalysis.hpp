#ifndef EQUILIBRATION_REANALYSIS_HPP
#define EQUILIBRATION_REANALYSIS_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "../src/struct_network.hpp"

struct ReanalysisConfig {
    int species_factor = 10000000;

    int smoothing_window = 7;
    int min_stable_steps = 20;

    double rel_tol = 2.0e-2;
    double abs_tol = 1.0e-6;
    double sigma_multiplier = 2.0;
};

struct SparseEncodedNetwork {
    int dim = 0;
    int num_colors = 0;
    int seed = 0;

    std::vector<int> shape;
    std::vector<double> rho;

    std::vector<int> active_idx;   // índice linear
    std::vector<int> active_val;   // valor codificado = species_factor*(c+1) + t

    std::unordered_map<int, int> encoded_value_by_idx;
    std::vector<std::unordered_set<int>> active_idx_by_color;

    std::size_t total_size = 0;
};

struct SparseSubgraph {
    int dim = 0;
    int num_colors = 0;

    std::vector<int> shape;
    std::vector<double> rho;

    // subgrafo já decodificado por cor
    std::vector<int> active_idx;   // índice linear
    std::vector<int> active_val;   // valor decodificado: 1 ou (c+2)

    std::unordered_map<int, int> value_by_idx;
    std::vector<std::unordered_set<int>> active_idx_by_color;

    std::size_t total_size = 0;
};

struct SubgraphAnalysis {
    SparseSubgraph net;

    std::vector<int> color_percolation;
    std::vector<int> percolation_order;

    std::vector<int> largest_component;
    std::vector<int> sp_len;
    std::vector<std::vector<int>> sp_path_lin;
};

struct ReanalysisResult {
    int t_eq = 0;

    SubgraphAnalysis pre_teq;
    SubgraphAnalysis post_teq;
};

TimeSeries load_timeseries_from_json(const std::string& json_path);

SparseEncodedNetwork load_sparse_encoded_network_from_npz(
    const std::string& npz_path,
    int species_factor);

int estimate_t_eq(const TimeSeries& ts, const ReanalysisConfig& cfg);
int estimate_t_eq_from_json(const std::string& json_path, const ReanalysisConfig& cfg);

SparseSubgraph build_preteq_sparse_subgraph(
    const SparseEncodedNetwork& encoded_net,
    int t_eq,
    int species_factor);

SparseSubgraph build_postteq_sparse_subgraph(
    const SparseEncodedNetwork& encoded_net,
    int t_eq,
    int species_factor);

ReanalysisResult reanalyze_animation(
    const std::string& json_path,
    const std::string& npz_path,
    const ReanalysisConfig& cfg);

#endif