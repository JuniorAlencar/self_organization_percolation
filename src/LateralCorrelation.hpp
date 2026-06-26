#ifndef LATERAL_CORRELATION_HPP
#define LATERAL_CORRELATION_HPP

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

struct LateralCorrelationRow {
    int t = 0;
    int r = 0;
    double f_t = 0.0;
    double G = 0.0;
    double C_raw = 0.0;
    double C_norm = 0.0;
    int valid_norm = 0;
    int pair_count = 0;
};

struct LateralSusceptibilityRow {
    int t = 0;
    double f_t = 0.0;
    int r_max = 0;
    double chi_raw_incl0 = 0.0;
    double chi_raw_excl0 = 0.0;
    double chi_norm_incl0 = 0.0;
    double chi_norm_excl0 = 0.0;
    int n_valid_norm = 0;
};

struct LateralObservablesSeries {
    std::string sample_id;
    int dim = 0;
    int L = 0;
    int r_max = 0;
    std::string boundary_mode = "periodic";
    std::vector<int> t;
    std::vector<double> f_t;
    std::vector<LateralCorrelationRow> correlation_rows;
    std::vector<LateralSusceptibilityRow> susceptibility_rows;

    double f_T = std::numeric_limits<double>::quiet_NaN();
    double p0 = std::numeric_limits<double>::quiet_NaN();
    double P0 = std::numeric_limits<double>::quiet_NaN();
    double c = std::numeric_limits<double>::quiet_NaN();
    std::string type_percolation;
    int seed = -1;
    double t_stat = std::numeric_limits<double>::quiet_NaN();
};

LateralObservablesSeries compute_lateral_observables(
    int dim,
    int L,
    int sx,
    int sy,
    int sz,
    const std::vector<std::uint32_t>& activation_time,
    const std::vector<int>& times,
    const std::string& boundary_mode = "periodic");

bool validate_lateral_observables(std::string* error_message = nullptr);

#endif
