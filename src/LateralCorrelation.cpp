#include "LateralCorrelation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int lateral_index_2d(const int x, const int y, const int L) {
    return x + L * y;
}

}  // namespace

LateralObservablesSeries compute_lateral_observables(
    const int dim,
    const int L,
    const int sx,
    const int sy,
    const int sz,
    const std::vector<std::uint32_t>& activation_time,
    const std::vector<int>& times,
    const std::string& boundary_mode,
    const bool store_correlation_rows)
{
    if (dim != 2 && dim != 3) {
        throw std::invalid_argument("compute_lateral_observables: dim must be 2 or 3");
    }
    if (L <= 0) {
        throw std::invalid_argument("compute_lateral_observables: L must be positive");
    }
    if (sx <= 0 || sy <= 0 || sz <= 0) {
        throw std::invalid_argument("compute_lateral_observables: grid dimensions must be positive");
    }

    LateralObservablesSeries result;
    result.dim = dim;
    result.L = L;
    result.r_max = std::max(0, L / 2);
    result.boundary_mode = boundary_mode.empty() ? "periodic" : boundary_mode;

    const int max_time = times.empty() ? 0 : *std::max_element(times.begin(), times.end());
    std::vector<std::vector<int>> sites_by_time(static_cast<std::size_t>(max_time + 1));

    for (int idx = 0; idx < static_cast<int>(activation_time.size()); ++idx) {
        const auto activation = activation_time[static_cast<std::size_t>(idx)];
        if (activation == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        const int t = static_cast<int>(activation);
        if (t < 0 || t > max_time) {
            continue;
        }
        sites_by_time[static_cast<std::size_t>(t)].push_back(idx);
    }

    const int lateral_size = (dim == 2) ? L : (L * L);
    const double normalizer = static_cast<double>((dim == 2) ? L : (L * L));

    for (const int t : times) {
        if (t < 0 || t > max_time) {
            continue;
        }

        const std::vector<int>& positions = sites_by_time[static_cast<std::size_t>(t)];
        std::vector<std::uint8_t> field(static_cast<std::size_t>(lateral_size), 0);
        for (const int idx : positions) {
            const int x = idx % sx;
            const int y = (dim >= 2) ? ((idx / sx) % sy) : 0;
            if (dim == 2) {
                field[static_cast<std::size_t>(x)] = 1;
            } else {
                field[static_cast<std::size_t>(lateral_index_2d(x, y, L))] = 1;
            }
        }

        std::vector<int> active_lateral;
        active_lateral.reserve(static_cast<std::size_t>(std::min(lateral_size, static_cast<int>(positions.size()))));
        for (int idx = 0; idx < lateral_size; ++idx) {
            if (field[static_cast<std::size_t>(idx)] != 0) {
                active_lateral.push_back(idx);
            }
        }

        const double f_t = static_cast<double>(active_lateral.size()) / normalizer;
        result.f_t.push_back(f_t);
        result.t.push_back(t);

        double chi_raw_incl0 = 0.0;
        double chi_raw_excl0 = 0.0;
        double chi_norm_incl0 = 0.0;
        double chi_norm_excl0 = 0.0;
        int n_valid_norm = 0;
        double c_norm_sum = 0.0;
        double c_norm_sumsq = 0.0;
        double c_norm_absmax = 0.0;
        int r_at_absmax = 0;
        double valid_norm_sum = 0.0;
        double pair_count_sum = 0.0;
        int n_corr_rows = 0;

        for (int r = 0; r <= result.r_max; ++r) {
            double G = 0.0;
            int pair_count = 0;

            if (dim == 2) {
                const bool periodic = (boundary_mode == "periodic");
                if (periodic) {
                    for (const int x : active_lateral) {
                        const int x2 = (x + r) % L;
                        G += static_cast<double>(field[static_cast<std::size_t>(x2)]);
                    }
                    pair_count = L;
                    G /= static_cast<double>(pair_count);
                } else {
                    for (const int x : active_lateral) {
                        const int x2 = x + r;
                        if (x2 < 0 || x2 >= L) {
                            continue;
                        }
                        G += static_cast<double>(field[static_cast<std::size_t>(x2)]);
                    }
                    pair_count = std::max(0, L - r);
                    if (pair_count > 0) {
                        G /= static_cast<double>(pair_count);
                    }
                }
            } else {
                const bool periodic = (boundary_mode == "periodic");
                if (periodic) {
                    for (const int idx_xy : active_lateral) {
                        const int x = idx_xy % L;
                        const int y = idx_xy / L;
                        const int x2 = (x + r) % L;
                        const int y2 = (y + r) % L;
                        const int idx_xp = lateral_index_2d(x2, y, L);
                        const int idx_yp = lateral_index_2d(x, y2, L);
                        G += static_cast<double>(field[static_cast<std::size_t>(idx_xp)]);
                        G += static_cast<double>(field[static_cast<std::size_t>(idx_yp)]);
                    }
                    pair_count = 2 * L * L;
                    G /= static_cast<double>(pair_count);
                } else {
                    for (const int idx_xy : active_lateral) {
                        const int x = idx_xy % L;
                        const int y = idx_xy / L;
                        if (x + r < L) {
                            const int idx_xp = lateral_index_2d(x + r, y, L);
                            G += static_cast<double>(field[static_cast<std::size_t>(idx_xp)]);
                        }
                        if (y + r < L) {
                            const int idx_yp = lateral_index_2d(x, y + r, L);
                            G += static_cast<double>(field[static_cast<std::size_t>(idx_yp)]);
                        }
                    }
                    pair_count = 2 * L * std::max(0, L - r);
                    if (pair_count > 0) {
                        G /= static_cast<double>(pair_count);
                    }
                }
            }

            const double c_raw = G - f_t * f_t;
            double c_norm = 0.0;
            int valid_norm = 0;
            const double denom = f_t * (1.0 - f_t);
            if (denom > 1.0e-14) {
                c_norm = c_raw / denom;
                valid_norm = 1;
            }

            if (store_correlation_rows) {
                LateralCorrelationRow row;
                row.t = t;
                row.r = r;
                row.f_t = f_t;
                row.G = G;
                row.C_raw = c_raw;
                row.C_norm = c_norm;
                row.valid_norm = valid_norm;
                row.pair_count = pair_count;
                result.correlation_rows.push_back(row);
            }

            chi_raw_incl0 += c_raw;
            if (r > 0) {
                chi_raw_excl0 += c_raw;
            }
            chi_norm_incl0 += c_norm;
            if (r > 0) {
                chi_norm_excl0 += c_norm;
            }
            if (valid_norm == 1) {
                ++n_valid_norm;
            }
            c_norm_sum += c_norm;
            c_norm_sumsq += c_norm * c_norm;
            if (std::fabs(c_norm) > c_norm_absmax) {
                c_norm_absmax = std::fabs(c_norm);
                r_at_absmax = r;
            }
            valid_norm_sum += static_cast<double>(valid_norm);
            pair_count_sum += static_cast<double>(pair_count);
            ++n_corr_rows;
        }

        LateralCorrelationSummaryRow corr_summary;
        corr_summary.t = t;
        corr_summary.f_t = f_t;
        corr_summary.r_max = result.r_max;
        corr_summary.n_rows = n_corr_rows;
        if (n_corr_rows > 0) {
            corr_summary.C_norm_mean = c_norm_sum / static_cast<double>(n_corr_rows);
            if (n_corr_rows > 1) {
                const double mean = corr_summary.C_norm_mean;
                const double variance =
                    std::max(0.0, (c_norm_sumsq - static_cast<double>(n_corr_rows) * mean * mean) /
                                      static_cast<double>(n_corr_rows - 1));
                corr_summary.C_norm_std = std::sqrt(variance);
            }
            corr_summary.C_norm_absmax = c_norm_absmax;
            corr_summary.r_at_absmax = r_at_absmax;
            corr_summary.valid_norm_mean = valid_norm_sum / static_cast<double>(n_corr_rows);
            corr_summary.pair_count_mean = pair_count_sum / static_cast<double>(n_corr_rows);
        }
        result.correlation_summary_rows.push_back(corr_summary);

        LateralSusceptibilityRow sus;
        sus.t = t;
        sus.f_t = f_t;
        sus.r_max = result.r_max;
        sus.chi_raw_incl0 = chi_raw_incl0;
        sus.chi_raw_excl0 = chi_raw_excl0;
        sus.chi_norm_incl0 = chi_norm_incl0;
        sus.chi_norm_excl0 = chi_norm_excl0;
        sus.n_valid_norm = n_valid_norm;
        result.susceptibility_rows.push_back(sus);
    }

    return result;
}

bool validate_lateral_observables(std::string* error_message)
{
    auto fail = [&](const std::string& message) {
        if (error_message != nullptr) {
            *error_message = message;
        }
        return false;
    };

    const std::vector<std::uint32_t> empty_layer(4, std::numeric_limits<std::uint32_t>::max());
    const auto empty = compute_lateral_observables(2, 4, 4, 1, 1, empty_layer, {0}, "periodic");
    if (empty.f_t.empty() || empty.f_t.front() != 0.0) {
        return fail("empty layer should have f_t = 0");
    }
    if (!empty.correlation_rows.empty()) {
        for (const auto& row : empty.correlation_rows) {
            if (row.C_raw != 0.0 || row.C_norm != 0.0 || row.valid_norm != 0) {
                return fail("empty layer should yield zero correlations");
            }
        }
    }

    const std::vector<std::uint32_t> full_layer(4, 0u);
    const auto full = compute_lateral_observables(2, 4, 4, 1, 1, full_layer, {0}, "periodic");
    if (full.f_t.empty() || full.f_t.front() != 1.0) {
        return fail("full layer should have f_t = 1");
    }
    if (!full.correlation_rows.empty()) {
        for (const auto& row : full.correlation_rows) {
            if (row.C_raw != 0.0 || row.C_norm != 0.0 || row.valid_norm != 0) {
                return fail("full layer should yield zero correlations");
            }
        }
    }

    const std::vector<std::uint32_t> alternating_layer = {
        0u,
        std::numeric_limits<std::uint32_t>::max(),
        0u,
        std::numeric_limits<std::uint32_t>::max()
    };
    const auto alt = compute_lateral_observables(2, 4, 4, 1, 1, alternating_layer, {0}, "periodic");
    if (alt.f_t.empty() || std::fabs(alt.f_t.front() - 0.5) > 1e-12) {
        return fail("alternating layer should have f_t = 0.5");
    }
    const auto& row0 = alt.correlation_rows[0];
    const auto& row1 = alt.correlation_rows[1];
    const auto& row2 = alt.correlation_rows[2];
    if (std::fabs(row0.C_raw - 0.25) > 1e-12 || row0.C_norm != 1.0 || row0.valid_norm != 1) {
        return fail("alternating layer should satisfy C_raw(0)=0.25 and C_norm(0)=1");
    }
    if (!(row1.C_raw < 0.0) || !(row1.C_norm < 0.0)) {
        return fail("alternating layer should have negative correlation at r=1");
    }
    if (!(row2.C_raw > 0.0) || !(row2.C_norm > 0.0)) {
        return fail("alternating layer should have positive correlation at r=2");
    }

    return true;
}
