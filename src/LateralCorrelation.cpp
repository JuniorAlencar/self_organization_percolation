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

void add_periodic_axis_hits(
    std::vector<int>& values,
    const int L,
    const int r_max,
    std::vector<double>& hits)
{
    const int n = static_cast<int>(values.size());
    if (n == 0) {
        return;
    }

    hits[0] += static_cast<double>(n);
    if (n == 1) {
        return;
    }

    std::sort(values.begin(), values.end());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int d = values[static_cast<std::size_t>(j)] -
                          values[static_cast<std::size_t>(i)];
            if (d <= r_max) {
                hits[static_cast<std::size_t>(d)] += 1.0;
            }
            const int wrapped_d = L - d;
            if (wrapped_d <= r_max) {
                hits[static_cast<std::size_t>(wrapped_d)] += 1.0;
            }
        }
    }
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
    const bool store_correlation_rows,
    const std::vector<int>* activated_indices)
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

    auto add_activated_index = [&](const int idx) {
        if (idx < 0 || idx >= static_cast<int>(activation_time.size())) {
            return;
        }
        const auto activation = activation_time[static_cast<std::size_t>(idx)];
        if (activation == std::numeric_limits<std::uint32_t>::max()) {
            return;
        }
        const int t = static_cast<int>(activation);
        if (t < 0 || t > max_time) {
            return;
        }
        sites_by_time[static_cast<std::size_t>(t)].push_back(idx);
    };

    if (activated_indices != nullptr) {
        for (const int idx : *activated_indices) {
            add_activated_index(idx);
        }
    } else {
        for (int idx = 0; idx < static_cast<int>(activation_time.size()); ++idx) {
            add_activated_index(idx);
        }
    }

    const int lateral_size = (dim == 2) ? L : (L * L);
    const double normalizer = static_cast<double>((dim == 2) ? L : (L * L));
    const bool periodic = (result.boundary_mode == "periodic");
    const int r_max = result.r_max;

    std::vector<int> stamp(static_cast<std::size_t>(lateral_size), -1);
    int stamp_id = 0;
    std::vector<int> active_lateral;
    active_lateral.reserve(static_cast<std::size_t>(lateral_size));

    std::vector<double> periodic_hits(static_cast<std::size_t>(r_max + 1), 0.0);
    std::vector<std::vector<int>> rows(static_cast<std::size_t>(dim == 3 ? L : 0));
    std::vector<std::vector<int>> cols(static_cast<std::size_t>(dim == 3 ? L : 0));
    std::vector<int> row_stamp(static_cast<std::size_t>(dim == 3 ? L : 0), -1);
    std::vector<int> col_stamp(static_cast<std::size_t>(dim == 3 ? L : 0), -1);
    std::vector<int> touched_rows;
    std::vector<int> touched_cols;

    auto is_active = [&](const int idx) {
        return stamp[static_cast<std::size_t>(idx)] == stamp_id;
    };

    for (const int t : times) {
        if (t < 0 || t > max_time) {
            continue;
        }

        const std::vector<int>& positions = sites_by_time[static_cast<std::size_t>(t)];
        if (stamp_id == std::numeric_limits<int>::max()) {
            std::fill(stamp.begin(), stamp.end(), -1);
            if (dim == 3) {
                std::fill(row_stamp.begin(), row_stamp.end(), -1);
                std::fill(col_stamp.begin(), col_stamp.end(), -1);
            }
            stamp_id = 0;
        }
        ++stamp_id;
        active_lateral.clear();
        touched_rows.clear();
        touched_cols.clear();

        for (const int idx : positions) {
            const int x = idx % sx;
            const int y = (dim >= 2) ? ((idx / sx) % sy) : 0;
            const int lateral_idx = (dim == 2) ? x : lateral_index_2d(x, y, L);
            if (stamp[static_cast<std::size_t>(lateral_idx)] == stamp_id) {
                continue;
            }
            stamp[static_cast<std::size_t>(lateral_idx)] = stamp_id;
            active_lateral.push_back(lateral_idx);

            if (dim == 3 && periodic) {
                if (row_stamp[static_cast<std::size_t>(y)] != stamp_id) {
                    row_stamp[static_cast<std::size_t>(y)] = stamp_id;
                    rows[static_cast<std::size_t>(y)].clear();
                    touched_rows.push_back(y);
                }
                rows[static_cast<std::size_t>(y)].push_back(x);

                if (col_stamp[static_cast<std::size_t>(x)] != stamp_id) {
                    col_stamp[static_cast<std::size_t>(x)] = stamp_id;
                    cols[static_cast<std::size_t>(x)].clear();
                    touched_cols.push_back(x);
                }
                cols[static_cast<std::size_t>(x)].push_back(y);
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

        bool use_sparse_periodic_hits = false;
        if (dim == 3 && periodic) {
            long double pair_work = 0.0L;
            for (const int y : touched_rows) {
                const auto n = static_cast<long double>(rows[static_cast<std::size_t>(y)].size());
                pair_work += n * (n - 1.0L) / 2.0L;
            }
            for (const int x : touched_cols) {
                const auto n = static_cast<long double>(cols[static_cast<std::size_t>(x)].size());
                pair_work += n * (n - 1.0L) / 2.0L;
            }

            const long double scan_work =
                static_cast<long double>(active_lateral.size()) *
                static_cast<long double>(r_max + 1);
            use_sparse_periodic_hits = (pair_work < scan_work);

            if (use_sparse_periodic_hits) {
                std::fill(periodic_hits.begin(), periodic_hits.end(), 0.0);
                for (const int y : touched_rows) {
                    add_periodic_axis_hits(
                        rows[static_cast<std::size_t>(y)],
                        L,
                        r_max,
                        periodic_hits);
                }
                for (const int x : touched_cols) {
                    add_periodic_axis_hits(
                        cols[static_cast<std::size_t>(x)],
                        L,
                        r_max,
                        periodic_hits);
                }
            }
        }

        for (int r = 0; r <= r_max; ++r) {
            double G = 0.0;
            int pair_count = 0;

            if (dim == 2) {
                if (periodic) {
                    for (const int x : active_lateral) {
                        const int x2 = (x + r) % L;
                        G += is_active(x2) ? 1.0 : 0.0;
                    }
                    pair_count = L;
                    G /= static_cast<double>(pair_count);
                } else {
                    for (const int x : active_lateral) {
                        const int x2 = x + r;
                        if (x2 < 0 || x2 >= L) {
                            continue;
                        }
                        G += is_active(x2) ? 1.0 : 0.0;
                    }
                    pair_count = std::max(0, L - r);
                    if (pair_count > 0) {
                        G /= static_cast<double>(pair_count);
                    }
                }
            } else {
                if (periodic) {
                    if (use_sparse_periodic_hits) {
                        G = periodic_hits[static_cast<std::size_t>(r)];
                    } else {
                        for (const int idx_xy : active_lateral) {
                            const int x = idx_xy % L;
                            const int y = idx_xy / L;
                            const int x2 = (x + r) % L;
                            const int y2 = (y + r) % L;
                            const int idx_xp = lateral_index_2d(x2, y, L);
                            const int idx_yp = lateral_index_2d(x, y2, L);
                            G += is_active(idx_xp) ? 1.0 : 0.0;
                            G += is_active(idx_yp) ? 1.0 : 0.0;
                        }
                    }
                    pair_count = 2 * L * L;
                    G /= static_cast<double>(pair_count);
                } else {
                    for (const int idx_xy : active_lateral) {
                        const int x = idx_xy % L;
                        const int y = idx_xy / L;
                        if (x + r < L) {
                            const int idx_xp = lateral_index_2d(x + r, y, L);
                            G += is_active(idx_xp) ? 1.0 : 0.0;
                        }
                        if (y + r < L) {
                            const int idx_yp = lateral_index_2d(x, y + r, L);
                            G += is_active(idx_yp) ? 1.0 : 0.0;
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
