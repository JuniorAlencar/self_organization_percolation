#include "write_save.hpp"
#include "equilibration_partition.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;

namespace {

// NPZ helpers removed. JSON/binary output is used instead.

template <typename T>
void write_json_array(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if constexpr (std::is_floating_point<T>::value) {
            os << std::setprecision(17) << v[i];
        } else {
            os << v[i];
        }
        if (i + 1 < v.size()) os << ", ";
    }
    os << "]";
}

void write_json_nullable_double(std::ostream& os, const double value)
{
    if (std::isfinite(value)) {
        os << std::setprecision(17) << value;
    } else {
        os << "null";
    }
}

void write_json_nullable_double_array(std::ostream& os,
                                      const std::vector<double>& v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        write_json_nullable_double(os, v[i]);
        if (i + 1 < v.size()) os << ", ";
    }
    os << "]";
}

void write_json_nullable_int_array(std::ostream& os,
                                   const std::vector<int>& v)
{
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] >= 0) {
            os << v[i];
        } else {
            os << "null";
        }
        if (i + 1 < v.size()) os << ", ";
    }
    os << "]";
}

void write_json_path_edge_pairs(std::ostream& os, const std::vector<int>& path)
{
    os << "[";
    if (path.size() >= 2) {
        for (size_t i = 1; i < path.size(); ++i) {
            os << "[" << path[i - 1] << ", " << path[i] << "]";
            if (i + 1 < path.size()) os << ", ";
        }
    }
    os << "]";
}

template <typename T>
void write_json_row(std::ostream& os,
                    const std::vector<std::vector<T>>& m,
                    const int row)
{
    if (row < 0 || row >= static_cast<int>(m.size())) {
        os << "[]";
        return;
    }
    write_json_array(os, m[static_cast<size_t>(row)]);
}

void write_subgraph_json(std::ostream& ofs,
                         const std::string& key,
                         const SubgraphAnalysis& sg,
                         const bool add_comma)
{
    ofs << "  \"" << key << "\": {\n";

    ofs << "    \"color_percolation\": ";
    write_json_array(ofs, sg.color_percolation);
    ofs << ",\n";

    ofs << "    \"percolation_order\": ";
    write_json_array(ofs, sg.percolation_order);
    ofs << ",\n";

    ofs << "    \"largest_component\": ";
    write_json_array(ofs, sg.largest_component);
    ofs << ",\n";

    ofs << "    \"sp_len\": ";
    write_json_array(ofs, sg.sp_len);
    ofs << ",\n";

    ofs << "    \"sp_path_lin\": [";
    for (size_t i = 0; i < sg.sp_path_lin.size(); ++i) {
        write_json_array(ofs, sg.sp_path_lin[i]);
        if (i + 1 < sg.sp_path_lin.size()) ofs << ", ";
    }
    ofs << "],\n";

    ofs << "    \"net_info\": {\n";
    ofs << "      \"dim\": " << sg.net.dim << ",\n";
    ofs << "      \"shape\": ";
    write_json_array(ofs, sg.net.shape);
    ofs << ",\n";
    ofs << "      \"num_colors\": " << sg.net.num_colors << ",\n";
    ofs << "      \"rho\": ";
    write_json_array(ofs, sg.net.rho);
    ofs << ",\n";
    ofs << "      \"num_active\": " << sg.net.active_idx.size() << "\n";
    ofs << "    }\n";

    ofs << "  }";
    if (add_comma) ofs << ",";
    ofs << "\n";
}

// NPZ payload writer removed. Use compact binary (`save_network_compact_bin`) or
// JSON outputs instead. Older API calls were migrated to compact/bin equivalents.

std::vector<int32_t> flatten_points4(const std::vector<Point3D>& pts)
{
    std::vector<int32_t> out;
    out.reserve(pts.size() * 4);

    for (const auto& p : pts) {
        out.push_back(static_cast<int32_t>(p.x));
        out.push_back(static_cast<int32_t>(p.y));
        out.push_back(static_cast<int32_t>(p.z));
        out.push_back(static_cast<int32_t>(p.color_index));
    }

    return out;
}

} // namespace

void save_data::save_network_as_npz(const NetworkPattern& net,
                                    const std::string& filename) const
{
    std::vector<int32_t> active_idx;
    std::vector<int32_t> active_val;

    active_idx.reserve(net.data.size() / 16);
    active_val.reserve(net.data.size() / 16);

    for (std::size_t i = 0; i < net.data.size(); ++i) {
        const int v = static_cast<int>(net.data[i]);

        if (v <= 0) continue;

        if (i > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "save_network_as_npz(NetworkPattern): índice linear excede int32");
        }

        active_idx.push_back(static_cast<int32_t>(i));
        active_val.push_back(static_cast<int32_t>(v));
    }

    // Convert to compact binary and save as .bin next to requested filename.
    NetworkCompact cnet;
    cnet.N = static_cast<NetworkCompact::index_t>(net.data.size());
    cnet.pos_flat.resize(cnet.N);
    cnet.species.resize(cnet.N);
    cnet.activation_time.resize(cnet.N);

    // Fill pos_flat and decode encoded values into species/time using
    // ANIMATION_SPECIES_FACTOR convention.
    const int species_factor = 10000000;
    for (NetworkCompact::index_t i = 0; i < cnet.N; ++i) {
        cnet.pos_flat[i] = i;
        const long long v = static_cast<long long>(net.data[i]);
        if (v <= 0) {
            cnet.species[i] = 0;
            cnet.activation_time[i] = 0;
        } else {
            const int color = static_cast<int>(v / species_factor);
            const int time = static_cast<int>(v % species_factor);
            cnet.species[i] = static_cast<uint8_t>(color > 255 ? 255 : color);
            cnet.activation_time[i] = static_cast<uint32_t>(time);
        }
    }

    std::string out = filename;
    if (out.size() >= 4 && out.substr(out.size()-4) == ".npz") {
        out = out.substr(0, out.size()-4) + ".bin";
    } else {
        out += ".bin";
    }
    save_network_compact_bin(cnet, out);
}

void save_data::save_network_as_npz(const SparseSubgraph& net,
                                    const std::string& filename) const
{
    if (net.active_idx.size() != net.active_val.size()) {
        throw std::runtime_error(
            "save_network_as_npz(SparseSubgraph): active_idx.size != active_val.size");
    }

    std::vector<int32_t> active_idx;
    std::vector<int32_t> active_val;

    active_idx.reserve(net.active_idx.size());
    active_val.reserve(net.active_val.size());

    for (std::size_t k = 0; k < net.active_idx.size(); ++k) {
        const int lin = net.active_idx[k];
        const int val = net.active_val[k];

        if (lin < 0) {
            throw std::runtime_error(
                "save_network_as_npz(SparseSubgraph): índice linear negativo");
        }
        if (val <= 0) {
            throw std::runtime_error(
                "save_network_as_npz(SparseSubgraph): active_val deve ser > 0");
        }

        active_idx.push_back(static_cast<int32_t>(lin));
        active_val.push_back(static_cast<int32_t>(val));
    }

    // Convert sparse subgraph to compact binary representation and save.
    NetworkCompact cnet;
    cnet.N = static_cast<NetworkCompact::index_t>(net.total_size);
    cnet.pos_flat.resize(cnet.N);
    cnet.species.assign(cnet.N, 0);
    cnet.activation_time.assign(cnet.N, 0);

    for (NetworkCompact::index_t i = 0; i < cnet.N; ++i) cnet.pos_flat[i] = i;
    for (size_t k = 0; k < static_cast<size_t>(net.active_idx.size()); ++k) {
        const int lin = net.active_idx[k];
        const int val = net.active_val[k];
        if (lin < 0 || lin >= static_cast<int>(cnet.N)) continue;
        const int color = val;
        cnet.species[static_cast<size_t>(lin)] = static_cast<uint8_t>(
            color > 255 ? 255 : (color < 0 ? 0 : color));
    }

    std::string out = filename;
    if (out.size() >= 4 && out.substr(out.size()-4) == ".npz") {
        out = out.substr(0, out.size()-4) + ".bin";
    } else {
        out += ".bin";
    }
    save_network_compact_bin(cnet, out);
}

void save_data::save_percolation_json(const PercolationSeries& ps,
                                      const TimeSeries& ts,
                                      const std::string& filename_json,
                                      const bool sort_by_order) const
{
    if (!ps.percolation_order.empty()) {
        const size_t m = ps.percolation_order.size();
        if (ps.color_percolation.size() != m) {
            throw std::runtime_error(
                "[save_percolation_json] vetores de eventos com tamanhos diferentes.");
        }
        if (!ps.M_size_at_perc.empty() && ps.M_size_at_perc.size() != m) {
            throw std::runtime_error(
                "[save_percolation_json] M_size_at_perc.size != #events.");
        }
    }

    std::ofstream ofs(filename_json);
    if (!ofs) {
        throw std::runtime_error(
            std::string("[save_percolation_json] não abriu: ") + filename_json);
    }

    std::vector<size_t> idx(ps.percolation_order.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

    if (sort_by_order) {
        std::sort(idx.begin(), idx.end(), [&](const size_t a, const size_t b) {
            return ps.percolation_order[a] < ps.percolation_order[b];
        });
    }

    const bool growth_test_order_mode =
        !ps.t_eq_by_species.empty() &&
        !ps.z_max_final.empty() &&
        ps.z_max_at_perc.empty();

    std::vector<double> t_eq_ordered_for_json;
    std::vector<int> z_max_ordered_for_json;
    std::vector<int> z_stat_ordered_for_json;
    if (growth_test_order_mode) {
        t_eq_ordered_for_json.reserve(static_cast<size_t>(ts.num_colors));
        z_max_ordered_for_json.reserve(idx.size());
        z_stat_ordered_for_json.reserve(static_cast<size_t>(ts.num_colors));

        for (const size_t i : idx) {
            const int color_1b = ps.color_percolation[i];
            const int crow = color_1b - 1;
            if (crow < 0 || crow >= static_cast<int>(ps.t_eq_by_species.size())) {
                continue;
            }
            const double t_eq_species =
                ps.t_eq_by_species[static_cast<size_t>(crow)];
            if (!std::isfinite(t_eq_species)) {
                continue;
            }

            t_eq_ordered_for_json.push_back(t_eq_species);
            if (crow < static_cast<int>(ps.z_max_final.size())) {
                z_max_ordered_for_json.push_back(
                    ps.z_max_final[static_cast<size_t>(crow)]);
            }
            if (crow < static_cast<int>(ps.z_stat_by_species.size())) {
                z_stat_ordered_for_json.push_back(
                    ps.z_stat_by_species[static_cast<size_t>(crow)]);
            }
        }

        while (static_cast<int>(t_eq_ordered_for_json.size()) < ts.num_colors) {
            t_eq_ordered_for_json.push_back(
                std::numeric_limits<double>::quiet_NaN());
        }
        while (static_cast<int>(z_stat_ordered_for_json.size()) < ts.num_colors) {
            z_stat_ordered_for_json.push_back(-1);
        }
    }

    ofs << "{\n";
    ofs << "  \"meta\": {\n";
    ofs << "    \"num_colors\": " << ts.num_colors << ",\n";
    ofs << "    \"rho\": ";
    write_json_array(ofs, ps.rho);
    ofs << ",\n";
    ofs << "    \"t_eq\": ";
    write_json_nullable_double(ofs, ps.t_eq);
    ofs << ",\n";
    if (!ps.t_eq_by_species.empty()) {
        ofs << "    \"t_eq_by_species\": ";
        if (growth_test_order_mode) {
            write_json_nullable_double_array(ofs, t_eq_ordered_for_json);
        } else {
            write_json_nullable_double_array(ofs, ps.t_eq_by_species);
        }
        ofs << ",\n";
        if (growth_test_order_mode) {
            ofs << "    \"growth_test_t_eq_by_species_by_color\": ";
            write_json_nullable_double_array(ofs, ps.t_eq_by_species);
            ofs << ",\n";
        }
    }
    if (!ps.z_max_final.empty()) {
        ofs << "    \"z_max\": ";
        if (growth_test_order_mode) {
            write_json_nullable_int_array(ofs, z_max_ordered_for_json);
        } else {
            write_json_array(ofs, ps.z_max_final);
        }
        ofs << ",\n";
        if (growth_test_order_mode) {
            ofs << "    \"growth_test_z_max_by_color\": ";
            write_json_array(ofs, ps.z_max_final);
            ofs << ",\n";
        }
        if (!growth_test_order_mode) {
            ofs << "    \"z_max_final\": ";
            write_json_array(ofs, ps.z_max_final);
            ofs << ",\n";
        }
        if (!ps.z_stat_by_species.empty()) {
            ofs << "    \"z_stat\": ";
            if (growth_test_order_mode) {
                write_json_nullable_int_array(ofs, z_stat_ordered_for_json);
            } else {
                write_json_nullable_int_array(ofs, ps.z_stat_by_species);
            }
            ofs << ",\n";
            if (growth_test_order_mode) {
                ofs << "    \"growth_test_z_stat_by_color\": ";
                write_json_nullable_int_array(ofs, ps.z_stat_by_species);
                ofs << ",\n";
            }
        }
        if (ps.dynamics_window_steps > 0) {
            ofs << "    \"growth_test_dynamics_window_steps\": "
                << ps.dynamics_window_steps << ",\n";
        }
        if (!ps.species_final_status.empty()) {
            ofs << "    \"growth_test_species_final_status_by_color\": ";
            write_json_array(ofs, ps.species_final_status);
            ofs << ",\n";
            ofs << "    \"growth_test_species_final_status_legend\": "
                << "\"1=stabilized,-1=died,0=not_stabilized_at_stop\",\n";
            if (ps.species_final_status.size() >= 4) {
                ofs << "    \"growth_test_species_4_died_marker\": "
                    << (ps.species_final_status[3] == -1 ? "true" : "false")
                    << ",\n";
            }
        }
        if (!ps.growth_test_stop_reason.empty()) {
            ofs << "    \"growth_test_stop_reason_actual\": \""
                << ps.growth_test_stop_reason << "\",\n";
        }
        if (ps.growth_test_stop_time >= 0) {
            ofs << "    \"growth_test_stop_time\": "
                << ps.growth_test_stop_time << ",\n";
        }
        if (ps.equilibrium_consecutive_steps > 0) {
            ofs << "    \"growth_test_stop_criterion\": \"alive_species_pt_derivative_stability_or_death\",\n";
            ofs << "    \"growth_test_equilibrium_consecutive_steps\": "
                << ps.equilibrium_consecutive_steps << ",\n";
            if (ps.post_equilibrium_extra_steps >= 0) {
                ofs << "    \"growth_test_post_equilibrium_extra_steps\": "
                    << ps.post_equilibrium_extra_steps << ",\n";
            }
            if (ps.dynamic_min_stop_height >= 0) {
                ofs << "    \"growth_test_dynamic_min_stop_height\": "
                    << ps.dynamic_min_stop_height << ",\n";
            }
            if (ps.dynamic_max_stop_height >= 0) {
                ofs << "    \"growth_test_dynamic_max_stop_height\": "
                    << ps.dynamic_max_stop_height << ",\n";
            }
            ofs << "    \"growth_test_t_eq_smoothing_window\": 15,\n";
            ofs << "    \"growth_test_t_eq_window_block\": 10,\n";
            ofs << "    \"growth_test_t_eq_min_stable_steps\": "
                << std::max(ps.equilibrium_consecutive_steps, 100) << ",\n";
            ofs << "    \"growth_test_t_eq_validation_window_steps\": "
                << std::max(std::max(0, ps.post_equilibrium_extra_steps), 200)
                << ",\n";
            ofs << "    \"growth_test_t_eq_validation\": \"global_tail_drift_validation_of_blocked_pt_variation\",\n";
            ofs << "    \"growth_test_t_eq_s_prime_threshold\": 0.00001,\n";
            ofs << "    \"growth_test_t_eq_series\": \"pt\",\n";
            ofs << "    \"growth_test_t_eq_requires_target\": false,\n";
            ofs << "    \"growth_test_t_eq_requires_global_height\": true,\n";
        }
        if (std::isfinite(ps.equilibrium_rel_tol)) {
            ofs << "    \"growth_test_equilibrium_rel_tol\": "
                << ps.equilibrium_rel_tol << ",\n";
            const double effective_rel_tol =
                ps.equilibrium_rel_tol * 0.01;
            ofs << "    \"growth_test_equilibrium_effective_rel_tol\": "
                << effective_rel_tol << ",\n";
            ofs << "    \"growth_test_equilibrium_rel_tol_scaling\": "
                << "\"fixed_base_tol_times_0p01\",\n";
        }
        if (std::isfinite(ps.equilibrium_abs_tol)) {
            ofs << "    \"growth_test_equilibrium_abs_tol\": "
                << ps.equilibrium_abs_tol << ",\n";
        }
        if (ps.z_max_at_perc.empty()) {
            ofs << "    \"results_convention\": \"growth_test_order_percolation_is_species_temporal_equilibration_order\",\n";
        }
    }
    if (!ps.z_max_at_perc.empty()) {
        ofs << "    \"z_max_at_perc\": ";
        write_json_array(ofs, ps.z_max_at_perc);
        ofs << ",\n";
    }
    if (!ps.initial_base_layout.empty()) {
        ofs << "    \"initial_base_layout\": \"" << ps.initial_base_layout << "\",\n";
    }
    if (!ps.fL_z_by_species.empty()) {
        ofs << "    \"fL_z_convention\": \"final_species_layer_fraction_N_i_z_over_lateral_layer_size\",\n";
    }
    ofs << "    \"pt_convention\": \"p_used_to_generate_same_time_step\",\n";
    ofs << "    \"nt_convention\": \"new_active_front_fraction_same_time_step\"\n";
    ofs << "  },\n";

    ofs << "  \"results\": {\n";
    for (size_t k = 0; k < idx.size(); ++k) {
        const size_t i = idx[k];
        const int order_i = ps.percolation_order[i];
        const int color_1b = ps.color_percolation[i];
        const int crow = std::max(0, color_1b - 1);

        const int M_size =
            ps.M_size_at_perc.empty() ? -1 : ps.M_size_at_perc[i];

        int shortest_path_lin_value = -1;
        if (crow < static_cast<int>(ps.sp_len.size()) && ps.sp_len[crow] >= 0) {
            shortest_path_lin_value = ps.sp_len[crow];
        }

        const int sp_lin_preteq =
            (crow < static_cast<int>(ps.sp_lin_preteq.size()))
                ? ps.sp_lin_preteq[crow]
                : -1;

        const int sp_lin_posteq =
            (crow < static_cast<int>(ps.sp_lin_posteq.size()))
                ? ps.sp_lin_posteq[crow]
                : -1;

        const int M_size_preteq =
            (crow < static_cast<int>(ps.M_size_preteq.size()))
                ? ps.M_size_preteq[crow]
                : -1;

        const int M_size_posteq =
            (crow < static_cast<int>(ps.M_size_posteq.size()))
                ? ps.M_size_posteq[crow]
                : -1;

        const double t_eq_species =
            (crow < static_cast<int>(ps.t_eq_by_species.size()))
                ? ps.t_eq_by_species[static_cast<std::size_t>(crow)]
                : std::numeric_limits<double>::quiet_NaN();

        const std::vector<double>* fL_z_ptr = nullptr;
        if (crow >= 0 && crow < static_cast<int>(ps.fL_z_by_species.size())) {
            const auto& row = ps.fL_z_by_species[static_cast<std::size_t>(crow)];
            if (!row.empty()) {
                fL_z_ptr = &row;
            }
        }

        const std::vector<int>* sp_path_lin_ptr = nullptr;
        if (crow < static_cast<int>(ps.sp_path_lin.size())) {
            sp_path_lin_ptr = &ps.sp_path_lin[crow];
        }
        
        const std::vector<int>* sp_path_lin_preteq_ptr = nullptr;
        if (crow < static_cast<int>(ps.sp_path_lin_preteq.size())) {
            sp_path_lin_preteq_ptr = &ps.sp_path_lin_preteq[crow];
        }

        const std::vector<int>* sp_path_lin_posteq_ptr = nullptr;
        if (crow < static_cast<int>(ps.sp_path_lin_posteq.size())) {
            sp_path_lin_posteq_ptr = &ps.sp_path_lin_posteq[crow];
        }

        ofs << "    \"order_percolation " << order_i << "\": {\n";
        ofs << "      \"data\": {\n";
        ofs << "        \"color\": " << color_1b << ",\n";
        ofs << "        \"time\": ";
        write_json_array(ofs, ts.t);
        ofs << ",\n";
        ofs << "        \"pt\": ";
        write_json_row(ofs, ts.p_t, crow);
        ofs << ",\n";
        ofs << "        \"nt\": ";
        write_json_row(ofs, ts.f_t, crow);
        ofs << ",\n";
        if (fL_z_ptr != nullptr) {
            ofs << "        \"fL_z\": ";
            write_json_array(ofs, *fL_z_ptr);
            ofs << ",\n";
        }
        if (!ps.t_eq_by_species.empty()) {
            ofs << "        \"t_eq_species\": ";
            write_json_nullable_double(ofs, t_eq_species);
            ofs << ",\n";
        }
        ofs << "        \"shortest_path_lin\": " << shortest_path_lin_value << ",\n";
        ofs << "        \"sp_path_lin\": ";
        if (sp_path_lin_ptr) write_json_array(ofs, *sp_path_lin_ptr);
            else ofs << "[]";
            ofs << ",\n";
        ofs << "        \"sp_edge_pairs_lin\": ";
        if (sp_path_lin_ptr) write_json_path_edge_pairs(ofs, *sp_path_lin_ptr);
            else ofs << "[]";
            ofs << ",\n";

        ofs << "        \"M_size\": " << M_size << ",\n";
        ofs << "        \"sp_lin_preteq\": " << sp_lin_preteq << ",\n";
        ofs << "        \"sp_path_lin_preteq\": ";
        if (sp_path_lin_preteq_ptr) write_json_array(ofs, *sp_path_lin_preteq_ptr);
            else ofs << "[]";
            ofs << ",\n";

        ofs << "        \"sp_lin_posteq\": " << sp_lin_posteq << ",\n";
        ofs << "        \"sp_path_lin_posteq\": ";
        if (sp_path_lin_posteq_ptr) write_json_array(ofs, *sp_path_lin_posteq_ptr);
            else ofs << "[]";
            ofs << ",\n";

        ofs << "        \"M_size_preteq\": " << M_size_preteq << ",\n";
        ofs << "        \"M_size_posteq\": " << M_size_posteq << "\n";
        ofs << "      }\n";
        ofs << "    }";
        if (k + 1 < idx.size()) ofs << ",";
        ofs << "\n";
    }

    ofs << "  }\n";
    ofs << "}\n";
}

void save_data::save_network_compact_bin(const NetworkCompact& net,
                                        const std::string& filename) const
{
    const std::filesystem::path out_path(filename);
    if (!out_path.parent_path().empty()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    if (!net.write_binary(filename)) {
        throw std::runtime_error("save_network_compact_bin: falha ao gravar " + filename);
    }
}

void save_data::save_lateral_observables_csv(
    const std::string& output_dir,
    const std::string& sample_id,
    const LateralObservablesSeries& observables) const
{
    const fs::path out_dir(output_dir);
    if (!out_dir.empty()) {
        fs::create_directories(out_dir);
    }

    const fs::path legacy_corr_path = out_dir / "lateral_correlation_time.csv";
    const fs::path legacy_sus_path = out_dir / "lateral_susceptibility_time.csv";
    std::error_code ec;
    fs::remove(legacy_corr_path, ec);
    fs::remove(legacy_sus_path, ec);

    const fs::path corr_path = out_dir / (sample_id + "_lateral_correlation_time.csv");
    std::ofstream corr(corr_path);
    if (!corr) {
        throw std::runtime_error("save_lateral_observables_csv: falha ao abrir " + corr_path.string());
    }

    corr << "sample_id,dim,L,t,f_t,r_max,n_rows,C_norm_mean,C_norm_std,C_norm_absmax,r_at_absmax,valid_norm_mean,pair_count_mean,boundary_mode";
    if (std::isfinite(observables.f_T)) corr << ",f_T";
    if (std::isfinite(observables.p0)) corr << ",p0";
    if (std::isfinite(observables.P0)) corr << ",P0";
    if (std::isfinite(observables.c)) corr << ",c";
    if (!observables.type_percolation.empty()) corr << ",type_perc";
    if (observables.seed >= 0) corr << ",seed";
    if (std::isfinite(observables.t_stat)) corr << ",t_stat";
    corr << "\n";

    if (!observables.correlation_summary_rows.empty()) {
        for (const auto& row : observables.correlation_summary_rows) {
            corr << sample_id << ',' << observables.dim << ',' << observables.L << ',' << row.t << ',';
            corr << std::setprecision(17) << row.f_t << ',' << row.r_max << ',' << row.n_rows << ','
                 << row.C_norm_mean << ',' << row.C_norm_std << ',' << row.C_norm_absmax << ','
                 << row.r_at_absmax << ',' << row.valid_norm_mean << ',' << row.pair_count_mean << ','
                 << observables.boundary_mode;
            if (std::isfinite(observables.f_T)) corr << ',' << observables.f_T;
            if (std::isfinite(observables.p0)) corr << ',' << observables.p0;
            if (std::isfinite(observables.P0)) corr << ',' << observables.P0;
            if (std::isfinite(observables.c)) corr << ',' << observables.c;
            if (!observables.type_percolation.empty()) corr << ',' << observables.type_percolation;
            if (observables.seed >= 0) corr << ',' << observables.seed;
            if (std::isfinite(observables.t_stat)) corr << ',' << observables.t_stat;
            corr << '\n';
        }
    } else {
        int current_t = std::numeric_limits<int>::min();
        int n_rows = 0;
        double c_sum = 0.0;
        double c_sumsq = 0.0;
        double c_absmax = 0.0;
        int r_at_absmax = 0;
        double valid_sum = 0.0;
        double pair_count_sum = 0.0;
        double f_t = 0.0;

        auto flush_summary = [&]() {
            if (n_rows <= 0) return;
            const double mean = c_sum / static_cast<double>(n_rows);
            const double variance = (n_rows > 1)
                ? std::max(0.0, (c_sumsq - static_cast<double>(n_rows) * mean * mean) /
                                  static_cast<double>(n_rows - 1))
                : 0.0;
            corr << sample_id << ',' << observables.dim << ',' << observables.L << ',' << current_t << ',';
            corr << std::setprecision(17) << f_t << ',' << observables.r_max << ',' << n_rows << ','
                 << mean << ',' << std::sqrt(variance) << ',' << c_absmax << ','
                 << r_at_absmax << ',' << (valid_sum / static_cast<double>(n_rows)) << ','
                 << (pair_count_sum / static_cast<double>(n_rows)) << ',' << observables.boundary_mode;
            if (std::isfinite(observables.f_T)) corr << ',' << observables.f_T;
            if (std::isfinite(observables.p0)) corr << ',' << observables.p0;
            if (std::isfinite(observables.P0)) corr << ',' << observables.P0;
            if (std::isfinite(observables.c)) corr << ',' << observables.c;
            if (!observables.type_percolation.empty()) corr << ',' << observables.type_percolation;
            if (observables.seed >= 0) corr << ',' << observables.seed;
            if (std::isfinite(observables.t_stat)) corr << ',' << observables.t_stat;
            corr << '\n';
        };

        for (const auto& row : observables.correlation_rows) {
            if (current_t != row.t) {
                flush_summary();
                current_t = row.t;
                n_rows = 0;
                c_sum = 0.0;
                c_sumsq = 0.0;
                c_absmax = 0.0;
                r_at_absmax = 0;
                valid_sum = 0.0;
                pair_count_sum = 0.0;
                f_t = row.f_t;
            }
            c_sum += row.C_norm;
            c_sumsq += row.C_norm * row.C_norm;
            if (std::fabs(row.C_norm) > c_absmax) {
                c_absmax = std::fabs(row.C_norm);
                r_at_absmax = row.r;
            }
            valid_sum += static_cast<double>(row.valid_norm);
            pair_count_sum += static_cast<double>(row.pair_count);
            ++n_rows;
        }
        flush_summary();
    }

    const fs::path sus_path = out_dir / (sample_id + "_lateral_susceptibility_time.csv");
    std::ofstream sus(sus_path);
    if (!sus) {
        throw std::runtime_error("save_lateral_observables_csv: falha ao abrir " + sus_path.string());
    }

    sus << "sample_id,dim,L,t,f_t,r_max,chi_raw_incl0,chi_raw_excl0,chi_norm_incl0,chi_norm_excl0,n_valid_norm,boundary_mode";
    if (std::isfinite(observables.f_T)) sus << ",f_T";
    if (std::isfinite(observables.p0)) sus << ",p0";
    if (std::isfinite(observables.P0)) sus << ",P0";
    if (std::isfinite(observables.c)) sus << ",c";
    if (!observables.type_percolation.empty()) sus << ",type_perc";
    if (observables.seed >= 0) sus << ",seed";
    if (std::isfinite(observables.t_stat)) sus << ",t_stat";
    sus << "\n";

    for (const auto& row : observables.susceptibility_rows) {
        sus << sample_id << ',' << observables.dim << ',' << observables.L << ',' << row.t << ',';
        sus << std::setprecision(17) << row.f_t << ',' << row.r_max << ',' << row.chi_raw_incl0 << ','
            << row.chi_raw_excl0 << ',' << row.chi_norm_incl0 << ',' << row.chi_norm_excl0 << ','
            << row.n_valid_norm << ',' << observables.boundary_mode;
        if (std::isfinite(observables.f_T)) sus << ',' << observables.f_T;
        if (std::isfinite(observables.p0)) sus << ',' << observables.p0;
        if (std::isfinite(observables.P0)) sus << ',' << observables.P0;
        if (std::isfinite(observables.c)) sus << ',' << observables.c;
        if (!observables.type_percolation.empty()) sus << ',' << observables.type_percolation;
        if (observables.seed >= 0) sus << ',' << observables.seed;
        if (std::isfinite(observables.t_stat)) sus << ',' << observables.t_stat;
        sus << '\n';
    }
}

void save_data::save_reanalysis_json(const ReanalysisResult& result,
                                     const std::string& filename_json) const
{
    std::ofstream ofs(filename_json);
    if (!ofs) {
        throw std::runtime_error(
            std::string("[save_reanalysis_json] não abriu: ") + filename_json);
    }

    ofs << "{\n";
    ofs << "  \"t_eq\": ";
    write_json_nullable_double(ofs, result.t_eq);
    ofs << ",\n";

    write_subgraph_json(ofs, "pre_teq", result.pre_teq, true);
    write_subgraph_json(ofs, "post_teq", result.post_teq, false);

    ofs << "}\n";
}

void save_data::save_reanalysis_networks(const ReanalysisResult& result,
                                         const std::string& filename_preteq_npz,
                                         const std::string& filename_posteq_npz) const
{
    // Save reanalysis networks as compact binaries (.bin)
    std::string pre = filename_preteq_npz;
    std::string post = filename_posteq_npz;
    if (pre.size() >= 4 && pre.substr(pre.size()-4) == ".npz") pre = pre.substr(0, pre.size()-4) + ".bin";
    if (post.size() >= 4 && post.substr(post.size()-4) == ".npz") post = post.substr(0, post.size()-4) + ".bin";
    save_network_as_npz(result.pre_teq.net, pre);
    save_network_as_npz(result.post_teq.net, post);
}

void save_data::save_surfaces_as_npz(const SurfacesCuts& surfaces,
                                     const std::string& filename) const
{
    // Save surfaces as JSON file instead of NPZ.
    std::string out = filename;
    if (out.size() >= 4 && out.substr(out.size()-4) == ".npz") {
        out = out.substr(0, out.size()-4) + ".json";
    } else if (out.size() >= 5 && out.substr(out.size()-5) == ".json") {
        // ok
    } else {
        out += ".json";
    }

    const fs::path out_path(out);
    if (!out_path.parent_path().empty()) fs::create_directories(out_path.parent_path());

    std::ofstream ofs(out);
    if (!ofs) throw std::runtime_error("save_surfaces_as_npz: falha abrindo " + out);

    ofs << "{\n";
    ofs << "  \"surface_preteq\": [\n";
    for (size_t i = 0; i < surfaces.surface_preteq.size(); ++i) {
        const auto &p = surfaces.surface_preteq[i];
        ofs << "    [" << p.x << ", " << p.y << ", " << p.z << ", " << p.color_index << "]";
        if (i + 1 < surfaces.surface_preteq.size()) ofs << ",\n";
        else ofs << "\n";
    }
    ofs << "  ],\n";
    ofs << "  \"surface_posteq\": [\n";
    for (size_t i = 0; i < surfaces.surface_posteq.size(); ++i) {
        const auto &p = surfaces.surface_posteq[i];
        ofs << "    [" << p.x << ", " << p.y << ", " << p.z << ", " << p.color_index << "]";
        if (i + 1 < surfaces.surface_posteq.size()) ofs << ",\n";
        else ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}
