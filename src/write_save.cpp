#include "write_save.hpp"
#include "equilibration_partition.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
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

    ofs << "{\n";
    ofs << "  \"meta\": {\n";
    ofs << "    \"num_colors\": " << ts.num_colors << ",\n";
    ofs << "    \"rho\": ";
    write_json_array(ofs, ps.rho);
    ofs << ",\n";
    ofs << "    \"t_eq\": " << ps.t_eq << "\n";
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
        ofs << "        \"shortest_path_lin\": " << shortest_path_lin_value << ",\n";
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

    // Make a copy and normalize species to small integer codes (0..255)
    NetworkCompact tmp = net;
    for (std::size_t i = 0; i < tmp.species.size(); ++i) {
        uint8_t s = tmp.species[i];
        // normalize: any non-zero -> keep as-is but clamp to 255
        tmp.species[i] = (s == 0) ? 0 : (s > 255 ? 255 : s);
    }

    if (!tmp.write_binary(filename)) {
        throw std::runtime_error("save_network_compact_bin: falha ao gravar " + filename);
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
    ofs << "  \"t_eq\": " << result.t_eq << ",\n";

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