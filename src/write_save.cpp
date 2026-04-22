#include "write_save.hpp"

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

#include <zip.h>

namespace fs = std::filesystem;

namespace {

template <typename T>
std::string npy_descr();

template <>
std::string npy_descr<int32_t>() { return "<i4"; }

template <>
std::string npy_descr<double>() { return "<f8"; }

inline std::string make_npy_header(const std::string& descr,
                                   const bool fortran_order,
                                   const std::vector<size_t>& shape)
{
    std::ostringstream dict;
    dict << "{'descr': '" << descr << "', 'fortran_order': "
         << (fortran_order ? "True" : "False") << ", 'shape': (";

    for (size_t i = 0; i < shape.size(); ++i) {
        dict << shape[i];
        if (shape.size() == 1) dict << ",";
        if (i + 1 < shape.size()) dict << ", ";
    }

    dict << "), }";

    std::string hdr = dict.str();

    const size_t preamble = 10;
    const size_t pad = (16 - ((preamble + 2 + hdr.size()) % 16)) % 16;
    hdr.append(pad, ' ');
    hdr.push_back('\n');

    return hdr;
}

template <typename T>
std::string make_npy_blob(const T* data,
                          const size_t count,
                          const std::vector<size_t>& shape,
                          const bool fortran_order = false)
{
    const std::string hdr = make_npy_header(npy_descr<T>(), fortran_order, shape);

    std::string blob;
    blob.reserve(10 + hdr.size() + count * sizeof(T));

    blob.append("\x93NUMPY", 6);
    blob.push_back(char(1));
    blob.push_back(char(0));

    const unsigned short hl = static_cast<unsigned short>(hdr.size());
    blob.push_back(static_cast<char>(hl & 0xFF));
    blob.push_back(static_cast<char>((hl >> 8) & 0xFF));

    blob.append(hdr);
    if (count > 0) {
        blob.append(reinterpret_cast<const char*>(data), count * sizeof(T));
    }
    return blob;
}

inline std::string zip_archive_error(zip_t* za)
{
    zip_error_t* err = zip_get_error(za);
    if (!err) return "erro libzip desconhecido";

    std::ostringstream oss;
    oss << zip_error_strerror(err)
        << " [zip_code=" << zip_error_code_zip(err)
        << ", sys_code=" << zip_error_code_system(err) << "]";
    return oss.str();
}

inline std::string zip_open_error_string(const int errcode)
{
    zip_error_t err;
    zip_error_init_with_code(&err, errcode);

    std::ostringstream oss;
    oss << zip_error_strerror(&err)
        << " [zip_code=" << zip_error_code_zip(&err)
        << ", sys_code=" << zip_error_code_system(&err) << "]";

    zip_error_fini(&err);
    return oss.str();
}

inline void zip_add_buffer(zip_t* za,
                           const std::string& entry_name,
                           const std::string& buffer)
{
    zip_source_t* src = zip_source_buffer(za, buffer.data(), buffer.size(), 0);
    if (!src) {
        throw std::runtime_error(
            "[npz] zip_source_buffer falhou em '" + entry_name +
            "': " + zip_archive_error(za));
    }

    const zip_int64_t idx = zip_file_add(
        za, entry_name.c_str(), src, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);

    if (idx < 0) {
        zip_source_free(src);
        throw std::runtime_error(
            "[npz] zip_file_add falhou em '" + entry_name +
            "': " + zip_archive_error(za));
    }
}

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

void save_sparse_npz_payload(const int dim,
                             const int num_colors,
                             const int seed,
                             const std::vector<int>& shape,
                             const std::vector<double>& rho,
                             const std::vector<int32_t>& active_idx,
                             const std::vector<int32_t>& active_val,
                             const std::string& filename)
{
    if (!(filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npz")) {
        throw std::runtime_error("save_network_as_npz: filename deve terminar com .npz");
    }

    if (active_idx.size() != active_val.size()) {
        throw std::runtime_error("save_network_as_npz: active_idx.size != active_val.size");
    }

    const fs::path out_path(filename);
    if (!out_path.parent_path().empty()) {
        fs::create_directories(out_path.parent_path());
    }

    const int32_t dim_i  = static_cast<int32_t>(dim);
    const int32_t nc_i   = static_cast<int32_t>(num_colors);
    const int32_t seed_i = static_cast<int32_t>(seed);

    const std::vector<size_t> scalar_shape{};
    const std::string npy_dim  = make_npy_blob(&dim_i,  1, scalar_shape);
    const std::string npy_nc   = make_npy_blob(&nc_i,   1, scalar_shape);
    const std::string npy_seed = make_npy_blob(&seed_i, 1, scalar_shape);

    std::vector<int32_t> shape_i32;
    shape_i32.reserve(shape.size());
    for (const int s : shape) {
        if (s <= 0) {
            throw std::runtime_error("save_network_as_npz: shape inválido");
        }
        shape_i32.push_back(static_cast<int32_t>(s));
    }

    std::vector<double> rho_f(rho.begin(), rho.end());

    const std::vector<size_t> shape_1d{shape_i32.size()};
    const std::vector<size_t> rho_1d{rho_f.size()};
    const std::vector<size_t> active_1d{active_idx.size()};

    const std::string npy_shape =
        make_npy_blob(shape_i32.data(), shape_i32.size(), shape_1d);

    const std::string npy_rho =
        make_npy_blob(rho_f.data(), rho_f.size(), rho_1d);

    const std::string npy_active_idx =
        make_npy_blob(active_idx.data(), active_idx.size(), active_1d);

    const std::string npy_active_val =
        make_npy_blob(active_val.data(), active_val.size(), active_1d);

    int errcode = 0;
    zip_t* za = zip_open(filename.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &errcode);
    if (!za) {
        throw std::runtime_error(
            "save_network_as_npz: não abriu zip '" + filename +
            "': " + zip_open_error_string(errcode));
    }

    zip_add_buffer(za, "dim.npy",        npy_dim);
    zip_add_buffer(za, "num_colors.npy", npy_nc);
    zip_add_buffer(za, "seed.npy",       npy_seed);
    zip_add_buffer(za, "shape.npy",      npy_shape);
    zip_add_buffer(za, "rho.npy",        npy_rho);
    zip_add_buffer(za, "active_idx.npy", npy_active_idx);
    zip_add_buffer(za, "active_val.npy", npy_active_val);

    if (zip_close(za) != 0) {
        const std::string err = zip_archive_error(za);
        zip_discard(za);
        throw std::runtime_error(
            "save_network_as_npz: zip_close falhou em '" + filename +
            "': " + err);
    }
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

    save_sparse_npz_payload(
        net.dim,
        net.num_colors,
        net.seed,
        net.shape,
        net.rho,
        active_idx,
        active_val,
        filename
    );
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

    save_sparse_npz_payload(
        net.dim,
        net.num_colors,
        0,
        net.shape,
        net.rho,
        active_idx,
        active_val,
        filename
    );
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
        write_json_row(ofs, ts.Nt, crow);
        ofs << ",\n";
        ofs << "        \"shortest_path_lin\": " << shortest_path_lin_value << ",\n";
        ofs << "        \"M_size\": " << M_size << ",\n";
        ofs << "        \"sp_lin_preteq\": " << sp_lin_preteq << ",\n";
        ofs << "        \"sp_lin_posteq\": " << sp_lin_posteq << ",\n";
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
    save_network_as_npz(result.pre_teq.net, filename_preteq_npz);
    save_network_as_npz(result.post_teq.net, filename_posteq_npz);
}