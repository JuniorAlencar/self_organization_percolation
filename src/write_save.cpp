#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <type_traits>
#include <cstring>
#include <algorithm>

#include "write_save.hpp"

static void write_npy_int32(const std::string& filename,
                            const std::vector<int>& shape,
                            const std::vector<int>& data)
{
    if (shape.empty()) throw std::runtime_error("[save_network_as_npz] shape vazio");
    size_t n = 1;
    for (int s : shape) {
        if (s <= 0) throw std::runtime_error("[save_network_as_npz] dimensão <=0");
        n *= static_cast<size_t>(s);
    }
    if (n != data.size()) throw std::runtime_error("[save_network_as_npz] shape != data.size()");

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error(std::string("[save_network_as_npz] não abriu: ")+filename);

    // magic + versão
    ofs.write("\x93NUMPY", 6);
    ofs.put(char(1)); ofs.put(char(0)); // v1.0

    // header dict
    std::ostringstream dict;
    dict << "{'descr': '|i4', 'fortran_order': False, 'shape': (";
    for (size_t i=0;i<shape.size();++i) {
        dict << shape[i];
        if (i+1<shape.size()) dict << ", ";
    }
    if (shape.size()==1) dict << ",";
    dict << "), }";
    std::string hdr = dict.str();

    // alinhamento 16B
    const size_t preamble = 10; // 6 magic + 2 ver + 2 header_len
    size_t pad = (16 - ((preamble + 2 + hdr.size()) % 16)) % 16;
    hdr.append(pad, ' ');
    hdr.push_back('\n');

    // header_len (LE uint16)
    unsigned short hl = static_cast<unsigned short>(hdr.size());
    ofs.put(static_cast<char>(hl & 0xFF));
    ofs.put(static_cast<char>((hl >> 8) & 0xFF));

    ofs.write(hdr.data(), hdr.size());
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size()*sizeof(int));
}



// =======================================================
// já existente no seu arquivo (mantido):
// static void write_npy_int32(...)
// ...

// =======================================================
// Helpers para gerar .npy em memória (v1.0, C-order)

namespace {

template <typename T>
std::string npy_descr(); // specializations abaixo

template <>
std::string npy_descr<int32_t>() { return "|i4"; }

template <>
std::string npy_descr<double>() { return "<f8"; } // little-endian float64

// Cria header v1.0 alinhado a 16B
inline std::string make_npy_header(const std::string& descr,
                                   bool fortran_order,
                                   const std::vector<size_t>& shape)
{
    std::ostringstream dict;
    dict << "{'descr': '" << descr << "', 'fortran_order': "
         << (fortran_order ? "True" : "False") << ", 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        dict << shape[i];
        if (shape.size() == 1) dict << ","; // tupla 1D precisa da vírgula
        if (i + 1 < shape.size()) dict << ", ";
    }
    if (shape.empty()) dict << ""; // escalar: shape=()
    dict << "), }";

    std::string hdr = dict.str();
    const size_t preamble = 10; // 6 magic + 2 ver + 2 header_len
    size_t pad = (16 - ((preamble + 2 + hdr.size()) % 16)) % 16;
    hdr.append(pad, ' ');
    hdr.push_back('\n');
    return hdr;
}

template <typename T>
std::string make_npy_blob(const T* data, size_t count,
                          const std::vector<size_t>& shape,
                          bool fortran_order = false)
{
    std::string blob;
    const std::string hdr = make_npy_header(npy_descr<T>(), fortran_order, shape);
    blob.reserve(10 + hdr.size() + count * sizeof(T));
    // magic + versão
    blob.append("\x93NUMPY", 6);
    blob.push_back(char(1));
    blob.push_back(char(0));
    // header length (LE uint16)
    const unsigned short hl = static_cast<unsigned short>(hdr.size());
    blob.push_back(static_cast<char>(hl & 0xFF));
    blob.push_back(static_cast<char>((hl >> 8) & 0xFF));
    // header
    blob.append(hdr);
    // data
    if (count) {
        blob.append(reinterpret_cast<const char*>(data), count * sizeof(T));
    }
    return blob;
}

// adiciona um arquivo ao zip a partir de um buffer na memória
inline void zip_add_buffer(zip_t* za, const std::string& entry_name, const std::string& buffer)
{
    zip_source_t* src = zip_source_buffer(za, buffer.data(), buffer.size(), 0);
    if (!src) throw std::runtime_error("[npz] zip_source_buffer falhou: " + entry_name);
    const zip_int64_t idx = zip_file_add(za, entry_name.c_str(), src, ZIP_FL_OVERWRITE | ZIP_FL_ENC_UTF_8);
    if (idx < 0) {
        zip_source_free(src);
        throw std::runtime_error("[npz] zip_file_add falhou: " + entry_name);
    }
}

} // namespace


// =======================================================
// NOVA implementação
void save_data::save_network_as_npz(const NetworkPattern& net,
                                    const std::string& filename) const
{
    // Retrocompatibilidade: se pedirem .npy, grava apenas os dados em i4
    if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npy") {
        write_npy_int32(filename, net.shape, net.data);
        return;
    }

    // Caso esperado: .npz
    if (!(filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npz")) {
        throw std::runtime_error("save_network_as_npz: filename deve terminar com .npz ou .npy");
    }

    // Preparar blobs .npy
    // 1) dim, num_colors, seed (escalares)
    const int32_t dim_i = static_cast<int32_t>(net.dim);
    const int32_t nc_i  = static_cast<int32_t>(net.num_colors);
    const int32_t seed_i= static_cast<int32_t>(net.seed);
    const std::vector<size_t> shape_scalar{}; // escalar: ()

    const std::string npy_dim  = make_npy_blob(&dim_i,  1, shape_scalar);
    const std::string npy_nc   = make_npy_blob(&nc_i,   1, shape_scalar);
    const std::string npy_seed = make_npy_blob(&seed_i, 1, shape_scalar);

    // 2) shape (int32 1D)
    std::vector<int32_t> shape_i32;
    shape_i32.reserve(net.shape.size());
    for (int s : net.shape) {
        if (s <= 0) throw std::runtime_error("save_network_as_npz: shape inválido");
        shape_i32.push_back(static_cast<int32_t>(s));
    }
    const std::vector<size_t> shape_1d{ shape_i32.size() };
    const std::string npy_shape = make_npy_blob(shape_i32.data(), shape_i32.size(), shape_1d);

    // 3) data (int32 1D, C-order flatten — mais portátil)
    const std::vector<size_t> data_shape_1d{ net.data.size() };
    const std::string npy_data = make_npy_blob(reinterpret_cast<const int32_t*>(net.data.data()),
                                               net.data.size(), data_shape_1d);

    // 4) rho (float64 1D)
    std::vector<double> rho_f(net.rho.begin(), net.rho.end());
    const std::vector<size_t> rho_shape_1d{ rho_f.size() };
    const std::string npy_rho = make_npy_blob(rho_f.data(), rho_f.size(), rho_shape_1d);

    // Abrir zip e adicionar entradas
    int errcode = 0;
    zip_t* za = zip_open(filename.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &errcode);
    if (!za) {
        std::ostringstream oss;
        oss << "save_network_as_npz: não abriu zip '" << filename << "' (err=" << errcode << ")";
        throw std::runtime_error(oss.str());
    }

    try {
        zip_add_buffer(za, "dim.npy",        npy_dim);
        zip_add_buffer(za, "num_colors.npy", npy_nc);
        zip_add_buffer(za, "seed.npy",       npy_seed);
        zip_add_buffer(za, "shape.npy",      npy_shape);
        zip_add_buffer(za, "data.npy",       npy_data);
        zip_add_buffer(za, "rho.npy",        npy_rho);

        if (zip_close(za) != 0) {
            throw std::runtime_error("save_network_as_npz: zip_close falhou");
        }
    } catch (...) {
        zip_discard(za);
        throw;
    }
}


// ========== helpers JSON ==========
namespace {
template <typename T>
void write_json_array(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i=0;i<v.size();++i) {
        if constexpr (std::is_floating_point<T>::value) os << std::setprecision(17) << v[i];
        else os << v[i];
        if (i+1<v.size()) os << ", ";
    }
    os << "]";
}
template <typename T>
void write_json_row(std::ostream& os, const std::vector<std::vector<T>>& m, int row) {
    if (row<0 || row>=(int)m.size()) { os << "[]"; return; }
    write_json_array(os, m[row]);
}
}

void save_data::save_percolation_json(const PercolationSeries& ps,
                                      const TimeSeries& ts,
                                      const std::string& filename_json,
                                      bool sort_by_order) const
{
    
    if (!ps.percolation_order.empty()) {
        const size_t m = ps.percolation_order.size();
        if (ps.color_percolation.size()!=m)
            throw std::runtime_error("[save_percolation_json] vetores de eventos com tamanhos diferentes.");
        if (!ps.M_size_at_perc.empty() && ps.M_size_at_perc.size()!=m)
            throw std::runtime_error("[save_percolation_json] M_size_at_perc.size != #events.");
    }

    std::ofstream ofs(filename_json);
    if (!ofs) throw std::runtime_error(std::string("[save_percolation_json] não abriu: ")+filename_json);

    // índices ordenados por percolation_order (se solicitado)
    std::vector<size_t> idx(ps.percolation_order.size());
    for (size_t i=0;i<idx.size();++i) idx[i] = i;
    if (sort_by_order) {
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
            return ps.percolation_order[a] < ps.percolation_order[b];
        });
    }

    
    // ---- HEADER JSON ----
    ofs << "{\n";
    ofs << "  \"meta\": {\n";
    ofs << "    \"num_colors\": " << ts.num_colors << ",\n";
    const double rho_meta = (!ps.rho.empty() ? ps.rho[0] : 0.0);
    ofs << "    \"rho\": " << std::setprecision(17) << rho_meta << "\n";
    ofs << "  },\n";

    ofs << "  \"results\": {\n";
    for (size_t k = 0; k < idx.size(); ++k) {
        const size_t i = idx[k];
        const int order_i  = ps.percolation_order[i];
        const int color_1b = ps.color_percolation[i];
        const int crow     = std::max(0, color_1b - 1);
        const int M_size   = ps.M_size_at_perc[i];

        int shortest_path_lin_value = -1;
        if (crow < (int)ps.sp_len.size() && ps.sp_len[crow] > 0)
            shortest_path_lin_value = ps.sp_len[crow] - 1;

        ofs << "    \"order_percolation " << order_i << "\": {\n";
        ofs << "      \"data\": {\n";
        ofs << "        \"color\": " << color_1b << ",\n";
        ofs << "        \"time\": "; write_json_array(ofs, ts.t); ofs << ",\n";
        ofs << "        \"pt\": ";   write_json_row(ofs, ts.p_t, crow); ofs << ",\n";
        ofs << "        \"nt\": ";   write_json_row(ofs, ts.Nt, crow);

        ofs << ",\n"; // vírgula separando do próximo campo obrigatório
        
        // -----------------------------------------------------------------------

        ofs << "        \"shortest_path_lin\": " << shortest_path_lin_value << ",\n";
        ofs << "        \"M_size\": " << M_size << "\n";
        ofs << "      }\n";
        ofs << "    }";
        if (k + 1 < idx.size()) ofs << ",";
        ofs << "\n";
    }
        ofs << "  }\n";   // fecha "results"
        ofs << "}\n";     // fecha objeto raiz

}
