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

void save_data::save_network_as_npz(const NetworkPattern& net,
                                    const std::string& filename) const
{
    write_npy_int32(filename, net.shape, net.data);
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
inline double rho_by_color_1b(const std::vector<double>& rho, int color_1b){
    int idx = std::max(0, color_1b - 1);
    return (idx<(int)rho.size() ? rho[idx] : 0.0);
}
} // anon

// ========== writer JSON (layout solicitado) ==========
void save_data::save_percolation_json(const PercolationSeries& ps,
                                      const TimeSeries& ts,
                                      const std::string& filename_json,
                                      bool sort_by_order) const
{
    // checagens básicas
    if ((int)ts.p_t.size()!=ts.num_colors || (int)ts.Nt.size()!=ts.num_colors || (int)ts.M_t.size()!=ts.num_colors)
        throw std::runtime_error("[save_percolation_json] TimeSeries inconsistente com num_colors.");

    // >>> NOVO: se existirem, Smax/Ni/chi também devem bater com num_colors
    if (!ts.Smax.empty() && (int)ts.Smax.size()!=ts.num_colors)
        throw std::runtime_error("[save_percolation_json] Smax.size != num_colors.");
    if (!ts.Ni.empty()   && (int)ts.Ni.size()!=ts.num_colors)
        throw std::runtime_error("[save_percolation_json] Ni.size != num_colors.");
    if (!ts.chi.empty()  && (int)ts.chi.size()!=ts.num_colors)
        throw std::runtime_error("[save_percolation_json] chi.size != num_colors.");
    // <<< NOVO

    if (!ps.percolation_order.empty()) {
        const size_t m = ps.percolation_order.size();
        if (ps.color_percolation.size()!=m || ps.time_percolation.size()!=m)
            throw std::runtime_error("[save_percolation_json] vetores de eventos com tamanhos diferentes.");
        if (!ps.M_size_at_perc.empty() && ps.M_size_at_perc.size()!=m)
            throw std::runtime_error("[save_percolation_json] M_size_at_perc.size != #events.");
    }

    std::ofstream ofs(filename_json);
    if (!ofs) throw std::runtime_error(std::string("[save_percolation_json] não abriu: ")+filename_json);

    // indices ordenados por percolation_order (se solicitado)
    std::vector<size_t> idx(ps.percolation_order.size());
    for (size_t i=0;i<idx.size();++i) idx[i] = i;
    if (sort_by_order) {
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){
            return ps.percolation_order[a] < ps.percolation_order[b];
        });
    }

    ofs << "{\n";
    ofs << "  \"meta\": {\n";
    ofs << "    \"num_colors\": " << ts.num_colors << "\n";
    ofs << "  },\n";

    ofs << "  \"results\": [\n";
    for (size_t k=0;k<idx.size();++k){
        const size_t i = idx[k];
        const int color_1b = ps.color_percolation[i];
        const int crow = std::max(0, color_1b - 1);
        const int order_i = ps.percolation_order[i];
        const int tperc   = ps.time_percolation[i];
        const double rho  = rho_by_color_1b(ps.rho, color_1b);

        // menor caminho (em nº de ARESTAS) = (#nós - 1)
        int shortest_path_lin_value = -1;
        if (crow<(int)ps.sp_len.size() && ps.sp_len[crow] > 0)
            shortest_path_lin_value = ps.sp_len[crow] - 1;

        ofs << "    {\n";
        ofs << "      \"order_percolation\": " << order_i << ",\n";
        ofs << "      \"data\": {\n";
        ofs << "        \"color\": " << color_1b << ",\n";
        ofs << "        \"rho\": " << std::setprecision(17) << rho << ",\n";
        ofs << "        \"time_percolation\": " << tperc << ",\n";
        ofs << "        \"time\": "; write_json_array(ofs, ts.t); ofs << ",\n";
        ofs << "        \"pt\": "; write_json_row(ofs, ts.p_t, crow); ofs << ",\n";
        ofs << "        \"nt\": "; write_json_row(ofs, ts.Nt, crow); ofs << ",\n";
        ofs << "        \"Mt\": "; write_json_row(ofs, ts.M_t, crow); ofs << ",\n";

        // >>> NOVO: séries por espécie (se vazias, write_json_row deve imprimir [] conforme seu helper)
        ofs << "        \"Smax\": "; write_json_row(ofs, ts.Smax, crow); ofs << ",\n";
        ofs << "        \"Ni\": ";   write_json_row(ofs, ts.Ni,   crow); ofs << ",\n";
        ofs << "        \"chi\": ";  write_json_row(ofs, ts.chi,  crow); ofs << ",\n";
        // <<< NOVO

        ofs << "        \"shortest_path_lin\": " << shortest_path_lin_value << "\n";
        ofs << "      }\n";
        ofs << "    }";
        if (k+1<idx.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
}
