#include "write_save.hpp"

// Utility function: write raw .npy header (version 1.0, little-endian, shape info)
static std::vector<unsigned char> generate_npy_data(const NetworkPattern& net) {
    const std::vector<int>& shape = net.shape;
    std::vector<unsigned char> result;

    // Magic string + version
    const char magic[] = "\x93NUMPY";
    result.insert(result.end(), magic, magic + 6);
    result.push_back(1);  // major version
    result.push_back(0);  // minor version

    // Header (shape-based)
    std::string header = "{'descr': '|i4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header += std::to_string(shape[i]);
        if (i != shape.size() - 1) header += ", ";
    }
    if (shape.size() == 1) header += ",";
    header += "), }";
    while ((header.size() + 10) % 16 != 0) header += ' ';
    header += '\n';

    // Header length
    uint16_t len = static_cast<uint16_t>(header.size());
    result.push_back(len & 0xFF);
    result.push_back((len >> 8) & 0xFF);

    // Add header
    result.insert(result.end(), header.begin(), header.end());

    // Reserve space for raw binary data
    size_t start_pos = result.size();
    size_t data_size = net.data.size() * sizeof(int);
    result.resize(start_pos + data_size);

    // Fill data in parallel
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < net.data.size(); ++i) {
        int val = net.data[i];
        std::memcpy(&result[start_pos + i * sizeof(int)], &val, sizeof(int));
    }

    return result;
}

void save_data::save_network_as_npz(const NetworkPattern& net, const std::string& filename) const {
    int errorp;
    zip_t* archive = zip_open(filename.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &errorp);
    if (!archive) throw std::runtime_error("Failed to open zip file for writing");

    std::vector<unsigned char> npy_data = generate_npy_data(net);
    zip_source_t* source = zip_source_buffer(archive, npy_data.data(), npy_data.size(), 0);
    if (!source) {
        zip_close(archive);
        throw std::runtime_error("Failed to create zip source");
    }

    if (zip_file_add(archive, "network.npy", source, ZIP_FL_OVERWRITE | ZIP_FL_ENC_GUESS) < 0) {
        zip_source_free(source);
        zip_close(archive);
        throw std::runtime_error("Failed to add .npy to archive");
    }

    if (zip_close(archive) != 0) {
        throw std::runtime_error("Failed to close zip archive");
    }

    std::cout << "Saving to file: " << filename << std::endl;
}

// ---- helpers ----
template <typename Tnum>
static std::vector<std::vector<Tnum>>
to_time_major_from_color_major(const std::vector<std::vector<Tnum>>& color_major, size_t T) {
    const size_t C = color_major.size();
    std::vector<std::vector<Tnum>> tm(T, std::vector<Tnum>(C));
    for (size_t c = 0; c < C; ++c) {
        if (color_major[c].size() != T)
            throw std::runtime_error("pilar CxT com T inconsistente");
        for (size_t i = 0; i < T; ++i) tm[i][c] = color_major[c][i];
    }
    return tm;
}
template <typename Tnum>
static bool is_time_major_rect(const std::vector<std::vector<Tnum>>& v, size_t T, size_t C) {
    if (v.size() != T) return false;
    for (size_t i = 0; i < T; ++i) if (v[i].size() != C) return false;
    return true;
}

void save_data::save_time_series_as_csv(const TimeSeries& ts,
                                        const std::string& filename_pt,
                                        const std::string& filename_Nt)
{
    std::cerr << "[TS-DBG] writer_v2_loaded\n";

    auto dbg = [&](const char* tag){
        std::cerr << "[TS-DBG] " << tag
                  << "  num_colors=" << ts.num_colors
                  << " | T=" << ts.t.size()
                  << " | p_outer=" << ts.p_t.size()
                  << " | Nt_outer=" << ts.Nt.size() << "\n";
        if (!ts.p_t.empty()) std::cerr << "  p_t[0].size=" << ts.p_t[0].size() << "\n";
        if (!ts.Nt.empty())  std::cerr << "  Nt[0].size="  << ts.Nt[0].size()  << "\n";
    };

    if (ts.t.empty() || ts.p_t.empty() || ts.Nt.empty()) {
        dbg("algum vetor vazio");
        std::cerr << "[Erro] TimeSeries inconsistente: vetor vazio.\n";
        return;
    }

    const size_t T = ts.t.size();
    const size_t C = static_cast<size_t>(ts.num_colors);

    std::vector<std::vector<double>> p_tm;
    std::vector<std::vector<int>>    Nt_tm;

    const bool looks_color_major = (ts.p_t.size()==C && ts.Nt.size()==C);
    const bool looks_time_major  = (ts.p_t.size()==T && ts.Nt.size()==T);

    try {
        if (looks_color_major) {
            p_tm  = to_time_major_from_color_major(ts.p_t, T);
            Nt_tm = to_time_major_from_color_major(ts.Nt,  T);
        } else if (looks_time_major &&
                   is_time_major_rect(ts.p_t, T, C) &&
                   is_time_major_rect(ts.Nt, T, C)) {
            p_tm  = ts.p_t;
            Nt_tm = ts.Nt;
        } else {
            dbg("forma incompatível");
            std::cerr << "[Erro] TimeSeries inconsistente: nem CxT nem TxC.\n";
            return;
        }
    } catch (const std::exception& e) {
        dbg("exceção");
        std::cerr << "[Erro] TimeSeries inconsistente: " << e.what() << "\n";
        return;
    }

    // --- p(t) ---
    {
        std::ofstream f(filename_pt);
        if (!f) { std::cerr << "Erro ao abrir " << filename_pt << "\n"; }
        else {
            f << "t"; for (size_t c = 0; c < C; ++c) f << ",pt_" << (c+1); f << "\n";
            for (size_t i = 0; i < T; ++i) {
                f << ts.t[i];
                for (size_t c = 0; c < C; ++c) f << "," << p_tm[i][c];
                f << "\n";
            }
            std::cout << "Saving CSV: " << filename_pt << std::endl;
        }
    }

    // --- N(t) ---
    {
        std::ofstream f(filename_Nt);
        if (!f) { std::cerr << "Erro ao abrir " << filename_Nt << "\n"; }
        else {
            f << "t"; for (size_t c = 0; c < C; ++c) f << ",Nt_" << (c+1); f << "\n";
            for (size_t i = 0; i < T; ++i) {
                f << ts.t[i];
                for (size_t c = 0; c < C; ++c) f << "," << Nt_tm[i][c];
                f << "\n";
            }
            std::cout << "Saving CSV: " << filename_Nt << std::endl;
        }
    }
}

void save_data::save_info_percolation(const PercolationSeries& ps,
                                      const std::string& filename_info)
{
    // Mapa: cor -> (tempo, ordem)
    const size_t m = std::min({ ps.color_percolation.size(),
                                ps.time_percolation.size(),
                                ps.percolation_order.size() });

    std::unordered_map<int, std::pair<int,int>> perc_by_color;
    perc_by_color.reserve(m);
    for (size_t i = 0; i < m; ++i) {
        perc_by_color[ ps.color_percolation[i] ] =
            { ps.time_percolation[i], ps.percolation_order[i] };
    }

    // Número de cores conhecido pelos vetores rho/pho
    const size_t C = std::max(ps.rho.size(), ps.pho.size());

    std::ofstream file_info(filename_info);
    if (!file_info.is_open()) {
        std::cerr << "Erro ao abrir " << filename_info << " para escrita\n";
        return;
    }

    file_info << "color rho pho time_percolation percolation_order\n";

    // Escreve 1 linha por cor conhecida (1..C)
    for (size_t c = 1; c <= C; ++c) {
        const double rho_val = (c-1 < ps.rho.size()) ? ps.rho[c-1] : 0.0;
        const double pho_val = (c-1 < ps.pho.size()) ? ps.pho[c-1] : 0.0;

        auto it = perc_by_color.find(static_cast<int>(c));
        if (it != perc_by_color.end()) {
            file_info << c << ' ' << rho_val << ' ' << pho_val << ' '
                      << it->second.first  << ' '  // time_percolation
                      << it->second.second << '\n'; // percolation_order
        } else {
            file_info << c << ' ' << rho_val << ' ' << pho_val << ' '
                      << -1 << ' ' << -1 << '\n';
        }
    }

    // Se por algum motivo houver registros de percolação com "cor" fora de 1..C,
    // também os gravamos para não perder informação.
    for (size_t i = 0; i < m; ++i) {
        int color = ps.color_percolation[i];
        if (color < 1 || static_cast<size_t>(color) > C) {
            const double rho_val = (static_cast<size_t>(color-1) < ps.rho.size())
                                   ? ps.rho[color-1] : 0.0;
            const double pho_val = (static_cast<size_t>(color-1) < ps.pho.size())
                                   ? ps.pho[color-1] : 0.0;
            file_info << color << ' ' << rho_val << ' ' << pho_val << ' '
                      << ps.time_percolation[i] << ' '
                      << ps.percolation_order[i] << '\n';
        }
    }

    file_info.close();
    // opcional: log amigável
    // std::cout << "Saving INFO: " << filename_info << std::endl;
}

// ------------------------------------------------------------
// Saving all results in .json


// ==================== Helpers locais (arquivo-local) ====================
namespace {

template <typename T>
void write_json_array(std::ostream& os, const std::vector<T>& arr) {
    os << "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        if constexpr (std::is_floating_point<T>::value) {
            os << std::setprecision(17) << arr[i];
        } else {
            os << arr[i];
        }
        if (i + 1 < arr.size()) os << ", ";
    }
    os << "]";
}

template <typename T>
void write_json_array_row(std::ostream& os, const std::vector<std::vector<T>>& mat, int row_0based) {
    if (row_0based < 0 || static_cast<size_t>(row_0based) >= mat.size()) {
        os << "[]";
        return;
    }
    write_json_array(os, mat[static_cast<size_t>(row_0based)]);
}

double get_by_color_or_fallback(const std::vector<double>& v, int color_1based) {
    if (v.empty()) return 0.0;
    const int idx0 = std::max(0, color_1based - 1); // tenta 1→0, 2→1, ...
    if (static_cast<size_t>(idx0) < v.size()) return v[static_cast<size_t>(idx0)];
    return v.front();
}

void check_ps_consistency(const PercolationSeries& ps) {
    const size_t m = ps.percolation_order.size();
    if (ps.color_percolation.size() != m || ps.time_percolation.size() != m) {
        throw std::runtime_error("[save_percolation_json] PercolationSeries inconsistente: tamanhos diferentes.");
    }
}

void check_ts_consistency(const TimeSeries& ts) {
    const size_t C = static_cast<size_t>(ts.num_colors >= 0 ? ts.num_colors : 0);
    if (C != ts.p_t.size() || C != ts.Nt.size()) {
        throw std::runtime_error("[save_percolation_json] TimeSeries inconsistente: num_colors != p_t.size() || Nt.size().");
    }
    for (size_t c = 0; c < C; ++c) {
        if (ts.p_t[c].size() != ts.t.size())
            throw std::runtime_error("[save_percolation_json] p_t[c].size() != t.size().");
        if (ts.Nt[c].size() != ts.t.size())
            throw std::runtime_error("[save_percolation_json] Nt[c].size() != t.size().");
    }
}

} // namespace
// =======================================================================

void save_data::save_percolation_json(const PercolationSeries& ps,
                                      const TimeSeries& ts,
                                      const std::string& filename_json,
                                      bool sort_by_order)
{
    check_ps_consistency(ps);
    check_ts_consistency(ts);

    if (ps.percolation_order.empty()) {
        std::cerr << "[WARN] save_percolation_json: nenhum evento de percolação; results ficará vazio\n";
    }

    std::ofstream ofs(filename_json);
    if (!ofs.is_open()) {
        throw std::runtime_error(std::string("[save_percolation_json] Não foi possível abrir: ") + filename_json);
    }

    const size_t M = ps.percolation_order.size();

    // ordenar por order_percolation (crescente)
    std::vector<size_t> idx(M);
    for (size_t i = 0; i < M; ++i) idx[i] = i;
    if (sort_by_order) {
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) {
                      return ps.percolation_order[a] < ps.percolation_order[b];
                  });
    }

    ofs << "{\n";
    ofs << "  \"results\": [\n";

    for (size_t k = 0; k < M; ++k) {
        const size_t i = idx[k];

        const int order_i   = ps.percolation_order[i];
        const int color_1b  = ps.color_percolation[i];     // armazenado 1-based no ps
        const int tperc_i   = ps.time_percolation[i];

        const double rho_i  = get_by_color_or_fallback(ps.rho, color_1b);
        const double pho_i  = get_by_color_or_fallback(ps.pho, color_1b);

        const int color_row = std::max(0, color_1b - 1);   // converter para acessar ts.p_t/Nt

        ofs << "    {\n";
        ofs << "      \"order_percolation\": " << order_i << ",\n";
        ofs << "      \"data\": {\n";
        ofs << "        \"color\": " << color_1b << ",\n";
        ofs << "        \"rho\": "   << std::setprecision(17) << rho_i << ",\n";
        ofs << "        \"pho\": "   << std::setprecision(17) << pho_i << ",\n";
        ofs << "        \"time_percolation\": " << tperc_i << ",\n";

        ofs << "        \"time\": ";
        write_json_array(ofs, ts.t);
        ofs << ",\n";

        ofs << "        \"pt\": ";
        write_json_array_row(ofs, ts.p_t, color_row);
        ofs << ",\n";

        ofs << "        \"nt\": ";
        write_json_array_row(ofs, ts.Nt, color_row);
        ofs << "\n";

        ofs << "      }\n";
        ofs << "    }";
        if (k + 1 < M) ofs << ",";
        ofs << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();
}





