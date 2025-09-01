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

void save_data::save_time_series_as_csv(const TimeSeries& ts,
                                        const std::string& filename_pt,
                                        const std::string& filename_Nt)
{
    auto bail = [&](const std::string& msg){
        std::cerr << "[Erro] TimeSeries inconsistente: " << msg << "\n";
    };

    if (ts.t.empty()) { bail("t vazio"); return; }
    if (ts.p_t.empty()) { bail("p_t vazio"); return; }
    if (ts.Nt.empty())  { bail("Nt vazio"); return; }

    const size_t T = ts.t.size();

    // Detecta orientação
    bool looks_color_major =
        (ts.p_t.size() == static_cast<size_t>(ts.num_colors)) &&
        (ts.Nt.size()  == static_cast<size_t>(ts.num_colors));

    bool ok_color_major = looks_color_major;
    if (looks_color_major) {
        for (int c = 0; c < ts.num_colors; ++c) {
            if (ts.p_t[c].size() != T || ts.Nt[c].size() != T) {
                ok_color_major = false; break;
            }
        }
    }

    bool looks_time_major =
        (ts.p_t.size() == T) && (ts.Nt.size() == T);

    bool ok_time_major = looks_time_major;
    if (looks_time_major) {
        for (size_t i = 0; i < T; ++i) {
            if (ts.p_t[i].size() != static_cast<size_t>(ts.num_colors) ||
                ts.Nt[i].size()  != static_cast<size_t>(ts.num_colors)) {
                ok_time_major = false; break;
            }
        }
    }

    if (!ok_color_major && !ok_time_major) {
        bail("nem cor-major nem time-major: "
             "p_t.outer=" + std::to_string(ts.p_t.size()) +
             ", Nt.outer=" + std::to_string(ts.Nt.size()) +
             ", T=" + std::to_string(T));
        return;
    }

    // ---- Salvar p(t) ----
    {
        std::ofstream f(filename_pt);
        if (!f) { std::cerr << "Erro ao abrir " << filename_pt << "\n"; }
        else {
            f << "t";
            for (int c = 0; c < ts.num_colors; ++c) f << ",pt_" << (c+1);
            f << "\n";

            if (ok_color_major) {
                for (size_t i = 0; i < T; ++i) {
                    f << ts.t[i];
                    for (int c = 0; c < ts.num_colors; ++c)
                        f << "," << ts.p_t[c][i];
                    f << "\n";
                }
            } else { // ok_time_major
                for (size_t i = 0; i < T; ++i) {
                    f << ts.t[i];
                    for (int c = 0; c < ts.num_colors; ++c)
                        f << "," << ts.p_t[i][c];
                    f << "\n";
                }
            }
            std::cout << "Saving CSV: " << filename_pt << std::endl;
        }
    }

    // ---- Salvar N(t) ----
    {
        std::ofstream f(filename_Nt);
        if (!f) { std::cerr << "Erro ao abrir " << filename_Nt << "\n"; }
        else {
            f << "t";
            for (int c = 0; c < ts.num_colors; ++c) f << ",Nt_" << (c+1);
            f << "\n";

            if (ok_color_major) {
                for (size_t i = 0; i < T; ++i) {
                    f << ts.t[i];
                    for (int c = 0; c < ts.num_colors; ++c)
                        f << "," << ts.Nt[c][i];
                    f << "\n";
                }
            } else { // ok_time_major
                for (size_t i = 0; i < T; ++i) {
                    f << ts.t[i];
                    for (int c = 0; c < ts.num_colors; ++c)
                        f << "," << ts.Nt[i][c];
                    f << "\n";
                }
            }
            std::cout << "Saving CSV: " << filename_Nt << std::endl;
        }
    }
}


void save_data::save_info_percolation(const PercolationSeries& ps,
                                      const std::string& filename_info){
    
    if (ps.color_percolation.empty() || ps.time_percolation.empty() || ps.percolation_order.empty() ||
    ps.color_percolation.size() != ps.time_percolation.size() || ps.percolation_order.size() != ps.color_percolation.size()) {
        std::cerr << "[Erro] TimeSeries inconsistente!\n";
        return;
    }
    std::ofstream file_info(filename_info);
    if (!file_info.is_open()) {
        std::cerr << "Erro to open " << filename_info << "to write\n";
    }
    
    else{
        file_info << "color" << " " << "rho" << " " << "pho" <<  " " << "time_percolation" << " " << "percolation_order\r\n";
        
        // minimum value of vectors (number of colors)
        const size_t n = std::min({ps.color_percolation.size(),
                                ps.time_percolation.size(),
                                ps.percolation_order.size()});
        // Run from vectors elements
        for (size_t i = 0; i < n; ++i) {
        file_info << ps.color_percolation[i] << ' '
                << ps.rho[i] << ' ' << ps.pho[i] << ' '<< ps.time_percolation[i] << ' '
                << ps.percolation_order[i]  << '\n';
        }

        file_info.close();

    }

}




