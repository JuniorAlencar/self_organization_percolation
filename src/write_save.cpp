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
                                        const std::string& filename_Nt) {
    if (ts.t.empty() || ts.p_t.empty() || ts.Nt.empty() ||
        ts.p_t.size() != ts.num_colors || ts.Nt.size() != ts.num_colors) {
        std::cerr << "[Erro] TimeSeries inconsistente!\n";
        return;
    }

    size_t T = ts.t.size();

    // --- Saving p(t) ---
    {
        std::ofstream file_pt(filename_pt);
        if (!file_pt.is_open()) {
            std::cerr << "Erro ao abrir " << filename_pt << " para escrita\n";
        } else {
            file_pt << "t";
            for (int c = 0; c < ts.num_colors; ++c)
                file_pt << ",pt_" << (c + 1);
            file_pt << "\n";

            for (size_t i = 0; i < T; ++i) {
                file_pt << ts.t[i];
                for (int c = 0; c < ts.num_colors; ++c)
                    file_pt << "," << ts.p_t[c][i];
                file_pt << "\n";
            }
            file_pt.close();
            std::cout << "Saving CSV: " << filename_pt << std::endl;
        }
    }

    // --- Saving N(t) ---
    {
        std::ofstream file_Nt(filename_Nt);
        if (!file_Nt.is_open()) {
            std::cerr << "Erro ao abrir " << filename_Nt << " para escrita\n";
        } else {
            file_Nt << "t";
            for (int c = 0; c < ts.num_colors; ++c)
                file_Nt << ",Nt_" << (c + 1);
            file_Nt << "\n";

            for (size_t i = 0; i < T; ++i) {
                file_Nt << ts.t[i];
                for (int c = 0; c < ts.num_colors; ++c)
                    file_Nt << "," << ts.Nt[c][i];
                file_Nt << "\n";
            }
            file_Nt.close();
            std::cout << "Saving CSV: " << filename_Nt << std::endl;
        }
    }
}




