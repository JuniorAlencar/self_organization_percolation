#include "write_save.hpp"
#include <zip.h>
#include <stdexcept>
#include <cstring>

// Utility function: write raw .npy header (version 1.0, little-endian, shape info)
static std::vector<unsigned char> generate_npy_data(const NetworkPattern& net) {
    int rows = net.shape[0];
    int cols = net.shape[1];
    std::vector<unsigned char> result;

    // Magic string + version
    const char magic[] = "\x93NUMPY";
    result.insert(result.end(), magic, magic + 6);
    result.push_back(1);  // major
    result.push_back(0);  // minor

    // Construct header dict (corrected dtype to 4-byte integer)
    std::string header = "{'descr': '|i4', 'fortran_order': False, 'shape': (";
    header += std::to_string(rows) + ", " + std::to_string(cols) + "), }";
    while ((header.size() + 10) % 16 != 0) header += ' ';
    header += '\n';

    // Header length
    uint16_t len = static_cast<uint16_t>(header.size());
    result.push_back(len & 0xFF);
    result.push_back((len >> 8) & 0xFF);

    // Add header
    result.insert(result.end(), header.begin(), header.end());

    // Add data as 4-byte integers
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int value = net.get({i, j});
            unsigned char* bytes = reinterpret_cast<unsigned char*>(&value);
            result.insert(result.end(), bytes, bytes + sizeof(int));
        }
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

void save_data::save_p_values_as_npy(const std::vector<int>& t_values,
                                     const std::vector<double>& p_values,
                                     const std::string& filename) {
    if (t_values.size() != p_values.size()) {
        std::cerr << "Erro: t_values e p_values têm tamanhos diferentes!\n";
        return;
    }

    size_t N = t_values.size();
    std::vector<double> combined(2 * N);

    for (size_t i = 0; i < N; ++i) {
        combined[2 * i]     = static_cast<double>(t_values[i]); // coluna 0: t
        combined[2 * i + 1] = p_values[i];                      // coluna 1: p_t
    }

    std::vector<size_t> shape = {N, 2}; // matriz N x 2: [t, p_t]
    cnpy::npy_save(filename, &combined[0], shape);

    std::cout << "Saving to file: " << filename << std::endl;
}

void save_data::save_Nt_values_as_npy(const std::vector<int>& t_values,
                                      const std::vector<int>& Nt_values,
                                      const std::string& filename) {
    if (t_values.size() != Nt_values.size()) {
        std::cerr << "Erro: t_values e Nt_values têm tamanhos diferentes!\n";
        return;
    }

    size_t N = t_values.size();
    std::vector<int> combined(2 * N);

    for (size_t i = 0; i < N; ++i) {
        combined[2 * i]     = t_values[i];     // coluna 0: t
        combined[2 * i + 1] = Nt_values[i];    // coluna 1: N_t
    }

    std::vector<size_t> shape = {N, 2}; // matriz N x 2: [t, N_t]
    cnpy::npy_save(filename, &combined[0], shape);

    std::cout << "Saving to file: " << filename << std::endl;
}


