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

void save_data::save_p_values_as_npy(const std::vector<int>& t_values,
                                     const std::vector<std::vector<double>>& p_values,
                                     const std::string& filename) {
    size_t N = t_values.size();
    size_t num_colors = p_values.empty() ? 0 : p_values[0].size();

    // Validação
    for (const auto& row : p_values) {
        if (row.size() != num_colors) {
            std::cerr << "Erro: Linhas de p_values têm número de colunas inconsistentes!\n";
            return;
        }
    }

    std::vector<double> combined(N * (1 + num_colors)); // [t, p1, p2, ..., pc]

    for (size_t i = 0; i < N; ++i) {
        combined[i * (1 + num_colors)] = static_cast<double>(t_values[i]);
        for (size_t j = 0; j < num_colors; ++j) {
            combined[i * (1 + num_colors) + j + 1] = p_values[i][j];
        }
    }

    std::vector<size_t> shape = {N, 1 + num_colors};
    cnpy::npy_save(filename, &combined[0], shape);
    std::cout << "Saving to file: " << filename << std::endl;
}


void save_data::save_Nt_values_as_npy(const std::vector<int>& t_values,
                                      const std::vector<std::vector<int>>& Nt_values,
                                      const std::string& filename) {
    size_t N = t_values.size();
    size_t num_colors = Nt_values.empty() ? 0 : Nt_values[0].size();

    for (const auto& row : Nt_values) {
        if (row.size() != num_colors) {
            std::cerr << "Erro: Linhas de Nt_values têm número de colunas inconsistentes!\n";
            return;
        }
    }

    std::vector<int> combined(N * (1 + num_colors)); // [t, N1, N2, ..., Nc]

    for (size_t i = 0; i < N; ++i) {
        combined[i * (1 + num_colors)] = t_values[i];
        for (size_t j = 0; j < num_colors; ++j) {
            combined[i * (1 + num_colors) + j + 1] = Nt_values[i][j];
        }
    }

    std::vector<size_t> shape = {N, 1 + num_colors};
    cnpy::npy_save(filename, &combined[0], shape);
    std::cout << "Saving to file: " << filename << std::endl;
}

