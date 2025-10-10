#ifndef write_save_hpp
#define write_save_hpp

// Minimal gzip-compressed npz (numpy .npz = zip of .npy files) writer for 1-bit sparse binary matrix
// This will write an uncompressed format inside a zip structure (basic version)
#include "network.hpp"
#include "struct_network.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <unordered_map>
#include <iomanip>
#include <type_traits>
#include <algorithm>
#include <sstream>
#include <omp.h>      // OpenMP
#include <zip.h> // Requires libzip-dev installed


class save_data {
    public:
        // Salva o campo "net" em arquivo .npy (int32, C-order).
        // Mantém o nome da API anterior por compatibilidade.
        void save_network_as_npz(const NetworkPattern& net, const std::string& filename) const;

        // Salva tudo em JSON (inclui séries p_t, Nt, M_t, M_t_total; eventos de percolação; SP).
        void save_percolation_json(const PercolationSeries& ps,
                                const TimeSeries& ts,
                                const std::string& filename_json,
                                bool& DSU_calculate_,
                                bool sort_by_order = true) const;
};

#endif // write_save_hpp