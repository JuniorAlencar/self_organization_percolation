#ifndef write_save_hpp
#define write_save_hpp

// Minimal gzip-compressed npz (numpy .npz = zip of .npy files) writer for 1-bit sparse binary matrix
// This will write an uncompressed format inside a zip structure (basic version)
#include "network.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <omp.h>      // OpenMP
#include <zip.h> // Requires libzip-dev installed
#include <cnpy.h>  // Make sure cnpy is installed and included

class save_data {
    public:
        // Save a binary matrix (NetworkPattern) into a compressed .npz file
        void save_network_as_npz(const NetworkPattern& net, 
                                const std::string& filename) const;
        void save_data::save_p_values_as_npy(const std::vector<int>& t_values,
                                            const std::vector<std::vector<double>>& p_values,
                                            const std::string& filename);

        void save_data::save_Nt_values_as_npy(const std::vector<int>& t_values,
                                      const std::vector<std::vector<int>>& Nt_values,
                                      const std::string& filename);

};

#endif // write_save_hpp