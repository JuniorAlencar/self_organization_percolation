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
#include <sstream>
#include <omp.h>      // OpenMP
#include <zip.h> // Requires libzip-dev installed


class save_data {
    public:
        // Save a binary matrix (NetworkPattern) into a compressed .npz file
        void save_network_as_npz(const NetworkPattern& net, 
                                 const std::string& filename) const;
        void save_time_series_as_csv(const TimeSeries& ts, 
                                    const std::string& filename_pt, 
                                    const std::string& filename_Nt);

};

#endif // write_save_hpp