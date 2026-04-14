#ifndef WRITE_SAVE_HPP
#define WRITE_SAVE_HPP

#pragma once

#include "network.hpp"
#include "struct_network.hpp"
#include "../analysis_equilibration/animation_reanalysis.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <zip.h>

class save_data {
public:
    void save_network_as_npz(const NetworkPattern& net,
                             const std::string& filename) const;

    void save_percolation_json(const PercolationSeries& ps,
                               const TimeSeries& ts,
                               const std::string& filename_json,
                               bool sort_by_order = true) const;
    void save_reanalysis_json(const ReanalysisResult& result,
                              const std::string& filename_json) const;

    void save_reanalysis_networks(const ReanalysisResult& result,
                                const std::string& filename_preteq_npz,
                                const std::string& filename_posteq_npz) const;
};

#endif // WRITE_SAVE_HPP
