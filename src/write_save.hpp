#ifndef WRITE_SAVE_HPP
#define WRITE_SAVE_HPP

#include <string>

#include "network.hpp"
#include "struct_network.hpp"
#include "network_partitions.hpp"
#include "equilibration_partition.hpp"

class save_data {
public:
    void save_network_as_npz(const NetworkPattern& net,
                             const std::string& filename) const;

    void save_network_as_npz(const SparseSubgraph& net,
                             const std::string& filename) const;

    void save_percolation_json(const PercolationSeries& ps,
                               const TimeSeries& ts,
                               const std::string& filename_json,
                               bool sort_by_order,
                               double c,
                               double f_T) const;

    void save_reanalysis_json(const ReanalysisResult& result,
                              const std::string& filename_json) const;

    void save_reanalysis_networks(const ReanalysisResult& result,
                                  const std::string& filename_preteq_npz,
                                  const std::string& filename_posteq_npz) const;

    void save_surfaces_as_npz(const SurfacesCuts& surfaces,
                              const std::string& filename) const;
};

#endif