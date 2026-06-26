#ifndef WRITE_SAVE_HPP
#define WRITE_SAVE_HPP

#include <string>

#include "network.hpp"
#include "struct_network.hpp"
#include "network_partitions.hpp"
#include "equilibration_partition.hpp"
#include "LateralCorrelation.hpp"

class save_data {
public:
    // NPZ format removed. Use `save_network_compact_bin` for compact networks.

    void save_percolation_json(const PercolationSeries& ps,
                               const TimeSeries& ts,
                               const std::string& filename_json,
                               bool sort_by_order) const;

    void save_reanalysis_json(const ReanalysisResult& result,
                              const std::string& filename_json) const;

    void save_reanalysis_networks(const ReanalysisResult& result,
                                  const std::string& filename_preteq_bin,
                                  const std::string& filename_posteq_bin) const;

    // Legacy-named helpers (still present but write compact .bin / JSON)
    void save_surfaces_as_npz(const SurfacesCuts& surfaces,
                              const std::string& filename) const;

    // Compatibility wrappers that convert encoded networks to compact .bin
    void save_network_as_npz(const NetworkPattern& net,
                             const std::string& filename) const;

    void save_network_as_npz(const SparseSubgraph& net,
                             const std::string& filename) const;

    void save_network_compact_bin(const NetworkCompact& net,
                                  const std::string& filename) const;

    void save_lateral_observables_csv(
        const std::string& output_dir,
        const std::string& sample_id,
        const LateralObservablesSeries& observables) const;
};

#endif