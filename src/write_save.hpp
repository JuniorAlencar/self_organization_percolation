#ifndef WRITE_SAVE_HPP
#define WRITE_SAVE_HPP

#include <string>

#include "network.hpp"
#include "../analysis_equilibration/equilibration_reanalysis.hpp"

class save_data {
public:
    // -------- save da rede original (pipeline principal) --------
    // Mantém compatibilidade com o fluxo atual do SOP.
    void save_network_as_npz(const NetworkPattern& net,
                             const std::string& filename) const;

    // -------- save da rede esparsa do reanalysis --------
    // Salva apenas os sítios ativos do subgrafo.
    void save_network_as_npz(const SparseSubgraph& net,
                             const std::string& filename) const;

    // -------- resultados da simulação principal --------
    void save_percolation_json(const PercolationSeries& ps,
                               const TimeSeries& ts,
                               const std::string& filename_json,
                               bool sort_by_order) const;

    // -------- resultados do EquilibrationReanalysis --------
    void save_reanalysis_json(const ReanalysisResult& result,
                              const std::string& filename_json) const;

    void save_reanalysis_networks(const ReanalysisResult& result,
                                  const std::string& filename_preteq_npz,
                                  const std::string& filename_posteq_npz) const;
};

#endif