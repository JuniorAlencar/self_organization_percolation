#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>

#include "network.hpp"
#include "struct_network.hpp"
#include "write_save.hpp"

int main(int argc, char** argv) {
    try {
        const int dim = 3;
        const int L = 128;
        const int num_samples = 500;
        const int num_colors = 1; // n_s = 1
        const double c = 0.01;
        const double f_T = 0.06;
        const int type_f_T = 0;
        const std::vector<double> p0 = {1.0};
        const double P0 = 0.1;
        const double a = 0.0;
        const double alpha = 1.0;
        const std::string type_percolation = "site";
        const std::vector<double> rho = {1.0};

        const int seed = 12345;
        all_random rng(seed);

        network nw(num_samples, num_colors);
        TimeSeries ts;
        PercolationSeries ps;

        std::cout << "Running create_network (this may take a few seconds)...\n";
        NetworkPattern net = nw.create_network(dim, L, num_samples, c, f_T, type_f_T,
                                              p0, P0, a, alpha, type_percolation,
                                              num_colors, rho, ts, ps, rng);

        // path used in create_network: results/network_compact_seed_<seed>_L_<L>_T_<T>.bin
        // Save percolation/time-series JSON for downstream inspection (even
        // if no percolation occurred). This provides t, p(t), f(t) in a file.
        std::ostringstream jsn;
        jsn << "results/percolation_seed_" << net.seed
            << "_L_" << L << "_T_" << num_samples << ".json";
        try {
            save_data sd2;
            sd2.save_percolation_json(ps, ts, jsn.str(), true);
            std::cout << "Saved percolation JSON: " << jsn.str() << "\n";
        } catch (const std::exception &e) {
            std::cerr << "Warning: failed to save percolation JSON: " << e.what() << "\n";
        }

        // If no species percolated, create_network will not have saved a compact
        // file and ps will have empty color_percolation. In that case we skip
        // reading/saving filtered outputs (sample is considered empty).
        if (ps.color_percolation.empty()) {
            std::cout << "No species percolated in this sample; nothing saved." << std::endl;
            return 0;
        }

        std::ostringstream pathss;
        pathss << "results/network_compact_seed_" << net.seed
               << "_L_" << L << "_T_" << num_samples << ".bin";
        const std::string path = pathss.str();

        std::cout << "Saved compact network path: " << path << "\n";

        NetworkCompact netc;
        if (!netc.read_binary(path)) {
            std::cerr << "Failed to read compact network from " << path << "\n";
            return 2;
        }

        std::cout << "Read compact network: N=" << netc.N << ", E=" << netc.num_edges() << "\n";

        // Filter active nodes and save filtered file
        NetworkCompact filtered = netc.filter_active();
        std::ostringstream outp;
        outp << "results/network_compact_active_seed_" << net.seed
            << "_L_" << L << "_T_" << num_samples << ".bin";
        save_data sd;
        sd.save_network_compact_bin(filtered, outp.str());

        std::cout << "Saved filtered active compact network: " << outp.str() << " (N=" << filtered.N << ")\n";

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
