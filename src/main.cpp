// main.cpp (limpo: tudo fora do main foi para helpers_print)

#include "network.hpp"
#include "rand_utils.hpp"
#include "write_save.hpp"
#include "create_folders.hpp"
#include "helpers_print.hpp"

#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <utility>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    if (argc >= 2) {
        if (helpers::is_help_token(argv[1])) {
            helpers::print_help(argv[0]);
            return 0;
        }
        if (std::strcmp(argv[1], "--version") == 0) {
            helpers::print_version();
            return 0;
        }
    }

    if (argc != 12) {
        std::cerr << "[ERROR] Invalid number of arguments (" << argc - 1 << ").\n";
        helpers::print_help(argv[0]);
        return 1;
    }

    try {
        int L = std::stoi(argv[1]);
        double pp0 = std::stod(argv[2]);
        int seed = std::stoi(argv[3]);
        std::string type_percolation = argv[4];
        double k = std::stod(argv[5]);
        int N_t = std::stoi(argv[6]);
        int dim = std::stoi(argv[7]);
        int num_colors = std::stoi(argv[8]);
        double rho_val = std::stod(argv[9]);
        double P0 = std::stod(argv[10]);
        std::string equilibration = argv[11];

        const bool animation = helpers::parse_bool(equilibration);

        int type_N_t = 0;
        double a = 0.0;
        double alpha = 0.0;

        if (dim != 2 && dim != 3) {
            std::cerr << "[ERROR] dim must be 2 or 3.\n";
            helpers::print_help(argv[0]);
            return 1;
        }

        if (type_percolation != "bond" && type_percolation != "node") {
            std::cerr << "[ERROR] type_percolation must be 'bond' or 'node'.\n";
            helpers::print_help(argv[0]);
            return 1;
        }

        if (num_colors < 1) {
            std::cerr << "[ERROR] num_colors must be >= 1.\n";
            helpers::print_help(argv[0]);
            return 1;
        }

        if (num_colors * rho_val > 1.0 + 1e-12) {
            std::cerr << "[ERROR] Constraint violated: num_colors * rho_val <= 1.0.\n"
                         "        You passed: num_colors=" << num_colors
                      << " and rho_val=" << rho_val
                      << " (product=" << num_colors * rho_val << ")\n";
            helpers::print_help(argv[0]);
            return 1;
        }

        if (seed == -1) {
            seed = all_random::generate_random_seed();
        }

        all_random rng(seed);

        TimeSeries ts;
        PercolationSeries ps;

        std::vector<double> rho(num_colors, rho_val);
        std::vector<double> p0(num_colors, pp0);

        int N_samples = 100000;
        network net_generator(N_samples, num_colors);

        NetworkPattern net = animation
            ? net_generator.animate_network(
                dim, L, N_samples, k, N_t, type_N_t,
                p0, P0, a, alpha, type_percolation,
                num_colors, rho, ts, ps, rng
            )
            : net_generator.create_network(
                dim, L, N_samples, k, N_t, type_N_t,
                p0, P0, a, alpha, type_percolation,
                num_colors, rho, ts, ps, rng
            );

        FolderCreator creator("./SOP_data");
        const auto [
            network_dir,
            data_dir,
            equilibration_dir,
            network_preteq,
            network_posteq
        ] = creator.create_structure(
                dim,
                type_N_t,
                N_t,
                k,
                L,
                num_colors,
                a,
                alpha,
                type_percolation,
                pp0,
                P0,
                rho_val,
                animation
            );

        std::cerr << "[DBG] ps sizes -> "
                  << "order=" << ps.percolation_order.size()
                  << ", color=" << ps.color_percolation.size()
                  << ", rho="   << ps.rho.size()
                  << ", rho_value=" << rho_val << "\n";

        std::cerr << "[DBG] ts sizes -> "
                  << "num_colors=" << ts.num_colors
                  << ", t="  << ts.t.size()
                  << ", p_t="<< ts.p_t.size()
                  << ", Nt=" << ts.Nt.size() << "\n";

        std::cout << "seed = " << seed << std::endl;

        const std::string machine_name = helpers::get_machine_name();
        const std::string timestamp_now = helpers::get_timestamp_now();

        std::ostringstream base_name;
        base_name << machine_name
                  << "_seed_" << seed
                  << "_ts_" << timestamp_now
                  << "_P0_" << std::fixed << std::setprecision(2) << P0
                  << "_p0_" << std::fixed << std::setprecision(2) << pp0;

        const std::string sample_base = base_name.str();
        std::string json_filename = data_dir + "/" + sample_base + ".json";

        save_data saver;

        if (animation == true) {
            std::string net_filename = network_dir + "/" + sample_base + ".npz";
            saver.save_network_as_npz(net, net_filename);
        }

        saver.save_percolation_json(ps, ts, json_filename, true);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[FATAL] Exception: " << e.what() << "\n";
        return 2;
    }
    catch (...) {
        std::cerr << "[FATAL] Unknown exception.\n";
        return 3;
    }
}