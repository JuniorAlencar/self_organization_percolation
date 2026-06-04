// main.cpp (limpo: tudo fora do main foi para helpers_print)

#include "network.hpp"
#include "rand_utils.hpp"
#include "write_save.hpp"
#include "create_folders.hpp"
#include "helpers_print.hpp"
#include "helpers_partitions.hpp"
#include "network_partitions.hpp"
#include "equilibration_partition.hpp"
#include "height_stop_config.hpp"

#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <utility>
#include <stdexcept>
#include <string>
#include <optional>
#include <algorithm>

namespace rh = reanalysis_helpers;

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

    // Allow either zero-argument (use defaults) or full-argument run.
    // Optional final flag enables expensive geometric/network properties.
    if (argc != 1 && argc != 12 && argc != 13 && argc != 14) {
        std::cerr << "[ERROR] Invalid number of arguments (" << argc - 1 << ").\n";
        helpers::print_help(argv[0]);
        return 1;
    }

    try {
        // If no arguments provided, use a set of reasonable defaults you can
        // edit here. If full argv are provided (11), parse them.
        int L = 128;
        double pp0 = 1.0;
        int seed = 12345;
        std::string type_percolation = "site";
        double c = 0.01;
        double f_T = 0.06;
        int dim = 3;
        int num_colors = 1;
        double rho_val = 1.0;
        double P0 = 0.1;
        std::string equilibration = "true";
        bool calculate_detailed_properties = false;
        std::string run_mode = "sop";
        
        if (argc == 12 || argc == 13 || argc == 14) {
            L = std::stoi(argv[1]);
            pp0 = std::stod(argv[2]);
            seed = std::stoi(argv[3]);
            type_percolation = argv[4];
            c = std::stod(argv[5]);
            f_T = std::stod(argv[6]);
            dim = std::stoi(argv[7]);
            num_colors = std::stoi(argv[8]);
            rho_val = std::stod(argv[9]);
            P0 = std::stod(argv[10]);
            equilibration = argv[11];
            if (argc >= 13) {
                calculate_detailed_properties = helpers::parse_bool(argv[12]);
            }
            if (argc == 14) {
                run_mode = argv[13];
            }
        }

        const bool teste = (run_mode == "growth_test");
        if (run_mode != "sop" && run_mode != "growth_test") {
            std::cerr << "[ERROR] run mode must be 'sop' or 'growth_test'.\n";
            helpers::print_help(argv[0]);
            return 1;
        }

        const bool animation = helpers::parse_bool(equilibration);

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
        GrowthStopConfig stop_config;
        if (teste) {
            stop_config.height_multiplier = HEIGHT_STOP_MULTIPLIER;
            stop_config.dynamic_height = true;
            stop_config.stop_at_percolation = false;
            stop_config.stop_at_equilibrium = true;
            stop_config.equilibrium_consecutive_steps = 10;
            stop_config.dynamics_window_steps =
                GROWTH_TEST_DYNAMICS_WINDOW_FACTOR *
                stop_config.equilibrium_consecutive_steps;
            if (const char* env_window =
                    std::getenv("GROWTH_TEST_DYNAMICS_WINDOW_STEPS")) {
                stop_config.dynamics_window_steps = std::stoi(env_window);
                if (stop_config.dynamics_window_steps <= 1) {
                    throw std::runtime_error(
                        "GROWTH_TEST_DYNAMICS_WINDOW_STEPS must be > 1");
                }
            }
        }
        const int SPECIES_FACTOR = 10000000;
        int type_f_T = 0;
        double a = 0.0, alpha = 0.0;
        //double alpha = 0.0;
        
        network net_generator(N_samples, num_colors);

        const bool build_full_network = animation && calculate_detailed_properties;

        NetworkPattern net = build_full_network
            ? net_generator.animate_network(
                    dim, L, N_samples, c, f_T, type_f_T,
                    p0, P0, a, alpha, type_percolation,
                    num_colors, rho, ts, ps, rng,
                    calculate_detailed_properties,
                    stop_config
              )
            : net_generator.create_network(
                    dim, L, N_samples, c, f_T, type_f_T,
                    p0, P0, a, alpha, type_percolation,
                    num_colors, rho, ts, ps, rng, false,
                    calculate_detailed_properties,
                    stop_config
              );

        FolderCreator creator("./SOP_data");
        const auto [
            network_dir,
            data_dir,
            surfaces_dir,
            network_preteq,
            network_posteq
        ] = creator.create_structure(
                dim,
                type_f_T,
                f_T,
                c,
                L,
                num_colors,
                a,
                alpha,
                type_percolation,
                pp0,
                P0,
                rho_val,
                teste,
                stop_config.dynamic_height,
                stop_config.height_extra_layers,
                stop_config.dynamics_window_steps
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
                  << ", f_t=" << ts.f_t.size() << "\n";

        std::cout << "seed = " << seed << std::endl;

        const std::string machine_name = helpers::get_machine_name();
        const std::string timestamp_now = helpers::get_timestamp_now();

        std::ostringstream base_name;
        base_name << machine_name
                  << "_seed_" << seed
                  << "_ts_" << timestamp_now
                  << "_P0_" << std::fixed << std::setprecision(2) << P0
                  << "_p0_" << std::fixed << std::setprecision(2) << pp0;
        save_data saver;
        
        const std::string sample_base = base_name.str();
        std::string json_filename = data_dir + "/" + sample_base + ".json";
        
        const bool has_percolation = !ps.color_percolation.empty();
        const bool write_large_artifacts =
            calculate_detailed_properties &&
            build_full_network &&
            has_percolation &&
            std::isfinite(ps.t_eq);

        std::optional<EquilibrationCutNetworks> cuts;
        if (write_large_artifacts) {
            cuts.emplace(build_equilibration_cut_networks(
                net,
                ps.t_eq,
                SPECIES_FACTOR
            ));
        }

        if (write_large_artifacts) {
            std::string surfaces_filename = surfaces_dir + "/" + sample_base + ".npz";
            SurfacesCuts surfaces =
                extract_exposed_surfaces(net, *cuts, SPECIES_FACTOR);
            saver.save_surfaces_as_npz(surfaces, surfaces_filename);
        } else if (build_full_network && !has_percolation) {
            std::cout << "[INFO] No percolating species found; skipping surface file."
                      << std::endl;
        } else if (animation && !calculate_detailed_properties) {
            std::cout << "[INFO] Time-series-only mode: skipping network/surface artifacts."
                      << std::endl;
        }

        if (write_large_artifacts) {
            // Helper: convert NetworkPattern -> NetworkCompact (decode encoded values)
            auto convert_to_compact = [&](const NetworkPattern& np) {
                NetworkCompact nc;
                const std::size_t total = np.data.size();
                nc.N = static_cast<NetworkCompact::index_t>(total);
                nc.pos_flat.resize(nc.N);
                for (NetworkCompact::index_t i = 0; i < nc.N; ++i) nc.pos_flat[i] = i;

                nc.species.resize(nc.N);
                nc.activation_time.resize(nc.N);

                for (NetworkCompact::index_t i = 0; i < nc.N; ++i) {
                    const long long code = static_cast<long long>(np.data[static_cast<std::size_t>(i)]);
                    if (code <= 0) {
                        nc.species[i] = 0;
                        nc.activation_time[i] = 0u;
                    } else {
                        const int color_1b = static_cast<int>(code / SPECIES_FACTOR);
                        const int time = static_cast<int>(code % SPECIES_FACTOR);
                        int color_idx = 0;
                        if (np.num_colors == 1) color_idx = 0;
                        else color_idx = std::max(0, std::min(np.num_colors - 1, color_1b - 1));
                        // store species as 1-based color id to match previous convention
                        nc.species[i] = static_cast<uint8_t>(color_idx + 1);
                        nc.activation_time[i] = static_cast<uint32_t>(time);
                    }
                }

                if (!np.edge_pairs.empty()) {
                    nc.build_csr_from_edge_pairs(np.edge_pairs);
                } else {
                    nc.edge_offsets.assign(nc.N + 1, 0);
                    nc.edges.clear();
                }

                return nc;
            };

            // full network (stored as compact)
            const std::string net_compact_filename = network_dir + "/" + sample_base + ".bin";
            try {
                NetworkCompact fullc = convert_to_compact(net);
                saver.save_network_compact_bin(fullc, net_compact_filename);
            } catch (const std::exception &e) {
                std::cerr << "Warning: failed to save full compact network: " << e.what() << '\n';
            }

            // percolating clusters (compact)
            try {
                NetworkPattern net_perc_clusters = net_generator.filter_percolating_clusters_from_encoded(net);
                NetworkCompact percc = convert_to_compact(net_perc_clusters);
                const std::string net_PERCOLATION_filename = network_dir + "/" + sample_base + "_PERCOLATION" + ".bin";
                saver.save_network_compact_bin(percc, net_PERCOLATION_filename);
            } catch (const std::exception &e) {
                std::cerr << "Warning: failed to save percolation compact network: " << e.what() << '\n';
            }

            // pre/post teq networks: prefer to preserve CSR edges from the full compact
            try {
                // Attempt to read the full compact file we saved above to reuse its CSR
                NetworkCompact base_full;
                bool have_csr = false;
                if (base_full.read_binary(net_compact_filename)) {
                    have_csr = true;
                }

                // Convert cuts to compact form (species + activation_time)
                NetworkCompact pre_c = convert_to_compact(cuts->pre_teq);
                NetworkCompact post_c = convert_to_compact(cuts->post_teq);

                auto rebuild_pre_post_csr = [](NetworkCompact& pre,
                                               NetworkCompact& post,
                                               const NetworkCompact& base) {
                    pre.edge_offsets.assign(pre.N + 1, 0);
                    post.edge_offsets.assign(post.N + 1, 0);

                    for (NetworkCompact::index_t u = 0; u < base.N; ++u) {
                        const bool pre_u = pre.species[u] != 0;
                        const bool post_u = post.species[u] != 0;
                        if (!pre_u && !post_u) continue;

                        const NetworkCompact::index_t start = base.neighbors_start(u);
                        const NetworkCompact::index_t end = base.neighbors_end(u);
                        for (NetworkCompact::index_t k = start; k < end; ++k) {
                            const NetworkCompact::index_t v =
                                base.edges[static_cast<std::size_t>(k)];
                            if (v >= base.N) continue;
                            if (pre_u && pre.species[v] != 0) {
                                ++pre.edge_offsets[static_cast<std::size_t>(u) + 1u];
                            }
                            if (post_u && post.species[v] != 0) {
                                ++post.edge_offsets[static_cast<std::size_t>(u) + 1u];
                            }
                        }
                    }

                    for (NetworkCompact::index_t i = 1; i <= pre.N; ++i) {
                        pre.edge_offsets[static_cast<std::size_t>(i)] +=
                            pre.edge_offsets[static_cast<std::size_t>(i - 1)];
                        post.edge_offsets[static_cast<std::size_t>(i)] +=
                            post.edge_offsets[static_cast<std::size_t>(i - 1)];
                    }

                    pre.edges.assign(pre.edge_offsets.back(), 0);
                    post.edges.assign(post.edge_offsets.back(), 0);

                    std::vector<NetworkCompact::index_t> pre_cursor(
                        pre.edge_offsets.begin(),
                        pre.edge_offsets.end());
                    std::vector<NetworkCompact::index_t> post_cursor(
                        post.edge_offsets.begin(),
                        post.edge_offsets.end());

                    for (NetworkCompact::index_t u = 0; u < base.N; ++u) {
                        const bool pre_u = pre.species[u] != 0;
                        const bool post_u = post.species[u] != 0;
                        if (!pre_u && !post_u) continue;

                        const NetworkCompact::index_t start = base.neighbors_start(u);
                        const NetworkCompact::index_t end = base.neighbors_end(u);
                        for (NetworkCompact::index_t k = start; k < end; ++k) {
                            const NetworkCompact::index_t v =
                                base.edges[static_cast<std::size_t>(k)];
                            if (v >= base.N) continue;
                            if (pre_u && pre.species[v] != 0) {
                                pre.edges[static_cast<std::size_t>(pre_cursor[u]++)] = v;
                            }
                            if (post_u && post.species[v] != 0) {
                                post.edges[static_cast<std::size_t>(post_cursor[u]++)] = v;
                            }
                        }
                    }
                };

                if (have_csr && base_full.N == pre_c.N) {
                    // Build CSR for pre and post by selecting only edges
                    // between active nodes, without materializing edge pairs.
                    rebuild_pre_post_csr(pre_c, post_c, base_full);
                }

                const std::string net_preteq_filename = network_preteq + "/" + sample_base + ".bin";
                const std::string net_posteq_filename = network_posteq + "/" + sample_base + ".bin";
                saver.save_network_compact_bin(pre_c, net_preteq_filename);
                saver.save_network_compact_bin(post_c, net_posteq_filename);

                // Additionally save filtered (reindexed) active-only compact networks.
                // Build and write them one at a time to avoid doubling the peak RAM.
                const std::string net_preteq_active = network_preteq + "/" + sample_base + "_active.bin";
                const std::string net_posteq_active = network_posteq + "/" + sample_base + "_active.bin";
                {
                    NetworkCompact pre_filtered = pre_c.filter_active();
                    saver.save_network_compact_bin(pre_filtered, net_preteq_active);
                }
                {
                    NetworkCompact post_filtered = post_c.filter_active();
                    saver.save_network_compact_bin(post_filtered, net_posteq_active);
                }
            } catch (const std::exception &e) {
                std::cerr << "Warning: failed to save pre/post teq compact networks: " << e.what() << '\n';
            }
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
