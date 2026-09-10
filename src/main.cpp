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
#include <limits>

namespace rh = reanalysis_helpers;

namespace {

constexpr int SPECIES_FACTOR = 10000000;

NetworkCompact convert_encoded_to_compact(const NetworkPattern& np)
{
    NetworkCompact nc;
    const std::size_t total = np.data.size();
    nc.N = static_cast<NetworkCompact::index_t>(total);
    nc.pos_flat.resize(nc.N);
    for (NetworkCompact::index_t i = 0; i < nc.N; ++i) nc.pos_flat[i] = i;

    nc.species.resize(nc.N);
    nc.activation_time.resize(nc.N);

    for (NetworkCompact::index_t i = 0; i < nc.N; ++i) {
        const long long code =
            static_cast<long long>(np.data[static_cast<std::size_t>(i)]);
        if (code <= 0) {
            nc.species[i] = 0;
            nc.activation_time[i] = 0u;
        } else {
            const int color_1b = static_cast<int>(code / SPECIES_FACTOR);
            const int time = static_cast<int>(code % SPECIES_FACTOR);
            int color_idx = 0;
            if (np.num_colors == 1) {
                color_idx = 0;
            } else {
                color_idx = std::max(0, std::min(np.num_colors - 1, color_1b - 1));
            }
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
}

NetworkCompact build_animation_window_compact_from_encoded(const NetworkPattern& np,
                                                           const PercolationSeries& ps,
                                                           const int dim,
                                                           const int L)
{
    if (L <= 0) {
        throw std::runtime_error("build_animation_window_compact_from_encoded: L invalido");
    }
    if (dim != 2 && dim != 3) {
        throw std::runtime_error("build_animation_window_compact_from_encoded: dim deve ser 2 ou 3");
    }
    if (static_cast<int>(np.shape.size()) != dim) {
        throw std::runtime_error("build_animation_window_compact_from_encoded: shape incompatível com dim");
    }

    const NetworkCompact::index_t invalid =
        std::numeric_limits<NetworkCompact::index_t>::max();
    const NetworkCompact::index_t layer_size =
        dim == 3
            ? static_cast<NetworkCompact::index_t>(L) * static_cast<NetworkCompact::index_t>(L)
            : static_cast<NetworkCompact::index_t>(L);
    const int max_z = np.shape[dim - 1] - 1;

    struct Window {
        int z0 = 0;
        int z1 = 0;
        std::size_t base = 0;
    };

    std::vector<Window> windows;
    windows.reserve(ps.z_stat_by_species.size());
    for (const int z_bottom : ps.z_stat_by_species) {
        if (z_bottom < 0) continue;
        const int z0 = std::max(0, z_bottom);
        const int z1 = std::min(max_z, z_bottom + L);
        if (z0 <= z1) windows.push_back(Window{z0, z1, 0});
    }
    if (windows.empty()) {
        NetworkCompact out;
        out.edge_offsets.assign(1, 0);
        return out;
    }

    std::sort(windows.begin(), windows.end(), [](const Window& a, const Window& b) {
        return a.z0 < b.z0 || (a.z0 == b.z0 && a.z1 < b.z1);
    });
    std::vector<Window> merged;
    for (const Window& w : windows) {
        if (merged.empty() || w.z0 > merged.back().z1 + 1) {
            merged.push_back(w);
        } else {
            merged.back().z1 = std::max(merged.back().z1, w.z1);
        }
    }

    std::size_t window_positions = 0;
    for (Window& w : merged) {
        w.base = window_positions;
        const std::size_t layers = static_cast<std::size_t>(w.z1 - w.z0 + 1);
        window_positions += layers * static_cast<std::size_t>(layer_size);
    }

    std::vector<NetworkCompact::index_t> remap(window_positions, invalid);
    NetworkCompact out;

    auto local_window_offset = [&](const NetworkCompact::index_t pos,
                                   std::size_t& local) -> bool {
        const int z = static_cast<int>(pos / layer_size);
        for (const Window& w : merged) {
            if (z < w.z0) return false;
            if (z > w.z1) continue;
            local = w.base
                + static_cast<std::size_t>(z - w.z0) * static_cast<std::size_t>(layer_size)
                + static_cast<std::size_t>(pos % layer_size);
            return local < remap.size();
        }
        return false;
    };

    for (const Window& w : merged) {
        for (int z = w.z0; z <= w.z1; ++z) {
            const std::size_t layer_begin =
                static_cast<std::size_t>(z) * static_cast<std::size_t>(layer_size);
            for (NetworkCompact::index_t d = 0; d < layer_size; ++d) {
                const std::size_t pos = layer_begin + static_cast<std::size_t>(d);
                if (pos >= np.data.size()) break;
                const long long code = static_cast<long long>(np.data[pos]);
                if (code <= 0) continue;
                if (out.N == invalid) {
                    throw std::runtime_error("build_animation_window_compact_from_encoded: muitos sitios ativos");
                }
                std::size_t local = 0;
                if (!local_window_offset(static_cast<NetworkCompact::index_t>(pos), local)) continue;
                remap[local] = out.N++;
                out.pos_flat.push_back(static_cast<NetworkCompact::index_t>(pos));
                const int color_1b = static_cast<int>(code / SPECIES_FACTOR);
                const int time = static_cast<int>(code % SPECIES_FACTOR);
                const int color_idx = np.num_colors == 1
                    ? 0
                    : std::max(0, std::min(np.num_colors - 1, color_1b - 1));
                out.species.push_back(static_cast<uint8_t>(color_idx + 1));
                out.activation_time.push_back(static_cast<uint32_t>(std::max(0, time)));
            }
        }
    }

    std::vector<std::pair<NetworkCompact::index_t, NetworkCompact::index_t>> pairs;
    for (const auto& edge : np.edge_pairs) {
        std::size_t lu = 0;
        std::size_t lv = 0;
        if (!local_window_offset(edge.first, lu) ||
            !local_window_offset(edge.second, lv)) {
            continue;
        }
        const NetworkCompact::index_t mu = remap[lu];
        const NetworkCompact::index_t mv = remap[lv];
        if (mu == invalid || mv == invalid) continue;
        pairs.emplace_back(mu, mv);
    }

    if (!pairs.empty()) {
        out.build_csr_from_edge_pairs(pairs);
    } else {
        out.edge_offsets.assign(out.N + 1, 0);
        out.edges.clear();
    }

    return out;
}

} // namespace

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
    if (argc != 1 && (argc < 12 || argc > 20)) {
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
        std::string equilibration = "false";
        bool calculate_detailed_properties = false;
        std::string run_mode = "growth_test";
        std::string initial_layout = "random";
        bool save_surface_observables = false;
        bool save_animation_window_only = false;
        std::string control_rule_name = "linear";
        double control_param = 0.0;
        double log_epsilon = 1.0e-12;
        
        if (argc >= 12) {
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
            if (argc >= 15) {
                run_mode = argv[13];
                initial_layout = argv[14];
            }
            if (argc == 16) {
                save_surface_observables = helpers::parse_bool(argv[15]);
            }
            if (argc == 17) {
                save_surface_observables = helpers::parse_bool(argv[15]);
                save_animation_window_only = helpers::parse_bool(argv[16]);
            }
            if (argc >= 18) {
                save_surface_observables = helpers::parse_bool(argv[15]);
                save_animation_window_only = helpers::parse_bool(argv[16]);
                control_rule_name = argv[17];
            }
            if (argc >= 19) {
                control_param = std::stod(argv[18]);
            }
            if (argc >= 20) {
                log_epsilon = std::stod(argv[19]);
            }
        }

        const bool teste = (run_mode == "growth_test");
        if (run_mode != "sop" && run_mode != "growth_test") {
            std::cerr << "[ERROR] run mode must be 'sop' or 'growth_test'.\n";
            helpers::print_help(argv[0]);
            return 1;
        }

        auto parse_initial_layout = [](const std::string& value) {
            if (value == "random") return InitialBaseLayout::Random;
            if (value == "blocks" || value == "quadrants" || value == "quadrantes") {
                return InitialBaseLayout::Blocks;
            }
            if (value == "alternating" || value == "alternado") {
                return InitialBaseLayout::Alternating;
            }
            throw std::invalid_argument(
                "initial layout must be 'random', 'blocks', or 'alternating'");
        };
        const InitialBaseLayout initial_base_layout =
            parse_initial_layout(initial_layout);

        auto canonical_control_rule_name = [](const std::string& value) {
            if (value == "linear") return std::string("linear");
            if (value == "log_saturated" || value == "log_sat" || value == "saturated_log") {
                return std::string("log_saturated");
            }
            if (value == "log_asymmetric" || value == "log_asym" || value == "asymmetric_log") {
                return std::string("log_asymmetric");
            }
            throw std::invalid_argument(
                "control rule must be 'linear', 'log_saturated', or 'log_asymmetric'");
        };
        control_rule_name = canonical_control_rule_name(control_rule_name);

        auto parse_feedback_control_rule = [](const std::string& value) {
            if (value == "linear") return FeedbackControlRule::Linear;
            if (value == "log_saturated") return FeedbackControlRule::LogSaturated;
            if (value == "log_asymmetric") return FeedbackControlRule::LogAsymmetric;
            throw std::invalid_argument(
                "control rule must be 'linear', 'log_saturated', or 'log_asymmetric'");
        };
        const FeedbackControlRule feedback_control_rule =
            parse_feedback_control_rule(control_rule_name);

        const bool return_encoded_network = helpers::parse_bool(equilibration);

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

        if (control_param < 0.0) {
            std::cerr << "[ERROR] control_param must be >= 0.\n";
            helpers::print_help(argv[0]);
            return 1;
        }
        if (log_epsilon < 0.0) {
            std::cerr << "[ERROR] log_epsilon must be >= 0.\n";
            helpers::print_help(argv[0]);
            return 1;
        }
        if (feedback_control_rule == FeedbackControlRule::LogSaturated &&
            control_param <= 0.0) {
            std::cerr << "[ERROR] control_param must be > 0 for log_saturated (positive p-step cap).\n";
            helpers::print_help(argv[0]);
            return 1;
        }
        if (feedback_control_rule == FeedbackControlRule::LogAsymmetric &&
            control_param <= 1.0) {
            std::cerr << "[ERROR] control_param must be > 1 for log_asymmetric (c_up = control_param * c_down).\n";
            helpers::print_help(argv[0]);
            return 1;
        }

        all_random rng(seed);

        TimeSeries ts;
        PercolationSeries ps;

        std::vector<double> rho(num_colors, rho_val);
        std::vector<double> p0(num_colors, pp0);

        int N_samples = teste ? std::max(100000, 20 * L) : 100000;
        GrowthStopConfig stop_config;
        stop_config.initial_base_layout = initial_base_layout;
        stop_config.feedback_control_rule = feedback_control_rule;
        stop_config.control_param = control_param;
        stop_config.log_epsilon = log_epsilon;
        if (teste) {
            stop_config.height_multiplier = HEIGHT_STOP_MULTIPLIER;
            stop_config.dynamic_height = true;
            stop_config.stop_at_percolation = false;
            stop_config.stop_at_equilibrium = true;
            stop_config.save_lateral_observables = false;
            stop_config.save_surface_observables = save_surface_observables;
            stop_config.equilibrium_consecutive_steps = 10;
            stop_config.dynamics_window_steps = -1;
        }
        int type_f_T = 0;
        double a = 0.0, alpha = 0.0;
        //double alpha = 0.0;
        network net_generator(N_samples, num_colors);

        const bool build_full_network = return_encoded_network;

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
            correlations_dir,
            network_preteq
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
                stop_config.dynamics_window_steps,
                control_rule_name,
                control_param,
                log_epsilon
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
        ps.feedback_control_rule = control_rule_name;
        ps.feedback_control_param = control_param;
        ps.feedback_log_epsilon = log_epsilon;

        const std::string machine_name = helpers::get_machine_name();
        const std::string timestamp_now = helpers::get_timestamp_now();

        std::ostringstream base_name;
        base_name << machine_name
                  << "_seed_" << seed
                  << "_ts_" << timestamp_now
                  << "_P0_" << std::fixed << std::setprecision(2) << P0
                  << "_p0_" << std::fixed << std::setprecision(2) << pp0;
        if (initial_layout != "random") {
            base_name << "_base_" << helpers::sanitize_for_filename(initial_layout);
        }
        save_data saver;
        
        const std::string sample_base = base_name.str();
        if (!ts.lateral_observables.t.empty() ||
            !ts.lateral_observables.correlation_summary_rows.empty() ||
            !ts.lateral_observables.susceptibility_rows.empty()) {
            ts.lateral_observables.sample_id = sample_base;
            ts.lateral_observables.dim = dim;
            ts.lateral_observables.L = L;
            ts.lateral_observables.r_max = std::max(0, L / 2);
            ts.lateral_observables.boundary_mode = "periodic";
            ts.lateral_observables.f_T = f_T;
            ts.lateral_observables.p0 = pp0;
            ts.lateral_observables.P0 = P0;
            ts.lateral_observables.c = c;
            ts.lateral_observables.type_percolation = type_percolation;
            ts.lateral_observables.seed = seed;
            ts.lateral_observables.t_stat = std::isfinite(ts.t_eq) ? ts.t_eq : -1.0;
            saver.save_lateral_observables_csv(correlations_dir, sample_base, ts.lateral_observables);
        }

        std::string json_filename = data_dir + "/" + sample_base + ".json";
        
        const bool has_percolation = !ps.color_percolation.empty();
        const bool dynamic_growth_artifacts =
            teste && stop_config.dynamic_height;
        const bool write_encoded_network_artifact =
            build_full_network && !net.data.empty();
        const bool write_large_artifacts =
            calculate_detailed_properties &&
            build_full_network &&
            has_percolation &&
            std::isfinite(ps.t_eq);
        const bool write_classic_large_artifacts =
            write_large_artifacts && !dynamic_growth_artifacts;

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
            if (dynamic_growth_artifacts) {
                std::cout << "[INFO] No stabilized species found; skipping surface file."
                          << std::endl;
            } else {
                std::cout << "[INFO] No percolating species found; skipping surface file."
                          << std::endl;
            }
        } else if (return_encoded_network && !calculate_detailed_properties) {
            std::cout << "[INFO] Encoded-network mode: skipping detailed surface artifacts."
                      << std::endl;
        }

        const std::string net_compact_filename =
            network_dir + "/" + sample_base + ".bin";

        if (write_encoded_network_artifact) {
            try {
                if (dynamic_growth_artifacts && save_animation_window_only) {
                    NetworkCompact windowc =
                        build_animation_window_compact_from_encoded(net, ps, dim, L);
                    const std::string overlay_filename =
                        network_dir + "/" + sample_base + "_animation_overlay.json";
                    saver.save_animation_overlay_json(
                        windowc,
                        ps,
                        dim,
                        L,
                        overlay_filename);
                    saver.save_network_compact_bin(windowc, net_compact_filename);
                    std::cout << "[INFO] Saved animation window compact network only: "
                              << windowc.N << " sites, "
                              << windowc.num_edges() << " edges."
                              << std::endl;
                } else {
                    NetworkCompact fullc = convert_encoded_to_compact(net);
                    if (dynamic_growth_artifacts) {
                        const std::string overlay_filename =
                            network_dir + "/" + sample_base + "_animation_overlay.json";
                        saver.save_animation_overlay_json(
                            fullc,
                            ps,
                            dim,
                            L,
                            overlay_filename);
                    }
                    saver.save_network_compact_bin(fullc, net_compact_filename);
                }
            } catch (const std::exception &e) {
                std::cerr << "Warning: failed to save encoded compact network: "
                          << e.what() << '\n';
            }
        }

        if (write_large_artifacts) {
            if (write_classic_large_artifacts) {
                // percolating clusters (compact)
                try {
                    NetworkPattern net_perc_clusters = net_generator.filter_percolating_clusters_from_encoded(net);
                    NetworkCompact percc = convert_encoded_to_compact(net_perc_clusters);
                    const std::string net_PERCOLATION_filename = network_dir + "/" + sample_base + "_PERCOLATION" + ".bin";
                    saver.save_network_compact_bin(percc, net_PERCOLATION_filename);
                } catch (const std::exception &e) {
                    std::cerr << "Warning: failed to save percolation compact network: " << e.what() << '\n';
                }
            }

            // pre/post teq networks: prefer to preserve CSR edges from the full compact
            if (write_classic_large_artifacts) try {
                // Attempt to read the full compact file we saved above to reuse its CSR
                NetworkCompact base_full;
                bool have_csr = false;
                if (base_full.read_binary(net_compact_filename)) {
                    have_csr = true;
                }

                // Convert cuts to compact form (species + activation_time)
                NetworkCompact pre_c = convert_encoded_to_compact(cuts->pre_teq);
                NetworkCompact post_c = convert_encoded_to_compact(cuts->post_teq);

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

                if (!network_preteq.empty()) {
                    const std::string net_preteq_filename = network_preteq + "/" + sample_base + ".bin";
                    const std::string net_posteq_filename = network_preteq + "/" + sample_base + "_posteq.bin";
                    saver.save_network_compact_bin(pre_c, net_preteq_filename);
                    saver.save_network_compact_bin(post_c, net_posteq_filename);

                    // Additionally save filtered (reindexed) active-only compact networks.
                    // Build and write them one at a time to avoid doubling the peak RAM.
                    const std::string net_preteq_active = network_preteq + "/" + sample_base + "_active.bin";
                    const std::string net_posteq_active = network_preteq + "/" + sample_base + "_posteq_active.bin";
                    {
                        NetworkCompact pre_filtered = pre_c.filter_active();
                        saver.save_network_compact_bin(pre_filtered, net_preteq_active);
                    }
                    {
                        NetworkCompact post_filtered = post_c.filter_active();
                        saver.save_network_compact_bin(post_filtered, net_posteq_active);
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "Warning: failed to save pre/post teq compact networks: " << e.what() << '\n';
            }
        }

        const std::string height_ts_filename = data_dir + "/" + sample_base + ".yts";
        saver.save_height_timeseries_bin(ts, height_ts_filename);
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
