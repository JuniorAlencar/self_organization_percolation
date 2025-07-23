#include "network.hpp"
#include "rand_utils.hpp"

double network::type_Nt_create(const int type_N_t, const int t_i, const double a, const double alpha){
    if(type_N_t == 0) return N_t;
    else if (type_N_t == 1) return a*pow(t_i, alpha);
    throw std::invalid_argument("Invalid type_N_t value: " + std::to_string(type_N_t));
}

double network::generate_p(const int type_N_t, const double p_t, const int t_i, const int N_current, const double k, const double a, const double alpha) {
    double N_T = type_Nt_create(type_N_t, t_i, a, alpha);  // get N_t(t)
    double p_next = p_t + k * (N_T - N_current);

    // Clamp p_next to [0, 1]
    if (p_next > 1.0) p_next = 1.0;
    if (p_next < 0.0) p_next = 0.0;

    return p_next;
}

NetworkPattern network::create_network(const int dim, const int lenght_network, const int num_of_samples,
                                       const double k, const double N_t, const int seed, const int type_N_t,
                                       const std::vector<double> p0, const double P0, const double a, const double alpha,
                                       const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
                                       TimeSeries& ts_out) {
    this->N_t = N_t;

    std::vector<int> shape = (dim == 2) ? std::vector<int>{num_of_samples, lenght_network}
                                        : std::vector<int>{num_of_samples, lenght_network, lenght_network};

    all_random rng(seed);
    NetworkPattern net(dim, shape, num_colors, rho, rng);

    std::vector<std::vector<double>> p_t(num_colors);
    std::vector<std::vector<int>> Nt_t(num_colors);
    for (int c = 0; c < num_colors; ++c) {
        p_t[c].push_back(p0[c]);
        Nt_t[c].push_back(0);
    }
    std::vector<int> t_list = {0};
    
    // Base to active initial nodes
    int base_size = (dim == 3) ? lenght_network * lenght_network : lenght_network;
    int seeds_per_color = (dim == 3) ? static_cast<int>(pow(P0,2) * base_size) : static_cast<int>(P0 * base_size);

    std::queue<std::vector<int>> borderland;
    // Counter N(t) for each color
    std::vector<int> N_current(num_colors, 0);
    
    // Initial active nodes (base)
    for (int c = 0; c < num_colors; ++c) {
        int activated = 0;
        int max_tries = base_size * 10;
        int tries = 0;

        while (activated < seeds_per_color && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            if (dim == 2) {
                coords[1] = rng.uniform_int(0, lenght_network - 1);
            } else {
                coords[1] = rng.uniform_int(0, lenght_network - 1);
                coords[2] = rng.uniform_int(0, lenght_network - 1);
            }

            size_t idx = net.to_index(coords);
            int target_val = (num_colors == 1) ? -1 : -(c + 2);
            int new_val    = (num_colors == 1) ?  1 :  (c + 2);

            if (net.data[idx] == target_val) {
                net.data[idx] = new_val;
                borderland.push(coords);
                Nt_t[c][0]++;
                N_current[c]++;
                activated++;
            }

            tries++;
        }
    }

    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    for (int t = 1; t < num_of_samples; ++t) {
        std::vector<int> N_current(num_colors, 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()) {
            const std::vector<int>& pos = borderland.front();
            int active_val = net.get(pos);

            if (active_val <= 0) {
                borderland.pop();
                continue;
            }

            int cor_idx = (num_colors == 1) ? 0 : std::abs(active_val) - 2;

            int v_idx = 0;
            for (int d = 0; d < dim; ++d) {
                for (int delta : {-1, 1}) {
                    std::vector<int>& viz = neighbor_buffer[v_idx++];
                    viz = pos;
                    viz[d] += delta;

                    bool valid = true;
                    for (int i = 0; i < dim; ++i) {
                        if (viz[i] < 0 || viz[i] >= shape[i]) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) continue;

                    int val_viz = net.get(viz);
                    if (val_viz > 0 || val_viz == 0) continue;

                    bool same_color = (num_colors == 1) || (val_viz == -(cor_idx + 2));
                    bool no_color = (val_viz == -1);
                    if (!same_color && !no_color) continue;

                    size_t idx = net.to_index(viz);
                    double r1 = rng.uniform_real(0.0, 1.0);
                    double r2 = rng.uniform_real(0.0, 1.0);
                    bool activate = false;

                    if (type_percolation == "node") {
                        if (r1 < p_t[cor_idx].back()) {
                            int new_val = (num_colors == 1) ? 1 : (cor_idx + 2);
                            net.set(viz, new_val);
                            new_borderland.push(viz);
                            N_current[cor_idx]++;
                        } else {
                            net.set(viz, 0); // Checado, mas não ativado
                        }
                    } else if (type_percolation == "bond") {
                        if (val_viz < 0 && (r1 < p_t[cor_idx].back())) {
                            int new_val = (num_colors == 1) ? 1 : (cor_idx + 2);
                            net.set(viz, new_val);
                            new_borderland.push(viz);
                            N_current[cor_idx]++;
                        }
                    }
                }
            }

            borderland.pop();
        }

        if (std::accumulate(N_current.begin(), N_current.end(), 0) == 0) break;

        borderland = std::move(new_borderland);

        for (int c = 0; c < num_colors; ++c) {
            double p_next = generate_p(type_N_t, p_t[c].back(), t, N_current[c], k, a, alpha);
            p_t[c].push_back(p_next);
            Nt_t[c].push_back(N_current[c]);
        }

        t_list.push_back(t);

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << ", p" << c + 1 << "(t)=" << p_t[c].back() << ",N_t" << c+1 << "(t)=" << Nt_t[c].back();
            std::cout << std::endl;
        }
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = p_t;
    ts_out.Nt = Nt_t;
    ts_out.t = t_list;

    return net;
}

NetworkPattern network::animate_network(const int dim, const int lenght_network, const int num_of_samples,
                                       const double k, const double N_t, const int seed, const int type_N_t,
                                       const std::vector<double> p0, const double P0, const double a, const double alpha,
                                       const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho) {
    this->N_t = N_t;

    std::vector<int> shape = (dim == 2) ? std::vector<int>{num_of_samples, lenght_network}
                                        : std::vector<int>{num_of_samples, lenght_network, lenght_network};

    all_random rng(seed);
    NetworkPattern net(dim, shape, num_colors, rho, rng);
    NetworkPattern net_animation(dim, shape, num_colors, rho, rng);

    std::vector<std::vector<double>> p_t(num_colors);
    for (int c = 0; c < num_colors; ++c) {
        p_t[c].push_back(p0[c]);
    }

    int base_size = (dim == 3) ? lenght_network * lenght_network : lenght_network;
    int seeds_per_color = (dim == 3) ? static_cast<int>(pow(P0, 2) * base_size) : static_cast<int>(P0 * base_size);

    std::queue<std::vector<int>> borderland;

    // Inicializa os sítios ativos na base
    for (int c = 0; c < num_colors; ++c) {
        int activated = 0;
        int max_tries = base_size * 10;
        int tries = 0;

        while (activated < seeds_per_color && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            if (dim == 2) {
                coords[1] = rng.uniform_int(0, lenght_network - 1);
            } else {
                coords[1] = rng.uniform_int(0, lenght_network - 1);
                coords[2] = rng.uniform_int(0, lenght_network - 1);
            }

            size_t idx = net.to_index(coords);
            int target_val = (num_colors == 1) ? -1 : -(c + 2);
            int new_val    = (num_colors == 1) ?  1 :  (c + 2);

            if (net.data[idx] == target_val) {
                net.data[idx] = new_val;
                borderland.push(coords);
                net_animation.set(coords, 0); // t = 0
                activated++;
            }
            tries++;
        }
    }

    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    for (int t = 1; t < num_of_samples; ++t) {
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()) {
            const std::vector<int>& pos = borderland.front();
            int active_val = net.get(pos);
            if (active_val <= 0) {
                borderland.pop();
                continue;
            }

            int cor_idx = (num_colors == 1) ? 0 : std::abs(active_val) - 2;

            int v_idx = 0;
            for (int d = 0; d < dim; ++d) {
                for (int delta : {-1, 1}) {
                    std::vector<int>& viz = neighbor_buffer[v_idx++];
                    viz = pos;
                    viz[d] += delta;

                    bool valid = true;
                    for (int i = 0; i < dim; ++i) {
                        if (viz[i] < 0 || viz[i] >= shape[i]) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) continue;

                    int val_viz = net.get(viz);
                    if (val_viz > 0 || val_viz == 0) continue;

                    bool same_color = (num_colors == 1) || (val_viz == -(cor_idx + 2));
                    bool no_color = (val_viz == -1);
                    if (!same_color && !no_color) continue;

                    double r1 = rng.uniform_real(0.0, 1.0);

                    if (type_percolation == "node") {
                        if (r1 < p_t[cor_idx].back()) {
                            int new_val = (num_colors == 1) ? 1 : (cor_idx + 2);
                            net.set(viz, new_val);
                            new_borderland.push(viz);
                            net_animation.set(viz, t);
                        } else {
                            net.set(viz, 0);
                        }
                    } else if (type_percolation == "bond") {
                        if (val_viz < 0 && (r1 < p_t[cor_idx].back())) {
                            int new_val = (num_colors == 1) ? 1 : (cor_idx + 2);
                            net.set(viz, new_val);
                            new_borderland.push(viz);
                            net_animation.set(viz, t);
                        }
                    }
                }
            }
            borderland.pop();
        }

        if (new_borderland.empty()) break;

        borderland = std::move(new_borderland);

        for (int c = 0; c < num_colors; ++c) {
            double p_next = generate_p(type_N_t, p_t[c].back(), t, 0, k, a, alpha); // N(t) não importa para animação
            p_t[c].push_back(p_next);
        }

        if (t < 10 || t % 100 == 0) {
            std::cout << "[ANIMATION] t = " << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << ", p" << c + 1 << "(t)=" << p_t[c].back();
            std::cout << std::endl;
        }
    }

    return net_animation;
}


NetworkPattern network::initialize_network(int dim, int length_network, int num_samples, int num_colors, double P0, const std::vector<double>& rho, int seed) {
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{num_samples, length_network}
        : std::vector<int>{num_samples, length_network, length_network};
    
    all_random rng(seed);
    NetworkPattern net(dim, shape, num_colors, rho, rng);
    

    int base_size = (dim == 3)
        ? length_network * length_network
        : length_network;

    int active_per_color = static_cast<int>(P0 * base_size);

    for (int c = 0; c < num_colors; ++c) {
        int activated = 0;
        int max_tries = base_size * 10;  // evita loop infinito
        int tries = 0;

        while (activated < active_per_color && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            if (dim == 2) {
                coords[1] = rng.uniform_int(0, length_network - 1);
            } else {
                coords[1] = rng.uniform_int(0, length_network - 1);
                coords[2] = rng.uniform_int(0, length_network - 1);
            }

            size_t idx = net.to_index(coords);
            int target_val = (num_colors == 1) ? -1 : -(c + 2);
            int new_val    = (num_colors == 1) ?  1 :  (c + 2);

            if (net.data[idx] == target_val) {
                net.data[idx] = new_val;
                activated++;
            }

            tries++;
        }
    }

    return net;
}

void network::print_initial_site_fractions(const NetworkPattern& net) {
    std::map<int, int> count;
    size_t total = net.data.size();

    for (int val : net.data) {
        count[val]++;
    }

    std::cout << "\n[ Fração inicial dos sítios ]\n";
    for (const auto& [val, n] : count) {
        double frac = static_cast<double>(n) / total;
        std::cout << "valor = " << val << " | fração = " << frac << std::endl;
    }
}






