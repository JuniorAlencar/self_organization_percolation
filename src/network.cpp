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

    const size_t total_sites = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    NetworkPattern net(dim, shape, num_colors, rho);
    all_random rng(seed);

    std::vector<std::vector<double>> p_t(num_colors, std::vector<double>());
    std::vector<std::vector<int>> Nt_t(num_colors, std::vector<int>());
    for (int c = 0; c < num_colors; ++c) {
        p_t[c].push_back(p0[c]);
        Nt_t[c].push_back(0);
    }
    std::vector<int> t_list = {0};

    int active_count = (dim == 3)
        ? static_cast<int>(std::pow(lenght_network * P0, 2))
        : static_cast<int>(P0 * lenght_network);

    std::queue<std::vector<int>> fronteira;
    std::vector<int> base_indices(lenght_network);
    std::iota(base_indices.begin(), base_indices.end(), 0);
    std::shuffle(base_indices.begin(), base_indices.end(), rng.get_gen());

    int seeds_per_color = active_count / num_colors;
    int rest = active_count % num_colors;
    int current = 0;

    for (int c = 0; c < num_colors; ++c) {
        int n_c = seeds_per_color + (c < rest ? 1 : 0);
        for (int i = 0; i < n_c && current < (int)base_indices.size(); ++i, ++current) {
            std::vector<int> coords(dim, 0);
            if (dim == 2) coords[1] = base_indices[current];
            else {
                coords.resize(3);
                coords[1] = base_indices[current % lenght_network];
                coords[2] = base_indices[(current / lenght_network) % lenght_network];
            }
            size_t idx = net.to_index(coords);

            int target_val = (num_colors == 1) ? -1 : -(c + 2);
            int new_val = (num_colors == 1) ? 1 : (c + 2);

            if (net.data[idx] == target_val) {
                net.data[idx] = new_val;
                fronteira.push(coords);
                Nt_t[c][0]++;
            }
        }
    }

    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    for (int t = 1; t < num_of_samples; ++t) {
        std::vector<int> N_current(num_colors, 0);
        std::queue<std::vector<int>> nova_fronteira;

        while (!fronteira.empty()) {
            const std::vector<int>& pos = fronteira.front();
            int active_val = net.get(pos);

            if (active_val <= 0) {
                fronteira.pop();
                continue;
            }

            int cor_idx;
            if (num_colors == 1)
                cor_idx = 0;
            else
                cor_idx = std::abs(active_val) - 2;

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
                    double r2 = rng.uniform_real(0.0, 1.0);
                    bool activate = false;

                    if (type_percolation == "node") {
                        activate = (r1 < p_t[cor_idx].back());
                    } else if (type_percolation == "bond") {
                        activate = (r1 < p_t[cor_idx].back()) && (r2 < p_t[cor_idx].back());
                    }

                    if (activate) {
                        net.set(viz, cor_idx + 2);
                        nova_fronteira.push(viz);
                        N_current[cor_idx]++;
                    } else {
                        net.set(viz, 0);
                    }
                }
            }

            fronteira.pop();
        }

        if (std::accumulate(N_current.begin(), N_current.end(), 0) == 0) break;

        fronteira = std::move(nova_fronteira);

        for (int c = 0; c < num_colors; ++c) {
            double Nt_i = (type_N_t == 0) ? N_t : a * std::pow(t, alpha);
            double p_next = std::clamp(p_t[c].back() + k * (Nt_i - N_current[c]), 0.0, 1.0);
            p_t[c].push_back(p_next);
            Nt_t[c].push_back(N_current[c]);
        }
        t_list.push_back(t);

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << ", p" << c + 1 << "(t)=" << p_t[c].back();
            std::cout << std::endl;
        }
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = p_t;
    ts_out.Nt = Nt_t;
    ts_out.t = t_list;

    return net;
}


void print_initial_network_info(const NetworkPattern& net) {
    int total_sites = net.size();
    std::map<int, int> counts;
    for (int val : net.data) counts[val]++;
    std::cout << "[Initial State] Total sites: " << total_sites << "\\n";
    for (const auto& [val, count] : counts) {
        std::cout << "  Value " << val << ": " << count << " sites\\n";
    }
}





