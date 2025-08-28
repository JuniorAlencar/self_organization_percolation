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

// Code to create network in 2D or 3D with num_colors >= 1
NetworkPattern network::create_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double k, const double N_t, const int seed, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out)
{
    this->N_t = N_t;

    // ===== Shape só espacial =====
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{lenght_network, lenght_network}
        : std::vector<int>{lenght_network, lenght_network, lenght_network};

    const int L = lenght_network;
    const int grow_axis = dim - 1; // direção de crescimento (y em 2D, z em 3D)

    auto wrap = [&](int coord, int Lax) {
        if (coord < 0) return Lax - 1;
        if (coord >= Lax) return 0;
        return coord;
    };

    all_random rng(seed);
    NetworkPattern net(dim, shape, num_colors, rho, rng);

    // Séries p_i(t) e N_i(t)
    std::vector<std::vector<double>> p_t(num_colors);
    std::vector<std::vector<int>>    Nt_t(num_colors);
    for (int c = 0; c < num_colors; ++c) {
        p_t[c].push_back(p0[c]);
        Nt_t[c].push_back(0);
    }
    std::vector<int> t_list = {0};

    // ===== Borda de partida (grow_axis = 0) =====
    int base_size = 1;
    for (int ax = 0; ax < dim - 1; ++ax) base_size *= shape[ax];

    const int seeds_per_color = static_cast<int>(std::round(P0 * base_size));

    std::queue<std::vector<int>> borderland; // fronteira
    std::vector<int> N_current(num_colors, 0);

    // Seeds por cor
    for (int c = 0; c < num_colors; ++c) {
        int activated = 0, tries = 0;
        const int max_tries = base_size * 20;

        while (activated < seeds_per_color && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            for (int ax = 0; ax < dim - 1; ++ax)
                coords[ax] = rng.uniform_int(0, shape[ax] - 1);
            coords[grow_axis] = 0;

            const int target_val = (num_colors == 1) ? -1 : -(c + 2);
            const int new_val    = (num_colors == 1) ?  1 :  (c + 2);

            if (net.get(coords) == target_val) {
                net.set(coords, new_val);
                borderland.push(coords);
                Nt_t[c][0]++; N_current[c]++; activated++;
            }
            tries++;
        }
    }

    // ===== Tracking de percolação =====
    std::vector<bool> percolated(num_colors, false);
    std::vector<int>  t_percolated(num_colors, -1);
    int order_counter = 0; // 1º, 2º, ...

    // Buffer de vizinhos
    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    // ===== Evolução temporal =====
    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()) {
            const std::vector<int> pos = borderland.front();
            borderland.pop();

            const int active_val = net.get(pos);
            if (active_val <= 0) continue; // expande só de ativo

            const int cor_idx = (num_colors == 1) ? 0 : (std::abs(active_val) - 2);

            int v_idx = 0;
            for (int ax = 0; ax < dim; ++ax) {
                for (int delta : {-1, 1}) {
                    std::vector<int>& viz = neighbor_buffer[v_idx++];
                    viz = pos;
                    viz[ax] += delta;

                    // Contorno: eixo de crescimento ABERTO, laterais PERIÓDICAS
                    bool valid = true;
                    for (int j = 0; j < dim; ++j) {
                        if (j == grow_axis) {
                            if (viz[j] < 0 || viz[j] >= shape[j]) { valid = false; break; }
                        } else {
                            viz[j] = wrap(viz[j], shape[j]);
                        }
                    }
                    if (!valid) continue;

                    const int val_viz = net.get(viz);
                    if (val_viz >= 0) continue; // >0 ativo; ==0 checado/falhou

                    const bool same_color = (num_colors == 1) || (val_viz == -(cor_idx + 2));
                    const bool no_color   = (val_viz == -1);
                    if (!same_color && !no_color) continue;

                    const double r = rng.uniform_real(0.0, 1.0);

                    if (type_percolation == "node") {
                        if (r < p_t[cor_idx].back()) {
                            const int new_val = (num_colors == 1) ? 1 : (cor_idx + 2);
                            net.set(viz, new_val);
                            new_borderland.push(viz);
                            N_current[cor_idx]++;

                            // Checa percolação dessa cor
                            if (viz[grow_axis] == L - 1 && !percolated[cor_idx]) {
                                percolated[cor_idx]   = true;
                                t_percolated[cor_idx] = t;
                                order_counter++;

                                std::cout << "[CREATE] Cor " << (cor_idx + 1)
                                          << " percolou em t=" << t
                                          << "  (ordem=" << order_counter << ")\n";

                                // grava na PercolationSeries (por referência)
                                ps_out.color_percolation.push_back(cor_idx + 1); // 1-based
                                ps_out.time_percolation.push_back(t);
                                ps_out.percolation_order.push_back(order_counter);
                            }
                        } else {
                            net.set(viz, 0); // checado e não ativado
                        }
                    } else if (type_percolation == "bond") {
                        // Placeholder igual ao node (implemente bonds únicos + flood-fill se desejar)
                        if (r < p_t[cor_idx].back()) {
                            const int new_val = (num_colors == 1) ? 1 : (cor_idx + 2);
                            net.set(viz, new_val);
                            new_borderland.push(viz);
                            N_current[cor_idx]++;
                            if (viz[grow_axis] == L - 1 && !percolated[cor_idx]) {
                                percolated[cor_idx]   = true;
                                t_percolated[cor_idx] = t;
                                order_counter++;

                                std::cout << "[CREATE] Cor " << (cor_idx + 1)
                                          << " percolou em t=" << t
                                          << "  (ordem=" << order_counter << ")\n";

                                ps_out.color_percolation.push_back(cor_idx + 1);
                                ps_out.time_percolation.push_back(t);
                                ps_out.percolation_order.push_back(order_counter);
                            }
                        }
                    }
                }
            }
        }

        // Sem crescimento → encerra
        const int grown_total = std::accumulate(N_current.begin(), N_current.end(), 0);
        if (grown_total == 0) break;

        borderland = std::move(new_borderland);

        // Atualiza p_i(t) e N_i(t)
        for (int c = 0; c < num_colors; ++c) {
            const double p_next = generate_p(type_N_t, p_t[c].back(), t, N_current[c], k, a, alpha);
            p_t[c].push_back(p_next);
            Nt_t[c].push_back(N_current[c]);
        }
        t_list.push_back(t);

        // critério: todas as cores percolaram (alcançaram grow_axis == L-1)
        if (std::all_of(percolated.begin(), percolated.end(), [](bool x){ return x; })) {
            std::cout << "[CREATE] Todas as cores percolaram em t=" << t << " (";
            for (int c = 0; c < num_colors; ++c) {
                std::cout << "c" << (c+1) << "=" << t_percolated[c];
                if (c + 1 < num_colors) std::cout << ", ";
            }
            std::cout << ")\n";
            break;
        }

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << ", p" << c + 1 << "(t)=" << p_t[c].back()
                          << ", N_t" << c + 1 << "(t)=" << Nt_t[c].back();
            std::cout << std::endl;
        }
    }

    // Saída
    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_t);
    ts_out.Nt  = std::move(Nt_t);
    ts_out.t   = std::move(t_list);

    return net;
}

NetworkPattern network::animate_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double k, const double N_t, const int seed, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out)
{
    this->N_t = N_t;

    // ===== SHAPE SOMENTE ESPACIAL =====
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{lenght_network, lenght_network}
        : std::vector<int>{lenght_network, lenght_network, lenght_network};

    const int L = lenght_network;
    const int grow_axis = dim - 1;

    auto wrap = [&](int coord, int Lax) {
        if (coord < 0) return Lax - 1;
        if (coord >= Lax) return 0;
        return coord;
    };

    all_random rng(seed);

    NetworkPattern net(dim, shape, num_colors, rho, rng);          // estados
    NetworkPattern net_animation(dim, shape, num_colors, rho, rng);// tempos de ativação
    std::fill(net_animation.data.begin(), net_animation.data.end(), -1);

    // Séries temporais
    std::vector<std::vector<double>> p_t(num_colors);
    std::vector<std::vector<int>>    Nt_t(num_colors);
    for (int c = 0; c < num_colors; ++c) {
        p_t[c].push_back(p0[c]);
        Nt_t[c].push_back(0);
    }
    std::vector<int> t_list = {0};

    // ===== SEEDS NA BORDA grow_axis == 0 =====
    int base_size = 1;
    for (int ax = 0; ax < dim - 1; ++ax) base_size *= shape[ax];
    const int seeds_per_color = static_cast<int>(std::round(P0 * base_size));

    std::queue<std::vector<int>> borderland;
    std::vector<int> N_current(num_colors, 0);

    for (int c = 0; c < num_colors; ++c) {
        int activated = 0, tries = 0;
        const int max_tries = base_size * 20;

        while (activated < seeds_per_color && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            for (int ax = 0; ax < dim - 1; ++ax)
                coords[ax] = rng.uniform_int(0, shape[ax] - 1);
            coords[grow_axis] = 0;

            const int target_val = (num_colors == 1) ? -1 : -(c + 2);
            const int new_val    = (num_colors == 1) ?  1 :  (c + 2);

            if (net.get(coords) == target_val) {
                net.set(coords, new_val);
                borderland.push(coords);
                net_animation.set(coords, 0);
                Nt_t[c][0]++; N_current[c]++; activated++;
            }
            tries++;
        }
    }

    // ===== Novos: tracking de percolação por cor =====
    std::vector<bool> percolated(num_colors, false);
    std::vector<int>  t_percolated(num_colors, -1);

    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    bool reached_top = false;   // parar na primeira cor que toca o topo

    // ===== EVOLUÇÃO =====
    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty() && !reached_top) {
            const std::vector<int> pos = borderland.front();
            borderland.pop();

            const int active_val = net.get(pos);
            if (active_val <= 0) continue;

            const int cor_idx = (num_colors == 1) ? 0 : (std::abs(active_val) - 2);

            int v_idx = 0;
            for (int ax = 0; ax < dim && !reached_top; ++ax) {
                for (int delta : {-1, 1}) {
                    std::vector<int>& viz = neighbor_buffer[v_idx++];
                    viz = pos;
                    viz[ax] += delta;

                    // Contorno: eixo de crescimento aberto, laterais periódicas
                    bool valid = true;
                    for (int j = 0; j < dim; ++j) {
                        if (j == grow_axis) {
                            if (viz[j] < 0 || viz[j] >= shape[j]) { valid = false; break; }
                        } else {
                            viz[j] = wrap(viz[j], shape[j]);
                        }
                    }
                    if (!valid) continue;

                    const int val_viz = net.get(viz);
                    if (val_viz >= 0) continue;

                    const bool same_color = (num_colors == 1) || (val_viz == -(cor_idx + 2));
                    const bool no_color   = (val_viz == -1);
                    if (!same_color && !no_color) continue;

                    const double r = rng.uniform_real(0.0, 1.0);

                    if (r < p_t[cor_idx].back()) {
                        const int new_val = (num_colors == 1) ? 1 : (cor_idx + 2);
                        net.set(viz, new_val);
                        new_borderland.push(viz);
                        net_animation.set(viz, t);
                        N_current[cor_idx]++;

                        // ===== Quando a cor alcança o topo =====
                        if (viz[grow_axis] == L - 1 && !percolated[cor_idx]) {
                            percolated[cor_idx]   = true;
                            t_percolated[cor_idx] = t;
                            std::cout << "[ANIMATION] Cor " << (cor_idx + 1)
                                      << " percolou em t=" << t
                                      << "  (p=" << p_t[cor_idx].back()
                                      << ", N_t=" << N_current[cor_idx] << ")"
                                      << std::endl;

                            reached_top = true;  // << comentar esta linha se quiser parar só quando TODAS percolarem
                        }
                    } else {
                        net.set(viz, 0);
                    }
                    if (reached_top) break;
                }
            }
        }

        const int grown_total = std::accumulate(N_current.begin(), N_current.end(), 0);
        if (grown_total == 0) break;

        // ===== Se optar por "todas as cores", imprime e para aqui =====
        if (std::all_of(percolated.begin(), percolated.end(), [](bool x){ return x; })) {
            std::cout << "[ANIMATION] Todas as cores percolaram em t=" << t << "  (";
            for (int c = 0; c < num_colors; ++c) {
                std::cout << "c" << (c+1) << "=" << t_percolated[c];
                if (c+1 < num_colors) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            break;
        }

        if (reached_top) break;  // parar ao detectar a primeira cor (comportamento atual)

        borderland = std::move(new_borderland);

        // Atualiza p_i(t) e N_i(t)
        for (int c = 0; c < num_colors; ++c) {
            const double p_next = generate_p(type_N_t, p_t[c].back(), t, N_current[c], k, a, alpha);
            p_t[c].push_back(p_next);
            Nt_t[c].push_back(N_current[c]);
        }
        t_list.push_back(t);

        if (t < 10 || t % 100 == 0) {
            std::cout << "[ANIMATION] t=" << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << "  p" << (c+1) << "(" << t << ")=" << p_t[c].back()
                          << "  N" << (c+1) << "(" << t << ")=" << Nt_t[c].back();
            std::cout << std::endl;
        }
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_t);
    ts_out.Nt  = std::move(Nt_t);
    ts_out.t   = std::move(t_list);

    return net_animation;
}





NetworkPattern network::initialize_network(int dim, int length_network, int num_colors, double P0, const std::vector<double>& rho, int seed) {
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{length_network, length_network}
        : std::vector<int>{length_network, length_network, length_network};
    
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






