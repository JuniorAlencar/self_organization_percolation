#include "network.hpp"


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

NetworkPattern network::create_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double k, const double N_t, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng)
{
    // ===== parâmetros e shape =====
    this->N_t = N_t;
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{lenght_network, lenght_network}
        : std::vector<int>{lenght_network, lenght_network, lenght_network};

    const int L = lenght_network;
    const int grow_axis = dim - 1;

    auto valid_coord = [&](std::vector<int>& v)->bool {
        for (int j=0;j<dim;++j) {
            if (j == grow_axis) {
                if (v[j] < 0 || v[j] >= shape[j]) return false; // aberto
            } else {
                if (v[j] < 0) v[j] = shape[j]-1; else if (v[j] >= shape[j]) v[j] = 0; // periódico
            }
        }
        return true;
    };

    auto get_gcoord = [&](const std::vector<int>& v)->int { return v[grow_axis]; };

    const int SX = shape[0];
    const int SY = (dim >= 2 ? shape[1] : 1);
    const int SZ = (dim == 3 ? shape[2] : 1);

    auto lin_index = [&](int x, int y, int z)->int {
        return x + SX*(y + SY*z);
    };

    // ===== rede =====
    NetworkPattern net(dim, shape, num_colors, rho);

    // ===== séries =====
    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<int>>    Nt_series(num_colors);
    std::vector<int>                 M_size_at_perc(num_colors, 0);
    std::vector<int>                 t_list; t_list.reserve(num_of_samples);
    std::vector<double>              p_curr = p0;

    const int GRID_N = SX*SY*SZ;
    std::vector<std::vector<int>> parent(num_colors, std::vector<int>(GRID_N, -2)); // -2: nunca visto; -1: raiz/seed

    auto commit_step = [&](int t_k, const std::vector<double>& p_vec, const std::vector<int>& Nt_vec)
    {
        t_list.push_back(t_k);
        for (int c=0;c<num_colors;++c){
            p_series[c].push_back(p_vec[c]);
            Nt_series[c].push_back(Nt_vec[c]);
        }
    };

    // ===== base measure =====
    long long base_size = 1;
    for (int ax = 0; ax < dim - 1; ++ax) base_size *= static_cast<long long>(shape[ax]);

    std::vector<int> seeds_quota(num_colors, 0);
    {
        const double Nt_target = static_cast<double>(this->N_t);
        (void)Nt_target; // silencioso (caso não use)
        for (int c = 0; c < num_colors; ++c) {
            long long q = std::llround(P0 * rho[c] * lenght_network);  // P0 * rho * L
            if (q < 0) q = 0;
            if (q > base_size) q = base_size;
            seeds_quota[c] = static_cast<int>(q);
        }
    }

    std::queue<std::vector<int>> borderland;
    std::vector<int> N_current(num_colors, 0);

    // ===== semeadura (t=0) =====
    for (int c = 0; c < num_colors; ++c) {
        int activated = 0, tries = 0;
        const int max_tries = static_cast<int>(base_size) * 20;

        const int prefer_neg = (num_colors == 1 ? -1 : -(c + 2));
        const int active_val = (num_colors == 1 ? 1 : (c + 2));

        while (activated < seeds_quota[c] && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            for (int ax = 0; ax < dim - 1; ++ax) {
                coords[ax] = rng.uniform_int(0, shape[ax] - 1);
            }
            coords[grow_axis] = 0;

            int idx = lin_index(coords[0], coords[1], coords[2]);
            int v = net.get(idx);
            if (v == prefer_neg || v == -1) {
                net.set(idx, active_val);
                borderland.push(coords);
                ++N_current[c];
                ++activated;

                parent[c][idx] = -1;  // Seed/root para shortest path
            }
            ++tries;
        }
    }

    // ===== percolação =====
    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, {});
    commit_step(0, p_curr, N_current);

    // buffer vizinhos
    std::vector<std::vector<int>> neighbor_buffer(2*dim, std::vector<int>(dim));

    std::vector<int> max_heights(num_colors, 0);
    auto update_max_height = [&](const std::vector<int>& pos, int cor_idx) {
        int current_height = pos[grow_axis];
        if (current_height > max_heights[cor_idx]) {
            max_heights[cor_idx] = current_height;
        }
    };

    // ordem real de percolação
    ps_out.color_percolation.clear();      // cores em ordem de chegada (1-based)
    ps_out.percolation_order.clear();      // 1, 2, 3, ...
    int order_ctr = 0;
    std::vector<bool> percolated(num_colors, false);

    // evolução
    for (int t = 1; t < num_of_samples; ++t) {
        bool any_growth = false;
        std::fill(N_current.begin(), N_current.end(), 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()) {
            std::vector<int> pos = borderland.front();
            borderland.pop();

            int idx = lin_index(pos[0], pos[1], pos[2]);
            int a_val = net.get(idx);
            if (a_val <= 0) continue;

            int cor_idx = (num_colors == 1 ? 0 : (std::abs(a_val) - 2));
            const int new_val = (num_colors == 1 ? 1 : (cor_idx + 2));

            int v_idx = 0;
            for (int ax = 0; ax < dim; ++ax) {
                for (int delta : {-1, 1}) {
                    std::vector<int>& viz = neighbor_buffer[v_idx++];
                    viz = pos;
                    viz[ax] += delta;
                    if (!valid_coord(viz)) continue;

                    int viz_idx = lin_index(viz[0], viz[1], viz[2]);
                    int vv = net.get(viz_idx);
                    if (vv >= 0) continue;

                    double r = rng.uniform_real(0.0, 1.0);

                    auto try_activate = [&]() {
                        net.set(viz_idx, new_val);
                        new_borderland.push(viz);
                        ++N_current[cor_idx];
                        update_max_height(viz, cor_idx);

                        // registra evento de percolação no instante da chegada ao topo
                        if (!percolated[cor_idx] && viz[grow_axis] == lenght_network - 1) {
                            percolated[cor_idx] = true;
                            ps_out.color_percolation.push_back(cor_idx + 1);   // 1-based
                            ps_out.percolation_order.push_back(++order_ctr);   // 1,2,3,...
                        }

                        parent[cor_idx][viz_idx] = idx;
                    };

                    if (r < p_curr[cor_idx]) {
                        try_activate();
                        any_growth = true;
                    } else {
                        // (mantido como solicitado)
                        net.set(vv, 0);
                    }
                }
            }
        }

        // critério de parada desta iteração
        bool all_top_reached = true;          // todas as cores chegaram ao topo?
        for (int c = 0; c < num_colors; ++c) {
            if (max_heights[c] != lenght_network - 1) {
                all_top_reached = false;
                break;
            }
        }
        bool all_stuck = !any_growth;         // ninguém cresceu nesta iteração?

        if (all_top_reached || all_stuck) {
            // finalização pós-simulação
            BiggestComponent bc;
            bc.compute_shortest_paths_to_base(net, dim, shape, grow_axis, num_colors, parent, ps_out);

            // tamanhos por COR; depois reordenamos para a ORDEM DE EVENTOS
            ps_out.M_size_at_perc = bc.largest_cluster_sizes(net, dim, shape, grow_axis, num_colors);

            std::vector<int> M_sizes_per_event;
            M_sizes_per_event.reserve(ps_out.color_percolation.size());
            for (int color_1b : ps_out.color_percolation) {
                int c = color_1b - 1;
                M_sizes_per_event.push_back(ps_out.M_size_at_perc[c]);
            }
            ps_out.M_size_at_perc = std::move(M_sizes_per_event);
            break;
        }

        borderland = std::move(new_borderland);

        std::vector<double> p_next(num_colors);
        for (int c = 0; c < num_colors; ++c)
            p_next[c] = generate_p(type_N_t, p_curr[c], t, N_current[c], k, a, alpha);

        commit_step(t, p_next, N_current);

        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << ", p" << c + 1 << "(t)=" << p_next[c]
                          << ", N_t" << c + 1 << "(t)=" << N_current[c]
                          << " max" << c + 1 << "(t)=" << max_heights[c];
            std::cout << std::endl;
        }

        p_curr.swap(p_next);
    }

    // ===== output =====
    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.Nt  = std::move(Nt_series);
    ts_out.t   = std::move(t_list);

    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < (int)rho.size()) ps_out.rho.push_back(rho[i]);
    }

    return net;
}

NetworkPattern network::animate_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double k, const double N_t, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng)
{
    this->N_t = N_t;

    // ===== SHAPE (igual ao create_network) =====
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{lenght_network, lenght_network}
        : std::vector<int>{lenght_network, lenght_network, lenght_network};

    const int L = lenght_network;
    const int grow_axis = dim - 1;

    auto valid_coord = [&](std::vector<int>& v)->bool {
        for (int j = 0; j < dim; ++j) {
            if (j == grow_axis) {
                // eixo de crescimento: aberto
                if (v[j] < 0 || v[j] >= shape[j]) return false;
            } else {
                // laterais: periódicas
                if (v[j] < 0) v[j] = shape[j] - 1;
                else if (v[j] >= shape[j]) v[j] = 0;
            }
        }
        return true;
    };

    const int SX = shape[0];
    const int SY = (dim >= 2 ? shape[1] : 1);
    const int SZ = (dim == 3 ? shape[2] : 1);

    auto lin_index = [&](int x, int y, int z)->int {
        return x + SX * (y + SY * z);
    };

    // ===== redes =====
    NetworkPattern net(dim, shape, num_colors, rho);          // estados reais
    NetworkPattern net_animation(dim, shape, num_colors, rho);// tempos codificados
    std::fill(net_animation.data.begin(), net_animation.data.end(), -1);

    int base_multi = (dim == 3) ? 100000000 : 1000000;    
    // ===== multiplicador por cor para codificar cor + tempo =====
    std::vector<int> color_mul(num_colors, 0);
    for (int c = 0; c < num_colors; ++c) {
        color_mul[c] = (c + 1) * base_multi;
    }

    // ===== séries temporais (como no create_network) =====
    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<int>>    Nt_series(num_colors);
    std::vector<int>                 t_list; t_list.reserve(num_of_samples);

    std::vector<double> p_curr = p0;
    std::vector<int>    N_current(num_colors, 0);

    auto commit_step = [&](int t_k, const std::vector<double>& p_vec, const std::vector<int>& Nt_vec)
    {
        t_list.push_back(t_k);
        for (int c = 0; c < num_colors; ++c) {
            p_series[c].push_back(p_vec[c]);
            Nt_series[c].push_back(Nt_vec[c]);
        }
    };

    // ===== base_size e seeds_quota (copiado do create_network) =====
    long long base_size = 1;
    for (int ax = 0; ax < dim - 1; ++ax)
        base_size *= static_cast<long long>(shape[ax]);

    std::vector<int> seeds_quota(num_colors, 0);
    {
        const double Nt_target = static_cast<double>(this->N_t);
        (void)Nt_target; // silencioso

        for (int c = 0; c < num_colors; ++c) {
            long long q = std::llround(P0 * rho[c] * lenght_network);  // P0 * rho * L
            if (q < 0) q = 0;
            if (q > base_size) q = base_size;
            seeds_quota[c] = static_cast<int>(q);
        }
    }

    const int GRID_N = SX * SY * SZ;
    std::vector<std::vector<int>> parent(num_colors, std::vector<int>(GRID_N, -2)); // -2: não visto, -1: raiz

    std::queue<std::vector<int>> borderland;

    // ===== inicializa PercolationSeries =====
    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, {});
    ps_out.color_percolation.clear();
    ps_out.percolation_order.clear();
    ps_out.M_size_at_perc.clear();

    int order_ctr = 0;
    std::vector<bool> percolated(num_colors, false);

    // ===== Semeadura t=0 (igual ao create_network, mas com tempo codificado) =====
    for (int c = 0; c < num_colors; ++c) {
        int activated = 0, tries = 0;
        const int max_tries = static_cast<int>(base_size) * 20;

        const int prefer_neg = (num_colors == 1 ? -1 : -(c + 2));
        const int active_val = (num_colors == 1 ?  1 :  (c + 2));

        while (activated < seeds_quota[c] && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            for (int ax = 0; ax < dim - 1; ++ax)
                coords[ax] = rng.uniform_int(0, shape[ax] - 1);
            coords[grow_axis] = 0;

            int idx = lin_index(coords[0], coords[1], coords[2]);
            int v   = net.get(idx);
            if (v == prefer_neg || v == -1) {
                net.set(idx, active_val);
                borderland.push(coords);
                ++N_current[c];
                ++activated;

                // tempo t=0 com código de cor
                int encoded_t = color_mul[c] + 0;
                net_animation.set(idx, encoded_t);

                parent[c][idx] = -1; // raiz (seed)
            }
            ++tries;
        }
    }

    // registra passo t = 0
    commit_step(0, p_curr, N_current);

    // ===== tracking de alturas (como no create_network) =====
    std::vector<int> max_heights(num_colors, 0);
    auto update_max_height = [&](const std::vector<int>& pos, int cor_idx) {
        int current_height = pos[grow_axis];
        if (current_height > max_heights[cor_idx]) {
            max_heights[cor_idx] = current_height;
        }
    };

    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    // ===== EVOLUÇÃO =====
    for (int t = 1; t < num_of_samples; ++t) {
        bool any_growth = false;
        std::fill(N_current.begin(), N_current.end(), 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()) {
            std::vector<int> pos = borderland.front();
            borderland.pop();

            int idx   = lin_index(pos[0], pos[1], pos[2]);
            int a_val = net.get(idx);
            if (a_val <= 0) continue;

            int cor_idx = (num_colors == 1 ? 0 : (std::abs(a_val) - 2));
            const int new_val = (num_colors == 1 ? 1 : (cor_idx + 2));

            int v_idx = 0;
            for (int ax = 0; ax < dim; ++ax) {
                for (int delta : {-1, 1}) {
                    std::vector<int>& viz = neighbor_buffer[v_idx++];
                    viz = pos;
                    viz[ax] += delta;

                    if (!valid_coord(viz)) continue;

                    int viz_idx = lin_index(viz[0], viz[1], viz[2]);
                    int vv      = net.get(viz_idx);
                    if (vv >= 0) continue;

                    bool same_color = (num_colors == 1) || (vv == -(cor_idx + 2));
                    bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    double r = rng.uniform_real(0.0, 1.0);

                    if (r < p_curr[cor_idx]) {
                        net.set(viz_idx, new_val);
                        new_borderland.push(viz);
                        ++N_current[cor_idx];
                        any_growth = true;

                        update_max_height(viz, cor_idx);

                        int encoded_t = color_mul[cor_idx] + t;
                        net_animation.set(viz_idx, encoded_t);

                        // registra percolação na chegada ao topo (igual ao create_network)
                        if (!percolated[cor_idx] && viz[grow_axis] == lenght_network - 1) {
                            percolated[cor_idx] = true;
                            ps_out.color_percolation.push_back(cor_idx + 1); // 1-based
                            ps_out.percolation_order.push_back(++order_ctr);
                        }

                        parent[cor_idx][viz_idx] = idx;
                    } else {
                        if (type_percolation == "node") {
                            net.set(viz_idx, 0);  // marca checado
                        }
                        // para "bond", não marcamos 0
                    }
                }
            }
        }

        // ===== critérios de parada iguais ao create_network =====
        bool all_top_reached = true;
        for (int c = 0; c < num_colors; ++c) {
            if (max_heights[c] != lenght_network - 1) {
                all_top_reached = false;
                break;
            }
        }
        bool all_stuck = !any_growth;

        if (all_top_reached || all_stuck) {
            // finalização pós-simulação: shortest paths + tamanhos de clusters
            BiggestComponent bc;
            bc.compute_shortest_paths_to_base(
                net, dim, shape, grow_axis, num_colors, parent, ps_out);

            // tamanhos por cor
            ps_out.M_size_at_perc = bc.largest_cluster_sizes(
                net, dim, shape, grow_axis, num_colors);

            // reordenar tamanhos na ordem de eventos
            std::vector<int> M_sizes_per_event;
            M_sizes_per_event.reserve(ps_out.color_percolation.size());
            for (int color_1b : ps_out.color_percolation) {
                int c = color_1b - 1;
                M_sizes_per_event.push_back(ps_out.M_size_at_perc[c]);
            }
            ps_out.M_size_at_perc = std::move(M_sizes_per_event);

            break;
        }

        // ===== se não parou, continua evolução =====
        borderland = std::move(new_borderland);

        std::vector<double> p_next(num_colors);
        for (int c = 0; c < num_colors; ++c)
            p_next[c] = generate_p(type_N_t, p_curr[c], t, N_current[c], k, a, alpha);

        commit_step(t, p_next, N_current);

        if (t < 10 || t % 100 == 0) {
            std::cout << "[ANIMATION] t = " << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << ", p" << c + 1 << "(t)=" << p_next[c]
                          << ", N_t" << c + 1 << "(t)=" << N_current[c]
                          << " max" << c + 1 << "(t)=" << max_heights[c];
            std::cout << std::endl;
        }

        p_curr.swap(p_next);
    }

    // ===== TimeSeries de saída =====
    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.Nt  = std::move(Nt_series);
    ts_out.t   = std::move(t_list);

    // ===== rho na PercolationSeries =====
    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < static_cast<int>(rho.size()))
            ps_out.rho.push_back(rho[i]);
    }

    // net_animation: (color_mul[c] + t) onde ativou, -1 onde nunca ativou
    return net_animation;
}


NetworkPattern network::create_shortest_paths_map(const NetworkPattern& net,
                                                  const PercolationSeries& ps_out)
{
    // Cria uma nova rede com mesmo dim/shape/num_colors/rho,
    // mas dados inicializados em zero.
    NetworkPattern sp_net(net.dim, net.shape, net.num_colors, net.rho);

    if (sp_net.data.size() != net.data.size()) {
        throw std::runtime_error("[create_shortest_paths_map] tamanho de data inconsistente");
    }

    std::fill(sp_net.data.begin(), sp_net.data.end(), 0);

    const int num_colors = net.num_colors;

    // Sanidade básica sobre PercolationSeries
    if (static_cast<int>(ps_out.sp_path_lin.size()) < num_colors ||
        static_cast<int>(ps_out.sp_len.size())      < num_colors) {
        throw std::runtime_error("[create_shortest_paths_map] PercolationSeries inconsistente");
    }

    const std::size_t N = sp_net.data.size();

    for (int c = 0; c < num_colors; ++c) {
        // Se não percolou ou não há caminho registrado, pula
        if (ps_out.sp_len[c] <= 0) continue;

        const std::vector<int>& path = ps_out.sp_path_lin[c];
        if (path.empty()) continue;

        // Valor numérico igual ao índice da cor (1, 2, 3, ...)
        const int color_label = (num_colors == 1 ? 1 : (c + 2));

        for (int idx : path) {
            if (idx < 0 || static_cast<std::size_t>(idx) >= N) {
                // segurança: ignora índices inválidos
                continue;
            }
            // Se um sítio estiver em mais de um caminho (raro), o último sobrescreve o anterior.
            sp_net.data[idx] = color_label;
        }
    }

    return sp_net;
}


// // ===== print_initial_site_fractions (compatível com o struct atual) =====
// void network::print_initial_site_fractions(const NetworkPattern& net) 
//     {
//     std::map<int, size_t> count;  // ordenado por chave (estado)
//     const size_t total = net.data.size();

//     for (int v : net.data) count[v]++;

//     std::cout << "\n[ Fração inicial dos sítios ]\n";
//     std::cout.setf(std::ios::fixed); std::cout << std::setprecision(6);

//     size_t check_sum = 0;
//     for (const auto& kv : count) {
//         int state = kv.first;
//         size_t n  = kv.second;
//         double frac = static_cast<double>(n) / static_cast<double>(total);
//         std::cout << "estado = " << std::setw(3) << state
//                   << " | contagem = " << std::setw(10) << n
//                   << " | fração = " << frac << '\n';
//         check_sum += n;
//     }

//     std::cout << "total sites = " << total
//               << " | soma contagens = " << check_sum
//               << (check_sum == total ? " (OK)\n" : " (INCONSISTENTE!)\n");
// }

