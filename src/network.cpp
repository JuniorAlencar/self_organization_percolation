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
    const double k, const double N_t, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng, bool &DSU_calculate_)
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

    auto lin_index = [&](int x,int y,int z)->int { return x + SX*(y + SY*z); };

    // ===== rede e DSUs =====
    NetworkPattern net(dim, shape, num_colors, rho, rng);

    std::vector<DSU> dsu;                           // opcional
    if (DSU_calculate_) {
        dsu.reserve(num_colors);
        for (int c = 0; c < num_colors; ++c)
            dsu.emplace_back(dim, SX, SY, SZ, grow_axis);
    }

    const bool is_bond = (type_percolation == "bond");
    const PercolationMode sp_mode = (is_bond ? PercolationMode::Bond : PercolationMode::Site);

    // ===== séries =====
    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<int>>    Nt_series(num_colors);
    std::vector<std::vector<double>> chi_series(num_colors);
    std::vector<std::vector<int>>   Smax_series(num_colors);
    std::vector<std::vector<int>>   Ni_series(num_colors);
    std::vector<int>                 t_list; t_list.reserve(num_of_samples);
    std::vector<double>              p_curr = p0;
    std::vector<int>                 M_curr(num_colors, 0);

    // ---- parents para shortest_path mesmo sem DSU ---- // <<<
    const int GRID_N = SX*SY*SZ;
    std::vector<std::vector<int>> parent(num_colors, std::vector<int>(GRID_N, -2)); // -2: nunca visto; -1: raiz/seed  // <<<

    auto commit_step = [&](int t_k, const std::vector<double>& p_vec, const std::vector<int>& Nt_vec)
    {
        t_list.push_back(t_k);
        for (int c=0;c<num_colors;++c){ p_series[c].push_back(p_vec[c]); Nt_series[c].push_back(Nt_vec[c]); }

        if (DSU_calculate_) {
            for (int c=0; c<num_colors; ++c) {
                dsu[c].append_stats_row(Smax_series[c], Ni_series[c], chi_series[c]);
            }
        }
        if (ts_out.M_t.empty()) ts_out.M_t.assign(num_colors,{});
        int gmax=0; for (int c=0;c<num_colors;++c){ ts_out.M_t[c].push_back(M_curr[c]); gmax = std::max(gmax, M_curr[c]); }
    };

    // ===== base measure =====
    long long base_size = 1;
    for (int ax=0; ax<dim-1; ++ax) base_size *= static_cast<long long>(shape[ax]);

    std::vector<int> seeds_quota(num_colors, 0);
    {
        const double Nt_target = static_cast<double>(this->N_t);
        for (int c=0;c<num_colors;++c){
            long long q = std::llround(P0 * Nt_target);
            if (q < 0) q = 0;
            if (q > base_size) q = base_size;
            seeds_quota[c] = static_cast<int>(q);
        }
    }

    std::queue<std::vector<int>> borderland;
    std::vector<int> N_current(num_colors, 0);

    // ===== semeadura (t=0) =====
    for (int c=0;c<num_colors;++c){
        int activated=0, tries=0;
        const int max_tries = static_cast<int>(base_size)*20;

        const int prefer_neg = (num_colors==1 ? -1 : -(c+2));
        const int active_val = (num_colors==1 ?  1 :  (c+2));

        while (activated < seeds_quota[c] && tries < max_tries) {
            std::vector<int> coords(dim,0);
            for (int ax=0; ax<dim-1; ++ax) coords[ax] = rng.uniform_int(0, shape[ax]-1);
            coords[grow_axis] = 0;

            int v = net.get(coords);
            if (v == prefer_neg || v == -1) {
                net.set(coords, active_val);
                borderland.push(coords);
                ++N_current[c]; ++activated;

                int x=(dim>=1?coords[0]:0), y=(dim>=2?coords[1]:0), z=(dim==3?coords[2]:0);
                int id0 = lin_index(x,y,z);

                if (DSU_calculate_) {
                    dsu[c].make_active(id0, get_gcoord(coords), L);
                    M_curr[c] = std::max(M_curr[c], dsu[c].sz[ dsu[c].find(id0) ]);
                }

                parent[c][id0] = -1; // seed/root para shortest_path sem DSU  // <<<
            }
            ++tries;
        }
    }

    // ===== percolação =====
    std::vector<bool> percolated(num_colors, false);
    std::vector<int>  t_percolated(num_colors, -1);
    int order_counter = 0;

    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, {});
    // ps_out.M_size_at_perc.clear();

    // commit t=0
    commit_step(0, p_curr, N_current);

    // buffer vizinhos
    std::vector<std::vector<int>> neighbor_buffer(2*dim, std::vector<int>(dim));

    auto rebuild_path_from_parent = [&](int c, int id_end)->std::vector<int> {         // <<<
        std::vector<int> path;
        int cur = id_end;
        while (cur != -1 && cur >= 0) {
            path.push_back(cur);
            cur = parent[c][cur];
        }
        std::reverse(path.begin(), path.end());
        return path;
    };                                                                                  // <<<

    // ===== evolução =====
    for (int t=1; t<num_of_samples; ++t){
        std::fill(N_current.begin(), N_current.end(), 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()){
            std::vector<int> pos = borderland.front(); borderland.pop();
            int a_val = net.get(pos);
            if (a_val <= 0) continue;

            int cor_idx = (num_colors==1 ? 0 : (std::abs(a_val)-2));
            const int new_val = (num_colors==1 ? 1 : (cor_idx+2));

            int v_idx = 0;
            for (int ax=0; ax<dim; ++ax){
                for (int delta : {-1,1}){
                    std::vector<int>& viz = neighbor_buffer[v_idx++]; viz = pos; viz[ax] += delta;
                    if (!valid_coord(viz)) continue;

                    int vv = net.get(viz);
                    if (vv >= 0) continue; // >0 ativo; ==0 checado/fracassou

                    const bool same_color = (num_colors==1) || (vv == -(cor_idx+2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    double r = rng.uniform_real(0.0, 1.0);

                    auto try_activate = [&](){
                        // ativa o nó vizinho
                        net.set(viz, new_val);
                        new_borderland.push(viz);
                        ++N_current[cor_idx];

                        int x=(dim>=1?viz[0]:0), y=(dim>=2?viz[1]:0), z=(dim==3?viz[2]:0);
                        int id_new = lin_index(x,y,z);

                        int px=(dim>=1?pos[0]:0), py=(dim>=2?pos[1]:0), pz=(dim==3?pos[2]:0);
                        int id_pos = lin_index(px,py,pz);

                        // registra pai SEMPRE (serve para bond/site)                              // <<<
                        parent[cor_idx][id_new] = id_pos;                                          // <<<

                        if (DSU_calculate_){
                            dsu[cor_idx].make_active(id_new, get_gcoord(viz), L);

                            if (is_bond){
                                dsu[cor_idx].open_bond(id_pos, id_new);
                            } else {
                                for (int ax2=0; ax2<dim; ++ax2){
                                    for (int dlt : {-1,1}){
                                        std::vector<int> vv2 = viz; vv2[ax2]+=dlt;
                                        if (!valid_coord(vv2)) continue;
                                        if (net.get(vv2) == new_val){
                                            int nx=(dim>=1?vv2[0]:0), ny=(dim>=2?vv2[1]:0), nz=(dim==3?vv2[2]:0);
                                            int nid = lin_index(nx,ny,nz);
                                            if (dsu[cor_idx].is_active(nid)) dsu[cor_idx].connect_if_site_adjacent(id_new, nid);
                                        }
                                    }
                                }
                            }

                            // maior componente
                            M_curr[cor_idx] = std::max(M_curr[cor_idx], dsu[cor_idx].sz[ dsu[cor_idx].find(id_new) ]);

                            // percolação via DSU (como antes)
                            int rroot = dsu[cor_idx].find(id_new);
                            if (dsu[cor_idx].spans(rroot) && !percolated[cor_idx]) {
                                percolated[cor_idx]   = true;
                                t_percolated[cor_idx] = t;
                                ++order_counter;

                                auto path = dsu[cor_idx].shortest_path_base_to_top(rroot, sp_mode);
                                ps_out.sp_len[cor_idx] = (int)path.size();
                                ps_out.sp_path_lin[cor_idx] = std::move(path);

                                ps_out.color_percolation.push_back(cor_idx + 1);
                                ps_out.time_percolation.push_back(t);
                                ps_out.percolation_order.push_back(order_counter);

                                std::cout << "[CREATE] Cor " << (cor_idx + 1)
                                          << " percolou em t=" << t
                                          << "  (ordem=" << order_counter
                                          << ", shortest_path_len=" << ps_out.sp_len[cor_idx]
                                          << ")\n";
                            }
                        } else {
                            // ---- Sem DSU: detectar chegada ao topo e reconstruir caminho ---- // <<<
                            if (!percolated[cor_idx] && get_gcoord(viz) == (L-1)) {
                                percolated[cor_idx]   = true;
                                t_percolated[cor_idx] = t;
                                ++order_counter;

                                auto path = rebuild_path_from_parent(cor_idx, id_new);
                                ps_out.sp_len[cor_idx] = (int)path.size();
                                ps_out.sp_path_lin[cor_idx] = std::move(path);

                                ps_out.color_percolation.push_back(cor_idx + 1);
                                ps_out.time_percolation.push_back(t);
                                ps_out.percolation_order.push_back(order_counter);

                                std::cout << "[CREATE-noDSU] Cor " << (cor_idx + 1)
                                          << " alcançou topo em t=" << t
                                          << "  (ordem=" << order_counter
                                          << ", shortest_path_len=" << ps_out.sp_len[cor_idx]
                                          << ")\n";
                            }                                                                             // <<<
                        }
                    };

                    if (!is_bond){ // SITE
                        if (r < p_curr[cor_idx]) { try_activate(); }
                        else { net.set(viz, 0); } // checado e falhou
                    } else {        // BOND real
                        if (r < p_curr[cor_idx]) { try_activate(); }
                        else {
                            // ligação fechada: não ativa
                        }
                    }
                }
            }
        }

        const int grown_total = std::accumulate(N_current.begin(), N_current.end(), 0);
        if (grown_total == 0) break;

        borderland = std::move(new_borderland);

        // atualiza p
        std::vector<double> p_next(num_colors);
        for (int c=0;c<num_colors;++c)
            p_next[c] = generate_p(type_N_t, p_curr[c], t, N_current[c], k, a, alpha);

        // commit
        commit_step(t, p_next, N_current);

        // todas percolaram?
        if (std::all_of(percolated.begin(), percolated.end(), [](bool x){ return x; })) {
            std::cout << "[CREATE] Todas as cores percolaram em t=" << t << " (";
            for (int c=0;c<num_colors;++c){ std::cout << "c" << (c+1) << "=" << t_percolated[c]; if (c+1<num_colors) std::cout << ", "; }
            std::cout << ")\n";
            break;
        }

        // logging opcional
        if (t < 10 || t % 100 == 0) {
            std::cout << "[" << type_percolation << "] t = " << t;
            for (int c=0;c<num_colors;++c)
                std::cout << ", p" << c+1 << "(t)=" << p_next[c]
                          << ", N_t" << c+1 << "(t)=" << N_current[c];
            std::cout << std::endl;
        }

        p_curr.swap(p_next);
    }

    // ===== saída =====
    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.Nt  = std::move(Nt_series);
    ts_out.t   = std::move(t_list);

    if (DSU_calculate_) {
        ts_out.Smax = std::move(Smax_series);
        ts_out.Ni   = std::move(Ni_series);
        ts_out.chi  = std::move(chi_series);
    } else {
        // sem DSU: manter vetores vazios (ou dimensões corretas vazias), como combinado
        ts_out.Smax.assign(num_colors, {});
        ts_out.Ni.assign(num_colors,   {});
        ts_out.chi.assign(num_colors,  {});
    }

    // copia rho
    ps_out.rho.clear();
    for (int i=0;i<num_colors;++i) if (i < (int)rho.size()) ps_out.rho.push_back(rho[i]);

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

    NetworkPattern net(dim, shape, num_colors, rho, rng);          // estados (propagação)
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

    // ===== SEEDS NA BORDA (grow_axis = 0), P0 dividida igualmente =====
    int base_size = 1;
    for (int ax = 0; ax < dim - 1; ++ax) base_size *= shape[ax];
    const int seeds_per_color = static_cast<int>(std::llround( (num_colors>0) ? (P0 * base_size / static_cast<double>(num_colors)) : 0.0 ));

    std::queue<std::vector<int>> borderland;
    std::vector<int> N_current(num_colors, 0);

    for (int c = 0; c < num_colors; ++c) {
        int activated = 0, tries = 0;
        const int max_tries = base_size * 20;

        const int prefer_neg = (num_colors == 1) ? -1 : -(c + 2);
        const int new_val    = (num_colors == 1) ?  1 :  (c + 2);

        while (activated < seeds_per_color && tries < max_tries) {
            std::vector<int> coords(dim, 0);
            for (int ax = 0; ax < dim - 1; ++ax)
                coords[ax] = rng.uniform_int(0, shape[ax] - 1);
            coords[grow_axis] = 0;

            const int v = net.get(coords);
            if (v == prefer_neg || v == -1) {
                net.set(coords, new_val);
                borderland.push(coords);
                net_animation.set(coords, 0); // grava tempo
                Nt_t[c][0]++; N_current[c]++; activated++;
            }
            ++tries;
        }
    }

    // Tracking de percolação por cor
    std::vector<bool> percolated(num_colors, false);
    std::vector<int>  t_percolated(num_colors, -1);

    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    // ===== EVOLUÇÃO =====
    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()) {
            const std::vector<int> pos = borderland.front();
            borderland.pop();

            const int active_val = net.get(pos);
            if (active_val <= 0) continue;

            const int cor_idx = (num_colors == 1) ? 0 : (std::abs(active_val) - 2);

            int v_idx = 0;
            for (int ax = 0; ax < dim; ++ax) {
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
                        net_animation.set(viz, t);  // grava tempo de ativação
                        N_current[cor_idx]++;

                        if (viz[grow_axis] == L - 1 && !percolated[cor_idx]) {
                            percolated[cor_idx]   = true;
                            t_percolated[cor_idx] = t;
                            std::cout << "[ANIMATION] Cor " << (cor_idx + 1)
                                      << " percolou em t=" << t
                                      << "  (p=" << p_t[cor_idx].back()
                                      << ", N_t=" << N_current[cor_idx] << ")"
                                      << std::endl;
                        }
                    } else {
                        if (type_percolation == "node") {
                            net.set(viz, 0);  // node: marca checado
                        } else {
                            // bond: não marca 0; permanece elegível
                        }
                    }
                }
            }
        }

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

        if (t < 10 || t % 100 == 0) {
            std::cout << "[ANIMATION] t=" << t;
            for (int c = 0; c < num_colors; ++c)
                std::cout << "  p" << (c+1) << "(" << t << ")=" << p_t[c].back()
                          << "  N" << (c+1) << "(" << t << ")=" << Nt_t[c].back();
            std::cout << std::endl;
        }

        // (Opcional) parar quando TODAS percolarem
        if (std::all_of(percolated.begin(), percolated.end(), [](bool x){ return x; })) {
            std::cout << "[ANIMATION] Todas as cores percolaram em t=" << t << "  (";
            for (int c = 0; c < num_colors; ++c) {
                std::cout << "c" << (c+1) << "=" << t_percolated[c];
                if (c+1 < num_colors) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            break;
        }
    }

    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_t);
    ts_out.Nt  = std::move(Nt_t);
    ts_out.t   = std::move(t_list);

    // net_animation contém apenas tempos de ativação (>=0), -1 onde nunca ativou
    return net_animation;
}


// Ativa a base conforme P0 (total) e p0 (fração por cor na base)
NetworkPattern network::initialize_network(int dim, int length_network, int num_colors,
                                           double P0,
                                           const std::vector<double>& rho,
                                           const std::vector<double>& p0,
                                           int seed)
{
    // ==== shape espacial ====
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{length_network, length_network}
        : std::vector<int>{length_network, length_network, length_network};

    all_random rng(seed);
    // Construtor: preenche data com -1 e rotula a base segundo 'rho' com -(c+2)
    NetworkPattern net(dim, shape, num_colors, rho, rng);

    // ==== checagens de p0 ====
    std::vector<double> p0_use = p0;
    if ((int)p0_use.size() != num_colors) {
        // fallback: distribui igualmente
        p0_use.assign(num_colors, 1.0 / std::max(1, num_colors));
    } else {
        // normaliza se necessário
        double s = std::accumulate(p0_use.begin(), p0_use.end(), 0.0);
        if (s <= 0.0) p0_use.assign(num_colors, 1.0 / std::max(1, num_colors));
        else for (double &x : p0_use) x /= s;
    }

    // ==== geometria da base (última dimensão = 0) ====
    int base_size = 1;
    for (int ax = 0; ax < dim - 1; ++ax) base_size *= shape[ax];

    auto base_coords_from_idx = [&](int idx_linear) {
        std::vector<int> coords(dim, 0);
        int rem = idx_linear;
        for (int ax = dim - 2; ax >= 0; --ax) {
            coords[ax] = rem % shape[ax];
            rem /= shape[ax];
        }
        coords.back() = 0;
        return coords;
    };

    // Total a ativar na base
    const int total_active_target = std::max(0, (int)std::llround(P0 * base_size));

    if (num_colors == 1) {
        // candidatos: -1 na base
        std::vector<size_t> candidates; candidates.reserve(base_size);
        for (int i = 0; i < base_size; ++i) {
            size_t idx = net.to_index(base_coords_from_idx(i));
            if (net.data[idx] == -1) candidates.push_back(idx);
        }
        std::shuffle(candidates.begin(), candidates.end(), rng.get_gen());
        int take = std::min<int>(total_active_target, (int)candidates.size());
        for (int k = 0; k < take; ++k) net.data[candidates[k]] = +1;
        return net;
    }

    // multi-cor: quota por cor segundo p0
    for (int c = 0; c < num_colors; ++c) {
        int quota = std::max(0, (int)std::llround(p0_use[c] * total_active_target));
        if (quota == 0) continue;

        const int label_neg = -(c + 2); // preferido
        const int label_pos =  (c + 2); // ativo dessa cor

        std::vector<size_t> preferred; preferred.reserve(base_size);
        std::vector<size_t> fallback;  fallback.reserve(base_size);

        for (int i = 0; i < base_size; ++i) {
            size_t idx = net.to_index(base_coords_from_idx(i));
            int v = net.data[idx];
            if (v == label_neg)      preferred.push_back(idx);
            else if (v == -1)        fallback.push_back(idx);
        }

        std::shuffle(preferred.begin(), preferred.end(), rng.get_gen());
        std::shuffle(fallback.begin(),  fallback.end(),  rng.get_gen());

        int need = quota;
        int take_pref = std::min<int>(need, (int)preferred.size());
        for (int k = 0; k < take_pref; ++k) net.data[preferred[k]] = label_pos;
        need -= take_pref;

        int take_fb = std::min<int>(need, (int)fallback.size());
        for (int k = 0; k < take_fb; ++k)  net.data[fallback[k]] = label_pos;
        // se ainda faltar, não há mais slots adequados na base — OK.
    }

    return net;
}



// ===== print_initial_site_fractions (compatível com o struct atual) =====
void network::print_initial_site_fractions(const NetworkPattern& net) {
    std::map<int, size_t> count;  // ordenado por chave (estado)
    const size_t total = net.data.size();

    for (int v : net.data) count[v]++;

    std::cout << "\n[ Fração inicial dos sítios ]\n";
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(6);

    size_t check_sum = 0;
    for (const auto& kv : count) {
        int state = kv.first;
        size_t n  = kv.second;
        double frac = static_cast<double>(n) / static_cast<double>(total);
        std::cout << "estado = " << std::setw(3) << state
                  << " | contagem = " << std::setw(10) << n
                  << " | fração = " << frac << '\n';
        check_sum += n;
    }

    std::cout << "total sites = " << total
              << " | soma contagens = " << check_sum
              << (check_sum == total ? " (OK)\n" : " (INCONSISTENTE!)\n");
}







