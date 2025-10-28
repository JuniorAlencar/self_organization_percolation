#include "network.hpp"
#include "rand_utils.hpp"
#include <vector>
#include <queue>
#include <cassert>

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

void network::colorize_base_by_rho(NetworkPattern& net, int grow_axis, all_random& rng) {
    const int dim = net.dim;
    const auto& shape = net.shape;
    const int nc = net.num_colors;

    // 1 cor: não colorir a base (fica tudo cinza por padrão)
    size_t base_size = 1;
    for (int ax = 0; ax < dim; ++ax) if (ax != grow_axis) base_size *= size_t(shape[ax]);
    if (base_size == 0 || nc <= 1) return;

    // 1) Quotas desejadas (em número de sítios) para cada cor
    std::vector<double> quotas(nc + 1, 0.0); // [0..nc-1]=cores, [nc]=cinza
    for (int c = 0; c < nc; ++c) {
        double r = (c < (int)net.rho.size() ? net.rho[c] : 0.0);
        if (r < 0.0) r = 0.0;
        quotas[c] = r * double(base_size);
    }

    double sum_colors = 0.0;
    for (int c = 0; c < nc; ++c) sum_colors += quotas[c];

    if (sum_colors <= double(base_size) + 1e-12) {
        // cinza recebe o resto
        quotas[nc] = double(base_size) - sum_colors; // pode ser zero
    } else {
        // cores "passaram" do tamanho da base — reescala proporcionalmente e zera cinza
        if (sum_colors > 0.0) {
            double s = double(base_size) / sum_colors;
            for (int c = 0; c < nc; ++c) quotas[c] *= s;
        }
        quotas[nc] = 0.0;
    }

    // 2) Hamilton (maiores restos) em (cores + cinza)
    std::vector<size_t> aloc(nc + 1, 0);
    size_t soma_floor = 0;
    for (int i = 0; i < nc + 1; ++i) {
        double q = quotas[i];
        if (q < 0.0) q = 0.0;
        aloc[i] = size_t(std::floor(q));
        soma_floor += aloc[i];
    }
    size_t leftover = (base_size > soma_floor) ? (base_size - soma_floor) : 0;

    std::vector<int> order(nc + 1);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){
        double fa = quotas[a] - std::floor(quotas[a]);
        double fb = quotas[b] - std::floor(quotas[b]);
        // desempate estável por índice
        return (fa == fb) ? (a < b) : (fa > fb);
    });

    for (size_t k = 0; k < leftover; ++k) {
        aloc[ order[k % (nc + 1)] ] += 1; // distribui resto também sobre "cinza"
    }

    // 3) Monta índices da base e embaralha
    std::vector<size_t> base_idx; base_idx.reserve(base_size);
    if (dim == 2) {
        int ax_other = (grow_axis == 0 ? 1 : 0);
        for (int i = 0; i < shape[ax_other]; ++i) {
            std::vector<int> v(2,0);
            v[grow_axis] = 0; v[ax_other] = i;
            base_idx.push_back(net.to_index(v));
        }
    } else { // 3D
        int ax_a = (grow_axis == 0 ? 1 : 0);
        int ax_b = (grow_axis == 2 ? 1 : 2);
        for (int ia = 0; ia < shape[ax_a]; ++ia) {
            for (int ib = 0; ib < shape[ax_b]; ++ib) {
                std::vector<int> v(3,0);
                v[grow_axis] = 0; v[ax_a] = ia; v[ax_b] = ib;
                base_idx.push_back(net.to_index(v));
            }
        }
    }
    std::shuffle(base_idx.begin(), base_idx.end(), rng.get_gen());

    // 4) Escreve só as cores inativas; "cinza" é o default e permanece
    size_t cursor = 0;
    for (int c = 0; c < nc; ++c) {
        size_t take = aloc[c];
        for (size_t k = 0; k < take && cursor < base_idx.size(); ++k, ++cursor) {
            net.data[ base_idx[cursor] ] = NetworkPattern::enc_inactive(c); // -(c+2) (inativo) na codificação compacta
        }
    }
    // aloc[nc] = quantidade de cinza — não precisa escrever nada; já é o valor default
}

// Code to create network in 2D or 3D with num_colors >= 1
// Requer: #include "network_utils.h"

NetworkPattern network::create_network(
    const int dim, const int lenght_network, const int num_of_samples,
    const double k, const double N_t, const int type_N_t,
    const std::vector<double> p0, const double P0, const double a, const double alpha,
    const std::string& type_percolation, const int& num_colors, const std::vector<double>& rho,
    TimeSeries& ts_out, PercolationSeries& ps_out, all_random& rng)
{
    this->N_t = N_t;

    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{lenght_network, lenght_network}
        : std::vector<int>{lenght_network, lenght_network, lenght_network};

    const int L = lenght_network;
    const int grow_axis = dim - 1;

    auto valid_coord = [&](std::vector<int>& v)->bool {
        for (int j=0;j<dim;++j) {
            if (j == grow_axis) {
                if (v[j] < 0 || v[j] >= shape[j]) return false;
            } else {
                if (v[j] < 0) v[j] = shape[j]-1; else if (v[j] >= shape[j]) v[j] = 0;
            }
        }
        return true;
    };
    auto get_gcoord = [&](const std::vector<int>& v)->int { return v[grow_axis]; };

    const int SX = shape[0];
    const int SY = (dim >= 2 ? shape[1] : 1);
    const int SZ = (dim == 3 ? shape[2] : 1);

    auto lin_index = [&](int x,int y,int z)->int { return x + SX*(y + SY*z); };
    auto id2xyz = [&](int id)->std::array<int,3>{
        int z = (dim==3 ? id / (SX*SY) : 0);
        int rem = (dim==3 ? id - z*(SX*SY) : id);
        int y = (dim>=2 ? rem / SX : 0);
        int x = rem - y*SX;
        return {x,y,z};
    };

    auto xy2id = [&](int x,int y)->int { return x + SX*y; };
    auto xyz2id = [&](int x,int y,int z)->int { return x + SX*(y + SY*z); };

    NetworkPattern net(dim, shape, num_colors, rho, rng);
    colorize_base_by_rho(net, grow_axis, rng);

    const bool is_bond = (type_percolation == "bond");

    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<int>>    Nt_series(num_colors);
    std::vector<int>                 t_list; t_list.reserve(num_of_samples);
    std::vector<double>              p_curr = p0;
    std::vector<int>                 M_curr(num_colors, 0);

    const int GRID_N = SX*SY*SZ;
    // pai compacto (direção): 0xFF = sem pai
    std::vector<uint8_t> parent_dir((size_t)GRID_N, 0xFF);

    // offsets dos vizinhos (até 6)
    std::vector<std::array<int,3>> neigh;
    if (dim == 2) {
        neigh = {{ {+1,0,0}, {-1,0,0}, {0,+1,0}, {0,-1,0} }};
    } else {
        neigh = {{ {+1,0,0}, {-1,0,0}, {0,+1,0}, {0,-1,0}, {0,0,+1}, {0,0,-1} }};
    }
    auto wrap = [&](int a, int L){ return (a<0? L-1 : (a>=L? 0 : a)); };

    auto dir_code = [&](int dx,int dy,int dz)->uint8_t{
        if (dim==2){
            if (dx==+1 && dy==0) return 0;
            if (dx==-1 && dy==0) return 1;
            if (dx==0  && dy==+1) return 2;
            return 3; // dx==0 && dy==-1
        } else {
            if (dx==+1 && dy==0 && dz==0) return 0;
            if (dx==-1 && dy==0 && dz==0) return 1;
            if (dx==0  && dy==+1 && dz==0) return 2;
            if (dx==0  && dy==-1 && dz==0) return 3;
            if (dx==0  && dy==0  && dz==+1) return 4;
            return 5; // dz==-1
        }
    };
    auto step_back = [&](int id, uint8_t code)->int{
        auto xyz = id2xyz(id);
        if (dim==2){
            if (code==0) return xy2id(xyz[0]-1, xyz[1]   );
            if (code==1) return xy2id(xyz[0]+1, xyz[1]   );
            if (code==2) return xy2id(xyz[0]  , xyz[1]-1 );
            return           xy2id(xyz[0]  , xyz[1]+1 );
        } else {
            if (code==0) return xyz2id(xyz[0]-1, xyz[1], xyz[2]);
            if (code==1) return xyz2id(xyz[0]+1, xyz[1], xyz[2]);
            if (code==2) return xyz2id(xyz[0], xyz[1]-1, xyz[2]);
            if (code==3) return xyz2id(xyz[0], xyz[1]+1, xyz[2]);
            if (code==4) return xyz2id(xyz[0], xyz[1], xyz[2]-1);
            return           xyz2id(xyz[0], xyz[1], xyz[2]+1);
        }
    };

    auto rebuild_path_from_parent = [&](int id_end)->std::vector<int> {
        std::vector<int> path;
        int cur = id_end;
        while (cur >= 0 && parent_dir[(size_t)cur] != 0xFF) {
            path.push_back(cur);
            cur = step_back(cur, parent_dir[(size_t)cur]);
        }
        if (cur >= 0) path.push_back(cur);
        std::reverse(path.begin(), path.end());
        return path;
    };

    // ---- Commit helper + print control ----
    auto commit_step = [&](int t_k, const std::vector<double>& p_vec, const std::vector<int>& Nt_vec)
    {
        t_list.push_back(t_k);
        
        for (int c=0;c<num_colors;++c){
            p_series[c].push_back(p_vec[c]);
            Nt_series[c].push_back(Nt_vec[c]);
            
        }
    };

    auto print_status = [&](int t, const std::vector<double>& p_vec, const std::vector<int>& Nt_vec){
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6);
        std::cout << "[create] t=" << t;
        for (int c = 0; c < num_colors; ++c) {
            std::cout << "  p" << (c+1) << "=" << p_vec[c]
                      << "  N" << (c+1) << "=" << Nt_vec[c];
        }
        std::cout << '\n';
    };

    // ===== base size / seeds =====
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

    // ===== fronteira por índices lineares =====
    std::vector<int> frontier, next_frontier;
    frontier.reserve(1<<20); next_frontier.reserve(1<<20);

    std::vector<int> N_current(num_colors, 0);

    // ===== semeadura t=0 =====
    for (int c=0;c<num_colors;++c){
        int activated=0, tries=0;
        const int max_tries = static_cast<int>(base_size)*20;

        const int prefer_neg_old = (num_colors==1 ? -1 : -(c+2));
        const int active_val_old = (num_colors==1 ?  1 :  (c+2));

        while (activated < seeds_quota[c] && tries < max_tries) {
            std::vector<int> coords(dim,0);
            for (int ax=0; ax<dim-1; ++ax) coords[ax] = rng.uniform_int(0, shape[ax]-1);
            coords[grow_axis] = 0;

            int v_old = net.get(coords);
            if (v_old == prefer_neg_old || v_old == -1) {
                net.set(coords, active_val_old);

                int id0 = (dim==2 ? lin_index(coords[0], coords[1], 0)
                                  : lin_index(coords[0], coords[1], coords[2]));
                frontier.push_back(id0);
                parent_dir[(size_t)id0] = 0xFF; // raiz
                ++N_current[c]; ++activated;
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
    std::vector<int> M_size_at_perc(num_colors, 0); // opcional (mantemos)

    // commit t=0 + print
    commit_step(0, p_curr, N_current);
    print_status(0, p_curr, N_current);

    // ===== evolução =====
    for (int t=1; t<num_of_samples; ++t){
        std::fill(N_current.begin(), N_current.end(), 0);
        next_frontier.clear();

        for (int id_pos : frontier) {
            if (id_pos < 0 || id_pos >= GRID_N) continue;
            uint8_t a_val = net.data[(size_t)id_pos];
            if (!NetworkPattern::is_active_color(a_val, num_colors)) continue;

            int cor_idx = NetworkPattern::color_of(a_val, num_colors);
            const uint8_t new_val_enc = NetworkPattern::enc_active(num_colors, cor_idx);

            auto xyz = id2xyz(id_pos);
            for (const auto &d : neigh) {
                int nx = xyz[0] + d[0];
                int ny = xyz[1] + d[1];
                int nz = xyz[2] + d[2];

                // contorno: laterais periódicas, eixo crescimento aberto
                if (dim >= 2) { if (d[0] != 0) nx = wrap(nx, SX); }
                if (dim >= 2) { if (d[1] != 0) ny = wrap(ny, SY); }
                if (dim == 3) {
                    if (d[2] != 0) {
                        if (nz < 0 || nz >= SZ) continue;
                    }
                } else { nz = 0; }

                int id_v = (dim==2 ? xy2id(nx, ny) : xyz2id(nx, ny, nz));
                uint8_t vv = net.data[(size_t)id_v];
                // Pode ativar: cinza (0) ou inativo da MESMA cor
                bool color_ok = NetworkPattern::is_gray(vv)
                                || (NetworkPattern::is_inactive_color(vv, num_colors)
                                    && NetworkPattern::color_of(vv, num_colors) == cor_idx);
                if (!color_ok) continue;

                double r = rng.uniform_real(0.0, 1.0);
                if (r < p_curr[cor_idx]) {
                    // ativa
                    net.data[(size_t)id_v] = new_val_enc;
                    parent_dir[(size_t)id_v] = dir_code(d[0], d[1], d[2]);
                    next_frontier.push_back(id_v);
                    ++N_current[cor_idx];

                    // chegou ao topo?
                    if (!percolated[cor_idx]) {
                        int g = (dim==3 ? nz : ny);
                        if (g == (L-1)) {
                            percolated[cor_idx]   = true;
                            t_percolated[cor_idx] = t;
                            ++order_counter;

                            auto path = rebuild_path_from_parent(id_v);
                            ps_out.sp_len[cor_idx] = (int)path.size();
                            ps_out.sp_path_lin[cor_idx] = std::move(path);

                            ps_out.color_percolation.push_back(cor_idx + 1);
                            ps_out.percolation_order.push_back(order_counter);
                        }
                    }
                } else {
                    if (!is_bond) {
                        // marca checado (não tenta de novo)
                        net.data[(size_t)id_v] = NetworkPattern::enc_checked();
                    } else {
                        // bond: elegível em iterações futuras (não marca 1)
                    }
                }
            }
        }

        if (next_frontier.empty()) break;

        frontier.swap(next_frontier);

        // atualiza p
        std::vector<double> p_next(num_colors);
        for (int c=0;c<num_colors;++c)
            p_next[c] = generate_p(type_N_t, p_curr[c], t, N_current[c], k, a, alpha);

        // commit + prints conforme solicitado
        commit_step(t, p_next, N_current);
        if (t < 10 || (t % 100) == 0) {
            print_status(t, p_next, N_current);
        }

        // All percolate?
        if (std::all_of(percolated.begin(), percolated.end(), [](bool x){ return x; })) {
            break;
        }

        p_curr.swap(p_next);
    }

    // ===== output =====
    ts_out.num_colors = num_colors;
    ts_out.p_t = std::move(p_series);
    ts_out.Nt  = std::move(Nt_series);
    ts_out.t   = std::move(t_list);
    ps_out.M_size_at_perc = std::move(M_size_at_perc);

    ps_out.rho.clear();
    for (int i=0;i<num_colors;++i){
        if (i < (int)rho.size()) ps_out.rho.push_back(rho[i]);
    }

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
    std::vector<int> shape = (dim == 2)
        ? std::vector<int>{length_network, length_network}
        : std::vector<int>{length_network, length_network, length_network};

    all_random rng(seed);
    NetworkPattern net(dim, shape, num_colors, rho, rng);

    const int grow_axis = dim - 1;

    // Colore a BASE com -(c+2) segundo rho (sobra permanece -1)
    colorize_base_by_rho(net, grow_axis, rng);

    // Índices lineares da base (para ativação P0/p0)
    auto base_coords_from_idx = [&](int idx_linear) {
        std::vector<int> coords(dim, 0);
        int rem = idx_linear;
        for (int ax = dim - 2; ax >= 0; --ax) {
            coords[ax] = rem % shape[ax];
            rem /= shape[ax];
        }
        coords[grow_axis] = 0;
        return coords;
    };
    int base_size = 1;
    for (int ax = 0; ax < dim - 1; ++ax) base_size *= shape[ax];

    std::vector<size_t> base_indices;
    base_indices.reserve((size_t)base_size);
    for (int i = 0; i < base_size; ++i) {
        base_indices.push_back(net.to_index(base_coords_from_idx(i)));
    }

    // Normaliza p0
    std::vector<double> p0_use = p0;
    if ((int)p0_use.size() != num_colors) {
        p0_use.assign(std::max(1, num_colors), (num_colors > 0 ? 1.0 / num_colors : 1.0));
    } else {
        double s = std::accumulate(p0_use.begin(), p0_use.end(), 0.0);
        if (s <= 0.0) p0_use.assign(num_colors, (num_colors > 0 ? 1.0 / num_colors : 1.0));
        else for (double &x : p0_use) x /= s;
    }

    const int total_active_target = std::max(0, (int)std::llround(P0 * base_size));

    if (num_colors <= 1) {
        std::vector<size_t> candidates;
        candidates.reserve((size_t)base_size);
        for (size_t idx_lin : base_indices) {
            if (net.data[idx_lin] == -1) candidates.push_back(idx_lin);
        }
        std::shuffle(candidates.begin(), candidates.end(), rng.get_gen());
        int take = std::min<int>(total_active_target, (int)candidates.size());
        for (int k = 0; k < take; ++k) net.data[candidates[k]] = +1;
        return net;
    }

    for (int c = 0; c < num_colors; ++c) {
        int quota = std::max(0, (int)std::llround(p0_use[c] * total_active_target));
        if (quota == 0) continue;

        const int label_neg = -(c + 2);
        const int label_pos =  (c + 2);

        std::vector<size_t> preferred; preferred.reserve((size_t)quota);
        std::vector<size_t> fallback;  fallback.reserve((size_t)quota);

        for (size_t idx_lin : base_indices) {
            int v = net.data[idx_lin];
            if (v == label_neg) preferred.push_back(idx_lin);
            else if (v == -1)   fallback.push_back(idx_lin);
        }

        std::shuffle(preferred.begin(), preferred.end(), rng.get_gen());
        std::shuffle(fallback.begin(),  fallback.end(),  rng.get_gen());

        int need = quota;
        int take_pref = std::min<int>(need, (int)preferred.size());
        for (int k = 0; k < take_pref; ++k) net.data[preferred[k]] = label_pos;
        need -= take_pref;

        int take_fb = std::min<int>(need, (int)fallback.size());
        for (int k = 0; k < take_fb; ++k)  net.data[fallback[k]] = label_pos;
    }

    return net;
}


// ===== print_initial_site_fractions (compatível com o struct atual) =====
void network::print_initial_site_fractions(const NetworkPattern& net) 
    {
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







