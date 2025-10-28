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
    const int num_colors = net.num_colors;

    // tamanho da base: produto dos eixos != grow_axis
    size_t base_size = 1;
    for (int ax = 0; ax < dim; ++ax) if (ax != grow_axis) base_size *= size_t(shape[ax]);
    if (base_size == 0 || num_colors <= 1) return;

    // Quotas "alvo" para cada cor na base (sem normalizar; permitimos sobra ou cap)
    std::vector<double> quotas(num_colors, 0.0);
    for (int c = 0; c < num_colors; ++c) {
        double q = (c < (int)net.rho.size() ? net.rho[c] : 0.0);
        if (q < 0.0) q = 0.0;
        quotas[c] = q * double(base_size);
    }

    // Floors + soma
    std::vector<size_t> aloc(num_colors, 0);
    size_t soma_floor = 0;
    for (int c = 0; c < num_colors; ++c) {
        aloc[c] = (quotas[c] > 0.0) ? size_t(std::floor(quotas[c])) : size_t(0);
        soma_floor += aloc[c];
    }
    // Ajuste por “maiores restos” até o máximo base_size
    size_t leftover_cap = (base_size > soma_floor) ? (base_size - soma_floor) : 0;
    std::vector<int> order(num_colors);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){
        double fa = quotas[a] - std::floor(quotas[a]);
        double fb = quotas[b] - std::floor(quotas[b]);
        if (fa == fb) return a < b;
        return fa > fb;
    });
    for (size_t k = 0; k < leftover_cap && k < (size_t)num_colors; ++k) {
        aloc[order[k]] += 1;
    }
    // Se soma(rho) > 1, cap: não exceder base_size
    size_t total_aloc = std::accumulate(aloc.begin(), aloc.end(), size_t{0});
    if (total_aloc > base_size) {
        // remove do fim de 'order' (menores restos) até caber
        size_t excess = total_aloc - base_size;
        for (int i = num_colors - 1; i >= 0 && excess > 0; --i) {
            size_t c = (size_t)order[i];
            size_t rem = std::min(excess, aloc[c]);
            aloc[c] -= rem;
            excess  -= rem;
        }
    }

    // Colete todos os índices lineares da BASE (grow_axis = 0)
    std::vector<size_t> base_indices;
    base_indices.reserve(base_size);

    // Gerar coordenadas da base: para ax != grow_axis percorre 0..shape[ax]-1, grow_axis = 0
    if (dim == 2) {
        const int ax_other = (grow_axis == 0 ? 1 : 0);
        for (int i = 0; i < shape[ax_other]; ++i) {
            std::vector<int> v(2, 0);
            v[grow_axis] = 0;
            v[ax_other]  = i;
            base_indices.push_back(net.to_index(v));
        }
    } else { // dim == 3
        int ax_a = (grow_axis == 0 ? 1 : 0);
        int ax_b = (grow_axis == 2 ? 1 : 2);
        for (int ia = 0; ia < shape[ax_a]; ++ia) {
            for (int ib = 0; ib < shape[ax_b]; ++ib) {
                std::vector<int> v(3, 0);
                v[grow_axis] = 0;
                v[ax_a]      = ia;
                v[ax_b]      = ib;
                base_indices.push_back(net.to_index(v));
            }
        }
    }

    // Embaralha os slots da base
    std::shuffle(base_indices.begin(), base_indices.end(), rng.get_gen());

    // Atribui -(c+2) para cada cor conforme aloc[c]; sobra permanece -1
    size_t cursor = 0;
    for (int c = 0; c < num_colors; ++c) {
        size_t take = aloc[c];
        int label = -(c + 2);
        for (size_t k = 0; k < take && cursor < base_indices.size(); ++k, ++cursor) {
            net.data[base_indices[cursor]] = label;
        }
    }
    // Qualquer posição restante (cursor..end) fica em -1 (cinza) — já está assim por default.
};

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

    NetworkPattern net(dim, shape, num_colors, rho, rng);
    colorize_base_by_rho(net, grow_axis, rng);
  
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
    const bool is_bond = (type_percolation == "bond");

    std::vector<std::vector<double>> p_series(num_colors);
    std::vector<std::vector<int>>    Nt_series(num_colors);
    std::vector<int>                 t_list; t_list.reserve(num_of_samples);
    std::vector<double>              p_curr = p0;
    std::vector<int>                 M_curr(num_colors, 0);

    const int GRID_N = SX*SY*SZ;
    std::vector<std::vector<int>> parent(num_colors, std::vector<int>(GRID_N, -2));

    auto rebuild_path_from_parent = [&](int c, int id_end)->std::vector<int> {
        std::vector<int> path; int cur = id_end;
        while (cur != -1 && cur >= 0) { path.push_back(cur); cur = parent[c][cur]; }
        std::reverse(path.begin(), path.end());
        return path;
    };

    auto component_size_from = [&](int start_id, int color_val)->int {
        if (start_id < 0 || start_id >= GRID_N) return 0;
        auto id_to_xyz = [&](int id)->std::array<int,3>{
            int z = (dim==3 ? id / (SX*SY) : 0);
            int rem = id - z*(SX*SY);
            int y = (dim>=2 ? rem / SX : 0);
            int x = rem - y*SX;
            return {x,y,z};
        };
        auto xyz_to_id = [&](int x,int y,int z)->int { return lin_index(x,y,z); };

        std::vector<char> vis(GRID_N, 0);
        std::vector<int> q; q.reserve(1024);

        if (dim==2) {
            auto xyz = id_to_xyz(start_id);
            if (net.get({xyz[0], xyz[1]}) != color_val) return 0;
        } else {
            auto xyz = id_to_xyz(start_id);
            if (net.get({xyz[0], xyz[1], xyz[2]}) != color_val) return 0;
        }

        vis[start_id] = 1; q.push_back(start_id);
        int comp=0; std::vector<int> deltas = {-1, 1};

        for (size_t head=0; head<q.size(); ++head) {
            int u = q[head]; ++comp; auto xyz = id_to_xyz(u);
            for (int ax=0; ax<dim; ++ax){
                for (int d : deltas){
                    int nx=xyz[0], ny=(dim>=2?xyz[1]:0), nz=(dim==3?xyz[2]:0);
                    if (ax==0) nx += d; else if (ax==1) ny += d; else nz += d;

                    std::vector<int> v = (dim==2) ? std::vector<int>{nx, ny}
                                                  : std::vector<int>{nx, ny, nz};
                    if (!valid_coord(v)) continue;
                    nx=v[0]; ny=(dim>=2?v[1]:0); nz=(dim==3?v[2]:0);

                    int vid = xyz_to_id(nx,ny,nz);
                    if (vis[vid]) continue;

                    int val = (dim==2) ? net.get({nx,ny}) : net.get({nx,ny,nz});
                    if (val != color_val) continue;

                    vis[vid] = 1; q.push_back(vid);
                }
            }
        }
        return comp;
    };

    auto commit_step = [&](int t_k, const std::vector<double>& p_vec, const std::vector<int>& Nt_vec)
    {
        t_list.push_back(t_k);
        
        for (int c=0;c<num_colors;++c){
            p_series[c].push_back(p_vec[c]);
            Nt_series[c].push_back(Nt_vec[c]);
        }
    };

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
                parent[c][id0] = -1;
            }
            ++tries;
        }
    }

    std::vector<bool> percolated(num_colors, false);
    std::vector<int>  t_percolated(num_colors, -1);
    int order_counter = 0;

    ps_out.sp_len.assign(num_colors, -1);
    ps_out.sp_path_lin.assign(num_colors, {});
    std::vector<int> M_size_at_perc(num_colors, 0);

    commit_step(0, p_curr, N_current);
    print_status(0, p_curr, N_current);
    
    std::vector<std::vector<int>> neighbor_buffer(2*dim, std::vector<int>(dim));

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
                    if (vv >= 0) continue;

                    const bool same_color = (num_colors==1) || (vv == -(cor_idx+2));
                    const bool no_color   = (vv == -1);
                    if (!same_color && !no_color) continue;

                    double r = rng.uniform_real(0.0, 1.0);

                    auto try_activate = [&](){
                        net.set(viz, new_val);
                        new_borderland.push(viz);
                        ++N_current[cor_idx];

                        int x=(dim>=1?viz[0]:0), y=(dim>=2?viz[1]:0), z=(dim==3?viz[2]:0);
                        int id_new = lin_index(x,y,z);

                        int px=(dim>=1?pos[0]:0), py=(dim>=2?pos[1]:0), pz=(dim==3?pos[2]:0);
                        int id_pos = lin_index(px,py,pz);
                        parent[cor_idx][id_new] = id_pos;

                        if (!percolated[cor_idx] && get_gcoord(viz) == (L-1)) {
                            percolated[cor_idx]   = true;
                            t_percolated[cor_idx] = t;
                            ++order_counter;

                            auto path = rebuild_path_from_parent(cor_idx, id_new);
                            ps_out.sp_len[cor_idx] = (int)path.size();
                            ps_out.sp_path_lin[cor_idx] = std::move(path);

                            ps_out.color_percolation.push_back(cor_idx + 1);
                            ps_out.percolation_order.push_back(order_counter);

                            M_size_at_perc[cor_idx] = component_size_from(id_new, new_val);
                        }
                    };

                    if (!is_bond){
                        if (r < p_curr[cor_idx]) { try_activate(); }
                        else { net.set(viz, 0); }
                    } else {
                        if (r < p_curr[cor_idx]) { try_activate(); }
                        else { /* ligação fechada */ }
                    }
                }
            }
        }

        const int grown_total = std::accumulate(N_current.begin(), N_current.end(), 0);
        if (grown_total == 0) break;

        borderland = std::move(new_borderland);

        std::vector<double> p_next(num_colors);
        for (int c=0;c<num_colors;++c)
            p_next[c] = generate_p(type_N_t, p_curr[c], t, N_current[c], k, a, alpha);

        commit_step(t, p_next, N_current);
        if (t < 10 || (t % 100) == 0) {
           print_status(t, p_next, N_current);
        }
        if (std::all_of(percolated.begin(), percolated.end(), [](bool x){ return x; })) break;

        p_curr.swap(p_next);
    }

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







