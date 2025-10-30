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
    std::vector<int>                M_size_at_perc(num_colors, 0);
    std::vector<int>                t_list; t_list.reserve(num_of_samples);
    std::vector<double>             p_curr = p0;

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
        for (int c = 0; c < num_colors; ++c) {
            long long q = std::llround(P0 * rho[c] * lenght_network);  // Corrigido para P0 * rho * L
            if (q < 0) q = 0;
            if (q > base_size) q = base_size;
            seeds_quota[c] = static_cast<int>(q);
        }
    }

    std::queue<std::vector<int>> borderland;
    std::vector<int> N_current(num_colors, 0);

    // Conta o tamanho do componente conectado a partir de start_id,
    // usando a API já existente para percolação.
    auto component_size_from = [&](int start_id)->int {
        const int N = GRID_N;
        if (start_id < 0 || start_id >= N) return 0;
        if (net.get(start_id) <= 0) return 0;

        std::vector<unsigned char> visited((size_t)N, 0); // N bytes
        std::vector<int> q;         q.reserve(1024);       // fila
        std::vector<int> neigh;     neigh.reserve(2 * dim);

        int comp = 0;
        visited[start_id] = 1;
        q.push_back(start_id);

        for (size_t head = 0; head < q.size(); ++head) {
            int u = q[head];
            ++comp;

            neigh.clear();          // IMPORTANTE: limpar antes de preencher
            for (int delta : {-1, 1}) {
                std::vector<int> pos = {u};  // Ajustar coordenadas para um índice válido
                if (valid_coord(pos)) {
                    int idx = lin_index(pos[0], pos[1], pos[2]); // Convertendo para índice linear
                    int v = net.get(idx);  // Usando índice linear para acessar o valor
                    if (v >= 0 && !visited[v]) {
                        visited[v] = 1;
                        q.push_back(v);
                    }
                }
            }
        }
        return comp;
    };

    // ===== semeadura (t=0) =====
    for (int c = 0; c < num_colors; ++c) {
        int activated = 0, tries = 0;
        const int max_tries = static_cast<int>(base_size) * 20;

        const int prefer_neg = (num_colors == 1 ? -1 : -(c + 2));
        const int active_val = (num_colors == 1 ? 1 : (c + 2));

        int n_color = static_cast<int>(rho[c] * lenght_network);
        int n_color_activated = static_cast<int>(P0 * n_color);

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

                parent[c][lin_index(coords[0], coords[1], coords[2])] = -1;  // Seed/root for shortest path
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
    commit_step(0, p_curr, N_current);

    // buffer vizinhos
    std::vector<std::vector<int>> neighbor_buffer(2*dim, std::vector<int>(dim));

    int max_height = 0;
    vector<int> max_heights(num_colors, 0);

    auto update_max_height = [&](const std::vector<int>& pos, int cor_idx) {
        int current_height = pos[grow_axis];  // A altura atual é dada pela coordenada no eixo de crescimento
        if (current_height > max_heights[cor_idx]) {
            max_heights[cor_idx] = current_height;  // Atualiza a maior altura para a cor específica
        }
    };

    bool all_top_reached = false;
    bool all_stuck = false;

    // evolução
    for (int t = 1; t < num_of_samples; ++t) {
        std::fill(N_current.begin(), N_current.end(), 0);
        std::queue<std::vector<int>> new_borderland;

        while (!borderland.empty()) {
            std::vector<int> pos = borderland.front();
            borderland.pop();
            int idx = lin_index(pos[0], pos[1], pos[2]);  // Convertendo coordenadas para índice linear
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

                    int viz_idx = lin_index(viz[0], viz[1], viz[2]);  // Convertendo coordenada para índice linear
                    int vv = net.get(viz_idx);  // Usando índice linear para acessar o valor
                    if (vv >= 0) continue;

                    double r = rng.uniform_real(0.0, 1.0);

                    auto try_activate = [&]() {
                        // Ativa o nó vizinho
                        net.set(viz_idx, new_val);
                        new_borderland.push(viz);
                        ++N_current[cor_idx];
                        update_max_height(viz, cor_idx);  // Atualiza a maior altura para a cor específica
                        parent[cor_idx][lin_index(viz[0], viz[1], viz[2])] = lin_index(pos[0], pos[1], pos[2]);  // Convertendo para índice linear
                    };

                    if (r < p_curr[cor_idx]) { 
                        try_activate(); 
                    }
                    else { 
                        net.set(vv, 0);  // Usando o índice linear aqui também
                    }
                }
            }
        }

        // Verificação de parada
        for (int c = 0; c < num_colors; ++c) {
            if (N_current[c] == 0)
                all_stuck = true;
            if (max_heights[c] == lenght_network - 1)
                all_top_reached = true;
        }

        if (!all_top_reached && !all_stuck) {
        // Continue a simulação
    } else {
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

    ps_out.M_size_at_perc = std::move(M_size_at_perc);

    ps_out.rho.clear();
    for (int i = 0; i < num_colors; ++i) {
        if (i < (int)rho.size()) ps_out.rho.push_back(rho[i]);
    }

    return net;
}





