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

const std::vector<double>& network::get_p() const {
        return p;
}

<<<<<<< HEAD
const std::vector<int>& network::get_N_t() const {
    return N_t_list;
}

const std::vector<int>& network::get_t() const {
    return t_list;
}
=======

#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <numeric>
#include <algorithm>
#include <iostream>
>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c

NetworkPattern network::create_network(const int dim, const int lenght_network, const int num_of_samples,
                                       const double k, const double N_t, const int seed, const int type_N_t,
                                       const double p0, const double P0, const double a, const double alpha,
                                       const std::string& type_percolation) {
<<<<<<< HEAD
=======

>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c
    this->N_t = N_t;
    NetworkPattern net(dim, {num_of_samples, lenght_network});
    all_random rng(seed);

<<<<<<< HEAD
    // Inicialização de p(t)
=======
>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c
    p.clear();
    p.resize(num_of_samples);
    p[0] = p0;

<<<<<<< HEAD
    // Limpa os vetores associados aos membros da classe
    t_list.clear();
    N_t_list.clear();

    auto to_hash = [=](int x, int y) {
        return static_cast<long long>(x) * lenght_network + y;
    };

    std::unordered_set<long long> visitados;

    // Inicializa a matriz com 0 (não checado)
    for (int i = 0; i < num_of_samples; ++i) {
        for (int j = 0; j < lenght_network; ++j) {
            net.set({i, j}, 0);
        }
    }

    // Inicializa sementes
    int active_count = static_cast<int>(P0 * lenght_network);
    std::vector<int> indices(lenght_network);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng.get_gen());

    std::queue<std::pair<int, int>> fronteira;
    for (int i = 0; i < active_count; ++i) {
        int col = indices[i];
        net.set({0, col}, 1);
        fronteira.push({0, col});
        visitados.insert(to_hash(0, col));
    }

    // ✅ Adiciona t = 0 e os valores correspondentes
    t_list.push_back(0);
    N_t_list.push_back(active_count);

=======
    std::unordered_set<long long> visitados;
    auto to_hash = [=](int x, int y) {
        return static_cast<long long>(x) * lenght_network + y;
    };

    // Inicializa sementes
    int active_count = static_cast<int>(P0 * lenght_network);
    std::vector<int> indices(lenght_network);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng.get_gen());

    std::queue<std::pair<int, int>> fronteira;
    for (int i = 0; i < active_count; ++i) {
        int col = indices[i];
        net.set({0, col}, 1);
        fronteira.push({0, col});
        visitados.insert(to_hash(0, col));
    }

>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c
    for (int t = 1; t < num_of_samples; ++t) {
        int N_current = 0;
        std::queue<std::pair<int, int>> nova_fronteira;

        while (!fronteira.empty()) {
            auto [x, y] = fronteira.front(); fronteira.pop();

<<<<<<< HEAD
            std::vector<std::pair<int, int>> candidatos = {
                {x + 1, y}, {x - 1, y}, {x, y - 1}, {x, y + 1}
            };
=======
            std::vector<std::pair<int, int>> candidatos;

            if (type_percolation == "bond") {
                // vertical
                candidatos.push_back({x + 1, y});
                // lateral
                candidatos.push_back({x, y - 1});
                candidatos.push_back({x, y + 1});
            } else {
                // site percolation - apenas posição abaixo
                candidatos.push_back({x + 1, y});
            }
>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c

            for (auto& [nx, ny] : candidatos) {
                if (nx < 0 || nx >= num_of_samples || ny < 0 || ny >= lenght_network) continue;
                long long h = to_hash(nx, ny);
                if (visitados.count(h)) continue;

                double r = rng.uniform_real(0.0, 1.0);
<<<<<<< HEAD

                if (type_percolation == "node") {
                    if (r < p[t - 1]) {
                        net.set({nx, ny}, 1);
                        nova_fronteira.push({nx, ny});
                        N_current++;
                    } else {
                        net.set({nx, ny}, -1);
                    }
                    visitados.insert(h);
                } else if (type_percolation == "bond") {
                    // só tenta ativar se ainda não foi ativado
                    if (net.get({nx, ny}) == 0) {
                        if (r < p[t - 1]) {
                            net.set({nx, ny}, 1);             // ativa o sítio
                            nova_fronteira.push({nx, ny});    // novo sítio ativo para próxima iteração
                            N_current++;
                            visitados.insert(h);              // marca como visitado APENAS SE ATIVADO
                        }
                        // do contrário: não faz nada; o sítio ainda pode ser acessado por outro caminho
                    }
=======
                if (r < p[t - 1]) {
                    net.set({nx, ny}, 1);
                    nova_fronteira.push({nx, ny});
                    N_current++;
>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c
                }
                visitados.insert(h);
            }
        }

        fronteira = nova_fronteira;
        double p_next = generate_p(type_N_t, p[t - 1], t, N_current, k, a, alpha);
        p[t] = p_next;

<<<<<<< HEAD
        t_list.push_back(t);
        N_t_list.push_back(N_current);

        if (t < 10 || t % 100 == 0)
            std::cout << "[" << type_percolation << "] t = " << t << ", p(t) = " << p[t] << ", N(t) = " << N_current << "\n";

=======
        if (t < 10 || t % 100 == 0)
            std::cout << "[" << type_percolation << "] t = " << t << ", p(t) = " << p[t] << ", N(t) = " << N_current << "\n";

>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c
        if (fronteira.empty()) break;
    }

    return net;
}



<<<<<<< HEAD
=======

>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c
