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

const std::vector<int>& network::get_N_t() const {
    return N_t_list;
}

const std::vector<int>& network::get_t() const {
    return t_list;
}

// NetworkPattern network::create_network(const int dim, const int lenght_network, const int num_of_samples,
//                                        const double k, const double N_t, const int seed, const int type_N_t,
//                                        const double p0, const double P0, const double a, const double alpha,
//                                        const std::string& type_percolation) {
//     this->N_t = N_t;

//     std::vector<int> shape;
//     if (dim == 2)
//         shape = {num_of_samples, lenght_network};
//     else if (dim == 3)
//         shape = {num_of_samples, lenght_network, lenght_network};
//     else
//         throw std::invalid_argument("Only dim=2 or dim=3 are supported.");

//     const size_t total_sites = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
//     NetworkPattern net(dim, shape);
//     all_random rng(seed);

//     // Initial probability
//     p.clear();
//     p.push_back(p0);

//     t_list.clear();
//     N_t_list.clear();

//     std::vector<uint8_t> visited(total_sites, 0);

//     std::fill(net.data.begin(), net.data.end(), 0);

//     // Adjust active seed count based on dimension
//     int active_count = 0;

//     if (dim == 3)
//         active_count = static_cast<int>(pow(lenght_network * P0, 2)); // agora P0 = número absoluto
//     else if (dim == 2)
//         active_count = static_cast<int>(P0 * lenght_network);


//     std::queue<std::vector<int>> fronteira;
//     if (dim == 2) {
//         std::vector<int> indices(lenght_network);
//         std::iota(indices.begin(), indices.end(), 0);
//         std::shuffle(indices.begin(), indices.end(), rng.get_gen());
//         for (int i = 0; i < active_count; ++i) {
//             std::vector<int> coords(dim, 0);
//             coords[1] = indices[i];
//             size_t idx = net.to_index(coords);
//             net.data[idx] = 1;
//             visited[idx] = 1;
//             fronteira.push(std::move(coords));
//         }
//     } else if (dim == 3) {
//         std::vector<std::pair<int, int>> base_coords;
//         for (int i = 0; i < lenght_network; ++i) {
//             for (int j = 0; j < lenght_network; ++j) {
//                 base_coords.emplace_back(i, j);
//             }
//         }
//         std::shuffle(base_coords.begin(), base_coords.end(), rng.get_gen());
//         for (int i = 0; i < active_count; ++i) {
//             std::vector<int> coords = {0, base_coords[i].first, base_coords[i].second};
//             size_t idx = net.to_index(coords);
//             net.data[idx] = 1;
//             visited[idx] = 1;
//             fronteira.push(std::move(coords));
//         }
//     }

//     t_list.push_back(0);
//     N_t_list.push_back(active_count);

//     std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

//     for (int t = 1; t < num_of_samples; ++t) {
//         int N_current = 0;
//         std::queue<std::vector<int>> nova_fronteira;

//         while (!fronteira.empty()) {
//             const std::vector<int>& pos = fronteira.front();

//             int v_idx = 0;
//             for (int d = 0; d < dim; ++d) {
//                 for (int delta : {-1, 1}) {
//                     std::vector<int>& viz = neighbor_buffer[v_idx++];
//                     viz = pos;
//                     viz[d] += delta;

//                     bool valid = true;
//                     for (int i = 0; i < dim; ++i) {
//                         if (viz[i] < 0 || viz[i] >= shape[i]) {
//                             valid = false;
//                             break;
//                         }
//                     }

//                     if (!valid) continue;

//                     size_t idx = net.to_index(viz);
//                     if (visited[idx]) continue;

//                     double r = rng.uniform_real(0.0, 1.0);
//                     if (type_percolation == "node") {
//                         if (r < p[t - 1]) {
//                             net.data[idx] = 1;
//                             nova_fronteira.push(viz);
//                             N_current++;
//                         } else {
//                             net.data[idx] = -1;
//                         }
//                         visited[idx] = 1;
//                     } else if (type_percolation == "bond") {
//                         if (net.data[idx] == 0 && r < p[t - 1]) {
//                             net.data[idx] = 1;
//                             nova_fronteira.push(viz);
//                             N_current++;
//                             visited[idx] = 1;
//                         }
//                     }
//                 }
//             }

//             fronteira.pop();
//         }

//         if (nova_fronteira.empty()) break;

//         fronteira = std::move(nova_fronteira);
//         double p_next = generate_p(type_N_t, p[t - 1], t, N_current, k, a, alpha);
//         p.push_back(p_next);

//         t_list.push_back(t);
//         N_t_list.push_back(N_current);

//         if (t < 10 || t % 100 == 0)
//             std::cout << "[" << type_percolation << "] t = " << t << ", p(t) = " << p[t] << ", N(t) = " << N_current << "\n";
//     }

//     if (type_percolation == "bond") {
//         for (int& val : net.data) {
//             if (val == 0) val = -1;
//         }
//     }

//     return net;
// }

NetworkPattern network::create_network(const int dim, const int lenght_network, const int num_of_samples,
                                       const double k, const double N_t, const int seed, const int type_N_t,
                                       const double p0, const double P0, const double a, const double alpha,
                                       const std::string& type_percolation) {
    this->N_t = N_t;

    std::vector<int> shape;
    if (dim == 2)
        shape = {num_of_samples, lenght_network};
    else if (dim == 3)
        shape = {num_of_samples, lenght_network, lenght_network};
    else
        throw std::invalid_argument("Only dim=2 or dim=3 are supported.");

    const size_t total_sites = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    NetworkPattern net(dim, shape);
    all_random rng(seed);

    p.clear();
    p.push_back(p0);

    t_list.clear();
    N_t_list.clear();

    std::vector<uint8_t> visited(total_sites, 0);
    std::fill(net.data.begin(), net.data.end(), 0);

    int active_count = 0;
    if (dim == 3)
        active_count = static_cast<int>(pow(lenght_network * P0, 2)); // P0 como valor absoluto
    else if (dim == 2)
        active_count = static_cast<int>(P0 * lenght_network);

    std::queue<std::vector<int>> fronteira;
    if (dim == 2) {
        std::vector<int> indices(lenght_network);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng.get_gen());
        for (int i = 0; i < active_count; ++i) {
            std::vector<int> coords(dim, 0);
            coords[1] = indices[i];
            size_t idx = net.to_index(coords);
            net.data[idx] = 1;
            visited[idx] = 1;
            fronteira.push(std::move(coords));
        }
    } else if (dim == 3) {
        std::vector<std::pair<int, int>> base_coords;
        for (int i = 0; i < lenght_network; ++i) {
            for (int j = 0; j < lenght_network; ++j) {
                base_coords.emplace_back(i, j);
            }
        }
        std::shuffle(base_coords.begin(), base_coords.end(), rng.get_gen());
        for (int i = 0; i < active_count; ++i) {
            std::vector<int> coords = {0, base_coords[i].first, base_coords[i].second};
            size_t idx = net.to_index(coords);
            net.data[idx] = 1;
            visited[idx] = 1;
            fronteira.push(std::move(coords));
        }
    }

    t_list.push_back(0);
    N_t_list.push_back(active_count);

    std::vector<std::vector<int>> neighbor_buffer(2 * dim, std::vector<int>(dim));

    for (int t = 1; t < num_of_samples; ++t) {
        int N_current = 0;
        std::queue<std::vector<int>> nova_fronteira;

        while (!fronteira.empty()) {
            const std::vector<int>& pos = fronteira.front();

            int v_idx = 0;
            for (int d = 0; d < dim; ++d) {
                for (int delta : {-1, 1}) {
                    std::vector<int>& viz = neighbor_buffer[v_idx++];
                    viz = pos;
                    viz[d] += delta;

                    bool valid = true;
                    for (int i = 0; i < dim; ++i) {
                        if (i == 0) {  // direção temporal: sem contorno periódico
                            if (viz[i] < 0 || viz[i] >= shape[i]) {
                                valid = false;
                                break;
                            }
                        } else {
                            if (viz[i] < 0)
                                viz[i] = shape[i] - 1;
                            else if (viz[i] >= shape[i])
                                viz[i] = 0;
                        }
                    }

                    if (!valid) continue;

                    size_t idx = net.to_index(viz);
                    if (visited[idx]) continue;

                    double r = rng.uniform_real(0.0, 1.0);
                    if (type_percolation == "node") {
                        if (r < p[t - 1]) {
                            net.data[idx] = 1;
                            nova_fronteira.push(viz);
                            N_current++;
                        } else {
                            net.data[idx] = -1;
                        }
                        visited[idx] = 1;
                    } else if (type_percolation == "bond") {
                        if (net.data[idx] == 0 && r < p[t - 1]) {
                            net.data[idx] = 1;
                            nova_fronteira.push(viz);
                            N_current++;
                            visited[idx] = 1;
                        }
                    }
                }
            }

            fronteira.pop();
        }

        if (nova_fronteira.empty()) break;

        fronteira = std::move(nova_fronteira);
        double p_next = generate_p(type_N_t, p[t - 1], t, N_current, k, a, alpha);
        p.push_back(p_next);

        t_list.push_back(t);
        N_t_list.push_back(N_current);

        if (t < 10 || t % 100 == 0)
            std::cout << "[" << type_percolation << "] t = " << t << ", p(t) = " << p[t] << ", N(t) = " << N_current << "\n";
    }

    if (type_percolation == "bond") {
        for (int& val : net.data) {
            if (val == 0) val = -1;
        }
    }

    return net;
}






