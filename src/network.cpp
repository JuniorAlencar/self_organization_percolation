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

NetworkPattern network::create_network(const int dim, const int lenght_network, const int num_of_samples, const double k, const double N_t, 
                                const int seed, const int type_N_t, const double p0, const double P0, const double a, const double alpha){

    
    this->N_t = N_t;
    // Create empty network (all values of 0) => num_of_samples:height, lenght_network:width
    NetworkPattern net(dim, {num_of_samples, lenght_network});

    // We use P0 to determine the fraction of sites that will be activated at the start of the simulation.

    // First activate ========================================

    // First activation (t = 0) ========================================

    // Pre-allocate p vector with size = num_of_samples
    p.resize(num_of_samples);

    // Set initial value p(0)
    p[0] = p0;

    // Determine how many active sites to initialize in the bottom row
    int active_count = static_cast<int>(P0 * lenght_network);

    // Generate a vector with column indices: [0, 1, ..., lenght_network - 1]
    std::vector<int> indices(lenght_network);
    std::iota(indices.begin(), indices.end(), 0);
    
    all_random rng(seed);
    
    // Shuffle the indices randomly using the RNG
    std::shuffle(indices.begin(), indices.end(), rng.get_gen());

    // Activate the first 'active_count' shuffled positions in the bottom row (row index 0)
    for (int i = 0; i < active_count; ++i) {
        int col = indices[i];
        net.set({0, col}, 1);  // Set site (0, col) as active
    }

    // Subsequent activations following p(t+1) ========================
    for (int t_i = 1; t_i < num_of_samples; ++t_i) {
        int N_current = 0;

        // Loop over each column in current row t_i
        for (int j = 0; j < lenght_network; ++j) {
            bool has_active_above = false;

            // Check if any of the three sites above (diagonal left, up, diagonal right) are active
            if (j > 0 && net.get({t_i - 1, j - 1}) == 1) has_active_above = true;
            if (net.get({t_i - 1, j}) == 1) has_active_above = true;
            if (j < lenght_network - 1 && net.get({t_i - 1, j + 1}) == 1) has_active_above = true;

            // If at least one site above is active, try to activate current site
            if (has_active_above) {
                double r = rng.uniform_real(0.0, 1.0);
                if (r < p[t_i - 1]) {
                    net.set({t_i, j}, 1);
                    N_current++;
                }
            }
        }

        // Compute p(t_i) using the SOP update rule
        double p_next = generate_p(type_N_t, p[t_i - 1], t_i, N_current, k, a, alpha);

        // Store p(t_i) directly at the corresponding position
        p[t_i] = p_next;
        if (t_i < 10) std::cout << "t = " << t_i << ", p(t) = " << p[t_i] << "\n";

    }

    
    // Return the initialized network
    return net;
}