#include "rand_utils.hpp"

// Constructor for the all_random class
// Initializes the generator using the provided seed
// If the seed is negative, it falls back to a fixed value (42) –
// but normally you should generate a seed before construction.
all_random::all_random(int seed) {
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        gen.seed(42);  // fallback; normally avoided
    }
}

// Returns reference to the internal generator
boost::random::mt19937& all_random::get_gen() {
    return gen;
}

// Generates a random integer uniformly distributed in [min, max]
int all_random::uniform_int(const int min, const int max) {
    boost::random::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}

// Generates a random double uniformly distributed in [min, max)
double all_random::uniform_real(const double min, const double max) {
    boost::random::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

// Static function that generates a random seed using the full range of int
// Uses std::random_device for entropy and std::mt19937 for fast generation
int all_random::generate_random_seed() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(
        1, 
        std::numeric_limits<int>::max()
    );
    return dist(gen);
}
