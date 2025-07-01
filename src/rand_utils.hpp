#ifndef RAND_UTILS_HPP
#define RAND_UTILS_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <random>
#include <limits>
#include <ctime>

// Template function to clip a value within a specified [min, max] range
template<typename T>
T clip(T value, T min_val, T max_val) {
    if (value < min_val) return min_val;
    else if (value > max_val) return max_val;
    else return value;
}

// Class that encapsulates random number generation functionality
class all_random {
private:
    // Mersenne Twister generator from Boost
    boost::random::mt19937 gen;

public:
    // Constructor using a fixed seed.
    // If seed < 0, the generator will still be initialized, but it's expected
    // that the caller uses generate_random_seed() to get a valid seed.
    explicit all_random(int seed);

    // Provides reference to the internal generator (useful for std::shuffle, etc.)
    boost::random::mt19937& get_gen();

    // Returns a uniformly distributed integer between [min, max]
    int uniform_int(int min, int max);

    // Returns a uniformly distributed double between [min, max)
    double uniform_real(const double min, const double max);

    // Static utility function that generates a random seed across the full range of int
    static int generate_random_seed();
};

#endif // RAND_UTILS_HPP
