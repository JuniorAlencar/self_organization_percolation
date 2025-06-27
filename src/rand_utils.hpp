#ifndef rand_utils_hpp
#define rand_utils_hpp

#include "network.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <ctime>
#include <cmath>
#include <random>


// Defining clip template for double variable
template<typename T>
T clip(T value, T min_val, T max_val) {
    if (value < min_val) {
        return min_val;
    } else if (value > max_val) {
        return max_val;
    } else {
        return value;
    }
}

#ifndef RAND_UTILS_HPP
#define RAND_UTILS_HPP

#include <boost/random.hpp>
#include <random>
#include <ctime>

class all_random {
private:
    boost::random::mt19937 gen;

public:
    // Construtor com seed fornecida
    explicit all_random(int seed) {
        if (seed > 0) {
            gen.seed(seed);
        } else {
            // Seed aleatória com base no dispositivo
            std::random_device rd;
            boost::mt19937 Gen(rd());
            boost::random::uniform_int_distribution<int> dist(1, 2147483647);
            int now = dist(Gen);
            gen.seed(now);
        }
    }

    // Acesso controlado ao gerador (para uso no std::shuffle, por exemplo)
    boost::random::mt19937& get_gen() { return gen; }

    // Gera inteiro uniforme entre min e max (inclusive)
    int uniform_int(int min, int max);

    // Gera número real uniforme entre min e max
    double uniform_real(const double min, const double max);
};

#endif



#endif // rand_utils_hpp