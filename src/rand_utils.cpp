#include "rand_utils.hpp"


double all_random::uniform_real(const double min, const double max) {
    boost::random::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

int all_random::uniform_int(const int min, const int max){
    boost::random::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}