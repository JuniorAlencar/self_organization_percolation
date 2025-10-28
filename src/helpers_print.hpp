#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include "struct_network.hpp"

using namespace std;

struct helpers{
    public: 
        void print_base_summary(const NetworkPattern& net);
        void print_slice(const NetworkPattern& net, int g_level, int max_w=80);
};


#endif // HELPERS_HPP