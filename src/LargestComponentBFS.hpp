#ifndef LargestComponentBFS_hpp
#define LargestComponentBFS_hpp

#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include "struct_network.hpp"

struct BiggestComponent {
public:
    void compute_shortest_paths_to_base(
        const NetworkPattern&               net,
        int                                 dim,
        const std::vector<int>&             shape,
        int                                 grow_axis,
        int                                 num_colors,
        const std::vector<int>&             parent,
        PercolationSeries&                  ps_out
    );

    std::vector<int> largest_cluster_sizes(
        const NetworkPattern&   net,
        int                     dim,
        const std::vector<int>& shape,
        int                     grow_axis,
        int                     num_colors
    );
};

#endif // LargestComponentBFS_hpp
