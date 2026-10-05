// Compile with -Isrc and build/libsop_core.a; includes internal topology helpers.
#include "../src/network.cpp"
#include "write_save.hpp"
#include <cassert>

int main(int argc, char** argv) {
    setenv("SOP_FRACTAL_FORCE_COUNTS", "1", 1);
    setenv("SOP_FRACTAL_ORIGINS", "2", 1);
    setenv("SOP_FRACTAL_NUM_OFFSETS", "2", 1);
    constexpr int L = 32, bottom = 7, top = bottom + L - 1;
    RawFractionsSeries output;
    output.L = L;
    for (int dim : {2, 3}) for (bool node : {true, false}) {
        std::unordered_map<std::uint64_t, std::int8_t, AbsSiteKeyHash> sites;
        std::unordered_set<AbsBondKey, AbsBondKeyHash> bonds;
        auto key = [&](int x, int h) { return dim == 2 ? abs_site_key(x, h, 0) : abs_site_key(x, 0, h); };
        auto add = [&](int x, int h) { sites[key(x, h)] = 1; };
        // Two separate vertical spanning clusters, masses 32 and 33.
        for (int h = bottom; h <= top; ++h) { add(0, h); add(8, h); }
        add(9, bottom);
        // A larger component touching neither boundary must be excluded.
        for (int x = 15; x <= 20; ++x)
            for (int h = bottom + 2; h <= top - 2; ++h) add(x, h);
        for (auto site : sites) {
            std::uint64_t neighbors[6]; int n = 0;
            collect_abs_neighbors(dim, L, site.first, neighbors, n);
            for (int i = 0; i < n; ++i)
                if (sites.count(neighbors[i])) bonds.insert(make_abs_bond_key(site.first, neighbors[i]));
        }
        auto result = compute_abs_slab_fractions(dim, L, sites, bonds, bottom, top, node, true, 44, 0, 100);
        assert(result.has_fractal_counts);
        auto& clusters = result.fractal_counts.spanning_clusters;
        assert(clusters.size() == 2);
        assert(clusters[0].largest_component_sites == L + 1);
        assert(clusters[1].largest_component_sites == L);
        for (auto& c : clusters) {
            assert(c.minimum_path_yardstick.defined);
            assert(c.minimum_path_yardstick.path_length == L - 1);
            assert(!c.component_box_counts.empty());
        }
        if (dim == 2) {
            std::vector<unsigned char> membership(L * L, 0);
            for (int h = bottom; h <= top; ++h) membership[(h - bottom) * L + 8] = 1;
            membership[9] = 1;
            auto traverse = [&](int x, int y, int nx, int ny) {
                return ny >= bottom && ny <= top && sites.count(key(nx, ny)) &&
                    (node || bonds.count(make_abs_bond_key(key(x, y), key(nx, ny))));
            };
            auto dense = compute_fractal_counts_2d(L, bottom, top, node, 44, 0, 100,
                sites.size(), bonds.size(), clusters[0].largest_component_bonds, membership, traverse);
            assert(dense.largest_component_sites == L + 1);
            assert(dense.minimum_path_yardstick.path_length == L - 1);
            assert(dense.component_box_counts.size() == clusters[0].component_box_counts.size());
            for (size_t i = 0; i < dense.component_box_counts.size(); ++i)
                assert(dense.component_box_counts[i].num_boxes == clusters[0].component_box_counts[i].num_boxes);
        }
        output.fractal_counts.push_back(result.fractal_counts);
        // Break every spanning path at the top (occupied sites remain for bond case).
        if (node) { sites.erase(key(0, top)); sites.erase(key(8, top)); }
        else { bonds.erase(make_abs_bond_key(key(0, top - 1), key(0, top)));
               bonds.erase(make_abs_bond_key(key(8, top - 1), key(8, top))); }
        result = compute_abs_slab_fractions(dim, L, sites, bonds, bottom, top, node, true, 44, 1, 101);
        assert(result.has_fractal_counts);
        assert(result.fractal_counts.spanning_clusters.empty());
        output.fractal_counts.push_back(result.fractal_counts);
    }
    if (argc > 1) save_data().save_fractal_counts_json(output, argv[1]);
}
