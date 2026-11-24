// Compile with -std=c++17 -O2 -Isrc tests/test_spanning_topology.cpp src/write_save.cpp ...
// Includes src/network.cpp directly to unit-test internal SlidingSlabGrid + compute_slab_stabilized.
#include "../src/network.cpp"
#include "write_save.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
    constexpr int L = 32;
    constexpr int bottom = 7;
    constexpr int top = bottom + L - 1;

    RawFractionsSeries output;
    output.L = L;
    output.dim = 2;
    output.num_colors = 1;
    output.collected_samples = 0;

    // 1. Single-species tests in 2D and 3D, node and bond modes + dynamic discard_below.
    for (int dim : {2, 3}) {
        for (bool is_node : {true, false}) {
            SlidingSlabGrid grid(dim, L, is_node);

            // Populate dummy discarded history layers [0, bottom - 1] and free them dynamically.
            for (int h = 0; h < bottom; ++h) {
                grid.set_site(h, grid.lat_index(0, 0), 1);
            }
            grid.discard_below(bottom, true);
            assert(grid.min_retained_h == bottom);
            assert(grid.get_site(bottom - 1, grid.lat_index(0, 0)) == 0);

            auto add_site = [&](int x, int y_lat, int h, int val = 1) {
                grid.set_site(h, grid.lat_index(x, (dim == 3) ? y_lat : 0), val);
            };
            auto connect_all_monochromatic = [&]() {
                if (is_node) return;
                for (int h = bottom; h <= top; ++h) {
                    for (std::size_t lat = 0; lat < grid.layer_sites; ++lat) {
                        const int v = grid.get_site(h, lat);
                        if (v <= 0) continue;
                        int x = 0, y_lat = 0;
                        grid.decode_lat(lat, x, y_lat);
                        const std::size_t lxp = grid.lat_index((x + 1) % L, y_lat);
                        if (grid.get_site(h, lxp) == v) grid.set_open_bond_dir(h, lat, 0);
                        if (dim == 3) {
                            const std::size_t lyp = grid.lat_index(x, (y_lat + 1) % L);
                            if (grid.get_site(h, lyp) == v) grid.set_open_bond_dir(h, lat, 1);
                        }
                        if (h < top && grid.get_site(h + 1, lat) == v) {
                            grid.set_open_bond_dir(h, lat, 2);
                        }
                    }
                }
            };

            // Two separate vertical spanning clusters:
            // - Cluster A at x=8 (plus (9,0,bottom)) -> mass L + 1 = 33
            // - Cluster B at x=0 -> mass L = 32
            for (int h = bottom; h <= top; ++h) {
                add_site(0, 0, h, 1);
                add_site(8, 0, h, 1);
            }
            add_site(9, 0, bottom, 1);

            // A much larger interior component touching neither bottom nor top -> MUST be excluded.
            for (int x = 15; x <= 20; ++x) {
                for (int h = bottom + 2; h <= top - 2; ++h) {
                    add_site(x, 0, h, 1);
                }
            }
            connect_all_monochromatic();

            auto res = compute_slab_stabilized(grid, 1, bottom, top, output.collected_samples++, 100);
            assert(res.num_spanning_clusters == 2);
            assert(res.largest_spanning_sites == L + 1);
            assert(res.largest_spanning_bonds == L);
            assert(res.largest_spanning_sp == L - 1);
            assert(res.largest_spanning_species == 1);
            assert(res.other_spanning_sites.size() == 1 && res.other_spanning_sites[0] == L);
            assert(res.other_spanning_bonds.size() == 1 && res.other_spanning_bonds[0] == L - 1);
            assert(res.other_spanning_sp.size() == 1 && res.other_spanning_sp[0] == L - 1);
            assert(res.largest_spanning_hull > 0);
            assert(res.largest_spanning_hull == res.largest_spanning_full_boundary);
            assert(res.interspecies_contact_bonds == 0);
            assert(res.interspecies_contact_sites == 0);
            assert(res.spanning_interspecies_bonds == 0);
            assert(res.spanning_interspecies_sites == 0);
            assert(res.largest_spanning_interspecies_bonds == 0);
            assert(res.other_spanning_interspecies_bonds.size() == 1 &&
                   res.other_spanning_interspecies_bonds[0] == 0);

            // Box counting and sphere yardstick counts must be OFF for all sizes.
            const auto& clusters = res.fractal_counts.spanning_clusters;
            assert(clusters.size() == 2);
            assert(clusters[0].largest_component_sites == L + 1);
            assert(clusters[1].largest_component_sites == L);
            for (const auto& c : clusters) {
                assert(c.component_box_counts.empty());
                assert(c.minimum_path_yardstick.defined);
                assert(c.minimum_path_yardstick.path_length == L - 1);
                assert(c.minimum_path_yardstick.counts.empty());
                assert(c.anchor_z == bottom);
                assert(c.slab_top_z == top);
            }
            output.fractal_counts.push_back(res.fractal_counts);

            // Break every spanning path at the top -> 0 spanning clusters.
            SlidingSlabGrid broken_grid(dim, L, is_node);
            for (int h = bottom; h <= (is_node ? top - 1 : top); ++h) {
                broken_grid.set_site(h, broken_grid.lat_index(0, 0), 1);
                broken_grid.set_site(h, broken_grid.lat_index(8, 0), 1);
            }
            if (!is_node) {
                for (int h = bottom; h < top - 1; ++h) {
                    broken_grid.set_open_bond_dir(h, broken_grid.lat_index(0, 0), 2);
                    broken_grid.set_open_bond_dir(h, broken_grid.lat_index(8, 0), 2);
                }
            }
            auto broken_res = compute_slab_stabilized(
                broken_grid, 1, bottom, top, output.collected_samples++, 101);
            assert(broken_res.num_spanning_clusters == 0);
            assert(broken_res.largest_spanning_sites == 0);
            assert(broken_res.largest_spanning_sp == -1);
            assert(broken_res.fractal_counts.spanning_clusters.empty());
            output.fractal_counts.push_back(broken_res.fractal_counts);
        }
    }

    // 2. Exterior Hull vs Full Boundary check with an enclosed hole/cavity in 2D (+4 edges) and 3D (+6 faces).
    {
        SlidingSlabGrid hole_grid(2, L, true);
        for (int h = bottom; h <= top; ++h) {
            hole_grid.set_site(h, hole_grid.lat_index(5, 0), 1);
        }
        // Build a 3x3 ring around empty center (6, bottom + 5):
        const int hc = bottom + 5;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dh = -1; dh <= 1; ++dh) {
                if (dx == 0 && dh == 0) continue; // enclosed empty hole at (6, hc) -> 4 interior edges
                hole_grid.set_site(hc + dh, hole_grid.lat_index(6 + dx, 0), 1);
            }
        }
        auto hole_res = compute_slab_stabilized(hole_grid, 1, bottom, top, 99, 200);
        assert(hole_res.num_spanning_clusters == 1);
        // Full boundary includes the 4 edges of the enclosed 1-site hole; exterior Hull excludes them!
        assert(hole_res.largest_spanning_full_boundary == hole_res.largest_spanning_hull + 4);

        SlidingSlabGrid hole_grid_3d(3, L, true);
        for (int h = bottom; h <= top; ++h) {
            hole_grid_3d.set_site(h, hole_grid_3d.lat_index(5, 6), 1);
        }
        // Build a 3x3x3 shell around empty center (7, 6, bottom + 5) touching x=5 at x=6:
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dh = -1; dh <= 1; ++dh) {
                    if (dx == 0 && dy == 0 && dh == 0) continue; // enclosed 1-site cavity -> 6 interior faces
                    hole_grid_3d.set_site(hc + dh, hole_grid_3d.lat_index(7 + dx, 6 + dy), 1);
                }
            }
        }
        auto hole_res_3d = compute_slab_stabilized(hole_grid_3d, 1, bottom, top, 100, 200);
        assert(hole_res_3d.num_spanning_clusters == 1);
        assert(hole_res_3d.largest_spanning_full_boundary == hole_res_3d.largest_spanning_hull + 6);
    }

    // 3. Two-species (num_colors == 2) strict monochromatic separation in 2D and 3D, node and bond.
    for (int dim : {2, 3}) {
        for (bool is_node : {true, false}) {
            SlidingSlabGrid grid2(dim, L, is_node);
            // Species 1 uses active_val = 2; Species 2 uses active_val = 3 (per color_to_active_value).
            const int val_sp1 = color_to_active_value(2, 0);
            const int val_sp2 = color_to_active_value(2, 1);
            assert(val_sp1 == 2 && val_sp2 == 3);

            // Place Species 1 column at x = 0 (mass L + 1 with extra site at x = L - 1, bottom)
            // and Species 2 column IMMEDIATELY ADJACENT at x = 1 (mass L).
            for (int h = bottom; h <= top; ++h) {
                grid2.set_site(h, grid2.lat_index(0, 0), val_sp1);
                grid2.set_site(h, grid2.lat_index(1, 0), val_sp2);
                if (!is_node && h < top) {
                    grid2.set_open_bond_dir(h, grid2.lat_index(0, 0), 2);
                    grid2.set_open_bond_dir(h, grid2.lat_index(1, 0), 2);
                }
            }
            grid2.set_site(bottom, grid2.lat_index(L - 1, 0), val_sp1);
            // Isolated non-spanning Species 2 site touching Species 1 column at (0, 0, bottom + 2):
            grid2.set_site(bottom + 2, grid2.lat_index(L - 1, 0), val_sp2);
            if (!is_node) {
                grid2.set_open_bond_dir(bottom, grid2.lat_index(L - 1, 0), 0);
                // Even if bond bit 0 at x=0 were set, different species must NEVER connect!
                grid2.set_open_bond_dir(bottom + 3, grid2.lat_index(0, 0), 0);
            }

            auto res2 = compute_slab_stabilized(grid2, 2, bottom, top, 200, 300);
            assert(res2.num_spanning_clusters == 2);
            assert(res2.largest_spanning_sites == L + 1);
            assert(res2.largest_spanning_species == 1);
            assert(res2.largest_spanning_sp == L - 1);
            assert(res2.other_spanning_sites.size() == 1 && res2.other_spanning_sites[0] == L);
            assert(res2.other_spanning_species.size() == 1 && res2.other_spanning_species[0] == 2);
            assert(res2.other_spanning_sp.size() == 1 && res2.other_spanning_sp[0] == L - 1);
            // Inter-species intersections:
            // - Spanning-to-spanning: L contacts between x=0 and x=1; 2*L touching spanning sites.
            assert(res2.spanning_interspecies_bonds == L);
            assert(res2.spanning_interspecies_sites == 2 * L);
            // - Total slab inter-species: L + 1 contacts (includes isolated (L-1, 0, bottom+2) <-> (0, 0, bottom+2))
            //   plus vertical contact (L-1, 0, bottom+1)? No, bottom and bottom+2 are 2 apart! So exact L + 1!
            assert(res2.interspecies_contact_bonds == L + 1);
            assert(res2.interspecies_contact_sites == 2 * L + 1);
            assert(res2.largest_spanning_interspecies_bonds == L + 1);
            assert(res2.other_spanning_interspecies_bonds.size() == 1 &&
                   res2.other_spanning_interspecies_bonds[0] == L);
            assert(res2.p_node_by_species.size() == 2);
            // Both species have L+1 sites total in the slab (Sp1: L+1 spanning; Sp2: L spanning + 1 isolated).
            assert(std::abs(res2.p_node_by_species[0] - res2.p_node_by_species[1]) < 1e-12);
        }
    }

    if (argc > 1) {
        save_data().save_fractal_counts_json(output, argv[1]);
    }
    std::cout << "All spanning topology tests (1/2 species, 2D/3D, node/bond, hull, SP) passed!\n";
    return 0;
}
