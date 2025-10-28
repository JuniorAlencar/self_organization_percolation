#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include "struct_network.hpp"
#include "helpers_print.hpp"

// ---------- helpers de impressão ----------
void helpers::print_base_summary(const NetworkPattern& net) {
    const int dim = net.dim;
    const int grow_axis = dim - 1;
    const int Lx = net.shape[0];
    const int Ly = (dim >= 2 ? net.shape[1] : 1);
    const int Lz = (dim == 3 ? net.shape[2] : 1);

    auto get = [&](int x, int y, int z)->int {
        if (dim==2) return net.get({x,y});
        return net.get({x,y,z});
    };

    // Tamanho da base (eixo de crescimento = 0)
    int base_size = 1;
    for (int ax = 0; ax < dim; ++ax) if (ax != grow_axis) base_size *= net.shape[ax];

    std::vector<long long> neg_color_counts(net.num_colors, 0);
    std::vector<long long> pos_color_counts(net.num_colors, 0);
    long long gray_count = 0;
    long long pos_single_color = 0; // caso num_colors == 1 e valor +1

    if (dim == 2) {
        // base é linha/coluna com coord[grow_axis] = 0
        if (grow_axis == 1) {
            int y = 0;
            for (int x = 0; x < Lx; ++x) {
                int v = get(x,y,0);
                if (v == -1) ++gray_count;
                else if (v > 0) {
                    if (net.num_colors == 1) ++pos_single_color;
                    else {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++pos_color_counts[c];
                    }
                } else { // v < 0
                    if (net.num_colors > 1) {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++neg_color_counts[c];
                    }
                }
            }
        } else { // grow_axis == 0
            int x = 0;
            for (int y = 0; y < Ly; ++y) {
                int v = get(x,y,0);
                if (v == -1) ++gray_count;
                else if (v > 0) {
                    if (net.num_colors == 1) ++pos_single_color;
                    else {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++pos_color_counts[c];
                    }
                } else {
                    if (net.num_colors > 1) {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++neg_color_counts[c];
                    }
                }
            }
        }
    } else { // dim == 3
        if (grow_axis == 2) {
            int z = 0;
            for (int y = 0; y < Ly; ++y) for (int x = 0; x < Lx; ++x) {
                int v = get(x,y,z);
                if (v == -1) ++gray_count;
                else if (v > 0) {
                    if (net.num_colors == 1) ++pos_single_color;
                    else {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++pos_color_counts[c];
                    }
                } else {
                    if (net.num_colors > 1) {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++neg_color_counts[c];
                    }
                }
            }
        } else if (grow_axis == 1) {
            int y = 0;
            for (int z = 0; z < Lz; ++z) for (int x = 0; x < Lx; ++x) {
                int v = get(x,y,z);
                if (v == -1) ++gray_count;
                else if (v > 0) {
                    if (net.num_colors == 1) ++pos_single_color;
                    else {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++pos_color_counts[c];
                    }
                } else {
                    if (net.num_colors > 1) {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++neg_color_counts[c];
                    }
                }
            }
        } else { // grow_axis == 0
            int x = 0;
            for (int z = 0; z < Lz; ++z) for (int y = 0; y < Ly; ++y) {
                int v = get(x,y,z);
                if (v == -1) ++gray_count;
                else if (v > 0) {
                    if (net.num_colors == 1) ++pos_single_color;
                    else {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++pos_color_counts[c];
                    }
                } else {
                    if (net.num_colors > 1) {
                        int c = std::abs(v) - 2;
                        if (c >= 0 && c < net.num_colors) ++neg_color_counts[c];
                    }
                }
            }
        }
    }

    std::cout << "\n=== Base summary ===\n";
    std::cout << "dim = " << dim << "  shape = {";
    for (size_t i=0;i<net.shape.size();++i){
        std::cout << net.shape[i] << (i+1<net.shape.size()? ", ":"");
    }
    std::cout << "}  grow_axis=" << grow_axis << "\n";
    std::cout << "base_size = " << base_size << "\n";

    if (net.num_colors <= 1) {
        std::cout << "gray (-1) on base : " << gray_count << "\n";
        std::cout << "active (+1) on base: " << pos_single_color << "\n";
    } else {
        for (int c=0;c<net.num_colors;++c) {
            std::cout << "color c=" << c+1
                      << "  neg label=-(c+2)=" << -(c+2)
                      << "  count(on base) = " << neg_color_counts[c] << "\n";
        }
        std::cout << "gray (-1) on base : " << gray_count << "\n";
        for (int c=0;c<net.num_colors;++c) {
            std::cout << "ACTIVE +(c+2)=" << (c+2)
                      << " on base      : " << pos_color_counts[c] << "\n";
        }
    }
    std::cout << "====================\n";
}

void helpers::print_slice(const NetworkPattern& net, int g_level, int max_w) {
    // Mostra uma fatia 2D no nível g_level do eixo de crescimento (dim 2: imprime linha; dim 3: imprime matriz).
    const int dim = net.dim;
    const int grow_axis = dim - 1;
    const int Lx = net.shape[0];
    const int Ly = (dim >= 2 ? net.shape[1] : 1);
    const int Lz = (dim == 3 ? net.shape[2] : 1);

    auto get = [&](int x, int y, int z)->int {
        if (dim==2) return net.get({x,y});
        return net.get({x,y,z});
    };

    std::cout << "\n=== Slice at grow_axis level " << g_level << " ===\n";

    if (dim == 2) {
        // imprime linha/coluna (dependendo do grow_axis)
        if (grow_axis == 1) {
            int y = g_level;
            for (int x = 0; x < Lx; ++x) {
                int v = get(x,y,0);
                char ch = (v==-1?'.' : (v==0?'o' : (v>0?'A':'a'))); // rótulo simples
                std::cout << ch;
                if (x+1>=max_w && Lx>max_w) { std::cout << " ..."; break; }
            }
        } else { // grow_axis == 0
            int x = g_level;
            for (int y = 0; y < Ly; ++y) {
                int v = get(x,y,0);
                char ch = (v==-1?'.' : (v==0?'o' : (v>0?'A':'a')));
                std::cout << ch;
                if (y+1>=max_w && Ly>max_w) { std::cout << " ..."; break; }
            }
        }
        std::cout << "\n";
    } else {
        // imprime matriz 2D
        if (grow_axis == 2) {
            int z = g_level;
            for (int y = 0; y < Ly; ++y) {
                for (int x = 0; x < Lx; ++x) {
                    int v = get(x,y,z);
                    char ch = (v==-1?'.' : (v==0?'o' : (v>0?'A':'a')));
                    std::cout << ch;
                    if (x+1>=max_w && Lx>max_w) { std::cout << " ..."; break; }
                }
                std::cout << "\n";
            }
        } else if (grow_axis == 1) {
            int y = g_level;
            for (int z = 0; z < Lz; ++z) {
                for (int x = 0; x < Lx; ++x) {
                    int v = get(x,y,z);
                    char ch = (v==-1?'.' : (v==0?'o' : (v>0?'A':'a')));
                    std::cout << ch;
                    if (x+1>=max_w && Lx>max_w) { std::cout << " ..."; break; }
                }
                std::cout << "\n";
            }
        } else { // grow_axis == 0
            int x = g_level;
            for (int z = 0; z < Lz; ++z) {
                for (int y = 0; y < Ly; ++y) {
                    int v = get(x,y,z);
                    char ch = (v==-1?'.' : (v==0?'o' : (v>0?'A':'a')));
                    std::cout << ch;
                    if (y+1>=max_w && Ly>max_w) { std::cout << " ..."; break; }
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << "==============================\n";
}
