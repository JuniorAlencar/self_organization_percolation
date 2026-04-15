#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <cstring>

#include "struct_network.hpp"
#include "helpers_print.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// ---------- helpers de impressão ----------
void helpers::print_base_summary(const NetworkPattern& net) {
    const int dim = net.dim;
    const int grow_axis = dim - 1;
    const int Lx = net.shape[0];
    const int Ly = (dim >= 2 ? net.shape[1] : 1);
    const int Lz = (dim == 3 ? net.shape[2] : 1);

    auto lin_index = [&](int x, int y, int z)->int {
        return x + Lx*(y + Ly*z);
    };
    auto get = [&](int x, int y, int z)->int {
        if (dim == 2) return net.get(lin_index(x, y, 0));
        return net.get(lin_index(x, y, z));
    };

    int base_size = 1;
    for (int ax = 0; ax < dim; ++ax) {
        if (ax != grow_axis) base_size *= net.shape[ax];
    }

    std::vector<long long> neg_color_counts(net.num_colors, 0);
    std::vector<long long> pos_color_counts(net.num_colors, 0);
    long long gray_count = 0;
    long long pos_single_color = 0;

    if (dim == 2) {
        if (grow_axis == 1) {
            int y = 0;
            for (int x = 0; x < Lx; ++x) {
                int v = get(x, y, 0);
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
        } else {
            int x = 0;
            for (int y = 0; y < Ly; ++y) {
                int v = get(x, y, 0);
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
    } else {
        if (grow_axis == 2) {
            int z = 0;
            for (int y = 0; y < Ly; ++y) {
                for (int x = 0; x < Lx; ++x) {
                    int v = get(x, y, z);
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
        } else if (grow_axis == 1) {
            int y = 0;
            for (int z = 0; z < Lz; ++z) {
                for (int x = 0; x < Lx; ++x) {
                    int v = get(x, y, z);
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
        } else {
            int x = 0;
            for (int z = 0; z < Lz; ++z) {
                for (int y = 0; y < Ly; ++y) {
                    int v = get(x, y, z);
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
    }

    std::cout << "\n=== Base summary ===\n";
    std::cout << "dim = " << dim << "  shape = {";
    for (size_t i = 0; i < net.shape.size(); ++i) {
        std::cout << net.shape[i] << (i + 1 < net.shape.size() ? ", " : "");
    }
    std::cout << "}  grow_axis=" << grow_axis << "\n";
    std::cout << "base_size = " << base_size << "\n";

    if (net.num_colors <= 1) {
        std::cout << "gray (-1) on base : " << gray_count << "\n";
        std::cout << "active (+1) on base: " << pos_single_color << "\n";
    } else {
        for (int c = 0; c < net.num_colors; ++c) {
            std::cout << "color c=" << c + 1
                      << "  neg label=-(c+2)=" << -(c + 2)
                      << "  count(on base) = " << neg_color_counts[c] << "\n";
        }
        std::cout << "gray (-1) on base : " << gray_count << "\n";
        for (int c = 0; c < net.num_colors; ++c) {
            std::cout << "ACTIVE +(c+2)=" << (c + 2)
                      << " on base      : " << pos_color_counts[c] << "\n";
        }
    }
    std::cout << "====================\n";
}

void helpers::print_slice(const NetworkPattern& net, int g_level, int max_w) {
    const int dim = net.dim;
    const int grow_axis = dim - 1;
    const int Lx = net.shape[0];
    const int Ly = (dim >= 2 ? net.shape[1] : 1);
    const int Lz = (dim == 3 ? net.shape[2] : 1);

    auto lin_index = [&](int x, int y, int z)->int {
        return x + Lx*(y + Ly*z);
    };

    auto get = [&](int x, int y, int z)->int {
        if (dim == 2) return net.get(lin_index(x, y, 0));
        return net.get(lin_index(x, y, z));
    };

    std::cout << "\n=== Slice at grow_axis level " << g_level << " ===\n";

    if (dim == 2) {
        if (grow_axis == 1) {
            int y = g_level;
            for (int x = 0; x < Lx; ++x) {
                int v = get(x, y, 0);
                char ch = (v == -1 ? '.' : (v == 0 ? 'o' : (v > 0 ? 'A' : 'a')));
                std::cout << ch;
                if (x + 1 >= max_w && Lx > max_w) { std::cout << " ..."; break; }
            }
        } else {
            int x = g_level;
            for (int y = 0; y < Ly; ++y) {
                int v = get(x, y, 0);
                char ch = (v == -1 ? '.' : (v == 0 ? 'o' : (v > 0 ? 'A' : 'a')));
                std::cout << ch;
                if (y + 1 >= max_w && Ly > max_w) { std::cout << " ..."; break; }
            }
        }
        std::cout << "\n";
    } else {
        if (grow_axis == 2) {
            int z = g_level;
            for (int y = 0; y < Ly; ++y) {
                for (int x = 0; x < Lx; ++x) {
                    int v = get(x, y, z);
                    char ch = (v == -1 ? '.' : (v == 0 ? 'o' : (v > 0 ? 'A' : 'a')));
                    std::cout << ch;
                    if (x + 1 >= max_w && Lx > max_w) { std::cout << " ..."; break; }
                }
                std::cout << "\n";
            }
        } else if (grow_axis == 1) {
            int y = g_level;
            for (int z = 0; z < Lz; ++z) {
                for (int x = 0; x < Lx; ++x) {
                    int v = get(x, y, z);
                    char ch = (v == -1 ? '.' : (v == 0 ? 'o' : (v > 0 ? 'A' : 'a')));
                    std::cout << ch;
                    if (x + 1 >= max_w && Lx > max_w) { std::cout << " ..."; break; }
                }
                std::cout << "\n";
            }
        } else {
            int x = g_level;
            for (int z = 0; z < Lz; ++z) {
                for (int y = 0; y < Ly; ++y) {
                    int v = get(x, y, z);
                    char ch = (v == -1 ? '.' : (v == 0 ? 'o' : (v > 0 ? 'A' : 'a')));
                    std::cout << ch;
                    if (y + 1 >= max_w && Ly > max_w) { std::cout << " ..."; break; }
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << "==============================\n";
}

// ---------- helpers gerais do executável ----------
void helpers::print_help(const char* prog) {
    std::cout <<
R"(To run:
  ./SOP <L> <p0> <seed> <type_percolation> <k> <N_t> <dim> <num_colors> <rho_val> <P0> <Equilibration>

Arguments:
  L                : Length of network (int)
  p0               : Initial Density (double)
  seed             : -1 to random seed (int)
  type_percolation : bond or node (string)
  k                : Check article (double)
  N_t              : Check article (int)
  dim              : Dimension of Network (2 or 3)
  num_colors       : Number of colors in network >= 1 (int)
  rho_val          : Density for each color (double)  [IMPORTANT => num_colors * rho_val <= 1.0]
  P0               : Fraction of active nodes in base [0 < P0 <= 1.0]
  Equilibration    : Return network with encoded time activation ['true' or 'false']
Examples:
  ./SOP 2000 1.0 -1 bond 1.0e-04 200 2 1 1.0 0.1 true
  ./SOP  500  0.05 42 node 1.0e-04 100 3 3 0.25 0.5 false

Tips:
  - Use seed = -1 to auto-generate a random seed.
  - 'bond' vs 'node' picks percolation type.
  - Check the article for recommended ranges of k and N_t.
)" << std::endl;
}

std::string helpers::sanitize_for_filename(std::string s) {
    for (char& c : s) {
        const bool ok =
            std::isalnum(static_cast<unsigned char>(c)) ||
            c == '-' || c == '_';
        if (!ok) c = '_';
    }
    return s;
}

std::string helpers::get_machine_name() {
#ifdef _WIN32
    char buffer[MAX_COMPUTERNAME_LENGTH + 1];
    DWORD size = sizeof(buffer);
    if (GetComputerNameA(buffer, &size)) {
        return sanitize_for_filename(std::string(buffer, size));
    }
    return "unknown_host";
#else
    char buffer[256];
    if (gethostname(buffer, sizeof(buffer)) == 0) {
        buffer[sizeof(buffer) - 1] = '\0';
        return sanitize_for_filename(std::string(buffer));
    }
    return "unknown_host";
#endif
}

std::string helpers::get_timestamp_now() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t tt = system_clock::to_time_t(now);

    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%dT%H%M%S");
    return oss.str();
}

bool helpers::is_help_token(const char* s) {
    return (std::strcmp(s, "!help") == 0 ||
            std::strcmp(s, "--help") == 0 ||
            std::strcmp(s, "-h") == 0);
}

void helpers::print_version() {
#ifdef SOP_VERSION
    std::cout << "SOP version " << SOP_VERSION << std::endl;
#else
    std::cout << "SOP version (unknown)" << std::endl;
#endif
}

bool helpers::parse_bool(const std::string& s) {
    std::string x = s;
    std::transform(x.begin(), x.end(), x.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (x == "1" || x == "true" || x == "t" || x == "yes" || x == "y") return true;
    if (x == "0" || x == "false" || x == "f" || x == "no"  || x == "n") return false;

    throw std::runtime_error("bool inválido: " + s);
}