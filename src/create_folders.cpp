#include "create_folders.hpp"
#include <filesystem>
#include <cstdio>

namespace fs = std::filesystem;

FolderCreator::FolderCreator(const std::string& base_path)
    : base_path(base_path) {}

std::tuple<std::string, std::string> FolderCreator::create_structure(
    int dim,
    int type_Nt,
    double N_t,
    double k,
    int L,
    int N_samples,
    double a,
    double alpha,
    double /* p0 */,
    double /* P0 */
) {
    char main_folder[256];

    // ❌ Removido: criação de pastas p0_/P0_
    sprintf(main_folder, "%s/dim_%d/L_%d_N_samples_%d", base_path.c_str(), dim, L, N_samples);

    std::string full_path;

    if (type_Nt == 0) {
        char sub[512];
        sprintf(sub, "%s/NT_constant/NT_%.0f/k_%.7f", main_folder, N_t, k);
        full_path = std::string(sub);
    } else {
        char sub[512];
        sprintf(sub, "%s/NT_variable/type_%d/a_%.2f/alpha_%.2f/k_%.7f",
                main_folder, type_Nt, a, alpha, k);
        full_path = std::string(sub);
    }

    std::string network_path = full_path + "/network";
    std::string pt_path = full_path + "/p_t";

    fs::create_directories(network_path);
    fs::create_directories(pt_path);

    return {network_path, pt_path};
}
