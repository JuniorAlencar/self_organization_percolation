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
    std::string type_percolation,
    double /* p0 */,
    double /* P0 */
) {
    char main_folder[256];

    sprintf(main_folder, "%s/%s_percolation/dim_%d/L_%d_N_samples_%d", 
            base_path.c_str(), type_percolation.c_str(), dim, L, N_samples);

    std::string full_path;

    std::ostringstream oss;
    if (type_Nt == 0) {
        oss << main_folder
            << "/NT_constant/NT_" << std::fixed << std::setprecision(0) << N_t
            << "/k_" << std::scientific << std::setprecision(1) << k;
    } else {
        oss << main_folder
            << "/NT_variable/type_" << type_Nt
            << "/a_" << std::fixed << std::setprecision(2) << a
            << "/alpha_" << std::fixed << std::setprecision(2) << alpha
            << "/k_" << std::scientific << std::setprecision(1) << k;
    }


full_path = oss.str();


    std::string network_path = full_path + "/network";
    std::string pt_path = full_path + "/p_t";

    fs::create_directories(network_path);
    fs::create_directories(pt_path);

    return {network_path, pt_path};
}
