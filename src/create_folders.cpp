#include "create_folders.hpp"

namespace fs = std::filesystem;

FolderCreator::FolderCreator(const std::string& base_path)
    : base_path(base_path) {}

std::tuple<std::string, std::string, std::string, std::string, std::string> FolderCreator::create_structure(
    int dim,
    int type_f_T,
    double f_T,
    double c,
    int L,
    int n_colors,
    double a,
    double alpha,
    std::string type_percolation,
    double /* p0 */,
    double /* P0 */,
    double rho,
    bool teste,
    int height_stop_multiplier
    )
{
    char main_folder[256];
    std::string raw_folder = "raw";
    if (teste) {
        raw_folder = "raw_" + std::to_string(height_stop_multiplier) + "L_stop";
    }


    sprintf(main_folder, "%s/%s/%s_percolation/num_colors_%d/dim_%d/L_%d",
            base_path.c_str(), raw_folder.c_str(), type_percolation.c_str(), n_colors, dim, L);
        

    std::string full_path;

    if (type_f_T == 0) {
        char sub[512];
        sprintf(sub, "%s/fT_constant/fT_%.6e/c_%.6e/rho_%.4e",
                main_folder, f_T, c, rho);
        full_path = std::string(sub);
    } else {
        char sub[512];
        sprintf(sub, "%s/fT_variable/type_%d/a_%.2f/alpha_%.2f/c_%.6e/rho_%.4e",
                main_folder, type_f_T, a, alpha, c, rho);
        full_path = std::string(sub);
    }

    std::string network_path = full_path + "/network";
    std::string data_path = full_path + "/data";
    std::string data_path_equilibration = full_path + "/data_surfaces";
    std::string data_network_preteq = full_path + "/network_preteq";
    std::string data_network_posteq = full_path + "/network_posteq";
    
    fs::create_directories(network_path);
    fs::create_directories(data_path);
    fs::create_directories(data_path_equilibration);
    fs::create_directories(data_network_preteq);
    fs::create_directories(data_network_posteq);

    return {network_path, data_path, data_path_equilibration, data_network_preteq, data_network_posteq};
}


