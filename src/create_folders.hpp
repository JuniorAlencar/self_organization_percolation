#ifndef CREATE_FOLDERS_HPP
#define CREATE_FOLDERS_HPP

#include <string>
#include <tuple>
#include <filesystem>
#include <string>
#include <sstream>
#include <filesystem>
#include <cstdio>

class FolderCreator {
    public:
        FolderCreator(const std::string& base_path);

        std::tuple<std::string, std::string, std::string, std::string, std::string> create_structure(
            int dim,
            int type_f_T,
            double f_T,
            double c,
            int L,
            int n_colors,
            double a,
            double alpha,
            std::string type_percolation,
            double p0,
            double P0,
            double rho,
            bool teste = false,
            int height_stop_multiplier = 1
        );

    private:
        std::string base_path;
};

#endif // CREATE_FOLDERS_HPP
