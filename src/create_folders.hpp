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
            int type_Nt,
            double N_t,
            double k,
            int L,
            int n_colors,
            double a,
            double alpha,
            std::string type_percolation,
            double p0,
            double P0,
            double rho,
            bool animation
        );

    private:
        std::string base_path;
};

#endif // CREATE_FOLDERS_HPP
