#ifndef CREATE_FOLDERS_HPP
#define CREATE_FOLDERS_HPP

#include <string>
#include <tuple>
<<<<<<< HEAD
#include <filesystem>
#include <string>
#include <sstream>
#include <filesystem>
#include <cstdio>
=======
#include <sstream>
#include <iomanip>
>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c

class FolderCreator {
    public:
        FolderCreator(const std::string& base_path);

<<<<<<< HEAD
        std::tuple<std::string, std::string, std::string> create_structure(
            int dim,
            int type_Nt,
            double N_t,
            double k,
            int L,
            int N_samples,
            double a,
            double alpha,
            std::string type_percolation,
            double p0,
            double P0
        );
=======
    // std::tuple<std::string, std::string> create_structure(
    //     int dim,
    //     int type_Nt,
    //     double N_t,
    //     double k,
    //     int L,
    //     int N_samples,
    //     double a = 0.0,
    //     double alpha = 0.0,
    //     std::string type_percolation,
    //     double p0 = 0.0,
    //     double P0 = 0.0,
    // );
    std::tuple<std::string, std::string> create_structure(
        int dim,
        int type_Nt,
        double N_t,
        double k,
        int L,
        int N_samples,
        double a = 0.0,
        double alpha = 0.0,
        std::string type_percolation = "node",
        double p0 = 0.0,
        double P0 = 0.0
    );
>>>>>>> 3199f00405dc30fc9383e3db35d3339145d2578c

    private:
        std::string base_path;
};

#endif // CREATE_FOLDERS_HPP
