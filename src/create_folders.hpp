#ifndef CREATE_FOLDERS_HPP
#define CREATE_FOLDERS_HPP

#include <string>
#include <tuple>
#include <sstream>
#include <iomanip>

class FolderCreator {
public:
    FolderCreator(const std::string& base_path);

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

private:
    std::string base_path;
};

#endif // CREATE_FOLDERS_HPP
