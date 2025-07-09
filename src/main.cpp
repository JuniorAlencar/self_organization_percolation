#include "network.hpp"
#include "rand_utils.hpp"
#include "write_save.hpp"
#include "create_folders.hpp"
#include <iomanip>
#include <sstream>

int main(int argc, char* argv[]){
    int L = stoi(argv[1]);
    int N_samples = stoi(argv[2]);
    double p0 = stod(argv[3]);
    int seed = stoi(argv[4]);
    string type_percolation = argv[5];
    double k = 1e-5;
    double N_t = 200;
    int type_N_t = 0;
    double a = 0;
    double alpha = 0;
    int dim = 2;
    //int seed = 1;
    //double p0 = 1.0;
    double P0 = 0.1;
    //string type_percolation = "site";
    
    if (argc != 6) {
        cout << "Usage: " << argv[0] << "<L> <N_samples> <seed>" << endl;
        return 1;
    }

    // If seed is negative (e.g. -1), generate a random seed
    if (seed < 0) {
        seed = all_random::generate_random_seed();
        std::cout << "[INFO] Random seed generated: " << seed << std::endl;
    }


    FolderCreator creator("./Data");

    auto [network_dir, pt_dir] = creator.create_structure(
        dim,
        type_N_t,
        N_t,
        k,
        L,
        N_samples,
        a,
        alpha,
        type_percolation,
        p0,
        P0
    );

    // Creating the network and vector p(t)
    network net_generator(N_samples);  // 👈 must pass num_samples to constructor
    
    NetworkPattern net = net_generator.create_network(dim, L, N_samples, k, N_t, seed, type_N_t, p0, P0, a, alpha, type_percolation);
        
    vector<double> p_values = net_generator.get_p();

    // Converting to 3 decimal places p0 and P0 and creating the file names
    std::ostringstream oss_net, oss_pt;
    oss_net << network_dir << "/P0_" << std::fixed << std::setprecision(2) << P0
            << "_p0_" << std::fixed << std::setprecision(2) << p0
            << "_seed_" << seed << ".npz";

    oss_pt << pt_dir << "/P0_" << std::fixed << std::setprecision(2) << P0
        << "_p0_" << std::fixed << std::setprecision(2) << p0
        << "_seed_" << seed << ".npy";

    std::string net_filename = oss_net.str();
    std::string pt_filename = oss_pt.str();
    
    // Saving network and p(t) file
    save_data saver;
    saver.save_network_as_npz(net, net_filename);
    saver.save_p_values_as_npy(p_values, pt_filename);
    return 0;
    
}
