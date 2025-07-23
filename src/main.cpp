#include "network.hpp"
#include "rand_utils.hpp"
#include "write_save.hpp"
#include "create_folders.hpp"
#include <iomanip>
#include <cstdlib>
#include <sstream>

int main(int argc, char* argv[]){
    int L = stoi(argv[1]);
    int N_samples = stoi(argv[2]);
    double pp0 = stod(argv[3]);
    int seed = stoi(argv[4]);
    string type_percolation = argv[5];
    double k = stod(argv[6]);
    int N_t = stoi(argv[7]);
    int dim = stoi(argv[8]);
    int type_N_t = 0;
    //double k = 1e-5;
    //double N_t = 200;
    double a = 0;
    double alpha = 0;
    double P0 = 0.1;

    if (argc != 9) {
        std::cout << "Usage: " << argv[0] << " <L> <N_samples> <p0> <seed> <type_percolation> <k> <N_t> <dim>" << std::endl;
        return 1;
    }

    if (seed < 0) {
        seed = all_random::generate_random_seed();
        std::cout << "[INFO] Random seed generated: " << seed << std::endl;
    }

    if(dim != 2 && dim != 3){
        std::cerr << "Error: only dim = 2 or dim = 3 are supported." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    FolderCreator creator("./Data");

    // 👇 Atualizado para receber os 3 caminhos
    auto [network_dir, pt_dir, nt_dir] = creator.create_structure(
        dim,
        type_N_t,
        N_t,
        k,
        L,
        N_samples,
        a,
        alpha,
        type_percolation,
        pp0,
        P0
    );
    
    TimeSeries ts;
    int num_colors = 1;
    vector<double> rho = {0.5, 0.5};
    vector<double> p0 = {pp0};

    network net_generator(N_samples, num_colors);
    
    //NetworkPattern net = net_generator.create_network(dim, L, N_samples, k, N_t, seed, type_N_t, p0, P0, a, alpha, type_percolation, num_colors, rho, ts);
    NetworkPattern net = net_generator.animate_network(dim, L, N_samples, k, N_t, seed, type_N_t, p0, P0, a, alpha, type_percolation, num_colors, rho);
    // Check initial ratio between types of nodes
    // NetworkPattern net = net_generator.initialize_network(dim, L, N_samples, num_colors, P0, rho, seed);
    // net_generator.print_initial_site_fractions(net);
    
    
    // Monta os nomes dos arquivos
    std::ostringstream oss_name;
    oss_name << "/P0_" << std::fixed << std::setprecision(2) << P0
             << "_p0_" << std::fixed << std::setprecision(2) << pp0
             << "_seed_" << seed << ".csv";

    std::ostringstream oss_net;
    oss_net << network_dir << "/P0_" << std::fixed << std::setprecision(2) << P0
            << "_p0_" << std::fixed << std::setprecision(2) << pp0
            << "_seed_" << seed << ".npz";

    std::string pt_filename = pt_dir + oss_name.str();
    std::string nt_filename = nt_dir + oss_name.str();
    std::string net_filename = oss_net.str();

    // Salva arquivos
    save_data saver;
    saver.save_network_as_npz(net, net_filename);
    
    saver.save_time_series_as_csv(ts, pt_filename, nt_filename);

    return 0;
}
