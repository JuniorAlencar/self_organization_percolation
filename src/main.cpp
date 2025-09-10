#include "network.hpp"
#include "rand_utils.hpp"
#include "write_save.hpp"
#include "create_folders.hpp"
#include <iomanip>
#include <cstdlib>
#include <sstream>


// --- imprime help/versão ---
static void print_help(const char* prog){
    std::cout <<
R"(To run:
  ./SOP <L> <p0> <seed> <type_percolation> <k> <N_t> <dim> <num_colors> <rho_val>

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

Examples:
  ./SOP 1000 0.10 -1 bond 1.0e-05 200 2 2 0.30
  ./SOP  500  0.05 42 node 1.0e-04 100 3 3 0.25

Tips:
  - Use seed = -1 to auto-generate a random seed.
  - 'bond' vs 'node' picks percolation type.
  - Check the article for recommended ranges of k and N_t.
)" << std::endl;
}

static bool is_help_token(const char* s){
    return (std::strcmp(s,"!help")==0 || std::strcmp(s,"--help")==0 || std::strcmp(s,"-h")==0);
}

static void print_version(){
#ifdef SOP_VERSION
    std::cout << "SOP version " << SOP_VERSION << std::endl;
#else
    std::cout << "SOP version (unknown)" << std::endl;
#endif
}


int main(int argc, char* argv[]){
    // ajuda/versão sem exigir todos os argumentos
    if (argc >= 2) {
        if (is_help_token(argv[1])) { print_help(argv[0]); return 0; }
        if (std::strcmp(argv[1],"--version")==0) { print_version(); return 0; }
    }
    if (argc != 10) {
        std::cerr << "[ERROR] Invalid number of arguments (" << argc-1 << ").\n";
        print_help(argv[0]);
        return 1;
    }
    
    int L = stoi(argv[1]);
    //int N_samples = stoi(argv[2]);
    int N_samples = 50000;
    double pp0 = stod(argv[2]);
    int seed = stoi(argv[3]);
    string type_percolation = argv[4];
    double k = stod(argv[5]); // 1.0e-04
    int N_t = stoi(argv[6]); // 200
    int dim = stoi(argv[7]); // 2
    int num_colors = stoi(argv[8]);
    double rho_val = stod(argv[9]);
    int type_N_t = 0; // if ==0 (Nt = constant), if==1 (Nt = at^{\alpha})
    double a = 0;
    double alpha = 0;
    double P0 = 0.1;

    // validações amigáveis
    if (dim != 2 && dim != 3){
        std::cerr << "[ERROR] dim must be 2 or 3.\n";
        print_help(argv[0]);
        return 1;
    }
    if (type_percolation != "bond" && type_percolation != "node"){
        std::cerr << "[ERROR] type_percolation must be 'bond' or 'node'.\n";
        print_help(argv[0]);
        return 1;
    }
    if (num_colors < 1){
        std::cerr << "[ERROR] num_colors must be >= 1.\n";
        print_help(argv[0]);
        return 1;
    }
    if (num_colors * rho_val > 1.0 + 1e-12){
        std::cerr << "[ERROR] Constraint violated: num_colors * rho_val <= 1.0.\n"
                     "        You passed: num_colors=" << num_colors
                  << " and rho_val=" << rho_val << " (product=" << num_colors*rho_val << ")\n";
        print_help(argv[0]);
        return 1;
    }
    // If seed < 0, return random seed 
    if (seed == -1) {
        seed = all_random::generate_random_seed();  // da sua rand_utils
    }
    
    // The generator
    all_random rng(seed);
    
    // Create folder Data
    FolderCreator creator("./Data");
    
    // 👇 Atualizado para receber os 2 caminhos
    auto [network_dir, data_dir] = creator.create_structure(
        dim,
        type_N_t,
        N_t,
        k,
        L,
        num_colors,
        a,
        alpha,
        type_percolation,
        pp0,
        P0,
        rho_val
    );
    // Struct to allocate TimeSeries
    TimeSeries ts;
    // Struct to allocate Percolation Informations
    PercolationSeries ps;
    
    // Density of network for each color
    vector<double> rho(num_colors, rho_val);
    // Initial probability for each color
    vector<double> p0(num_colors, pp0);

    network net_generator(N_samples, num_colors);
    
    // Network
    NetworkPattern net = net_generator.create_network(dim, L, N_samples, k, N_t, type_N_t, p0, P0, a, alpha, type_percolation, num_colors, rho, ts, ps, rng);
    
    std::cerr << "[DBG] ps sizes -> "
          << "order=" << ps.percolation_order.size()
          << ", color=" << ps.color_percolation.size()
          << ", time="  << ps.time_percolation.size()
          << ", rho="   << ps.rho.size()
          << ", pho="   << ps.pho.size() << "\n";

    std::cerr << "[DBG] ts sizes -> "
            << "num_colors=" << ts.num_colors
            << ", t=" << ts.t.size()
            << ", p_t=" << ts.p_t.size()
            << ", Nt=" << ts.Nt.size() << "\n";


    // Check initial ratio between types of nodes
//    NetworkPattern net = net_generator.initialize_network(dim, L, num_colors, P0, rho, p0, seed);
    //net_generator.print_initial_site_fractions(net);
    
    // Create name of files
    std::ostringstream oss_name;
    oss_name << "/P0_" << std::fixed << std::setprecision(2) << P0
             << "_p0_" << std::fixed << std::setprecision(2) << pp0
             << "_seed_" << seed << ".json";

    std::ostringstream oss_net;
    oss_net << network_dir << "/P0_" << std::fixed << std::setprecision(2) << P0
            << "_p0_" << std::fixed << std::setprecision(2) << pp0
            << "_seed_" << seed << ".npz";
    
    std::string json_filename = data_dir + oss_name.str();
    std::string net_filename = oss_net.str();
    
    // Animation network
    // NetworkPattern net = net_generator.animate_network(dim, L, N_samples, k, N_t, seed, type_N_t, p0, P0, a, alpha, type_percolation, num_colors, rho, ts);
    
    // Saving files
    save_data saver;
    // Network
    // saver.save_network_as_npz(net, net_filename);
    
    // Results
    saver.save_percolation_json(ps, ts, json_filename, true);
    cout << "file save with name:\t" <<  oss_name.str() << endl;
    return 0;
}
