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
    double k = stod(argv[6]);
    int N_t = stoi(argv[7]);
    int type_N_t = 0;
    //double k = 1e-5;
    //double N_t = 200;
    double a = 0;
    double alpha = 0;
    int dim = 2;
    double P0 = 0.1;

    if (argc != 8) {
        std::cout << "Usage: " << argv[0] << " <L> <N_samples> <p0> <seed> <type_percolation> <k> <N_t>" << std::endl;
        return 1;
    }

    if (seed < 0) {
        seed = all_random::generate_random_seed();
        std::cout << "[INFO] Random seed generated: " << seed << std::endl;
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
        p0,
        P0
    );

    network net_generator(N_samples);
    NetworkPattern net = net_generator.create_network(dim, L, N_samples, k, N_t, seed, type_N_t, p0, P0, a, alpha, type_percolation);

    // Obtém dados de saída
    std::vector<double> p_values = net_generator.get_p();
    std::vector<int> t_values = net_generator.get_t();
    std::vector<int> N_t_values = net_generator.get_N_t();

    // Monta os nomes dos arquivos
    std::ostringstream oss_name;
    oss_name << "/P0_" << std::fixed << std::setprecision(2) << P0
             << "_p0_" << std::fixed << std::setprecision(2) << p0
             << "_seed_" << seed << ".npy";

    std::ostringstream oss_net;
    oss_net << network_dir << "/P0_" << std::fixed << std::setprecision(2) << P0
            << "_p0_" << std::fixed << std::setprecision(2) << p0
            << "_seed_" << seed << ".npz";

    std::string pt_filename = pt_dir + oss_name.str();
    std::string nt_filename = nt_dir + oss_name.str();
    std::string net_filename = oss_net.str();

    // Salva arquivos
    save_data saver;
    saver.save_network_as_npz(net, net_filename);
    saver.save_p_values_as_npy(t_values, p_values, pt_filename);
    saver.save_Nt_values_as_npy(t_values, N_t_values, nt_filename);

    return 0;
}
