#include "network.hpp"
#include "rand_utils.hpp"
#include "write_save.hpp"

int main(int argc, char* argv[]){
    int L = stoi(argv[1]);
    int N_samples = stoi(argv[2]);
    int seed = stoi(argv[3]);
    double k = pow(10,-4);
    double N_t = 200;
    int type_N_t = 0;
    double a = 0;
    double alpha = 0;
    int dim = 2;
    //int seed = 1;
    double p0 = 0.8;
    double P0 = 0.7;
    
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " file.json" << endl;
        return 1;
    }
    
    network net_generator(N_samples);  // 👈 must pass num_samples to constructor
    
    NetworkPattern net = net_generator.create_network(dim, L, N_samples, k, N_t, seed, type_N_t, p0, P0, a, alpha);
        
    vector<double> p_values = net_generator.get_p();
    
    save_data saver;
    saver.save_network_as_npz(net, "rede.npz");
    saver.save_p_values_as_npy(p_values, "p.npy");
    return 0;
    
}
