// main.cpp (atualizado para o novo write do JSON)

#include "network.hpp"
#include "rand_utils.hpp"
#include "write_save.hpp"
#include "create_folders.hpp"
#include "helpers_print.hpp"

#include <iomanip>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <iostream>
#include <utility>   // para structured bindings (C++17)

#include <chrono>
#include <ctime>
#include <string>
#include <algorithm>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif


static void print_help(const char* prog){
    std::cout <<
R"(To run:
  ./SOP <L> <p0> <seed> <type_percolation> <k> <N_t> <dim> <num_colors> <rho_val> <P0> <Equilibration>

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
  P0               : Fraction of active nodes in base [0 < P0 <= 1.0]
  Equilibration    : Return network with encoded time activation ['true' or 'false']
Examples:
  ./SOP 2000 1.0 -1 bond 1.0e-04 200 2 1 1.0 0.1 true
  ./SOP  500  0.05 42 node 1.0e-04 100 3 3 0.25 0.5 false

Tips:
  - Use seed = -1 to auto-generate a random seed.
  - 'bond' vs 'node' picks percolation type.
  - Check the article for recommended ranges of k and N_t.
)" << std::endl;
}

static std::string sanitize_for_filename(std::string s) {
    for (char& c : s) {
        const bool ok =
            std::isalnum(static_cast<unsigned char>(c)) ||
            c == '-' || c == '_';
        if (!ok) c = '_';
    }
    return s;
}

static std::string get_machine_name() {
#ifdef _WIN32
    char buffer[MAX_COMPUTERNAME_LENGTH + 1];
    DWORD size = sizeof(buffer);
    if (GetComputerNameA(buffer, &size)) {
        return sanitize_for_filename(std::string(buffer, size));
    }
    return "unknown_host";
#else
    char buffer[256];
    if (gethostname(buffer, sizeof(buffer)) == 0) {
        buffer[sizeof(buffer) - 1] = '\0';
        return sanitize_for_filename(std::string(buffer));
    }
    return "unknown_host";
#endif
}

static std::string get_timestamp_now() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t tt = system_clock::to_time_t(now);

    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%dT%H%M%S");
    return oss.str();
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

bool parse_bool(const std::string& s) {
    std::string x = s;
    std::transform(x.begin(), x.end(), x.begin(), ::tolower);
    if (x=="1"||x=="true"||x=="t"||x=="yes"||x=="y") return true;
    if (x=="0"||x=="false"||x=="f"||x=="no" ||x=="n") return false;
    throw std::runtime_error("bool inválido: " + s);
}

int main(int argc, char* argv[]){
    // ajuda/versão sem exigir todos os argumentos
    if (argc >= 2) {
        if (is_help_token(argv[1])) { print_help(argv[0]); return 0; }
        if (std::strcmp(argv[1],"--version")==0) { print_version(); return 0; }
    }
    if (argc != 12) {
        std::cerr << "[ERROR] Invalid number of arguments (" << argc-1 << ").\n";
        print_help(argv[0]);
        return 1;
    }

    try {
        int L = std::stoi(argv[1]);
        double pp0 = std::stod(argv[2]);
        int seed = std::stoi(argv[3]);
        string type_percolation = argv[4];
        double k = std::stod(argv[5]);    // 1.0e-04
        int N_t = std::stoi(argv[6]);     // 200
        int dim = std::stoi(argv[7]);     // 2
        int num_colors = std::stoi(argv[8]);
        double rho_val = std::stod(argv[9]);
        double P0 = std::stod(argv[10]);
        string equilibration = argv[11];
        
        bool animation;

        if (equilibration == "true" || equilibration == "1") {
            animation = true;
        } else if (equilibration == "false" || equilibration == "0") {
            animation = false;
        } else {
            std::cerr << "Valor invalido para bool: " << equilibration << "\n";
            return 1;
        }

        int type_N_t = 0;   // 0 => Nt constante; 1 => Nt = a * t^alpha
        double a = 0.0;
        double alpha = 0.0;
        
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

        // seed automática se < 0
        if (seed == -1) {
            seed = all_random::generate_random_seed();
        }

        // gerador
        all_random rng(seed);

        // estruturas de saída
        TimeSeries ts;
        PercolationSeries ps;

        // densidades por cor
        std::vector<double> rho(num_colors, rho_val);
        // prob. inicial por cor
        std::vector<double> p0(num_colors, pp0);
        int N_samples = 100000;
        network net_generator(N_samples, num_colors);
        
        // select if I used animation or just create
        // bool animation = false;

        NetworkPattern net = animation
            ? net_generator.animate_network(
                dim, L, N_samples, k, N_t, type_N_t,
                p0, P0, a, alpha, type_percolation,
                num_colors, rho, ts, ps, rng
            )
            : net_generator.create_network(
                dim, L, N_samples, k, N_t, type_N_t,
                p0, P0, a, alpha, type_percolation,
                num_colors, rho, ts, ps, rng
            );

        // cria pastas Data
        FolderCreator creator("./SOP_data");
        const auto [
            network_dir,
            data_dir,
            equilibration_dir,
            network_preteq,
            network_posteq
        ] = creator.create_structure(
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
                rho_val,
                animation
            );
        
        std::cerr << "[DBG] ps sizes -> "
                  << "order=" << ps.percolation_order.size()
                  << ", color=" << ps.color_percolation.size()
                  << ", rho="   << ps.rho.size()
                  << ", rho_value=" << rho_val  << "\n";

        std::cerr << "[DBG] ts sizes -> "
                  << "num_colors=" << ts.num_colors
                  << ", t="  << ts.t.size()
                  << ", p_t="<< ts.p_t.size()
                  << ", Nt=" << ts.Nt.size() << "\n";
        
                  cout << "seed = " << seed << endl;

        const std::string machine_name = get_machine_name();
        const std::string timestamp_now = get_timestamp_now();

        std::ostringstream base_name;
        base_name << machine_name
                << "_seed_" << seed
                << "_ts_" << timestamp_now
                << "_P0_" << std::fixed << std::setprecision(2) << P0
                << "_p0_" << std::fixed << std::setprecision(2) << pp0;

        const std::string sample_base = base_name.str();
        std::string json_filename = data_dir + "/" + sample_base + ".json";
        
        // // salvar (novo writer JSON)
        save_data saver;
        // 1) rede (Numpy .npy)
        if(animation==true){
            std::string net_filename  = network_dir + "/" + sample_base + ".npz";
            saver.save_network_as_npz(net, net_filename);
            std::string shortest_filename = network_dir + "/map_shortest_" + sample_base + ".npz";
        }
    

        // 2) resultados (JSON novo)
        //sort_by_order = true para ordenar por percolation_order
        saver.save_percolation_json(ps, ts, json_filename, /*sort_by_order=*/true);
        // std::string net_filename  = network_dir + "/" + sample_base + ".npz";
        // saver.save_network_as_npz(net, net_filename);
        // 3) save shortest map(Numpy .npy)
        // NetworkPattern sp_net = net_generator.create_shortest_paths_map(net, ps);
        // saver.save_network_as_npz(sp_net, shortest_filename);
        
        // TESTS ----------------

    //     FolderCreator creator_tests("./Data_tests");
    //     auto [network_dir_tests, data_dir_tests] = creator_tests.create_structure(
    //         dim, type_N_t, N_t, k, L, num_colors, a, alpha,
    //         type_percolation, pp0, P0, rho_val
    //     );
                
    //     std::ostringstream oss_name_tests;
    //     oss_name_tests << "/P0_" << std::fixed << std::setprecision(2) << P0
    //              << "_p0_" << std::fixed << std::setprecision(2) << pp0
    //              << "_seed_" << seed << ".json";
        
    //     std::ostringstream oss_net_tests;
    //     oss_net_tests << network_dir_tests << "/P0_" << std::fixed << std::setprecision(2) << P0
    //       << "_p0_" << std::fixed << std::setprecision(2) << pp0
    //        << "_seed_" << seed << ".npz"; // writer grava .npy
        
    //     string json_filename_tests = data_dir_tests + oss_name_tests.str();
    //     saver.save_percolation_json(ps, ts, json_filename_tests, /*sort_by_order=*/true);
    //     //string net_filename_tests  = oss_net_tests.str();
    //     //saver.save_network_as_npz(net, net_filename_tests);
        
    //    std::cout << "file save with name:\t" << oss_name_tests.str() << std::endl;
        
        return 0;
        // END TESTS -----
    
    }
    catch (const std::exception& e){
        std::cerr << "[FATAL] Exception: " << e.what() << "\n";
        return 2;
    }
    catch (...){
        std::cerr << "[FATAL] Unknown exception.\n";
        return 3;
    }
}
