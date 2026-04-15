#include "animation_reanalysis.hpp"
#include "helpers.hpp"
#include "../src/write_save.hpp"
#include "../src/create_folders.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
namespace rh = reanalysis_helpers;

int main(int argc, char** argv)
{
    try {
        if (argc < 11) {
            std::cerr
                << "Uso:\n"
                << "  " << argv[0]
                << " <L> <p0> <seed> <type_percolation> <k> <N_t> <dim> <num_colors> <rho> <P0>"
                << " [species_factor] [smoothing_window] [min_stable_steps]"
                << " [rel_tol] [abs_tol] [sigma_multiplier]\n\n"
                << "Observacao:\n"
                << "  seed >= 0  -> processa apenas aquele seed\n"
                << "  seed < 0   -> processa todos os arquivos compativeis em data_path\n";
            return 1;
        }

        const int L = std::stoi(argv[1]);
        const double pp0 = std::stod(argv[2]);
        const int seed = std::stoi(argv[3]);
        const std::string type_percolation = argv[4];
        const double k = std::stod(argv[5]);
        const double N_t = std::stod(argv[6]);
        const int dim = std::stoi(argv[7]);
        const int num_colors = std::stoi(argv[8]);
        const double rho_val = std::stod(argv[9]);
        const double P0 = std::stod(argv[10]);

        ReanalysisConfig cfg;
        if (argc > 11) cfg.species_factor   = std::stoi(argv[11]);
        if (argc > 12) cfg.smoothing_window = std::stoi(argv[12]);
        if (argc > 13) cfg.min_stable_steps = std::stoi(argv[13]);
        if (argc > 14) cfg.rel_tol          = std::stod(argv[14]);
        if (argc > 15) cfg.abs_tol          = std::stod(argv[15]);
        if (argc > 16) cfg.sigma_multiplier = std::stod(argv[16]);

        const std::string base_path = "./SOP_data";
        const bool animation = true;

        const int type_Nt = 0;
        const double a = 0.0;
        const double alpha = 0.0;

        FolderCreator folder_creator(base_path);
        const auto [
            network_path_str,
            data_path_str,
            equilibration_dir_str,
            network_preteq_str,
            network_posteq_str
        ] = folder_creator.create_structure(
                dim,
                type_Nt,
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

        const fs::path network_path(network_path_str);
        const fs::path data_path(data_path_str);
        const fs::path equilibration_dir(equilibration_dir_str);
        const fs::path network_preteq_dir(network_preteq_str);
        const fs::path network_posteq_dir(network_posteq_str);

        const std::vector<fs::path> json_files =
            rh::collect_json_files(data_path, P0, pp0, seed);

        if (json_files.empty()) {
            throw std::runtime_error(
                "Nenhum JSON compativel encontrado em: " + data_path.string());
        }

        const auto npz_index = rh::build_npz_filename_index(network_path);

        save_data saver;

        int n_ok = 0;
        int n_fail = 0;

        for (const auto& json_path : json_files) {
            try {
                const fs::path npz_path =
                    rh::find_matching_npz(json_path, data_path, network_path, npz_index);

                ReanalysisResult result =
                    reanalyze_animation(json_path.string(), npz_path.string(), cfg);

                const fs::path out_json =
                    rh::build_output_json_path(json_path, data_path, equilibration_dir);

                const fs::path out_preteq_npz =
                    rh::build_output_network_path(json_path, data_path, network_preteq_dir);

                const fs::path out_posteq_npz =
                    rh::build_output_network_path(json_path, data_path, network_posteq_dir);

                saver.save_reanalysis_json(result, out_json.string());
                saver.save_reanalysis_networks(
                    result,
                    out_preteq_npz.string(),
                    out_posteq_npz.string()
                );

                std::cout << "[OK]\n";
                std::cout << "  input json         : " << json_path << '\n';
                std::cout << "  input npz          : " << npz_path << '\n';
                std::cout << "  output json        : " << out_json << '\n';
                std::cout << "  output pre_teq npz : " << out_preteq_npz << '\n';
                std::cout << "  output post_teq npz: " << out_posteq_npz << '\n';
                std::cout << "  t_eq               : " << result.t_eq << '\n';

                rh::print_valid_path_colors("cores com caminho em pre_teq ", result.pre_teq.sp_len);
                rh::print_valid_path_colors("cores com caminho em post_teq", result.post_teq.sp_len);

                std::cout << '\n';

                ++n_ok;
            } catch (const std::exception& e) {
                std::cerr << "[ERRO] Falha ao processar: " << json_path << '\n';
                std::cerr << "       Motivo: " << e.what() << "\n\n";
                ++n_fail;
            }
        }

        std::cout << "Resumo final:\n";
        std::cout << "  processados com sucesso = " << n_ok << '\n';
        std::cout << "  falhas                  = " << n_fail << '\n';

        return (n_ok > 0) ? 0 : 2;
    }
    catch (const std::exception& e) {
        std::cerr << "[AnimationReanalysis] erro: " << e.what() << '\n';
        return 2;
    }
}