#include "animation_reanalysis.hpp"
#include "../src/write_save.hpp"
#include "../src/create_folders.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace fs = std::filesystem;

namespace {

std::string format_fixed_2(const double x)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << x;
    return oss.str();
}

std::string format_sci_1(const double x)
{
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(1) << x;
    return oss.str();
}

std::string find_matching_file(const std::string& dir,
                               const std::string& extension,
                               const int seed,
                               const double P0,
                               const double p0)
{
    if (!fs::exists(dir)) {
        throw std::runtime_error("Diretorio nao existe: " + dir);
    }

    const std::string token_seed = "seed_" + std::to_string(seed);
    const std::string token_P0   = "P0_" + format_fixed_2(P0);
    const std::string token_p0   = "p0_" + format_fixed_2(p0);

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != extension) continue;

        const std::string name = entry.path().filename().string();

        if (name.find(token_seed) != std::string::npos &&
            name.find(token_P0)   != std::string::npos &&
            name.find(token_p0)   != std::string::npos) {
            return entry.path().string();
        }
    }

    throw std::runtime_error(
        "Nao encontrei arquivo " + extension +
        " com tokens [" + token_seed + ", " + token_P0 + ", " + token_p0 +
        "] em: " + dir);
}

std::string build_output_json_path(const std::string& input_json)
{
    fs::path p(input_json);
    return (p.parent_path() / (p.stem().string() + "_reanalysis.json")).string();
}

std::string build_output_npz_path(const std::string& input_npz)
{
    fs::path p(input_npz);
    return (p.parent_path() / (p.stem().string() + "_teq_filtered.npz")).string();
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 11) {
            std::cerr
                << "Uso:\n"
                << "  " << argv[0]
                << " <L> <p0> <seed> <type_percolation> <k> <N_t> <dim> <num_colors> <rho> <P0>"
                << " [species_factor] [smoothing_window] [min_stable_steps]"
                << " [rel_tol] [abs_tol] [sigma_multiplier]\n";
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

        // Ajuste este base_path para o mesmo usado no CreateFolders do seu projeto
        const std::string base_path = "./SOP_data";

        // Para este reanalysis estamos lendo dados do animation
        const bool animation = true;

        // Mesmo shape lógico do create_folders
        const int type_Nt = 0;
        const double a = 0.0;
        const double alpha = 0.0;

        FolderCreator folder_creator(base_path);
        const auto [network_path, data_path] = folder_creator.create_structure(
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

        const std::string json_path = find_matching_file(data_path, ".json", seed, P0, pp0);
        const std::string npz_path  = find_matching_file(network_path, ".npz", seed, P0, pp0);

        const std::string out_summary_json = build_output_json_path(json_path);
        const std::string out_filtered_npz = build_output_npz_path(npz_path);

        ReanalysisResult result = reanalyze_animation(json_path, npz_path, cfg);

        save_data saver;
        saver.save_reanalysis_json(result, out_summary_json);
        saver.save_network_as_npz(result.filtered_net, out_filtered_npz);

        std::cout << "input json  = " << json_path << '\n';
        std::cout << "input npz   = " << npz_path << '\n';
        std::cout << "output json = " << out_summary_json << '\n';
        std::cout << "output npz  = " << out_filtered_npz << '\n';
        std::cout << "t_eq = " << result.t_eq << '\n';

        std::cout << "Cores percolantes: ";
        for (std::size_t i = 0; i < result.color_percolation.size(); ++i) {
            std::cout << result.color_percolation[i];
            if (i + 1 < result.color_percolation.size()) std::cout << ", ";
        }
        std::cout << '\n';

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[AnimationReanalysis] erro: " << e.what() << '\n';
        return 2;
    }
}