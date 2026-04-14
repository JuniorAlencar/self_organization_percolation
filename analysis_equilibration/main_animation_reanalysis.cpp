#include "animation_reanalysis.hpp"
#include "../src/write_save.hpp"
#include "../src/create_folders.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string format_fixed_2(const double x)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << x;
    return oss.str();
}

bool contains_token(const std::string& name, const std::string& token)
{
    return name.find(token) != std::string::npos;
}

bool matches_requested_sample(
    const std::string& filename,
    const double P0,
    const double p0,
    const int seed)
{
    const std::string token_P0 = "P0_" + format_fixed_2(P0);
    const std::string token_p0 = "p0_" + format_fixed_2(p0);

    if (!contains_token(filename, token_P0)) return false;
    if (!contains_token(filename, token_p0)) return false;

    // Compatível com dois modos:
    //  seed < 0  -> processa todos os seeds
    //  seed >= 0 -> filtra apenas aquele seed
    if (seed >= 0) {
        const std::string token_seed = "seed_" + std::to_string(seed);
        if (!contains_token(filename, token_seed)) return false;
    }

    return true;
}

std::vector<fs::path> collect_json_files(
    const fs::path& data_path,
    const double P0,
    const double p0,
    const int seed)
{
    if (!fs::exists(data_path)) {
        throw std::runtime_error("Diretorio data_path nao existe: " + data_path.string());
    }

    std::vector<fs::path> json_files;

    for (const auto& entry : fs::recursive_directory_iterator(data_path)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".json") continue;

        const std::string filename = entry.path().filename().string();

        if (!matches_requested_sample(filename, P0, p0, seed)) continue;

        json_files.push_back(entry.path());
    }

    std::sort(json_files.begin(), json_files.end());
    return json_files;
}

std::unordered_map<std::string, fs::path> build_npz_filename_index(const fs::path& network_path)
{
    if (!fs::exists(network_path)) {
        throw std::runtime_error("Diretorio network_path nao existe: " + network_path.string());
    }

    std::unordered_map<std::string, fs::path> index;

    for (const auto& entry : fs::recursive_directory_iterator(network_path)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".npz") continue;

        // índice por nome do arquivo, ex.:
        // light_seed_..._P0_0.10_p0_1.00.npz
        index[entry.path().filename().string()] = entry.path();
    }

    return index;
}

fs::path find_matching_npz(
    const fs::path& json_path,
    const fs::path& data_path,
    const fs::path& network_path,
    const std::unordered_map<std::string, fs::path>& npz_index)
{
    // 1) Tenta manter a mesma estrutura relativa entre data_path e network_path
    fs::path rel = fs::relative(json_path, data_path);
    fs::path candidate = network_path / rel;
    candidate.replace_extension(".npz");

    if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
        return candidate;
    }

    // 2) Fallback: procura por mesmo nome-base em qualquer lugar de network_path
    fs::path npz_name = json_path.filename();
    npz_name.replace_extension(".npz");

    auto it = npz_index.find(npz_name.string());
    if (it != npz_index.end()) {
        return it->second;
    }

    throw std::runtime_error(
        "Nao encontrei o .npz correspondente para: " + json_path.string());
}

fs::path build_output_json_path(
    const fs::path& json_path,
    const fs::path& data_path,
    const fs::path& equilibration_dir)
{
    fs::path rel = fs::relative(json_path, data_path);

    fs::path out_path = equilibration_dir / rel;
    const std::string stem = out_path.stem().string();
    out_path.replace_filename(stem + "_process.json");

    fs::create_directories(out_path.parent_path());
    return out_path;
}

fs::path build_output_network_path(
    const fs::path& json_path,
    const fs::path& data_path,
    const fs::path& network_dir)
{
    fs::path rel = fs::relative(json_path, data_path);

    fs::path out_path = network_dir / rel;
    out_path.replace_extension(".npz");

    fs::create_directories(out_path.parent_path());
    return out_path;
}

} // namespace

void print_colors(const std::string& label, const std::vector<int>& colors)
{
    std::cout << "  " << label << ": ";
    if (colors.empty()) {
        std::cout << "nenhuma";
    } else {
        for (std::size_t i = 0; i < colors.size(); ++i) {
            std::cout << colors[i];
            if (i + 1 < colors.size()) std::cout << ", ";
        }
    }
    std::cout << '\n';
}

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
            collect_json_files(data_path, P0, pp0, seed);

        if (json_files.empty()) {
            throw std::runtime_error(
                "Nenhum JSON compativel encontrado em: " + data_path.string());
        }

        const auto npz_index = build_npz_filename_index(network_path);

        save_data saver;

        int n_ok = 0;
        int n_fail = 0;

        for (const auto& json_path : json_files) {
        try {
            const fs::path npz_path =
                find_matching_npz(json_path, data_path, network_path, npz_index);

            ReanalysisResult result =
                reanalyze_animation(json_path.string(), npz_path.string(), cfg);

            const fs::path out_json =
                build_output_json_path(json_path, data_path, equilibration_dir);

            const fs::path out_preteq_npz =
                build_output_network_path(json_path, data_path, network_preteq_dir);

            const fs::path out_posteq_npz =
                build_output_network_path(json_path, data_path, network_posteq_dir);

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

            print_colors("cores perc post_teq", result.post_teq.color_percolation);

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