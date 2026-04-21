#pragma once

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace reanalysis_helpers {

namespace fs = std::filesystem;

inline std::string format_fixed_2(const double x)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << x;
    return oss.str();
}

inline bool contains_token(const std::string& name, const std::string& token)
{
    return name.find(token) != std::string::npos;
}

inline bool matches_requested_sample(
    const std::string& filename,
    const double P0,
    const double p0,
    const int seed)
{
    const std::string token_P0 = "P0_" + format_fixed_2(P0);
    const std::string token_p0 = "p0_" + format_fixed_2(p0);

    if (!contains_token(filename, token_P0)) return false;
    if (!contains_token(filename, token_p0)) return false;

    if (seed >= 0) {
        const std::string token_seed = "seed_" + std::to_string(seed);
        if (!contains_token(filename, token_seed)) return false;
    }

    return true;
}

inline std::vector<fs::path> collect_json_files(
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

inline std::unordered_map<std::string, fs::path> build_npz_filename_index(const fs::path& network_path)
{
    if (!fs::exists(network_path)) {
        throw std::runtime_error("Diretorio network_path nao existe: " + network_path.string());
    }

    std::unordered_map<std::string, fs::path> index;

    for (const auto& entry : fs::recursive_directory_iterator(network_path)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".npz") continue;

        index[entry.path().filename().string()] = entry.path();
    }

    return index;
}

inline fs::path find_matching_npz(
    const fs::path& json_path,
    const fs::path& data_path,
    const fs::path& network_path,
    const std::unordered_map<std::string, fs::path>& npz_index)
{
    fs::path rel = fs::relative(json_path, data_path);
    fs::path candidate = network_path / rel;
    candidate.replace_extension(".npz");

    if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
        return candidate;
    }

    fs::path npz_name = json_path.filename();
    npz_name.replace_extension(".npz");

    auto it = npz_index.find(npz_name.string());
    if (it != npz_index.end()) {
        return it->second;
    }

    throw std::runtime_error(
        "Nao encontrei o .npz correspondente para: " + json_path.string());
}

inline fs::path build_output_json_path(
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

inline fs::path build_output_network_path(
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

inline void print_valid_path_colors(const std::string& label, const std::vector<int>& sp_len)
{
    std::cout << "  " << label << ": ";

    bool first = true;
    for (std::size_t i = 0; i < sp_len.size(); ++i) {
        if (sp_len[i] < 0) continue;

        if (!first) std::cout << ", ";
        std::cout << (i + 1);
        first = false;
    }

    if (first) {
        std::cout << "nenhuma";
    }

    std::cout << '\n';
}

} // namespace reanalysis_helpers