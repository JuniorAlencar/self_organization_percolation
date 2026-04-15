#pragma once

#include <string>
#include "struct_network.hpp"

namespace helpers {

// ---------- helpers de impressão da rede ----------
void print_base_summary(const NetworkPattern& net);
void print_slice(const NetworkPattern& net, int g_level, int max_w = 80);

// ---------- helpers gerais do executável ----------
void print_help(const char* prog);
std::string sanitize_for_filename(std::string s);
std::string get_machine_name();
std::string get_timestamp_now();
bool is_help_token(const char* s);
void print_version();
bool parse_bool(const std::string& s);

} // namespace helpers