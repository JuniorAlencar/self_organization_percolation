#ifndef STRUCT_NETWORK_HPP
#define STRUCT_NETWORK_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <numeric>
#include <random>
#include <algorithm>
#include "rand_utils.hpp"

using namespace std;

struct NetworkPattern {
    int dim;
    int num_colors;
    int seed;
    std::vector<int> shape;   // (2D) {L, L}  |  (3D) {L, L, L}
    std::vector<int> data;    // estado por sítio
    std::vector<double> rho;  // fração global por cor (agora aplicada na REDE TODA)
    // Convenções de estado:
    //   >0 : ativo, +1 (1 cor) ou +(c+2) (multi-cor)
    //   -1 : inativo sem cor (pode virar qualquer cor)
    //   -(c+2) : inativo com rótulo da cor c (c=0..num_colors-1)
    //    0 : checado e não ativado (não conecta mais)

    NetworkPattern(int dim_, const std::vector<int>& shape_, int num_colors_,
                   const std::vector<double>& rho_, all_random& rng_)
    : dim(dim_), shape(shape_), num_colors(num_colors_), rho(rho_), seed(rng_.get_seed())
    {
        // tamanho total
        const size_t total_sites = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                                   std::multiplies<size_t>());
        data.clear();
        data.resize(total_sites, -1); // default: sem cor

        if (num_colors_ <= 1) {
            // Caso 1 cor: mantém toda a rede -1 (sem cor), como no comportamento anterior.
            // (A ativação inicial via P0/p0 acontece depois, fora do construtor.)
            return;
        }

        // ---------- Multi-cor: distribuir -(c+2) e -1 na REDE TODA conforme rho ----------
        // Normaliza rho (se vier com soma != 1, mantém proporções)
        std::vector<double> rho_use = rho_;
        if ((int)rho_use.size() != num_colors_) {
            rho_use.assign(num_colors_, 1.0 / std::max(1, num_colors_));
        } else {
            for (double &x : rho_use) if (x < 0.0) x = 0.0;
            double s = std::accumulate(rho_use.begin(), rho_use.end(), 0.0);
            if (s <= 0.0) rho_use.assign(num_colors_, 1.0 / std::max(1, num_colors_));
            else for (double &x : rho_use) x /= s; // soma=1
        }

        // Se desejar permitir soma<1 para sobrar -1, basta comentar a normalização acima
        // e usar a linha abaixo para "fracao_none". Aqui, como normalizamos rho para 1,
        // fracao_none = 0.0 quando soma(rho) = 1.
        double fracao_none = std::max(0.0, 1.0 - std::accumulate(rho_.begin(), rho_.end(), 0.0));

        // Monta pesos: cores + "sem cor"
        std::vector<double> pesos;
        pesos.reserve(num_colors_ + 1);
        for (int c = 0; c < num_colors_; ++c) pesos.push_back(rho_use[c]);
        pesos.push_back(fracao_none);

        // Quotas proporcionais
        std::vector<double> quotas(num_colors_ + 1, 0.0);
        for (size_t i = 0; i < quotas.size(); ++i) quotas[i] = pesos[i] * double(total_sites);

        // Maiores restos (Hamilton): floors + distribui o restante pelos maiores decimais
        std::vector<size_t> aloc(num_colors_ + 1, 0);
        size_t soma_floor = 0;
        for (size_t i = 0; i < quotas.size(); ++i) {
            aloc[i] = static_cast<size_t>(std::floor(quotas[i]));
            soma_floor += aloc[i];
        }
        size_t leftover = (total_sites > soma_floor) ? (total_sites - soma_floor) : 0;

        // índices por fração decimal descrescente
        std::vector<size_t> idxs(quotas.size());
        std::iota(idxs.begin(), idxs.end(), 0);
        std::sort(idxs.begin(), idxs.end(),
                  [&](size_t a, size_t b){
                      double fa = quotas[a] - std::floor(quotas[a]);
                      double fb = quotas[b] - std::floor(quotas[b]);
                      if (fa == fb) return a < b;
                      return fa > fb;
                  });
        for (size_t k = 0; k < leftover; ++k) {
            aloc[idxs[k]] += 1;
        }

        // aloc[0..num_colors_-1] = contagem de -(c+2)
        // aloc[num_colors_]       = contagem de -1
        std::vector<int> labels;
        labels.reserve(total_sites);

        for (int c = 0; c < num_colors_; ++c) {
            size_t cnt = aloc[c];
            int lab = -(c + 2); // −2, −3, ...
            for (size_t i = 0; i < cnt; ++i) labels.push_back(lab);
        }
        // "sem cor"
        if (aloc[num_colors_] > 0) {
            labels.insert(labels.end(), aloc[num_colors_], -1);
        }

        // Embaralha e grava
        std::shuffle(labels.begin(), labels.end(), rng_.get_gen());
        if (labels.size() != total_sites) {
            // segurança: ajusta (por raríssimo drift numérico)
            labels.resize(total_sites, -1);
        }
        data = std::move(labels);
    }

    size_t to_index(const std::vector<int>& coords) const {
        if ((int)coords.size() != dim) {
            throw std::invalid_argument("Número de coordenadas incompatível com a dimensão");
        }
        size_t idx = 0;
        size_t stride = 1;
        for (int i = dim - 1; i >= 0; --i) {
            if (coords[i] < 0 || coords[i] >= shape[i])
                throw std::out_of_range("Coordenada fora dos limites");
            idx += size_t(coords[i]) * stride;
            stride *= size_t(shape[i]);
        }
        return idx;
    }

    int get(const std::vector<int>& coords) const { return data[to_index(coords)]; }
    void set(const std::vector<int>& coords, int value) { data[to_index(coords)] = value; }
    size_t size() const { return data.size(); }
};


// Struct to load (t, pt_i) and (t, Nt_i)
struct TimeSeries {
    int num_colors;
    std::vector<std::vector<double>> p_t;   // [cor][t]
    std::vector<std::vector<int>>    Nt;    // [cor][t]
    std::vector<std::vector<int>>    M_t;   // [cor][t]
    std::vector<int>                 t;     // [t]
    std::vector<int>                 M_size;// [cor] maior cluster final da cor
};

struct PercolationSeries {
    std::vector<int> color_percolation;   // 1-based
    std::vector<int> time_percolation;
    std::vector<int> percolation_order;
    std::vector<double> rho;              // por cor (0-based)
    std::vector<int> M_size_at_perc;      // por evento

    // SP armazenado internamente (opcional no writer)
    std::vector<int>              sp_len;       // [cor] = #nós no caminho (ou -1)
    std::vector<std::vector<int>> sp_path_lin;  // [cor] ids lineares (não será escrito)
};

#endif // network_hpp
