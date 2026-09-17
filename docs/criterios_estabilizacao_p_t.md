---
title: "Critérios atuais de estabilização e correção em p(t)"
author: "self_organization_percolation"
date: "2026-09-14"
lang: pt-BR
geometry: margin=2.2cm
fontsize: 11pt
---

# Escopo

Este documento resume o critério de parada atualmente implementado no código fonte em `src/`, isto é, no caminho usado para gerar as amostras e os arquivos compactos `.bin`. A análise se concentra no modo `growth_test`, configurado em `src/main.cpp`, e na rotina principal de crescimento em `src/network.cpp`.

# Configuração do modo `growth_test`

No executável principal, o modo padrão é `growth_test`. Quando esse modo é usado, a configuração efetiva é:

- `dynamic_height = true`: a altura da rede cresce dinamicamente durante a simulação.
- `stop_at_percolation = false`: a simulação não para quando uma espécie atinge `z = L`.
- `stop_at_equilibrium = true`: a parada passa a depender da estabilização dinâmica.
- `equilibrium_consecutive_steps = 10`: valor base configurado para estabilidade.
- `N_samples = max(100000, 20L)`: limite superior de tempo usado como `hard_max_steps`, salvo quando sobrescrito.

Referência principal: `src/main.cpp`, linhas 238-242 e 361-373.

# Série observada para estabilização

A estabilização é calculada por espécie usando a série `p_i(t)`, armazenada em `p_series[c]`. A cada passo de tempo, antes da atualização para o próximo tempo, o código registra:

$$
t,\quad p_i(t),\quad f_i(t)
$$

com:

$$
f_i(t) = \frac{N_i(t)}{A},
$$

onde \(N_i(t)\) é o número de novos sítios ativados pela espécie \(i\) no passo \(t\), e \(A\) é a área lateral da base: \(L\) em 2D ou \(L^2\) em 3D.

Referência: `src/network.cpp`, linhas 2000-2003, 2132-2140, 2779-2781 e 2796.

# Estimativa de \(t_{\mathrm{eq}}\)

O candidato a tempo de estabilização é calculado por `estimate_t_eq_from_series`. No `growth_test`, essa rotina é chamada separadamente para cada espécie viva, usando `p_series[c]`.

A rotina faz os seguintes passos:

1. Suaviza a cauda da série \(p_i(t)\) com média móvel centrada de janela 15.
2. Divide a série suavizada em blocos regulares de 10 passos.
3. Calcula a média de cada bloco:

$$
j_w(k) = \langle p_i(t) \rangle_{\text{bloco } k}.
$$

4. Calcula a variação entre blocos consecutivos:

$$
s(k) = |j_w(k+1) - j_w(k)|.
$$

5. Aplica uma validação global de deriva na cauda. Com a configuração atual, essa validação é a parte determinante, porque `validation_window_steps > 0`.

Referência: `src/network.cpp`, linhas 1178-1208 e 1230-1311.

# Critério atual de estabilidade

Para o `growth_test`, os parâmetros efetivos usados na chamada são:

- janela de suavização: 15 passos;
- tamanho do bloco: 10 passos;
- tolerância relativa: `equilibrium_rel_tol = 2.5e-2`;
- tolerância absoluta: `equilibrium_abs_tol = 1.0e-6`;
- limiar de derivada formal: \(10^{-5}\);
- alvo explícito: desativado (`require_target = false`);
- validação em cauda: ativada.

Embora a função ainda contenha o critério local baseado em \(|s'(t)| < 10^{-5}\), no caminho atual do `growth_test` a chamada fornece uma janela de validação positiva. Por isso, a função entra no ramo de validação global da cauda e retorna \(t_{\mathrm{eq}}\) quando encontra uma janela em que:

$$
|\bar{j}_{\mathrm{final}} - \bar{j}_{\mathrm{inicial}}|
\leq
\max(\epsilon_{\mathrm{abs}}, \epsilon_{\mathrm{rel}}\,S),
$$

onde \(S = \max(|\bar{j}_{\mathrm{inicial}}|, |\bar{j}_{\mathrm{final}}|, 10^{-12})\).

Além disso, a inclinação linear ajustada dentro da janela também precisa produzir uma deriva projetada menor que o mesmo limiar:

$$
|\mathrm{slope}|\,\Delta t
\leq
\max(\epsilon_{\mathrm{abs}}, \epsilon_{\mathrm{rel}}\,S).
$$

A janela candidata também precisa passar quando comparada contra a cauda completa da série. Se isso ocorre, a função retorna o centro temporal do primeiro bloco estável como \(t_{\mathrm{eq}}\).

Referência: `src/network.cpp`, linhas 1237-1311.

# Critérios de parada da simulação

Em cada passo de tempo, depois de registrar \(p_i(t)\) e \(f_i(t)\), o código avalia as condições de parada globais:

1. `all_dead` ou `all_finished`: todas as espécies morreram ou todas terminaram o crescimento.
2. `all_percolated`: todas percolaram. No modo `growth_test` atual, esta condição fica desativada na prática porque `stop_at_percolation = false`.
3. `partial_percolation`: pelo menos uma percolou e todas as demais espécies não percoladas morreram. Também depende de `stop_at_percolation`, então não é o mecanismo principal do `growth_test` atual.
4. `stop_equilibrated`: todas as espécies ainda vivas tiveram \(t_{\mathrm{eq}}\) detectado e depois cresceram até uma altura pós-estabilização.
5. `hard_max_steps`: o limite duro de passos foi atingido.

Referência: `src/network.cpp`, linhas 2798-2857 e 3014-3035.

# Parada por estabilização e crescimento pós-estabilização

Uma espécie só pode ser marcada como estabilizada se:

1. ela ainda não morreu;
2. ela já atingiu altura global suficiente, isto é, em altura dinâmica `max_height >= L`;
3. a análise de \(p_i(t)\) retorna um \(t_{\mathrm{eq}}\) finito.

Quando isso acontece, o código salva:

- `t_eq_online_by_species[c]`: o tempo de estabilização da espécie;
- `z_stat_by_species[c]`: a altura máxima da espécie no momento da detecção;
- `equilibrium_detection_time_by_species[c]`: o tempo em que a estabilização foi detectada.

Depois da detecção, a simulação ainda não para imediatamente. Para cada espécie estabilizada, define-se:

$$
z_{\mathrm{stop}, i} = z_{\mathrm{stat}, i} + \left\lceil 2.5L \right\rceil.
$$

A parada global por estabilização ocorre apenas quando todas as espécies vivas estão estabilizadas e todas alcançaram sua respectiva altura pós-estabilização. A checagem final também é feita em passos múltiplos do bloco de 10 tempos.

Referência: `src/network.cpp`, linhas 2873-2912 e 2996-3011.

# Estados finais salvos

No resultado salvo, cada espécie recebe um status:

- `1`: estabilizou;
- `-1`: morreu;
- `0`: não estabilizou até a parada.

O motivo real da parada é salvo em `growth_test_stop_reason_actual`, com valores como:

- `all_dead`;
- `all_finished`;
- `post_equilibrium_height_reached`;
- `hard_max_steps`.

Referência: `src/network.cpp`, linhas 3151-3178, e `src/write_save.cpp`, linhas 430-448.

# Modificação atual em \(p(t)\)

A atualização de \(p(t)\) fica em `network::generate_p`. Para a regra padrão `relative`, o alvo é \(f_T\), e a correção é:

$$
\Delta p
=
c\frac{f_T - f(t)}{f_T},
\quad f_T > 0.
$$

Se \(f_T \leq 0\), o código usa a forma linear:

$$
\Delta p = c\,[f_T - f(t)].
$$

Depois disso, foram adicionadas duas proteções importantes.

Primeiro, o passo é limitado simetricamente:

$$
-c \leq \Delta p \leq c.
$$

Isso impede que um pico momentâneo em \(f(t)\), comum em redes pequenas por flutuações fortes, derrube \(p(t)\) profundamente em uma única iteração.

Segundo, quando \(\Delta p > 0\), foi adicionada uma atenuação perto do teto supercrítico efetivo:

$$
p_{\max}^{\mathrm{eff}} = 0.75.
$$

Para \(p(t) \leq 0.60\), o ganho positivo é mantido integralmente. Para \(0.60 < p(t) < 0.75\), o ganho positivo é multiplicado por:

$$
\frac{0.75 - p(t)}{0.75 - 0.60}.
$$

Para \(p(t) \geq 0.75\), qualquer incremento positivo é zerado. Por fim, o valor final ainda é limitado ao intervalo físico:

$$
0 \leq p(t+1) \leq 1.
$$

Referência: `src/network.cpp`, linhas 1921-1976.

# Interpretação da correção para redes pequenas

Em redes pequenas, \(f(t)\) é mais ruidoso porque \(N(t)\) é normalizado por uma área lateral menor. Isso pode causar dois efeitos artificiais:

1. picos de \(f(t)\) produzem quedas exageradas em \(p(t)\);
2. vales ou frentes estreitas produzem aumentos exagerados em \(p(t)\), levando \(p(t)\) rapidamente para 1.

A regra atual reduz esses dois problemas:

- o limite \(\Delta p \geq -c\) impede quedas bruscas;
- o limite \(\Delta p \leq c\) mantém simetria com o caso \(f(t)=0\);
- a atenuação acima de \(p(t)=0.60\) impede que redes pequenas ou frentes momentaneamente pobres sejam empurradas rapidamente para o teto \(p=1\);
- o teto efetivo de incremento positivo em \(p(t)=0.75\) mantém a dinâmica em uma faixa supercrítica controlada, sem congelar quedas quando \(f(t)\) volta a crescer.

Assim, a modificação não muda o observável de estabilização: a parada continua baseada na estabilidade de \(p_i(t)\). O que muda é a lei de controle que gera \(p_i(t+1)\), tornando a série menos sensível a flutuações finitas de redes pequenas.

# Resumo operacional

No `growth_test` atual, a amostra cresce com altura dinâmica. Cada espécie viva é monitorada pela estabilidade da sua própria série \(p_i(t)\), após atingir ao menos altura \(L\). A estabilização é aceita quando a cauda blocada de \(p_i(t)\) passa no teste global de deriva, com tolerância relativa \(2.5\times 10^{-2}\), tolerância absoluta \(10^{-6}\), suavização 15 e blocos de 10 tempos. Após detectada a estabilização, a espécie ainda cresce até \(z_{\mathrm{stat}}+\lceil 2.5L\rceil\). A simulação para quando todas as espécies vivas satisfazem esse pós-crescimento, quando todas morrem/terminam, ou quando o limite duro de tempo é alcançado.
