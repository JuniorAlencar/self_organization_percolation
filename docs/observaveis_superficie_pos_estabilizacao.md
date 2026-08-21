---
title: "Observaveis Pos-Estabilizacao"
subtitle: "Superficie, volume e massa de clusters"
author: "self_organization_percolation"
date: "2026-08-11"
geometry: margin=2.2cm
fontsize: 11pt
---

# Escopo

Este documento resume os observaveis calculados quando a flag de observaveis de superficie esta ligada.

O objetivo e analisar a rede depois que o criterio de estabilizacao foi atingido. No instante em que uma especie estabiliza, o codigo salva:

$$z_{stat} = \text{altura maxima da rede no equilibrio}.$$

Depois disso, a rede continua crescendo ate:

$$z_{stop} = z_{stat} + \lceil 2.5L \rceil.$$

As propriedades de superficie e volume sao amostradas somente ate a regiao destacada:

$$z_{stat} \le z \le z_{stat}+L.$$

Esta e uma janela discreta inclusiva. Portanto, para contagens de sitios, o numero de camadas verticais consideradas e:

$$H = (z_{stat}+L) - z_{stat} + 1 = L+1.$$

Em 2D, a area lateral e $A=L$. Em 3D, a area lateral e $A=L^2$.

# Tempos Salvos

## `deltaT_sur`

Intervalo inteiro de passos usado entre amostras da superficie.

O valor padrao atual e:

$$\Delta T_{sur}=15.$$

## `t_sur`

Lista de indices lineares das amostras de superficie:

$$t_{sur} = [1,2,3,\ldots].$$

Esse tempo nao e o tempo fisico bruto da simulacao. Ele e um indice de amostra. Duas entradas consecutivas correspondem, em geral, a um intervalo de aproximadamente `deltaT_sur` passos fisicos.

## `t_vol`

Lista de indices lineares dos intervalos usados no calculo de volume entre duas superficies consecutivas:

$$t_{vol} = [1,2,3,\ldots].$$

Como o volume e definido entre duas superficies consecutivas, `t_vol` normalmente tem uma entrada a menos que `t_sur`.

# Superficie Superior

## `h_sur`

Perfil de altura da superficie superior dentro da regiao:

$$z_{stat} \le z \le z_{stat}+L.$$

Em 2D, `h_sur` tem um valor por coluna lateral $x$.

Em 3D, `h_sur` tem um valor por coluna lateral $(x,y)$, salvo de forma achatada pelo indice:

$$i = x + L y.$$

Para cada coluna, o codigo salva a maior altura ocupada pela especie dentro do slab. Se nao existir sitio ativo daquela especie na coluna dentro da janela, o valor salvo e `-1`.

## `f_sur`

Fracao de sitios da superficie superior exposta ao ar conectado pelo topo, restrita ao slab:

$$z_{stat} \le z \le z_{stat}+L.$$

Ela e normalizada pela area lateral:

$$f_{sur} = \frac{N_{superficie}}{A}.$$

Como uma superficie rugosa pode ter mais sitios expostos do que a area lateral projetada, `f_sur` pode ser maior que 1.

## `S_sur`

Numero bruto de sitios da superficie superior em cada amostra:

$$S_{sur} = f_{sur} A.$$

Esse observavel mede o tamanho da interface superior exposta, nao a massa do cluster.

# Rugosidade

## `w_sur`

Largura, ou rugosidade, do perfil `h_sur` em cada amostra:

$$w = \sqrt{\langle h^2\rangle - \langle h\rangle^2}.$$

O calculo usa apenas entradas validas de `h_sur`, isto e, ignora colunas com valor `-1`.

## `grad_sur`

Gradiente medio absoluto da superficie superior.

Em 2D:

$$grad = \langle |h(x+1)-h(x)| \rangle.$$

Em 3D:

$$grad = \langle |h(x+1,y)-h(x,y)|, |h(x,y+1)-h(x,y)| \rangle.$$

O calculo usa condicoes periodicas laterais e ignora pares onde pelo menos uma das alturas e `-1`.

Portanto, depois da correcao atual, `grad_sur` e calculado apenas usando o perfil `h_sur` restrito a:

$$z_{stat} \le z \le z_{stat}+L.$$

# Volume Entre Superficies

## `f_vol`

Fracao ocupada entre duas superficies superiores consecutivas.

Entre as amostras $k-1$ e $k$, o codigo conta os sitios ativos que ficaram entre os dois perfis:

$$h_{k-1}(x) < z \le h_k(x)$$

em 2D, ou

$$h_{k-1}(x,y) < z \le h_k(x,y)$$

em 3D.

O calculo tambem restringe os sitios ao slab:

$$z_{stat} \le z \le z_{stat}+L.$$

A normalizacao usa o volume discreto total do slab:

$$V_{max}=A(H),$$

com:

$$H=L+1.$$

Assim:

$$f_{vol} = \frac{N_{ocupado\ entre\ superficies}}{V_{max}}.$$

# Massa Para Dimensao Fractal

## `M_L`

Este e o observavel recomendado para estimar a dimensao fractal do cluster de percolacao.

O codigo encontra todos os clusters conectados da especie dentro do slab:

$$z_{stat} \le z \le z_{stat}+L.$$

Depois identifica quais clusters atravessam verticalmente a regiao, isto e, tocam simultaneamente:

$$z=z_{stat}$$

e

$$z=z_{stat}+L.$$

O valor salvo em `M_L` e a massa do maior cluster atravessante:

$$M(L) = \max\{s_i : C_i \text{ toca a base e o topo}\}.$$

Se nenhum cluster atravessa da base ao topo, o codigo salva:

$$M_L = 0.$$

Para extrair a dimensao fractal, a relacao esperada e:

$$M(L) \sim L^{D_f}.$$

Logo, em escala log-log:

$$\log M(L) = D_f \log L + \text{constante}.$$

## `M_cluster_sizes`

Lista com o tamanho de todos os clusters conectados encontrados dentro do slab:

$$z_{stat} \le z \le z_{stat}+L.$$

A lista e salva em ordem decrescente. Ela pode ser usada para estudar distribuicoes de tamanho de cluster, por exemplo $n_s$, ou para verificar se existe um cluster dominante.

# Processamento

O processamento separado dos observaveis de superficie gera um bundle proprio, sem poluir o dataframe principal.

As principais keys agregadas sao:

- `f_sur_mean`, `f_sur_std`, `f_sur_sem`
- `w_sur_mean`, `w_sur_std`, `w_sur_sem`
- `grad_sur_mean`, `grad_sur_std`, `grad_sur_sem`
- `S_sur_mean`, `S_sur_std`, `S_sur_sem`
- `f_vol_mean`, `f_vol_std`, `f_vol_sem`
- `M`
- `M_values`
- `M_cluster_sizes`
- `M_cluster_sizes_flat`

`M_values` guarda os valores de `M_L` de cada amostra. `M_cluster_sizes` guarda uma lista por amostra. `M_cluster_sizes_flat` junta todos os tamanhos de cluster em uma unica lista, util para histogramas.

# Resumo Conceitual

`f_sur`, `S_sur`, `h_sur`, `w_sur` e `grad_sur` descrevem a interface superior.

`f_vol` descreve o preenchimento entre duas interfaces consecutivas.

`M_L` descreve a massa do maior cluster atravessante dentro da regiao destacada e e o observavel apropriado para estimar a dimensao fractal de percolacao.

`M_cluster_sizes` descreve todos os clusters no subgrafo analisado, nao apenas o maior.
