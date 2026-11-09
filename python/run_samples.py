import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.run_samples_functions import (
    shell_data,
    custom_range,
    find_dynamic_file,
    load_filtered_dynamic,
    resolve_existing_ft_and_samples,
    build_all_convergence_jobs,
    generate_convergence_shells,
)

# Garante que os scripts shell sejam gerados na pasta shells/ do projeto
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# PARAMETROS DA SIMULACAO
# =============================================================================

seed = -1
dim = 2
nc = 1                      # Numero de cores (ex: 1, 2, 3, 4...)

type_lst = ['node', 'bond']  # 'node' (site) e/ou 'bond' percolation

# Lista de tamanhos L e orcamento de amostras por tamanho
# Para L=16384 em modo fractions, cada run gera 'fraction_samples' (ex: 30) fatias gigantes;
# recomenda-se 3 a 5 runs por script.
L_lst = [16384]
num_runs_lst = [50]
num_runs_por_L = dict(zip(L_lst, num_runs_lst))

c = 0.05
multi = True
Equilibration = 'false'
Properties = 'false'

# Modo e Layout para propriedades topologicas (d_bulk, d_min):
Mode = 'growth_test'           # 'fractions' calcula fatias topologicas; use 'growth_test' para crescimento continuo
InitialLayout = 'random'     # 'random' e o padrao para fracoes/topologia; 'clustered' para crescimento
ControlRule = 'linear'       # 'linear' ou 'relative'
ForceTopologicalCounts = False

# Parametros das fatias (slabs) no modo fractions:
fraction_samples = 30        # Quantidade de fatias estabilizadas coletadas por run
fraction_gap = 2.0           # Espacamento vertical entre fatias em unidades de L

# Fracao de densidade por cor (para nc cores simetricas: 1/nc)
rho = [1.0 / nc]

p0 = 0.8
P0 = 0.2                     # 0.2 e o padrao nas analises topologicas

# Valores de f_{T,min} para L=16384 (c = 0.05, dim = 2)
# Obtidos dos ajustes empiricos refinados:
#ft_por_tipo = {
    #'bond': 0.0874002,
    #'node': 0.0720032,
#}
ft_lst = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# =============================================================================
# GERACAO DOS SCRIPTS SHELL
# =============================================================================

for L in L_lst:
    num_runs = num_runs_por_L.get(L, 3)
    for type_perc in type_lst:
        #ftmin = ft_por_tipo.get(type_perc, 0.1)
        #ft_lst = np.linspace(ftmin, 0.3, 10)
        for ft in ft_lst:
            mode_tag = "" if Mode == "sop" else f"_{Mode}"
            layout_tag = "" if InitialLayout == "clustered" else f"_{InitialLayout}"
            exec_name = (
                f"topological_L_{L}_ft_{ft:.7g}_c_{c}_nc_{nc}_dim_{dim}"
                f"_p0_{p0}_P0_{P0:.3f}_M_{fraction_samples}"
                f"_gap_{fraction_gap:.3f}L{mode_tag}{layout_tag}_type_{type_perc}.sh"
            )

            shell_data(
                L=L,
                type_perc=type_perc,
                p0=p0,
                seed=seed,
                c=c,
                f_T=ft,
                dim=dim,
                num_colors=nc,
                num_runs=num_runs,
                rho=rho,
                exec_name=exec_name,
                P0=P0,
                equlibration=Equilibration,
                multi=multi,
                properties=Properties,
                mode=Mode,
                initial_layout=InitialLayout,
                control_rule=ControlRule,
                fraction_samples=fraction_samples,
                fraction_gap_over_L=fraction_gap,
                force_topological_counts=ForceTopologicalCounts,
            )
