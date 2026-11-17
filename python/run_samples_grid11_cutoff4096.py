import os
from pathlib import Path
import numpy as np

from src.run_samples_functions import shell_data

# Garante que os scripts shell sejam gerados na pasta shells/ do projeto
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# SCRIPT TEMPORARIO: GRADE DE 11 PONTOS PARA L_min_cutoff = 4096
# =============================================================================
# Grade completa (11 pontos):
#   Existentes (5): 4096, 5793, 8192, 11585, 16384
#   Novos (6):      4598, 5161, 6502, 7298, 9742, 13777
#
# Orcamento de amostras N_s seguindo o padrao (4096->400, 5793->300, 8192->200, 11585->150, 16384->100):
#   L = 4598  -> N_s = 360
#   L = 5161  -> N_s = 330
#   L = 6502  -> N_s = 265
#   L = 7298  -> N_s = 230
#   L = 9742  -> N_s = 175
#   L = 13777 -> N_s = 125

seed = -1
dim = 2
nc = 1
c = 0.05
p0 = 0.8
P0 = 0.2
rho = [1.0 / nc]
multi = True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'
InitialLayout = 'alternating'
ControlRule = 'linear'

L_new_por_N = {
    4598: 360,
    5161: 330,
    6502: 265,
    7298: 230,
    9742: 175,
    13777: 125,
}

ft_by_type = {
    'bond': np.round(np.arange(0.04, 0.305, 0.01), 2),
    'node': np.round(np.arange(0.03, 0.305, 0.01), 2),
}

count = 0
for type_perc, ft_lst in ft_by_type.items():
    for ft in ft_lst:
        for L, num_runs in L_new_por_N.items():
            exec_name = (
                f"grid11_L_{L}_ft_{ft:.2f}_c_{c}_nc_{nc}_dim_{dim}"
                f"_p0_{p0}_P0_{P0:.3f}_{Mode}_{InitialLayout}_{ControlRule}_type_{type_perc}.sh"
            )
            shell_data(
                L=L,
                type_perc=type_perc,
                p0=p0,
                seed=seed,
                c=c,
                f_T=float(ft),
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
            )
            count += 1

print(f"Total de scripts shell gerados em shells/: {count}")
