import os
from pathlib import Path
import numpy as np
from src.run_samples_functions import shell_data

# Garante que os scripts shell sejam gerados na pasta shells/ do projeto
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# PARAMETROS DA SIMULACAO (2D 1-COLOR)
# =============================================================================
seed = -1
dim = 2
nc = 1
c = 0.05
f_T = 0.0
P0 = 0.2
rho = [1.0 / nc]
multi = True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'
InitialLayout = 'alternating'
ControlRule = 'linear'
#max_steps_lst = [8000, 16000]  # TMAX
max_steps_lst = [16000]  # TMAX
#L_lst = [4096, 16384]
#N_samples = [200, 50]
L_lst = [16384]
N_samples = [50]
#p0 = 0.8
p0 = 0.8
#p0_lst = np.linspace(0.01, 1.0, 40)
ft_lst = np.linspace(0.05, 0.5, 40)
type_perc_lst = ['bond', 'node']

count = 0
for type_perc in type_perc_lst:
    for L, num_runs, max_steps in zip(L_lst, N_samples, max_steps_lst):
        for f_T in ft_lst:
#        for p0 in p0_lst:
            #p0_val = float(p0)
            exec_name = (
                f"L_{L}_ft_{f_T:.4g}_c_{c}_nc_{nc}_dim_{dim}"
                f"_p0_{p0:.4f}_P0_{P0:.3f}_tmax_{max_steps}_{Mode}_{InitialLayout}_{ControlRule}_type_{type_perc}.sh"
            )
            shell_data(
                L=L,
                type_perc=type_perc,
                p0=p0,
                seed=seed,
                c=c,
                f_T=f_T,
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
                max_steps=max_steps,
            )
            count += 1

print(f"Sucesso! {count} scripts shell gerados na pasta shells/ (TMAX={max_steps}).")
