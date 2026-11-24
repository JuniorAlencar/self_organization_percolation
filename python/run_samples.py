import os
from pathlib import Path
import numpy as np
from src.run_samples_functions import shell_data

# Garante que os scripts shell sejam gerados na pasta shells/ do projeto
os.chdir(Path(__file__).resolve().parent)
# DIM = 2 ==========================================
# L = 512, N_samples = 1400, max_steps = 1500
# L = 1024, N_samples = 1000, max_steps = 2000
# L = 1448, N_samples = 9000, max_steps = 2400
# L = 2048, N_samples = 800, max_steps = 3000
# L = 2896, N_samples = 600, max_steps = 3800
# L = 4096, N_samples = 400, max_steps = 4500
# L = 4598, N_samples = 360, max_steps = 4800
# L = 5161, N_samples = 330, max_steps = 5200
# L = 5793, N_samples = 300, max_steps = 5600
# L = 6502, N_samples = 265, max_steps = 6000
# L = 7298, N_samples = 230, max_steps = 6500
# L = 8192, N_samples = 200, max_steps = 7000
# L = 9742, N_samples = 175, max_steps = 7800
# L = 11585, N_samples = 150, max_steps = 8800
# L = 13777, N_samples = 125, max_steps = 9800
# L = 16384, N_samples = 100, max_steps = 11000

# DIM = 3 ==========================================
# L = 64, N_samples = 700, max_steps = 700
# L = 128, N_samples = 400, max_steps = 800
# L = 256, N_samples = 200, max_steps = 1000
# L = 512, N_samples = 100, max_steps = 1200
# L = 1024, N_samples = 50, max_steps = 1500
# L = 2048, N_samples = 25, max_steps = 2000

# ==============================================================================


# =============================================================================
# PARAMETROS DA SIMULACAO (2D 1-COLOR)
# =============================================================================
seed = -1
dim = 2
nc = 2
c = 0.00
f_T = 0.0
P0 = 0.2

rho = [1.0 / nc]
multi = True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'
InitialLayout = 'alternating'
ControlRule = 'linear'

# Configuração otimizada para cálculo das médias das caudas (DIM = 2)
CONFIG_2D = {
    #   L:    (N_samples, max_steps)
      512:  (1400,  1500),
     1024:  (1000,  2000),
     1448:  ( 900,  2400),
     2048:  ( 800,  3000),
     2896:  ( 600,  3800),
     4096:  ( 400,  4500),
     4598:  ( 360,  4800),
     5161:  ( 330,  5200),
     5793:  ( 300,  5600),
     6502:  ( 265,  6000),
     7298:  ( 230,  6500),
     8192:  ( 200,  7000),
     9742:  ( 175,  7800),
    11585:  ( 150,  8800),
    13777:  ( 125,  9800),
    16384:  ( 100, 11000),
}

# Configuração otimizada para cálculo das médias das caudas (DIM = 3)
# Configuração otimizada para cálculo das médias das caudas (DIM = 3)
CONFIG_3D = {
    #    L:    (N_samples, max_steps)
        64:  ( 700,   1000),
       128:  ( 400,   1200),
       256:  ( 200,  1300),
       512:  ( 100,  1400),
       645:  (  80,  1500),
       813:  (  65,  1600),
      1024:  (  50,  1700),
      1290:  (  40,  1750),
      1625:  (  30,  1800),
      2048:  (  25,  2000),
}

config = CONFIG_2D if dim == 2 else CONFIG_3D

#N_samples = [200, 50]
#L_lst = [16384]
#N_samples = [200, 50]
#p0 = 0.8
#p0 = 0.5
p0_2D = 0.8
p0_3D = 0.5
p0 = p0_2D if dim == 2 else p0_3D
p0_lst = np.linspace(0.01, 1.0, 40)
#ft_lst = np.linspace(0.01, 0.5, 40)
type_perc_lst = ['bond', 'node']

count = 0

for type_perc in type_perc_lst:
    for L, (num_runs, max_steps) in config.items():
#        for f_T in ft_lst:
        for p0 in p0_lst:
            p0_val = float(p0)
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
