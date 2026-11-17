import os
from pathlib import Path

from src.run_samples_functions import shell_data

# Garante que os scripts shell sejam gerados na pasta shells/ do projeto
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# SCRIPT TEMPORARIO: AMOSTRAS RESTANTES PARA OUTLIERS DE nu_fit
# =============================================================================
# Relacao esperada L -> N_s:
# L = 2896  -> N_s = 600
# L = 4096  -> N_s = 400
# L = 5793  -> N_s = 300
# L = 8192  -> N_s = 200
# L = 11585 -> N_s = 150
# L = 16384 -> N_s = 100

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

# Lista de (type_perc, f_T, L, num_runs_restante)
missing_jobs = [
    # ---- bond ----
    ('bond', 0.04, 2896, 600),
    ('bond', 0.04, 4096, 96),
    ('bond', 0.04, 8192, 46),
    ('bond', 0.05, 4096, 96),
    ('bond', 0.05, 8192, 46),
    ('bond', 0.06, 4096, 96),
    ('bond', 0.06, 8192, 46),
    ('bond', 0.08, 4096, 96),
    ('bond', 0.08, 8192, 46),
    ('bond', 0.12, 4096, 96),
    ('bond', 0.12, 5793, 76),
    ('bond', 0.12, 8192, 46),
    ('bond', 0.12, 11585, 74),
    ('bond', 0.18, 4096, 96),
    ('bond', 0.18, 5793, 110),
    ('bond', 0.18, 8192, 46),
    ('bond', 0.18, 11585, 74),
    ('bond', 0.27, 4096, 96),
    ('bond', 0.27, 8192, 46),
    ('bond', 0.27, 11585, 58),
    # ---- node ----
    ('node', 0.04, 8192, 67),
    ('node', 0.05, 8192, 102),
    ('node', 0.05, 11585, 64),
    ('node', 0.06, 2896, 125),
    ('node', 0.06, 5793, 134),
    ('node', 0.06, 8192, 129),
    ('node', 0.06, 11585, 113),
    ('node', 0.07, 2896, 239),
    ('node', 0.07, 5793, 248),
    ('node', 0.07, 8192, 145),
    ('node', 0.07, 11585, 137),
    ('node', 0.08, 8192, 154),
    ('node', 0.09, 8192, 164),
    ('node', 0.10, 8192, 161),
    ('node', 0.10, 16384, 49),
    ('node', 0.11, 8192, 193),
    ('node', 0.12, 8192, 193),
]

for type_perc, ft, L, num_runs in missing_jobs:
    exec_name = (
        f"nu_outlier_L_{L}_ft_{ft:.4g}_c_{c}_nc_{nc}_dim_{dim}"
        f"_p0_{p0}_P0_{P0:.3f}_{Mode}_{InitialLayout}_{ControlRule}_type_{type_perc}.sh"
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
    )

print(f"Total de scripts shell gerados em shells/: {len(missing_jobs)}")
