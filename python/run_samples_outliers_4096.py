import os
from pathlib import Path
from src.run_samples_functions import shell_data

# Garante que a execucao ocorra no diretorio python/ para gravar em shells/
os.chdir(Path(__file__).resolve().parent)

# =============================================================================
# PARAMETROS DA SIMULACAO (PADRAO 2D 1-COLOR)
# =============================================================================
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

# =============================================================================
# AMOSTRAGEM CIRURGICA PARA OS OUTLIERS DE L_cutoff = 4096 (faixa de 95%)
# Meta: Reduzir chi2_red para <= 2.114 (nu_dof = 9) nos pontos discrepantes
# =============================================================================
# Cada tupla contem: (type_perc, f_T, L, num_runs_novas)
outlier_jobs_4096 = [
    # -------------------------------------------------------------------------
    # 1. BOND: f_T = 0.08 (chi2_red atual = 2.296 | meta: <= 2.114)
    # Culpado: L = 9742 (residuo r = -3.71, N_s atual: 352) -> dobrar amostras
    # Reforco preventivo: L = 4096 (r = +1.65)
    # -------------------------------------------------------------------------
    ('bond', 0.08, 9742, 350),
    ('bond', 0.08, 4096, 200),

    # -------------------------------------------------------------------------
    # 2. BOND: f_T = 0.11 (chi2_red atual = 2.864 | meta: <= 2.114)
    # Culpado: L = 8192 (residuo r = -3.46, N_s atual: 355) -> dobrar amostras
    # Reforco preventivo: L = 6502 (r = -1.90)
    # -------------------------------------------------------------------------
    ('bond', 0.11, 8192, 350),
    ('bond', 0.11, 6502, 200),

    # -------------------------------------------------------------------------
    # 3. BOND: f_T = 0.39 (chi2_red atual = 2.764 | meta: <= 2.114)
    # Culpados: L = 8192 (r = -2.90) e L = 6502 (r = -2.24)
    # -------------------------------------------------------------------------
    ('bond', 0.39, 8192, 200),
    ('bond', 0.39, 6502, 200),
    ('bond', 0.39, 5161, 170),
    ('bond', 0.39, 11585, 150),

    # -------------------------------------------------------------------------
    # 4. NODE: f_T = 0.13 (chi2_red atual = 2.883 | meta: <= 2.114)
    # Culpados: L = 5161 (r = -2.78) e L = 9742 (r = +2.74)
    # Reforco preventivo: L = 8192 (r = +1.93)
    # -------------------------------------------------------------------------
    ('node', 0.13, 5161, 300),
    ('node', 0.13, 9742, 250),
    ('node', 0.13, 8192, 200),
]

# (Opcional) Se quiser aumentar o R2 na regiao quase-estatica (f_T <= 0.05),
# descomente as linhas abaixo para reduzir a incerteza estatistica em f_T baixo:
optional_low_ft_jobs = [
    # ('bond', 0.04, 5161, 200),
    # ('bond', 0.04, 5793, 200),
    # ('bond', 0.05, 5161, 200),
    # ('bond', 0.05, 6502, 200),
]

jobs_to_run = outlier_jobs_4096 + optional_low_ft_jobs

count = 0
for type_perc, ft, L, num_runs in jobs_to_run:
    exec_name = (
        f"outlier4096_L_{L}_ft_{ft:.4g}_c_{c}_nc_{nc}_dim_{dim}"
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
    count += 1

print(f"Sucesso! {count} scripts shell gerados na pasta shells/ para os outliers de L_cutoff=4096.")
