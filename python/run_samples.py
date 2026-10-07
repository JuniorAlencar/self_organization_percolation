"""Gera shells de simulacao no f_T_min para bond e node a partir do dataframe de minimos.

Na raiz do projeto: python3 python/run_samples.py
Depois, em shells/:
AUTO_RAM_JOBS=0 ./run_all.sh './minimum_node_*.sh' './minimum_bond_*.sh'
ou:
AUTO_RAM_JOBS=0 ./run_all.sh './minimum_*.sh'

Os shells gerados executam amostras exatamente no ft_min dos plots do artigo,
acumulando estatistica nas series temporais para reduzir as incertezas de p*.
"""

import os
from pathlib import Path
import pandas as pd

from src.run_samples_functions import shell_data


SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# PARAMETROS DA SIMULACAO (alinhados a jupyter/1Color_2D.ipynb)
# =============================================================================
seed = -1
dim = 2
nc = 1
c = 0.05
p0 = 0.8
P0 = 0.2
multi = True
Equilibration = 'false'
Properties = 'false'
Mode = 'growth_test'
InitialLayout = 'clustered'
ControlRule = 'linear'
type_lst = ['node', 'bond']  # node e bond percolation

# Lista de tamanhos L
L_lst = [512, 645, 813, 1024, 1448, 2048, 2896, 4096, 5793, 8192, 11585, 16384]

# Orcamento de amostras adicionais por tamanho L
num_runs_por_L = {
    512: 700,
    645: 600,
    813: 550,
    1024: 500,
    1448: 450,
    2048: 400,
    2896: 200,
    4096: 150,
    5793: 100,
    8192: 75,
    11585: 60,
    16384: 50,
}


def find_csv(type_perc: str) -> Path:
    """Localiza o arquivo CSV de minimos para o tipo de percolacao."""
    candidates = [
        SCRIPT_DIR.parent / f"SOP_data/ft_min_max_2D_{type_perc}_{ControlRule}.csv",
        Path(f"../SOP_data/ft_min_max_2D_{type_perc}_{ControlRule}.csv"),
        Path(f"SOP_data/ft_min_max_2D_{type_perc}_{ControlRule}.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Arquivo CSV para {type_perc} nao encontrado em: {candidates}")


def build_jobs():
    """Carrega o dataframe de minimos para cada type_perc e extrai f_T_min."""
    jobs = []
    for type_perc in type_lst:
        csv_path = find_csv(type_perc)
        df = pd.read_csv(csv_path)

        # Filtra pela concentracao e parametros de controle
        df_sub = df[(df['c'] == c) & (df['f0'] == P0)]

        for L in L_lst:
            row = df_sub[df_sub['L'] == L]
            if row.empty:
                print(f"[AVISO] L={L} nao encontrado para {type_perc} em {csv_path.name}.")
                continue
            fT = float(row.iloc[0]['f_T_min'])
            jobs.append((type_perc, L, fT))

    return jobs


def main():
    jobs = build_jobs()

    # shell_data escreve em ../shells: permite chamar o gerador de qualquer pasta.
    previous_dir = Path.cwd()
    try:
        os.chdir(SCRIPT_DIR)
        for type_perc, L, ft in jobs:
            num_runs = num_runs_por_L[L]
            exec_name = (
                f'minimum_{type_perc}_L_{L}_ft_{ft:.6e}_c_{c:g}_nc_{nc}_dim_{dim}'
                f'_p0_{p0:g}_P0_{P0:g}_{ControlRule}_{Mode}_{InitialLayout}.sh'
            )
            shell_data(
                L, type_perc, p0, seed, c, ft, dim, nc, num_runs, [1/nc],
                exec_name, P0, Equilibration, multi,
                properties=Properties, mode=Mode,
                initial_layout=InitialLayout, control_rule=ControlRule,
            )
    finally:
        os.chdir(previous_dir)

    total_runs = sum(num_runs_por_L[L] for _, L, _ in jobs)
    print(f'Gerados {len(jobs)} shells de {", ".join(type_lst)}, {total_runs} simulacoes no total.')
    print(f'Amostras por ponto em cada L: { {L: num_runs_por_L[L] for L in L_lst} }')
    print('Em shells/, execute:')
    filters = ' '.join(f"'./minimum_{type_perc}_*.sh'" for type_perc in type_lst)
    print(f'AUTO_RAM_JOBS=0 ./run_all.sh {filters}')


if __name__ == '__main__':
    main()
