"""Gera uma campanha curta em f_T para bond e node, sem executar simulacoes.

Na raiz do projeto: python3 python/run_samples.py
Depois, em shells/:
AUTO_RAM_JOBS=0 ./run_all.sh './minimum_bond_*.sh' './minimum_node_*.sh'

O filtro seleciona esta campanha, sem executar os shells antigos de bond/node.
O shell gerado ja faz sua propria medicao de RAM; AUTO_RAM_JOBS=0 evita
que run_all.sh execute uma sondagem adicional fora do numero de amostras.
"""

import os
from pathlib import Path

from src.run_samples_functions import shell_data


SCRIPT_DIR = Path(__file__).resolve().parent

# Mesma configuracao da secao MINIMUM ANALYSIS de jupyter/1Color_2D.ipynb.
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
type_lst = ['bond', 'node']  # use ['node'] para gerar somente os novos testes de node

# Centros exploratorios: 0.270709756315223 * (L/512)**(-0.305950695008207).
# Sao candidatos a testar, nao novos minimos ja validados.
ft_centro_bond_por_L = {
    512: 0.2707098,
    645: 0.2522436,
    813: 0.2349972,
    1024: 0.2189796,
    1448: 0.1969553,
    2048: 0.1771345,
    2896: 0.1593189,
    4096: 0.1432857,
    5793: 0.1288677,
    8192: 0.1159051,
    11585: 0.1042450,
    16384: 0.09375668,
}

# Tendencia suave de node: 0.20935383311317587 * (L/512)**(-0.2965301912047261).
# Suavidade nao garante estabilizacao; os novos valores precisam ser simulados.
ft_centro_node_por_L = {
    L: float(f'{0.20935383311317587 * (L/512)**(-0.2965301912047261):.6e}')
    for L in ft_centro_bond_por_L
}
ft_centro_por_tipo = {
    'bond': ft_centro_bond_por_L,
    'node': ft_centro_node_por_L,
}

L_lst = list(ft_centro_bond_por_L)
# Para uma primeira campanha menor:
# L_lst = [813, 2896, 4096, 16384]

# Apenas tres pontos por L: centro -0.5%, centro, centro +0.5%.
# Faixa mais estreita que os +/-2% da proposta inicial do notebook.
passo_relativo = 0.005
offsets = (-2, -1, 0, 1, 2)

# Orcamento original por tamanho, aplicado a cada f_T de bond e node.
# O mapa permanece correto mesmo ao selecionar apenas um subconjunto de L_lst.
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


def main():
    # Canoniza na precisao usada pelos diretorios fT_{ft:.6e}.
    jobs = []
    for type_perc in type_lst:
        for L in L_lst:
            valores = [
                float(f'{ft_centro_por_tipo[type_perc][L] * (1 + offset * passo_relativo):.6e}')
                for offset in offsets
            ]
            if len(set(valores)) != len(valores):
                raise ValueError(f'Passo pequeno demais: tipo={type_perc}, L={L}')
            if not all(0 < ft < 1 for ft in valores):
                raise ValueError(f'f_T fora do intervalo (0, 1): tipo={type_perc}, L={L}')
            jobs.extend((type_perc, L, ft) for ft in valores)

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
