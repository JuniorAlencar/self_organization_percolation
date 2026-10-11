"""Gera shells de simulacao para equalizar N_samples em todos os conjuntos de f_T analisados
(truncando sempre L >= 1024 em todos os casos):
  1) minimum_power: pontos de f_{T,min}(L) (ajuste Power-law do artigo, L >= 1024)
  2) conv_const:    cortes de f_T constante decrescente (Tabela 1 de convergencia, L >= 1024)
  3) conv_scaled:   curvas proporcionais f_T(L) = lambda * f_{T,min}(L) (Tabela 2 de convergencia, L >= 1024)

Uso:
    python3 python/run_samples.py

Depois, em shells/:
    # Para rodar eventual diferenca no f_{T,min}(L) (L >= 1024):
    AUTO_RAM_JOBS=0 ./run_all.sh './minimum_power_*.sh'

    # Para rodar os pontos de f_T constante (Tabela 1, L >= 1024):
    AUTO_RAM_JOBS=0 ./run_all.sh './conv_const_*.sh'

    # Para rodar os pontos de f_T(L) = lambda * f_{T,min}(L) (Tabela 2, L >= 1024):
    AUTO_RAM_JOBS=0 ./run_all.sh './conv_scaled_*.sh'
"""

import os
from pathlib import Path
import numpy as np
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
type_lst = ['node', 'bond']
L_MIN = 1024

# Metas totais de amostras solicitadas por tamanho L (para L >= 1024)
target_samples_por_L = {
    1024: 1000,
    1448: 900,
    2048: 800,
    2896: 600,
    4096: 500,
    5793: 400,
    8192: 300,
    11585: 200,
    16384: 100,
}

# Valores de f_T constante da grade analisados na Tabela 1 (onde todos os 9 tamanhos L >= 1024 estao ativos)
CONST_FT_GRID = {
    'node': [
        0.3000000,
        0.2847368,
        0.2694737,
        0.2542105,
        0.2389474,
        0.2236842,
        0.2084211,
        0.1931579,
        0.1778947,
        0.1626316,
    ],
    'bond': [
        0.3000000,
        0.2847368,
        0.2694737,
        0.2542105,
        0.2389474,
        0.2236842,
        0.2084211,
        0.1931579,
    ],
}


def find_dynamic_file() -> Path:
    candidates = [
        SCRIPT_DIR.parent / 'SOP_data/all_data_dynamic.dat',
        Path('../SOP_data/all_data_dynamic.dat'),
        Path('SOP_data/all_data_dynamic.dat'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Nao foi possivel encontrar 'SOP_data/all_data_dynamic.dat'.")


def load_filtered_dynamic() -> pd.DataFrame:
    p_data = find_dynamic_file()
    raw = pd.read_csv(p_data, sep=r'\s+')
    mask = (
        (raw['dim'] == dim)
        & (raw['nc'] == nc)
        & (raw['order'] == 0)
        & (raw['stat_window'] == 0)
        & (raw['control_rule'] == ControlRule)
        & np.isclose(raw['c'], c)
        & np.isclose(raw['p0'], p0)
        & np.isclose(raw['P0'], P0)
        & (raw['L'] >= L_MIN)
    )
    sub = raw.loc[mask].copy()
    sub['stab'] = sub['N_samples_perc'] / sub['N_samples']
    return sub


def resolve_existing_ft_and_samples(
    df_sub: pd.DataFrame, type_perc: str, L: int, ft_target: float, atol: float = 1e-4
) -> tuple[float, int]:
    """
    Procura em all_data_dynamic.dat se ja existe uma serie em (type_perc, L) com f_T
    dentro de `atol` de `ft_target`. Se existir, retorna o f_T exato gravado e seu N_samples.
    Caso contrario, retorna (round(ft_target, 7), 0).
    """
    cand = df_sub[(df_sub['type_perc'] == type_perc) & (df_sub['L'] == L)].copy()
    if cand.empty:
        return round(float(ft_target), 7), 0
    cand['diff'] = (cand['f_T'] - ft_target).abs()
    near = cand[cand['diff'] <= atol].sort_values('diff')
    if near.empty:
        return round(float(ft_target), 7), 0
    best = near.iloc[0]
    return float(best['f_T']), int(best['N_samples'])


def build_all_convergence_jobs(df_sub: pd.DataFrame):
    """
    Monta todos os pontos (prefixo, descricao_grupo, type_perc, L, f_T, curr_samples, target_samples, needed)
    truncando sempre L >= 1024 para:
      1) minimum_power: pontos de f_{T,min}(L) (L >= 1024)
      2) conv_const:    cortes de f_T constante decrescente (Tabela 1, L >= 1024)
      3) conv_scaled:   escalonamento proporcional f_T(L) = lambda * f_{T,min}(L) (Tabela 2, L >= 1024)
    evitando duplicatas de (type_perc, L, f_T).
    """
    seen_keys = set()
    all_records = []

    def add_point(prefix: str, group_desc: str, type_perc: str, L: int, ft_req: float, atol: float = 1e-4):
        if int(L) < L_MIN or int(L) not in target_samples_por_L:
            return
        ft_exact, curr = resolve_existing_ft_and_samples(df_sub, type_perc, L, ft_req, atol=atol)
        key = (type_perc, int(L), f'{ft_exact:.6e}')
        if key in seen_keys:
            return
        seen_keys.add(key)
        target = target_samples_por_L[int(L)]
        needed = max(0, target - curr)
        all_records.append({
            'prefix': prefix,
            'group': group_desc,
            'type_perc': type_perc,
            'L': int(L),
            'f_T': ft_exact,
            'curr': curr,
            'target': target,
            'needed': needed,
        })

    # 1. Pontos de f_{T,min}(L) (smooth_refined.csv, L >= 1024)
    for tp in type_lst:
        ref_path = SCRIPT_DIR.parent / f'SOP_data/ft_min_max_2D_{tp}_smooth_refined.csv'
        if ref_path.exists():
            df_ref = pd.read_csv(ref_path)
            df_ref = df_ref[df_ref['L'] >= L_MIN].sort_values('L')
            for _, r in df_ref.iterrows():
                add_point('minimum_power', 'f_T,min(L)', tp, int(r['L']), float(r['f_T']), atol=2e-5)

    # 2. Pontos de f_T constante (Tabela 1, L >= 1024)
    df_stab = df_sub[(df_sub['stab'] >= 0.98) & (df_sub['N_samples'] >= 5) & (df_sub['p_mean'] < 0.9)]
    for tp in type_lst:
        for ft_c in CONST_FT_GRID[tp]:
            s_ft = df_stab[
                (df_stab['type_perc'] == tp)
                & (df_stab['L'] >= L_MIN)
                & np.isclose(df_stab['f_T'], ft_c, atol=2e-5)
            ].sort_values('L')
            for _, r in s_ft.iterrows():
                add_point('conv_const', f'const f_T={ft_c:.4f}', tp, int(r['L']), float(r['f_T']), atol=2e-5)

    # 3. Pontos de f_T(L) = lambda * f_{T,min}(L) (Tabela 2, L >= 1024)
    for tp in type_lst:
        ref_path = SCRIPT_DIR.parent / f'SOP_data/ft_min_max_2D_{tp}_smooth_refined.csv'
        df_ref = pd.read_csv(ref_path)
        df_ref = df_ref[df_ref['L'] >= L_MIN].sort_values('L')
        max_lam = 0.30 / df_ref['f_T'].max()
        lams = np.linspace(max_lam, 1.0, 6)[:-1]  # lambda = 1.0 ja esta em minimum_power
        for lam in lams:
            for _, r in df_ref.iterrows():
                L_val = int(r['L'])
                ft_lam = float(lam * r['f_T'])
                add_point('conv_scaled', f'lambda={lam:.2f}*f_T,min', tp, L_val, ft_lam, atol=1e-4)

    return pd.DataFrame(all_records)


def main():
    df_sub = load_filtered_dynamic()
    df_jobs = build_all_convergence_jobs(df_sub)

    print('=' * 105)
    print(f'VERIFICACAO DE AMOSTRAS E GERACAO DE SHELLS PARA L >= {L_MIN} (TODOS OS CASOS)')
    print('=' * 105)

    # Limpa eventuais shells antigos com esses prefixos antes de regerar
    shells_dir = SCRIPT_DIR.parent / 'shells'
    shells_dir.mkdir(parents=True, exist_ok=True)
    for pat in ['minimum_power_*.sh', 'conv_const_*.sh', 'conv_scaled_*.sh']:
        for old_f in shells_dir.glob(pat):
            old_f.unlink()

    active_jobs = []
    for _, row in df_jobs.iterrows():
        prefix = row['prefix']
        grp = row['group']
        tp = row['type_perc']
        L = int(row['L'])
        ft = float(row['f_T'])
        curr = int(row['curr'])
        target = int(row['target'])
        needed = int(row['needed'])

        exec_name = (
            f'{prefix}_{tp}_L_{L}_ft_{ft:.6e}_c_{c:g}_nc_{nc}_dim_{dim}'
            f'_p0_{p0:g}_P0_{P0:g}_{ControlRule}_{Mode}_{InitialLayout}.sh'
        )
        if needed > 0:
            active_jobs.append((prefix, grp, tp, L, ft, curr, target, needed, exec_name))

    # Resumo por grupo e tipo
    for prefix, title in [
        ('minimum_power', f'1) PONTOS DE f_{{T,min}}(L) (L >= {L_MIN})'),
        ('conv_const', f'2) TABELA 1: PONTOS DE f_T CONSTANTE DECRESCENTE (L >= {L_MIN})'),
        ('conv_scaled', f'3) TABELA 2: PONTOS DE f_T(L) = lambda * f_{{T,min}}(L) (L >= {L_MIN})'),
    ]:
        sub_df = df_jobs[df_jobs['prefix'] == prefix]
        pend_df = sub_df[sub_df['needed'] > 0]
        print(f'\n{title}')
        print(f'  Total de pontos verificados: {len(sub_df)} | Com amostras faltantes: {len(pend_df)} | Amostras a rodar: {pend_df["needed"].sum()}')
        if not pend_df.empty:
            print('-' * 105)
            print(f"{'Grupo':>22} | {'Tipo':>5} | {'L':>6} | {'f_T':>12} | {'Atual':>6} | {'Alvo':>6} | {'Diferenca (+N)':>14}")
            print('-' * 105)
            for _, r in pend_df.iterrows():
                print(
                    f"{r['group']:>22} | {r['type_perc']:>5} | {int(r['L']):6d} | "
                    f"{r['f_T']:12.7f} | {int(r['curr']):6d} | {int(r['target']):6d} | +{int(r['needed']):13d}"
                )

    # Grava os scripts shell
    previous_dir = Path.cwd()
    try:
        os.chdir(SCRIPT_DIR)
        for prefix, grp, tp, L, ft, curr, target, num_runs, exec_name in active_jobs:
            shell_data(
                L, tp, p0, seed, c, ft, dim, nc, num_runs, [1 / nc],
                exec_name, P0, Equilibration, multi,
                properties=Properties, mode=Mode,
                initial_layout=InitialLayout, control_rule=ControlRule,
            )
    finally:
        os.chdir(previous_dir)

    print('\n' + '=' * 105)
    for prefix in ['minimum_power', 'conv_const', 'conv_scaled']:
        cnt = sum(1 for j in active_jobs if j[0] == prefix)
        runs = sum(j[7] for j in active_jobs if j[0] == prefix)
        print(f"  Prefixo '{prefix}_*.sh': {cnt:3d} arquivos gerados ({runs:6d} amostras faltantes)")
    print(f"  TOTAL GERAL (L >= {L_MIN}): {len(active_jobs)} arquivos shell gerados em shells/ ({sum(j[7] for j in active_jobs)} amostras)")
    print('=' * 105)
    print('\nComandos para executar em shells/:')
    print('  cd shells')
    print("  # 1. Para completar o unico ponto faltante de f_{T,min}(L) (node L=5793, +102 samples):")
    print("  AUTO_RAM_JOBS=0 ./run_all.sh './minimum_power_*.sh'\n")
    print("  # 2. Para igualar todas as amostras da Tabela 1 (f_T constante decrescente, L >= 1024):")
    print("  AUTO_RAM_JOBS=0 ./run_all.sh './conv_const_*.sh'\n")
    print("  # 3. Para igualar todas as amostras da Tabela 2 (f_T(L) = lambda * f_{T,min}(L), L >= 1024):")
    print("  AUTO_RAM_JOBS=0 ./run_all.sh './conv_scaled_*.sh'\n")


if __name__ == '__main__':
    main()
