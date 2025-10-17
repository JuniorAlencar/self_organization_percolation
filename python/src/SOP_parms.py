# sop_params.py
# ------------------------------------------------------------
# Parâmetros do modelo SOP (multicor) com NT fixo por L
# e k escalando com n_c^beta (beta padrão = 0.5).
# - NT é global (independe de n_c): NT = epsilon0 * L^(d-1).
# - k = (kappa0 * n_c**beta) / NT.
# - NT arredondado p/ 2 algarismos significativos (ex.: 819 -> 820; 3277 -> 3300).
# - k arredondado p/ "mantissa inteira" na notação científica (ex.: 1.22e-05 -> 1.00e-05;
#   8.62e-06 -> 9.00e-06; 9.6e-07 -> 1.00e-06).
# ------------------------------------------------------------

from __future__ import annotations
import math
from dataclasses import dataclass

# ---------------------------
# Helpers de arredondamento
# ---------------------------

def _round_sig(x: float, sig: int = 2) -> float:
    """Arredonda x para 'sig' algarismos significativos (preserva potência de 10)."""
    if x == 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(abs(x)))
    factor = 10 ** (e - sig + 1)
    return round(x / factor) * factor

def _k_round_and_format(x: float, mantissa_decimals: int = 0) -> tuple[float, str]:
    """
    Arredonda 'x' para mantissa científica inteira e retorna (k_float, k_str).
    Ex.:
      1.22e-05 -> (1.0e-05, '1e-05')        [mantissa_decimals=0]
      8.62e-06 -> (9.0e-06, '9e-06')
      9.6e-07  -> (1.0e-06, '1e-06')
    Se mantissa_decimals=2, a string vira '1.00e-05', '9.00e-06', etc.
    """
    if x == 0 or not math.isfinite(x):
        s = f"{x:.{mantissa_decimals}e}"
        return x, s

    sign = -1 if x < 0 else 1
    a = abs(x)
    exp = math.floor(math.log10(a))
    mant = a / (10 ** exp)          # 1 <= mant < 10
    mant_i = int(round(mant))       # inteiro mais próximo

    # Ajustes para bordas: 0 -> 1e(exp-1), 10 -> 1e(exp+1)
    if mant_i == 0:
        exp -= 1
        mant_i = 1
    if mant_i == 10:
        mant_i = 1
        exp += 1

    k_float = sign * mant_i * (10 ** exp)
    if mantissa_decimals > 0:
        k_str = f"{sign*mant_i:.{mantissa_decimals}f}e{exp:+d}"
    else:
        # sem casas — mantissa inteira
        k_str = f"{sign*mant_i:d}e{exp:+d}"

    # limpa "e+0" / "e-0" se aparecer
    k_str = k_str.replace("+0", "+").replace("-0", "-")
    return k_float, k_str

# ---------------------------
# Dataclass de retorno "completo"
# ---------------------------

@dataclass
class SOPParams:
    # entradas usadas
    L: int
    d: int
    n_c: int
    epsilon0: float
    kappa0: float
    beta: float
    # saídas cruas
    NT_raw: float
    k_raw: float
    # saídas arredondadas
    NT: int
    k: float
    k_str: str
    # observações
    notes: str

# ---------------------------
# Funções públicas
# ---------------------------

def sop_choose_params(
    L: int,
    n_c: int,
    d: int = 3,
    beta: float = 0.5,
    epsilon0: float = 0.05,
    kappa0: float = 0.010,
    mantissa_decimals: int = 0,
) -> SOPParams:
    """
    Calcula NT e k para o SOP multicor com NT fixo por L.

    Definições:
      - Frente global (indep. de n_c):
           NT_raw = epsilon0 * L^(d-1)
      - NT arredondado p/ 2 algarismos significativos (ex.: 819->820; 3277->3300)
      - Ganho cru:
           k_raw  = (kappa0 * n_c**beta) / NT
      - k arredondado p/ mantissa científica inteira (ex.: 1.22e-05->1.00e-05)

    Recomendações usuais:
      - epsilon0 ∈ [0.03, 0.06]  (fração da área da frente)
      - kappa0   ∈ [0.007, 0.015] (ganho adimensional alvo κ = k·NT)
      - beta = 0.0 (sem compensação), 0.5 (**recomendado**), 1.0 (compensação total)

    Parâmetros:
      L, n_c, d, beta, epsilon0, kappa0: conforme descrito.
      mantissa_decimals: casas decimais para a mantissa textual de k ('1e-05' vs '1.00e-05').

    Retorna:
      SOPParams com campos crus e arredondados + nota explicativa.
    """
    NT_raw = float(epsilon0) * (L ** (d - 1))
    NT = int(_round_sig(NT_raw, sig=2))

    k_raw = (float(kappa0) * (n_c ** float(beta))) / NT if NT > 0 else float("nan")
    k, k_str = _k_round_and_format(k_raw, mantissa_decimals=mantissa_decimals)

    notes = (
        "NT é global (não depende de n_c) e segue NT = epsilon0 * L^(d-1). "
        "k cresce como n_c^beta: k = (kappa0 * n_c^beta) / NT. "
        "Faixas recomendadas: epsilon0 ∈ [0.03, 0.06], kappa0 ∈ [0.007, 0.015], "
        "beta ∈ {0.0 (sem compensação), 0.5 (parcial, recomendado), 1.0 (total)}. "
        f"Usado: epsilon0={epsilon0}, kappa0={kappa0}, beta={beta}."
    )

    return SOPParams(
        L=L, d=d, n_c=n_c,
        epsilon0=epsilon0, kappa0=kappa0, beta=beta,
        NT_raw=NT_raw, k_raw=k_raw,
        NT=NT, k=k, k_str=k_str,
        notes=notes,
    )

def sop_choose_NT_k(
    L: int,
    n_c: int,
    d: int = 3,
    beta: float = 0.5,
    epsilon0: float = 0.05,
    kappa0: float = 0.010,
    mantissa_decimals: int = 0,
) -> tuple[int, float, str]:
    """
    Retorna apenas (NT, k_float, k_str), já com os arredondamentos.

      NT  = round_sig(epsilon0 * L^(d-1), 2)
      k   = (kappa0 * n_c**beta) / NT  -> arredondado p/ mantissa inteira
      k_str = representação limpa em notação científica ('3e-05' ou '3.00e-05' se mantissa_decimals=2)
    """
    NT_raw = float(epsilon0) * (L ** (d - 1))
    NT = int(_round_sig(NT_raw, sig=2))
    k_raw = (float(kappa0) * (n_c ** float(beta))) / NT if NT > 0 else float("nan")
    k_float, k_str = _k_round_and_format(k_raw, mantissa_decimals=mantissa_decimals)
    return NT, k_float, k_str

# ---------------------------
# Demonstração rápida
# ---------------------------
if __name__ == "__main__":
    # Exemplos: L=128 e 512, n_c=4 e 8, beta default (0.5)
    for L in (128, 512):
        for n_c in (4, 8):
            NT, k, kstr = sop_choose_NT_k(L=L, n_c=n_c, mantissa_decimals=0)
            print(f"L={L:4d}, n_c={n_c:2d} -> NT={NT:5d}, k={kstr}")
