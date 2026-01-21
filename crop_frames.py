#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from PIL import Image

# ===== CONFIGURAÇÃO =====

# Pasta de entrada (onde estão os frame_####.png)
INPUT_DIR = Path(
    "/home/light/Documents/self_organization_percolation/animation/"
    "3D_L256_nc4_rho0.25_k1.0e-06_Nt3000/positions_vs_time"
)

# Pasta de saída (onde serão salvos os recortes)
OUTPUT_DIR = Path(
    "/home/light/Documents/self_organization_percolation/animation/"
    "3D_L256_nc4_rho0.25_k1.0e-06_Nt3000/position_vs_time_without_background"
)

# Ponto onde você "clicaria" para pegar a cor de fundo
# (coordenadas em pixels a partir do canto superior esquerdo, começando em 0)
CLICK_X = 5
CLICK_Y = 5

# Tolerância para considerar um pixel como "fundo"
# 0 = só exatamente a mesma cor
# 10 ou 20 = permite variação pequena (útil se o fundo tiver anti-aliasing / gradiente leve)
THRESHOLD = 5


def is_foreground(pixel, bg, threshold):
    """
    Retorna True se o pixel for diferente do fundo, considerando um threshold
    simples na distância máxima entre canais.
    """
    # Normaliza para tupla
    if isinstance(pixel, int):
        pixel = (pixel,)
    if isinstance(bg, int):
        bg = (bg,)

    n = min(len(pixel), len(bg))
    diff = max(abs(pixel[i] - bg[i]) for i in range(n))
    return diff > threshold


def crop_image_remove_background(path_in, path_out,
                                 click_x=CLICK_X,
                                 click_y=CLICK_Y,
                                 threshold=THRESHOLD):
    """
    Abre uma imagem PNG, pega a cor de fundo em (click_x, click_y),
    encontra o retângulo mínimo que contém todos os pixels "não fundo"
    e salva o recorte em path_out.
    """
    img = Image.open(path_in).convert("RGBA")
    width, height = img.size
    pixels = img.load()

    # Pega cor de fundo na coordenada desejada
    if click_x < 0 or click_x >= width or click_y < 0 or click_y >= height:
        raise ValueError("CLICK_X/CLICK_Y fora dos limites da imagem.")

    bg_color = pixels[click_x, click_y]

    # Inicializa bounding box do conteúdo
    min_x, min_y = width, height
    max_x, max_y = -1, -1

    for y in range(height):
        for x in range(width):
            p = pixels[x, y]
            if is_foreground(p, bg_color, threshold):
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    # Se não achou nenhum pixel de conteúdo, copia a imagem inteira
    if max_x == -1:
        print(f"[AVISO] Nenhum conteúdo detectado em {path_in}, copiando inteira.")
        cropped = img
    else:
        # +1 em max_x/max_y porque o box do Pillow é [esquerda, superior, direita, inferior)
        box = (min_x, min_y, max_x + 1, max_y + 1)
        cropped = img.crop(box)

    # Garante pasta de saída
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Salva como PNG
    cropped.save(path_out, format="PNG")


def main():
    if not INPUT_DIR.is_dir():
        raise SystemExit(f"Pasta de entrada não existe: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    )

    if not files:
        print("Nenhum .png encontrado em", INPUT_DIR)
        return

    for f in files:
        out_path = OUTPUT_DIR / f.name
        print(f"Processando {f.name} -> {out_path.name}")
        crop_image_remove_background(f, out_path)

    print("Concluído.")


if __name__ == "__main__":
    main()

