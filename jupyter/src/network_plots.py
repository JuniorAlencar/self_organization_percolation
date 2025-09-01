import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw, ImageFont
import os, gc
from PIL import Image
import pandas as pd
import imageio.v2 as imageio

def animation_network(start_frame, end_frame, filepath, output_dir, font_path, font_size=28):
    # === SETTINGS ===
    os.makedirs(output_dir, exist_ok=True)

    # Load network data
    data = np.load(filepath)
    activation_times = data["network"]
    nrows, ncols = activation_times.shape

    # Group segments by activation time
    segments_by_time = {t: [] for t in range(end_frame + 1)}

    for i in range(nrows):
        for j in range(ncols):
            t = activation_times[i, j]
            if t >= 0:
                # Horizontal bond
                if j + 1 < ncols and activation_times[i, j + 1] >= 0:
                    t_bond = int(max(t, activation_times[i, j + 1]))  # 🔧 Force Python int
                    if t_bond <= end_frame:
                        segments_by_time[t_bond].append([(j, i), (j + 1, i)])
                # Vertical bond
                if i + 1 < nrows and activation_times[i + 1, j] >= 0:
                    t_bond = int(max(t, activation_times[i + 1, j]))  # 🔧 Force Python int
                    if t_bond <= end_frame:
                        segments_by_time[t_bond].append([(j, i), (j, i + 1)])

    # Load accumulated black segments up to the starting frame
    accumulated_black = []
    for t in range(start_frame):
        accumulated_black.extend(segments_by_time.get(t, []))

    # Load font for timestamp
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found at: {font_path}")
    font = ImageFont.truetype(font_path, font_size)

    # Generate frames from start_frame to end_frame - 1
    for t in range(start_frame, end_frame):
        red_segments = segments_by_time[t]
        black_segments = accumulated_black

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, ncols)
        ax.set_ylim(0, nrows)  # 🔁 Y axis goes bottom → top
        ax.set_aspect("equal")
        ax.axis("off")

        if black_segments:
            ax.add_collection(LineCollection(black_segments, colors="black", linewidths=0.25))
        if red_segments:
            ax.add_collection(LineCollection(red_segments, colors="red", linewidths=0.5))

        filename = os.path.join(output_dir, f"frame_{t:03d}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Add timestamp after saving figure
        image = Image.open(filename).convert("RGB")
        draw = ImageDraw.Draw(image)
        text = f"t = {t}"
        img_width, img_height = image.size
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = img_width - text_width - 10
        y = 10

        # Optional: white box background for visibility
        draw.rectangle(
            [x - 4, y - 2, x + text_width + 4, y + text_height + 2],
            fill="white"
        )
        draw.text((x, y), text, fill="black", font=font)
        image.save(filename)

        # Accumulate newly activated segments
        accumulated_black.extend(red_segments)
        del fig, ax, red_segments
        gc.collect()

        print(f"[✔] Frame {t} saved with timestamp.")

    print(f"[✅] All frames generated and timestamped in: {output_dir}")


def animation_properties(filename, output_dir, start_frame, end_frame):
    dir = "../animation/"
    # === Carregar os dados ===
    pt_df = pd.read_csv(dir + "pt_" + filename)
    nt_df = pd.read_csv(dir + "Nt_" + filename)

    t_values = pt_df.iloc[:, 0].values
    pt_values = pt_df.iloc[:, 1:].values
    nt_values = nt_df.iloc[:, 1:].values
    labels = pt_df.columns[1:]

    # === Estilo gráfico ===
    fontsize_ticks = 13
    thickness = 1.4
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    os.makedirs(output_dir, exist_ok=True)

    # === Loop principal para gerar os frames ===
    for frame in range(start_frame, end_frame + 1):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

        # --- Subplot N(t) ---
        for i in range(nt_values.shape[1]):
            axs[0].plot(t_values[:frame], nt_values[:frame, i], color=colors[i], lw=2)
        axs[0].axhline(y=200, color="gray", linestyle="--", linewidth=1.2)
        axs[0].set_xlim(0, 1000)
        axs[0].set_ylim(0, 1000)
        axs[0].set_ylabel("$N(t)$", fontsize=fontsize_ticks)
        axs[0].tick_params(axis='both', labelsize=fontsize_ticks,
                        width=thickness, length=4.0, direction='in')
        for spine in axs[0].spines.values():
            spine.set_linewidth(thickness)

        # --- Subplot p(t) ---
        for i in range(pt_values.shape[1]):
            axs[1].plot(t_values[:frame], pt_values[:frame, i], color=colors[i], lw=2)
        axs[1].axhline(y=0.5, color="gray", linestyle="--", linewidth=1.2)
        axs[1].set_xlim(0, 1000)
        axs[1].set_ylim(0.3, 1.0)
        axs[1].set_xlabel("t", fontsize=fontsize_ticks)
        axs[1].set_ylabel("$p(t)$", fontsize=fontsize_ticks)
        axs[1].tick_params(axis='both', labelsize=fontsize_ticks,
                        width=thickness, length=4.0, direction='in')
        for spine in axs[1].spines.values():
            spine.set_linewidth(thickness)

        # === Salvar ===
        plt.tight_layout()
        filename = os.path.join(output_dir, f"frame_{frame:03d}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[✔] Frame {frame} salvo: {filename}")


def create_gif_with_cleanup(frames_folder, output_name, delay=0.05, scale=1.0):
    """
    Create a GIF from a sequence of PNG frames, optionally resizing them.
    Releases memory explicitly for Jupyter notebook usage.
    
    Parameters:
        frames_folder (str): Path to the folder containing .png frames.
        output_name (str): Output file name for the resulting GIF.
        delay (float): Time (in seconds) between frames.
        scale (float): Rescaling factor (1.0 = original size).
    """
    # Collect all .png files in sorted order
    frame_files = sorted([
        os.path.join(frames_folder, f)
        for f in os.listdir(frames_folder)
        if f.endswith(".png")
    ])

    print(f"[ℹ] Total frames: {len(frame_files)}")
    print(f"[▶] Generating GIF: {output_name}")

    # Open GIF writer
    with imageio.get_writer(output_name, mode='I', duration=delay) as writer:
        for i, frame_path in enumerate(frame_files, 1):
            # Open image
            img = Image.open(frame_path)

            # Optionally resize
            if scale != 1.0:
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.ANTIALIAS
                )

            # Write image to GIF
            writer.append_data(np.array(img))
            print(f"[✔] Frame {i}/{len(frame_files)} added: {os.path.basename(frame_path)}")

            # Release memory for this frame
            img.close()
            del img
            gc.collect()

    # Final cleanup
    del frame_files
    gc.collect()

    print(f"[✅] Final GIF saved: {output_name}")

def plot_bond_network(
    filepath,
    num_colors: int,
    savepath=None,
    dpi=600,
    min_density=1,
    color_map=None,          # dict[int,str] opcional: {valor_ativo: cor}
    linewidth=0.25,
    figsize=(8, 10),
    show_legend=True
):
    """
    Plota ligações entre sítios ativos de MESMA cor, conforme num_colors.

    Regras:
      - num_colors = 1 -> ativos = {+1} (cor cinza por padrão)
      - num_colors >= 2 -> ativos = {+2, +3, ..., +(num_colors+1)}
      - valores negativos e -1 são ignorados
    """
    if num_colors < 1:
        raise ValueError("num_colors deve ser >= 1")

    # Define o conjunto de valores ativos conforme a regra solicitada
    if num_colors == 1:
        active_values = (1,)
    else:
        active_values = tuple(range(2, 2 + num_colors))

    data = np.load(filepath)
    network = data["network"].T  # “em pé”
    nrows, ncols = network.shape

    # Paleta padrão
    if color_map is None:
        if num_colors == 1:
            color_map = {1: "0.4"}  # cinza
        else:
            base = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                    "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
            color_map = {val: base[(i % len(base))] for i, val in enumerate(active_values)}

    segments_by_color = {val: [] for val in active_values}
    density_per_row = np.zeros(nrows, dtype=int)

    # Varre e cria segmentos apenas se vizinho tem o MESMO valor ativo
    for i in range(nrows):
        for j in range(ncols):
            v = network[i, j]
            if v in active_values:
                # horizontal (direita)
                if j + 1 < ncols and network[i, j + 1] == v:
                    segments_by_color[v].append([(j, i), (j + 1, i)])
                    density_per_row[i] += 1
                # vertical (cima)
                if i + 1 < nrows and network[i + 1, j] == v:
                    segments_by_color[v].append([(j, i), (j, i + 1)])
                    density_per_row[i] += 1

    active_rows = np.where(density_per_row >= min_density)[0]
    if active_rows.size == 0:
        print("[!] No row found with minimum density.")
        return
    row_start, row_end = active_rows[0], active_rows[-1]

    # Recorte vertical
    for val in active_values:
        segments_by_color[val] = [
            [(x0, y0), (x1, y1)]
            for (x0, y0), (x1, y1) in segments_by_color[val]
            if row_start <= y0 <= row_end and row_start <= y1 <= row_end
        ]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    handles, labels = [], []
    for val in active_values:
        segs = segments_by_color[val]
        if not segs:
            continue
        lc = LineCollection(segs, colors=color_map.get(val, "black"),
                            linewidths=linewidth, label=f"+{val}")
        ax.add_collection(lc)
        handles.append(lc)
        labels.append(f"+{val}")

    ax.set_xlim(0, ncols)
    ax.set_ylim(row_end, row_start)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    if show_legend and handles:
        ax.legend(handles=handles, labels=labels, loc="upper right",
                  frameon=False, fontsize=8)

    plt.tight_layout(pad=0)
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0)
        print(f"[✔] Image saved to: {savepath}")
    else:
        plt.show()