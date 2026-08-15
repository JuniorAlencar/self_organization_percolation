#!/usr/bin/env python3
import argparse
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


MAGIC_NETG = 0x4E455447


def parse_hex_color(value):
    s = value.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise argparse.ArgumentTypeError("cor deve estar no formato #RRGGBB")
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cor deve estar no formato #RRGGBB") from exc


def read_compact_bin_header(path):
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != MAGIC_NETG:
            raise ValueError(f"Arquivo nao parece ser NetworkCompact .bin: magic={hex(magic)}")
        n_sites = struct.unpack("<I", f.read(4))[0]
        n_edges = struct.unpack("<Q", f.read(8))[0]
    return n_sites, n_edges


def read_compact_bin_arrays(path):
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != MAGIC_NETG:
            raise ValueError(f"Arquivo nao parece ser NetworkCompact .bin: magic={hex(magic)}")
        n_sites = struct.unpack("<I", f.read(4))[0]
        n_edges = struct.unpack("<Q", f.read(8))[0]

        pos_flat = np.fromfile(f, dtype=np.uint32, count=n_sites)
        species = np.fromfile(f, dtype=np.uint8, count=n_sites)
        activation_time = np.fromfile(f, dtype=np.uint32, count=n_sites)

    return {
        "N": int(n_sites),
        "E": int(n_edges),
        "pos_flat": pos_flat,
        "species": species,
        "activation_time": activation_time,
    }


def choose_frame_times(times, max_frames=None, stride=1):
    unique_times = np.unique(times)
    unique_times = unique_times[unique_times > 0]
    if unique_times.size == 0:
        unique_times = np.unique(times)
    if stride > 1:
        unique_times = unique_times[:: int(stride)]
    if max_frames is not None and unique_times.size > int(max_frames):
        idx = np.linspace(0, unique_times.size - 1, int(max_frames)).round().astype(np.int64)
        unique_times = unique_times[idx]
    return unique_times.astype(np.uint32, copy=False)


def even_dimension(value):
    value = int(value)
    return value if value % 2 == 0 else value - 1


def build_video(frames_dir, output, fps, pattern="frame_%06d.png"):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg nao encontrado no PATH")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(Path(frames_dir) / pattern),
        "-vf",
        "format=yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def render_growth_animation(args):
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    info = read_compact_bin_arrays(input_path)
    n_sites = info["N"]
    if n_sites % args.L != 0:
        raise ValueError(f"N={n_sites} nao e multiplo de L={args.L}; informe o L correto")

    height = n_sites // args.L
    out_w = even_dimension(args.output_width)
    out_h = even_dimension(args.output_height or round(out_w * height / args.L))
    out_h = max(2, out_h)

    active_mask = info["species"] > 0
    pos = info["pos_flat"][active_mask]
    times = info["activation_time"][active_mask]
    active_count = int(active_mask.sum())

    print(f"[info] arquivo: {input_path}")
    print(f"[info] L={args.L}, H={height}, N={n_sites}, ativos={active_count}, E={info['E']}")
    print(f"[info] tempo: min={int(times.min())}, max={int(times.max())}, unicos={np.unique(times).size}")
    print(f"[info] saida: {out_w}x{out_h}")

    if args.info:
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_times = choose_frame_times(times, max_frames=args.max_frames, stride=args.frame_stride)
    if frame_times.size == 0:
        raise ValueError("nenhum tempo de ativacao encontrado")
    print(f"[render] frames={frame_times.size}, primeiro_t={int(frame_times[0])}, ultimo_t={int(frame_times[-1])}")

    x = (pos % args.L).astype(np.uint32, copy=False)
    y = (pos // args.L).astype(np.uint32, copy=False)
    px = ((x.astype(np.uint64) * out_w) // args.L).astype(np.int32)
    py = ((y.astype(np.uint64) * out_h) // height).astype(np.int32)
    del x, y, pos

    order = np.argsort(times, kind="stable")
    times = times[order]
    px = px[order]
    py = py[order]

    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = args.background_color

    red = np.array(args.active_color, dtype=np.uint8)
    front = np.array(args.front_color, dtype=np.uint8)
    previous_end = 0
    previous_front = None

    for frame_idx, t in enumerate(frame_times):
        end = int(np.searchsorted(times, t, side="right"))

        if previous_front is not None:
            fy, fx = previous_front
            canvas[fy, fx] = red

        if end > previous_end:
            canvas[py[previous_end:end], px[previous_end:end]] = red

        front_y = py[previous_end:end]
        front_x = px[previous_end:end]
        if front_y.size:
            canvas[front_y, front_x] = front
            previous_front = (front_y.copy(), front_x.copy())
        else:
            previous_front = None

        img = Image.fromarray(np.flipud(canvas), mode="RGB")
        frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
        img.save(frame_path, optimize=False)

        previous_end = end
        if (frame_idx + 1) % args.report_every == 0 or frame_idx + 1 == frame_times.size:
            pct = 100.0 * (frame_idx + 1) / frame_times.size
            print(f"[render] {frame_idx + 1}/{frame_times.size} frames ({pct:.1f}%)")

    if args.video:
        output_video = output_dir / args.video_name
        build_video(frames_dir, output_video, fps=args.fps)
        print(f"[done] video: {output_video}")
    else:
        print(f"[done] frames: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Anima o crescimento 2D de uma rede NetworkCompact .bin codificada por tempo de ativacao."
    )
    parser.add_argument("input", help="arquivo .bin salvo na pasta network")
    parser.add_argument("--L", type=int, required=True, help="largura/base da rede")
    parser.add_argument("--output-dir", default="animation_2D_growth", help="diretorio de saida")
    parser.add_argument("--output-width", type=int, default=1600, help="largura renderizada em pixels")
    parser.add_argument("--output-height", type=int, default=None, help="altura renderizada em pixels; default preserva aspecto")
    parser.add_argument("--max-frames", type=int, default=300, help="numero maximo de frames")
    parser.add_argument("--frame-stride", type=int, default=1, help="usa um a cada N tempos unicos antes do max-frames")
    parser.add_argument("--fps", type=int, default=24, help="FPS do video")
    parser.add_argument("--active-color", type=parse_hex_color, default=parse_hex_color("#c9252d"))
    parser.add_argument("--front-color", type=parse_hex_color, default=parse_hex_color("#ffd166"))
    parser.add_argument("--background-color", type=parse_hex_color, default=parse_hex_color("#fff7ef"))
    parser.add_argument("--video", action="store_true", help="monta MP4 com ffmpeg ao final")
    parser.add_argument("--video-name", default="growth_2D.mp4")
    parser.add_argument("--info", action="store_true", help="mostra metadados e nao renderiza")
    parser.add_argument("--report-every", type=int, default=10)
    args = parser.parse_args()

    render_growth_animation(args)


if __name__ == "__main__":
    main()
