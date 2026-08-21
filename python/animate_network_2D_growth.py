#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def build_video(frames_dir, output, fps, pattern="frame_%06d.png", hold_seconds=0):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg nao encontrado no PATH")

    filters = []
    if hold_seconds and float(hold_seconds) > 0.0:
        filters.append(f"tpad=stop_mode=clone:stop_duration={float(hold_seconds):.6g}")
    filters.append("format=yuv420p")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(Path(frames_dir) / pattern),
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def load_overlay(path, color_index=None):
    if path is None:
        return None

    overlay_path = Path(path).expanduser().resolve()
    with overlay_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("by_color", [])
    if color_index is not None:
        for row in rows:
            if int(row.get("color_index", -1)) == int(color_index):
                return row
        raise ValueError(f"color_index={color_index} nao encontrado em {overlay_path}")

    for row in rows:
        if int(row.get("z_stab", -1)) >= 0:
            return row
    return rows[0] if rows else None


def load_time_series(path, color_index=None):
    if path is None:
        return None

    data_path = Path(path).expanduser().resolve()
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("results", {})
    wanted_color = None if color_index is None else int(color_index) + 1
    selected = None
    for item in rows.values():
        row = item.get("data", {})
        if wanted_color is None or int(row.get("color", -1)) == wanted_color:
            selected = row
            break
    if selected is None:
        raise ValueError(f"serie temporal nao encontrada em {data_path}")

    time = np.asarray(selected.get("time", []), dtype=np.float64)
    pt = np.asarray(selected.get("pt", []), dtype=np.float64)
    if time.size == 0 or pt.size == 0 or time.size != pt.size:
        raise ValueError(f"serie temporal invalida em {data_path}")

    t_eq = selected.get("t_eq_species", data.get("meta", {}).get("t_eq"))
    return {
        "time": time,
        "pt": pt,
        "t_eq": None if t_eq is None else float(t_eq),
    }


def first_time_at_or_above_z(pos, times, L, z):
    if z is None or int(z) < 0:
        return None
    heights = pos // np.uint64(L)
    mask = heights >= np.uint64(int(z))
    if not np.any(mask):
        return None
    return int(times[mask].min())


def fit_physical_view(view_width, view_height, physical_width, physical_height):
    scale = min(view_width / max(1, physical_width), view_height / max(1, physical_height))
    draw_w = max(1, int(round(physical_width * scale)))
    draw_h = max(1, int(round(physical_height * scale)))
    off_x = max(0, (view_width - draw_w) // 2)
    off_y = max(0, (view_height - draw_h) // 2)
    return draw_w, draw_h, off_x, off_y


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def line_y_for_z(z, viewport_bottom, viewport_top, out_h):
    span = max(1, int(viewport_top) - int(viewport_bottom))
    py_bottom_origin = int(round((int(z) - int(viewport_bottom)) * (out_h - 1) / span))
    return int(np.clip(out_h - 1 - py_bottom_origin, 0, out_h - 1))


def draw_layer_lines(
    img,
    overlay,
    viewport_bottom,
    viewport_top,
    color=(35, 35, 35),
    current_t=None,
    layer_times=None,
):
    if overlay is None:
        return

    draw = ImageDraw.Draw(img)
    width, height = img.size
    for key in ("z_stab", "z_stab_plus_L", "z_stab_plus_2_5L", "z_stab_plus_2L"):
        z = int(overlay.get(key, -1))
        if z < viewport_bottom or z > viewport_top:
            continue
        if current_t is not None and layer_times is not None:
            first_t = layer_times.get(key)
            if first_t is not None and int(current_t) < int(first_t):
                continue
        y = line_y_for_z(z, viewport_bottom, viewport_top, height)
        draw.line([(0, y), (width - 1, y)], fill=color, width=3)


def positions_to_pixels(pos, L, viewport_bottom, viewport_top, out_w, out_h):
    pos = np.asarray(pos, dtype=np.uint64)
    if pos.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    x = pos % np.uint64(L)
    z = pos // np.uint64(L)
    mask = (z >= np.uint64(viewport_bottom)) & (z <= np.uint64(viewport_top))
    if not np.any(mask):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    x = x[mask]
    z = z[mask] - np.uint64(viewport_bottom)
    span = max(1, int(viewport_top) - int(viewport_bottom))
    px = ((x * np.uint64(out_w)) // np.uint64(L)).astype(np.int32)
    py = np.rint(z.astype(np.float64) * float(out_h - 1) / float(span)).astype(np.int32)
    px = np.clip(px, 0, out_w - 1)
    py = np.clip(py, 0, out_h - 1)
    return px, py


def render_viewport_frame(
    pos,
    times,
    t,
    L,
    viewport_bottom,
    viewport_top,
    out_w,
    out_h,
    background_color,
    active_color,
    overlay=None,
    giant_color=None,
    path_color=None,
    path_fraction=1.0,
    path_width=9,
    path_radius=4,
    layer_times=None,
    line_color=(35, 35, 35),
):
    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[:, :] = background_color

    visible = times <= t
    active_pos = pos[visible]
    px, py = positions_to_pixels(active_pos, L, viewport_bottom, viewport_top, out_w, out_h)
    if px.size:
        canvas[py, px] = active_color

    if overlay is not None and giant_color is not None:
        gx, gy = positions_to_pixels(
            overlay.get("giant_component_pos_flat", []),
            L,
            viewport_bottom,
            viewport_top,
            out_w,
            out_h,
        )
        if gx.size:
            canvas[gy, gx] = giant_color

    img = Image.fromarray(np.flipud(canvas), mode="RGB")
    draw_layer_lines(
        img,
        overlay,
        viewport_bottom,
        viewport_top,
        color=line_color,
        current_t=t,
        layer_times=layer_times,
    )

    if overlay is not None and path_color is not None:
        path = overlay.get("shortest_path_pos_flat", [])
        path_x, path_y = positions_to_pixels(
            path,
            L,
            viewport_bottom,
            viewport_top,
            out_w,
            out_h,
        )
        if path_x.size:
            draw_path_pixels(
                img,
                path_x,
                path_y,
                path_color,
                path_fraction,
                path_width,
                path_radius,
            )

    return img


def paste_preserving_physical_scale(base, network_img, physical_width, physical_height, box):
    x0, y0, x1, y1 = box
    draw_w, draw_h, off_x, off_y = fit_physical_view(x1 - x0, y1 - y0, physical_width, physical_height)
    resized = network_img.resize((draw_w, draw_h), Image.Resampling.NEAREST)
    base.paste(resized, (x0 + off_x, y0 + off_y))
    return (x0 + off_x, y0 + off_y, x0 + off_x + draw_w, y0 + off_y + draw_h)


def moving_average(values, window):
    window = int(window)
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def draw_time_series_panel(
    img,
    series,
    current_t,
    t_stab,
    color,
    stab_color,
    box,
    smooth_window=1,
    plot_height=None,
    pc_value=0.5,
    y_min=None,
    y_max=None,
    series_line_width=5.0,
):
    if series is None:
        return

    x0, y0, x1, y1 = box
    panel_w = max(2, x1 - x0 + 1)
    panel_h = max(2, y1 - y0 + 1)
    chart_h = int(plot_height or panel_h)
    chart_h = max(320, min(chart_h, panel_h))
    chart_w = panel_w
    chart_y = y0 + (panel_h - chart_h) // 2
    if chart_w <= 2 or chart_h <= 2:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    time = series["time"]
    pt = moving_average(series["pt"], smooth_window)
    t_max = max(float(time[-1]), 1.0)
    if y_min is not None and y_max is not None and float(y_max) > float(y_min):
        p_min = float(y_min)
        p_max = float(y_max)
    else:
        finite_pt = pt[np.isfinite(pt)]
        p_min = float(np.percentile(finite_pt, 1.0))
        p_max = float(np.percentile(finite_pt, 99.0))
        if p_max <= p_min:
            p_min = float(np.nanmin(pt))
            p_max = float(np.nanmax(pt))
        pad = max(1e-6, 0.06 * (p_max - p_min))
        p_min = min(p_min - pad, pc_value - pad)
        p_max = max(p_max + pad, pc_value + pad)

    visible = time <= float(current_t)
    fig = plt.figure(figsize=(chart_w / 120.0, chart_h / 120.0), dpi=120)
    bg_rgb = (255, 247, 239)
    bg = tuple(c / 255.0 for c in bg_rgb)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0.24, 0.22, 0.66, 0.66])
    ax.set_facecolor(bg)

    if np.any(visible):
        tt = time[visible]
        yy = pt[visible]
        step = max(1, int(np.ceil(tt.size / 2200)))
        yy = np.clip(yy, p_min, p_max)
        ax.plot(tt[::step], yy[::step], color=np.asarray(color) / 255.0, linewidth=float(series_line_width))

    ax.axhline(pc_value, color="#7f1d1d", linewidth=4.0)
    if t_stab is not None and float(current_t) >= float(t_stab):
        ax.axvline(float(t_stab), color=np.asarray(stab_color) / 255.0, linewidth=4.5)

    ax.set_xlim(0.0, t_max)
    ax.set_ylim(p_min, p_max)
    ax.set_xlabel(r"$t$", fontsize=34, labelpad=14)
    ax.set_ylabel(r"$p(t)$", fontsize=34, labelpad=18)
    xticks = np.linspace(0, t_max, 4)
    yticks = np.linspace(p_min, p_max, 4)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([rf"${int(x):d}$" for x in xticks], fontsize=24)
    ax.set_yticklabels([rf"${y:.3f}$" for y in yticks], fontsize=24)
    ax.tick_params(axis="both", which="major", length=10, width=2.4, direction="out")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="both", colors="black")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    chart = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)

    img.paste(chart, (x0, chart_y))


def render_scaled_view_frame(
    pos,
    times,
    t,
    L,
    viewport_bottom,
    viewport_top,
    frame_w,
    frame_h,
    background_color,
    active_color,
    overlay=None,
    giant_color=None,
    path_color=None,
    path_fraction=1.0,
    path_width=9,
    path_radius=4,
    layer_times=None,
    line_color=(35, 35, 35),
):
    span = max(1, int(viewport_top) - int(viewport_bottom))
    view_w, view_h, off_x, off_y = fit_physical_view(frame_w, frame_h, L, span)
    view = render_viewport_frame(
        pos,
        times,
        t,
        L,
        viewport_bottom,
        viewport_top,
        view_w,
        view_h,
        background_color,
        active_color,
        overlay=overlay,
        giant_color=giant_color,
        path_color=path_color,
        path_fraction=path_fraction,
        path_width=path_width,
        path_radius=path_radius,
        layer_times=layer_times,
        line_color=line_color,
    )
    img = Image.new("RGB", (frame_w, frame_h), background_color)
    img.paste(view, (off_x, off_y))
    return img


def draw_path_pixels(img, path_x, path_y, color, fraction=1.0, width=9, radius=4):
    if path_x.size == 0:
        return

    draw = ImageDraw.Draw(img)
    out_h = img.size[1]
    n = int(np.clip(round(path_x.size * float(fraction)), 1, path_x.size))
    display_points = [
        (int(x), int(out_h - 1 - y))
        for x, y in zip(path_x[:n], path_y[:n])
    ]
    if len(display_points) >= 2:
        draw.line(display_points, fill=color, width=max(1, int(width)), joint="curve")
    r = max(1, int(radius))
    for x, y in display_points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


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
    overlay = load_overlay(args.overlay_json, args.overlay_color_index)
    if overlay is not None and args.output_height is None:
        out_h = out_w
    time_series = load_time_series(args.time_series_json, args.overlay_color_index)

    print(f"[info] arquivo: {input_path}")
    print(f"[info] L={args.L}, H={height}, N={n_sites}, ativos={active_count}, E={info['E']}")
    print(f"[info] tempo: min={int(times.min())}, max={int(times.max())}, unicos={np.unique(times).size}")
    print(f"[info] saida: {out_w}x{out_h}")
    if overlay is not None:
        print(
            "[info] overlay: "
            f"z_stab={overlay.get('z_stab')}, "
            f"z_stab+L={overlay.get('z_stab_plus_L')}, "
            f"z_stab+2.5L={overlay.get('z_stab_plus_2_5L', overlay.get('z_stab_plus_2L'))}, "
            f"componente={len(overlay.get('giant_component_pos_flat', []))}, "
            f"caminho={len(overlay.get('shortest_path_pos_flat', []))}"
        )
    if time_series is not None:
        print(
            "[info] serie temporal: "
            f"pontos={time_series['time'].size}, "
            f"t_stab={time_series['t_eq']}"
        )

    if args.info:
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_times = choose_frame_times(times, max_frames=args.max_frames, stride=args.frame_stride)
    if frame_times.size == 0:
        raise ValueError("nenhum tempo de ativacao encontrado")
    if args.preview_main_frame:
        frame_times = np.asarray([frame_times[-1]], dtype=np.uint32)
    print(f"[render] frames={frame_times.size}, primeiro_t={int(frame_times[0])}, ultimo_t={int(frame_times[-1])}")

    layer_times = {}
    if overlay is not None:
        for key in ("z_stab", "z_stab_plus_L", "z_stab_plus_2_5L", "z_stab_plus_2L"):
            z = int(overlay.get(key, -1))
            layer_times[key] = first_time_at_or_above_z(pos, times, args.L, z) if z >= 0 else None
        print(
            "[info] tempos das retas: "
            + ", ".join(f"{k}={v}" for k, v in layer_times.items() if v is not None)
        )

    main_has_series = time_series is not None
    plot_w = int(args.plot_width) if main_has_series else 0
    plot_w = max(0, min(plot_w, out_w - 64))
    net_panel_w = out_w - plot_w
    net_draw_w, net_draw_h, net_off_x, net_off_y = fit_physical_view(
        net_panel_w,
        out_h,
        args.L,
        height,
    )
    net_off_x = 0

    x = (pos % args.L).astype(np.uint32, copy=False)
    y = (pos // args.L).astype(np.uint32, copy=False)
    px = ((x.astype(np.uint64) * net_draw_w) // args.L).astype(np.int32)
    py = ((y.astype(np.uint64) * net_draw_h) // height).astype(np.int32)
    del x, y

    order = np.argsort(times, kind="stable")
    times = times[order]
    pos = pos[order]
    px = px[order]
    py = py[order]

    canvas = np.empty((net_draw_h, net_draw_w, 3), dtype=np.uint8)
    canvas[:, :] = args.background_color

    active_rgb = np.array(args.active_color, dtype=np.uint8)
    front = np.array(args.front_color, dtype=np.uint8)
    main_done = (frames_dir / f"frame_{frame_times.size - 1:06d}.png").exists()
    if args.skip_main_frames and main_done:
        print(f"[render] pulando {frame_times.size} frames principais ja existentes")
    else:
        previous_end = 0
        previous_front = None

        for frame_idx, t in enumerate(frame_times):
            end = int(np.searchsorted(times, t, side="right"))

            if previous_front is not None:
                fy, fx = previous_front
                canvas[fy, fx] = active_rgb

            if end > previous_end:
                canvas[py[previous_end:end], px[previous_end:end]] = active_rgb

            front_y = py[previous_end:end]
            front_x = px[previous_end:end]
            if front_y.size:
                canvas[front_y, front_x] = front
                previous_front = (front_y.copy(), front_x.copy())
            else:
                previous_front = None

            network_img = Image.fromarray(np.flipud(canvas), mode="RGB")
            draw_layer_lines(
                network_img,
                overlay,
                0,
                height - 1,
                color=args.stab_line_color,
                current_t=int(t),
                layer_times=layer_times,
            )
            img = Image.new("RGB", (out_w, out_h), args.background_color)
            img.paste(network_img, (net_off_x, net_off_y))
            if main_has_series:
                draw_time_series_panel(
                    img,
                    time_series,
                    int(t),
                    layer_times.get("z_stab"),
                    args.series_color,
                    args.stab_line_color,
                    (net_panel_w, 0, out_w - 1, out_h - 1),
                    smooth_window=args.series_smooth_window,
                    plot_height=args.plot_height,
                    pc_value=args.pc_value,
                    y_min=args.plot_y_min,
                    y_max=args.plot_y_max,
                    series_line_width=args.series_line_width,
                )
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            img.save(frame_path, optimize=False)
            if args.preview_main_frame:
                preview_path = Path(args.preview_main_frame).expanduser().resolve()
                if preview_path.parent:
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(frame_path, preview_path)
                print(f"[done] preview: {preview_path}")
                return

            previous_end = end
            if (frame_idx + 1) % args.report_every == 0 or frame_idx + 1 == frame_times.size:
                pct = 100.0 * (frame_idx + 1) / frame_times.size
                print(f"[render] {frame_idx + 1}/{frame_times.size} frames ({pct:.1f}%)")

    next_frame_idx = int(frame_times.size)
    if overlay is not None:
        final_t = int(frame_times[-1])
        z_bottom = int(overlay.get("z_stab", -1))
        z_top = int(overlay.get("z_stab_plus_L", -1))
        if z_bottom >= 0 and z_top >= z_bottom:
            zoom_frames = max(0, int(args.zoom_frames))
            freeze_frames = max(0, int(args.freeze_frames))
            path_frames = max(0, int(args.path_frames))
            for i in range(zoom_frames):
                alpha = (i + 1) / max(1, zoom_frames)
                smooth = alpha * alpha * (3.0 - 2.0 * alpha)
                view_bottom = int(round((1.0 - smooth) * 0 + smooth * z_bottom))
                view_top = int(round((1.0 - smooth) * (height - 1) + smooth * z_top))
                img = render_scaled_view_frame(
                    pos,
                    times,
                    final_t,
                    args.L,
                    view_bottom,
                    view_top,
                    out_w,
                    out_h,
                    args.background_color,
                    args.active_color,
                    overlay=overlay,
                    giant_color=args.giant_color,
                    path_color=None,
                    layer_times=layer_times,
                    line_color=args.stab_line_color,
                )
                img.save(frames_dir / f"frame_{next_frame_idx:06d}.png", optimize=False)
                next_frame_idx += 1

            freeze_img = render_scaled_view_frame(
                pos,
                times,
                final_t,
                args.L,
                z_bottom,
                z_top,
                out_w,
                out_h,
                args.background_color,
                args.active_color,
                overlay=overlay,
                giant_color=args.giant_color,
                path_color=None,
                layer_times=layer_times,
                line_color=args.stab_line_color,
            )
            path_x, path_y = positions_to_pixels(
                overlay.get("shortest_path_pos_flat", []),
                args.L,
                z_bottom,
                z_top,
                out_w,
                out_h,
            )
            for i in range(path_frames):
                img = freeze_img.copy()
                draw_path_pixels(
                    img,
                    path_x,
                    path_y,
                    args.path_color,
                    (i + 1) / max(1, path_frames),
                    args.path_width,
                    args.path_radius,
                )
                img.save(frames_dir / f"frame_{next_frame_idx:06d}.png", optimize=False)
                next_frame_idx += 1

            if freeze_frames:
                freeze_img = freeze_img.copy()
                draw_path_pixels(
                    freeze_img,
                    path_x,
                    path_y,
                    args.path_color,
                    1.0,
                    args.path_width,
                    args.path_radius,
                )
            for _ in range(freeze_frames):
                freeze_img.save(frames_dir / f"frame_{next_frame_idx:06d}.png", optimize=False)
                next_frame_idx += 1
            print(
                f"[render] frames finais: zoom={zoom_frames}, caminho={path_frames}, congelado={freeze_frames}, "
                f"total={next_frame_idx}"
            )

    if args.video:
        output_video = output_dir / args.video_name
        build_video(frames_dir, output_video, fps=args.fps, hold_seconds=args.hold_seconds)
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
    parser.add_argument("--active-color", type=parse_hex_color, default=parse_hex_color("#1f77b4"))
    parser.add_argument("--front-color", type=parse_hex_color, default=parse_hex_color("#62b6ff"))
    parser.add_argument("--background-color", type=parse_hex_color, default=parse_hex_color("#fff7ef"))
    parser.add_argument("--giant-color", type=parse_hex_color, default=parse_hex_color("#2a9d8f"))
    parser.add_argument("--path-color", type=parse_hex_color, default=parse_hex_color("#7f1d1d"))
    parser.add_argument("--stab-line-color", type=parse_hex_color, default=parse_hex_color("#202020"))
    parser.add_argument("--series-color", type=parse_hex_color, default=parse_hex_color("#1f77b4"))
    parser.add_argument("--overlay-json", default=None, help="JSON *_animation_overlay.json gerado pelo SOP")
    parser.add_argument("--time-series-json", default=None, help="JSON de dados com a serie temporal p(t)")
    parser.add_argument("--overlay-color-index", type=int, default=None, help="cor a usar no overlay; default usa a primeira estabilizada")
    parser.add_argument("--plot-width", type=int, default=560, help="largura do painel de serie temporal durante o crescimento")
    parser.add_argument("--plot-height", type=int, default=None, help="altura do grafico p(t), centralizado no painel lateral")
    parser.add_argument("--pc-value", type=float, default=0.5, help="valor de p_c para reta horizontal no painel p(t)")
    parser.add_argument("--plot-y-min", type=float, default=None, help="limite inferior fixo do eixo y do painel p(t)")
    parser.add_argument("--plot-y-max", type=float, default=None, help="limite superior fixo do eixo y do painel p(t)")
    parser.add_argument("--series-line-width", type=float, default=4.0, help="espessura da curva p(t)")
    parser.add_argument("--series-smooth-window", type=int, default=301, help="janela de media movel visual para p(t)")
    parser.add_argument("--zoom-frames", type=int, default=48, help="frames de zoom final ate [z_stab,z_stab+L]")
    parser.add_argument("--path-frames", type=int, default=96, help="frames para desenhar o shortest path da base ao topo")
    parser.add_argument("--freeze-frames", type=int, default=72, help="frames congelados no zoom final")
    parser.add_argument("--hold-seconds", type=float, default=0.0, help="segundos extras clonando o ultimo frame no MP4")
    parser.add_argument("--path-width", type=int, default=11, help="espessura do traco do shortest path")
    parser.add_argument("--path-radius", type=int, default=4, help="raio dos sitios desenhados no shortest path")
    parser.add_argument("--skip-main-frames", action="store_true", help="pula os frames principais se frame final ja existir")
    parser.add_argument("--video", action="store_true", help="monta MP4 com ffmpeg ao final")
    parser.add_argument("--video-name", default="growth_2D.mp4")
    parser.add_argument("--info", action="store_true", help="mostra metadados e nao renderiza")
    parser.add_argument("--preview-main-frame", default=None, help="salva apenas o ultimo frame da fase principal neste PNG")
    parser.add_argument("--report-every", type=int, default=10)
    args = parser.parse_args()

    render_growth_animation(args)


if __name__ == "__main__":
    main()
