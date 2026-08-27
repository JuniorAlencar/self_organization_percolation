#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import time
from pathlib import Path


CASES = {
    "BOND": {
        512: 0.4631579,
        1024: 0.2394655,
        2048: 0.1807087,
        4096: 0.1343507,
        8192: 0.1041379,
        16384: 0.06873726,
    },
    "NODE": {
        512: 0.3051789,
        1024: 0.2001948,
        2048: 0.1475368,
        4096: 0.1238368,
        8192: 0.1047474,
        16384: 0.08565789,
    },
}


def newest_matching(root, stem, suffix):
    matches = sorted(root.rglob(f"{stem}*{suffix}"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def newest_artifact_set(root):
    overlays = sorted(root.glob("*_animation_overlay.json"), key=lambda p: p.stat().st_mtime)
    for overlay in reversed(overlays):
        stem = overlay.name.removesuffix("_animation_overlay.json")
        bin_path = root / f"{stem}.bin"
        json_path = root / f"{stem}.json"
        if bin_path.exists():
            return bin_path, overlay, json_path if json_path.exists() else None
    return None


def newest_raw_artifact_set(raw_root):
    overlays = sorted(
        raw_root.rglob("network/*_animation_overlay.json"),
        key=lambda p: p.stat().st_mtime,
    )
    for overlay in reversed(overlays):
        stem = overlay.name.removesuffix("_animation_overlay.json")
        bin_path = overlay.parent / f"{stem}.bin"
        json_matches = sorted(
            raw_root.rglob(f"data/{stem}.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if bin_path.exists() and json_matches:
            return bin_path, overlay, json_matches[-1]
    return None


def run_case(
    project_root,
    mode,
    L,
    f_t,
    seed,
    overwrite,
    active_alpha,
    site_pixel_size,
    visual_reference_L,
    background_color,
    active_color,
    giant_color,
    path_color,
    cluster_colors,
    cluster_alpha,
):
    sop = project_root / "build" / "SOP"
    target_dir = project_root / "networks_c0.01_ftmin" / mode / f"L_{L}"
    if not target_dir.exists() and (project_root / "networks_c0.01_ftmin" / mode / str(L)).exists():
        target_dir = project_root / "networks_c0.01_ftmin" / mode / str(L)
    target_dir.mkdir(parents=True, exist_ok=True)

    final_png = target_dir / "final_frame_zstab_white.png"
    existing_artifacts = newest_artifact_set(target_dir)
    if final_png.exists() and existing_artifacts and not overwrite:
        print(f"[skip] {mode} L={L}: ja existe {final_png}")
        return final_png

    raw_root = (
        project_root
        / "SOP_data"
        / "raw_growth_test_dynamic"
        / f"{mode.lower()}_percolation"
        / "num_colors_1"
        / "dim_2"
        / f"L_{L}"
    )

    if existing_artifacts:
        copied_bin, copied_overlay, copied_json = existing_artifacts
        print(f"[reuse] {mode} L={L}: usando {copied_bin.name}")
        if copied_json:
            print(f"[reuse] {copied_json.name}")
    else:
        raw_artifacts = newest_raw_artifact_set(raw_root)
        if raw_artifacts:
            bin_path, overlay_path, json_path = raw_artifacts
            copied_bin = target_dir / bin_path.name
            copied_overlay = target_dir / overlay_path.name
            copied_json = target_dir / json_path.name
            shutil.copy2(bin_path, copied_bin)
            shutil.copy2(overlay_path, copied_overlay)
            shutil.copy2(json_path, copied_json)
            print(f"[reuse-raw] {mode} L={L}: copiando dados existentes")
            print(f"[copy] {copied_bin.name}")
            print(f"[copy] {copied_overlay.name}")
            print(f"[copy] {copied_json.name}")
        else:
            before = time.time()
            cmd = [
                str(sop),
                str(L),
                "0.8",
                str(seed),
                mode.lower(),
                "0.01",
                f"{f_t:.8g}",
                "2",
                "1",
                "1.0",
                "0.2",
                "true",
                "true",
                "growth_test",
                "random",
                "false",
                "true",
            ]
            print(f"[run] {mode} L={L} fT={f_t:.8g}")
            subprocess.run(cmd, cwd=project_root, check=True)

            new_bins = [p for p in raw_root.rglob("network/*.bin") if p.stat().st_mtime >= before]
            new_overlays = [p for p in raw_root.rglob("network/*_animation_overlay.json") if p.stat().st_mtime >= before]
            new_jsons = [p for p in raw_root.rglob("data/*.json") if p.stat().st_mtime >= before]
            if not new_bins or not new_overlays or not new_jsons:
                stem = "light_seed_44_ts_"
                new_bins = new_bins or [newest_matching(raw_root, stem, ".bin")]
                new_overlays = new_overlays or [newest_matching(raw_root, stem, "_animation_overlay.json")]
                new_jsons = new_jsons or [newest_matching(raw_root, stem, ".json")]

            bin_path = max((p for p in new_bins if p), key=lambda p: p.stat().st_mtime)
            overlay_path = max((p for p in new_overlays if p), key=lambda p: p.stat().st_mtime)
            json_path = max((p for p in new_jsons if p and not p.name.endswith("_animation_overlay.json")), key=lambda p: p.stat().st_mtime)

            copied_bin = target_dir / bin_path.name
            copied_overlay = target_dir / overlay_path.name
            copied_json = target_dir / json_path.name
            shutil.copy2(bin_path, copied_bin)
            shutil.copy2(overlay_path, copied_overlay)
            shutil.copy2(json_path, copied_json)
            print(f"[copy] {copied_bin.name}")
            print(f"[copy] {copied_overlay.name}")
            print(f"[copy] {copied_json.name}")

    render_cmd = [
        "python3",
        "python/animate_network_2D_growth.py",
        str(copied_bin),
        "--L",
        str(L),
        "--overlay-json",
        str(copied_overlay),
        "--output-dir",
        str(target_dir),
        "--final-frame-only",
        "--final-frame-name",
        final_png.name,
        "--background-color",
        background_color,
        "--active-color",
        active_color,
        "--active-alpha",
        f"{active_alpha:g}",
        "--giant-color",
        giant_color,
        "--path-color",
        path_color,
        "--stab-line-color",
        "#202020",
        "--site-pixel-size",
        f"{site_pixel_size:g}",
        "--visual-reference-L",
        str(visual_reference_L),
    ]
    if cluster_colors:
        render_cmd.extend([
            "--cluster-colors",
            "--cluster-alpha",
            f"{cluster_alpha:g}",
        ])
    subprocess.run(render_cmd, cwd=project_root, check=True)
    return final_png


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["BOND", "NODE"], action="append")
    parser.add_argument("--L", type=int, action="append")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--active-alpha", type=float, default=0.35)
    parser.add_argument("--background-color", default="#FFFFFF")
    parser.add_argument("--active-color", default="#d7e7e3")
    parser.add_argument("--giant-color", default="#0f766e")
    parser.add_argument("--path-color", default="#7f1d1d")
    parser.add_argument("--cluster-colors", action="store_true")
    parser.add_argument("--cluster-alpha", type=float, default=1.0)
    parser.add_argument("--site-pixel-size", type=float, default=1.5)
    parser.add_argument("--visual-reference-L", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    modes = args.mode or ["BOND", "NODE"]
    generated = []
    for mode in modes:
        for L, f_t in CASES[mode].items():
            if args.L and L not in args.L:
                continue
            generated.append(
                run_case(
                    project_root,
                    mode,
                    L,
                    f_t,
                    args.seed,
                    args.overwrite,
                    args.active_alpha,
                    args.site_pixel_size,
                    args.visual_reference_L,
                    args.background_color,
                    args.active_color,
                    args.giant_color,
                    args.path_color,
                    args.cluster_colors,
                    args.cluster_alpha,
                )
            )

    print("[done] imagens geradas:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
