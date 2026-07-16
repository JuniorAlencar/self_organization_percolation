import os
from src.network_functions import create_multiple_quality_videos

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

L = 64
nc = 1

types_list = ['alternating', 'blocks']

for type_base in types_list:
    frames_dir = os.path.join(
        PROJECT_ROOT, "animate", f"L_{L}_nc{nc}", f"type_base_{type_base}"
    )

    # Verificar se o diretório de frames existe
    if not os.path.isdir(frames_dir):
        print(f"[SKIP] Diretório não encontrado: {frames_dir}")
        continue

    cropped_dir = os.path.join(frames_dir, "cropped")
    if os.path.isdir(cropped_dir):
        print(f"\n[animation_3D] Usando frames recortados de {cropped_dir}")
        input_dir = cropped_dir
        output_dir = os.path.join(frames_dir, "videos")
    else:
        print(f"\n[animation_3D] Usando frames originais de {frames_dir}")
        input_dir = frames_dir
        output_dir = frames_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"[animation_3D] Montando animações para: {type_base}")
    print(f"  Frames: {input_dir}")
    print(f"  Saída: {output_dir}")

    videos = create_multiple_quality_videos(
        frames_dir=input_dir,
        output_base_dir=output_dir,
        fps=10,
        frame_pattern="frame_*.png",
        quality_presets=[
            {"name": "hq", "codec": "libx265", "crf": 15, "preset": "slow"},
            {"name": "mq", "codec": "libx264", "crf": 23, "preset": "medium"},
            {"name": "lq", "codec": "libx264", "crf": 28, "preset": "fast"},
        ],
    )

    print(f"✓ Animações criadas para {type_base}")

print("\n✓ Todas as animações foram criadas com sucesso!")
