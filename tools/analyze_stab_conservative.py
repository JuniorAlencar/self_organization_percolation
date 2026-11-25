import glob
import json
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import matplotlib.pyplot as plt
import numpy as np
from tools.process_dynamic_growth import estimate_t_eq_species

targets = [
    ('2D', 512, 0.06),
    ('2D', 512, 0.5),
    ('2D', 16384, 0.04),
    ('2D', 16384, 0.5),
    ('3D', 64, 0.02),
    ('3D', 64, 0.5),
    ('3D', 2048, 0.02),
    ('3D', 2048, 0.5)
]

os.makedirs('figures/stab_analysis', exist_ok=True)

# -------------------------------------------------------------
# 1. Overview Figure (all 8 cases)
# -------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=False)
axes = axes.flatten()

loaded_data = {}

for idx, (dim_str, L, ft) in enumerate(targets):
    ax = axes[idx]
    dim = int(dim_str[0])
    ft_sci = f'{ft:.6e}'
    pattern = f'SOP_data/tests_data/linear/node_percolation/num_colors_1/dim_{dim}/L_{L}/fT_constant/fT_{ft_sci}/**/data/*.json'
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not matches:
        continue
    fpath = matches[-1]
    with open(fpath) as fp:
        d = json.load(fp)
    meta = d['meta']
    data = d['results']['order_percolation 0']['data']
    t = np.array(data['time'])
    pt = np.array(data['pt'])
    nt = np.array(data['nt'])
    
    loaded_data[(dim_str, L, ft)] = (meta, t, pt, nt)

    w = 15
    p_smooth = np.convolve(pt, np.ones(w)/w, mode='valid')
    t_smooth = t[w//2 : len(pt) - (w - w//2 - 1)]
    
    cpp_teq = meta.get('t_eq')
    py_teq = estimate_t_eq_species(t, pt, meta=meta)
    py_teq_val50 = estimate_t_eq_species(t, pt, validation_window_steps=50, meta=meta)
    py_teq_raw = estimate_t_eq_species(t, pt, validation_window_steps=0, min_stable_steps=15, meta=meta)

    ax.plot(t, pt, color='lightgray', alpha=0.7, label=r'$p(t)$')
    ax.plot(t_smooth, p_smooth, color='tab:blue', lw=1.5, label='Moving avg (w=15)')
    
    if cpp_teq is not None and not np.isnan(cpp_teq):
        ax.axvline(cpp_teq, color='red', linestyle='--', lw=2.5, label=f'C++ online $t_{{eq}}={cpp_teq:.1f}$')
    else:
        ax.plot([], [], ' ', label='C++ online: None ($h < L$)')
        
    if py_teq is not None:
        ax.axvline(py_teq, color='green', linestyle='-', lw=2.0, label=f'C++ algo offline $t_{{eq}}={py_teq:.1f}$')
    if py_teq_val50 is not None and (py_teq is None or abs(py_teq_val50 - py_teq) > 1):
        ax.axvline(py_teq_val50, color='orange', linestyle='-.', lw=1.5, label=f'Algo (val=50) $t_{{eq}}={py_teq_val50:.1f}$')
    if py_teq_raw is not None and (py_teq is None or abs(py_teq_raw - py_teq) > 1):
        ax.axvline(py_teq_raw, color='purple', linestyle=':', lw=1.5, label=f'Algo (raw derivative) $t_{{eq}}={py_teq_raw:.1f}$')

    title = f'{dim_str} | L={L} | $f_T={ft}$'
    if cpp_teq is None:
        title += ' [C++ online: None ($h < L$)]'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('t')
    ax.set_ylabel('p(t)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

plt.tight_layout()
plt.savefig('figures/stab_analysis/overview_8_cases.png', dpi=150)
plt.close(fig)
print('Saved overview_8_cases.png')

# -------------------------------------------------------------
# 2. Zoomed-in 2D Figure (4 subplots)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
cases_2d = [
    ('2D', 512, 0.06, 600),
    ('2D', 512, 0.5, 300),
    ('2D', 16384, 0.04, 2000),
    ('2D', 16384, 0.5, 300)
]

for idx, (dim_str, L, ft, t_zoom) in enumerate(cases_2d):
    ax = axes[idx // 2, idx % 2]
    meta, t, pt, nt = loaded_data[(dim_str, L, ft)]
    
    mask = t <= t_zoom
    t_sub = t[mask]
    pt_sub = pt[mask]
    
    w = 15
    p_smooth = np.convolve(pt_sub, np.ones(w)/w, mode='valid')
    t_smooth = t_sub[w//2 : len(pt_sub) - (w - w//2 - 1)]
    
    cpp_teq = meta.get('t_eq')
    py_teq = estimate_t_eq_species(t, pt, meta=meta)
    py_teq_val50 = estimate_t_eq_species(t, pt, validation_window_steps=50, meta=meta)
    py_teq_raw = estimate_t_eq_species(t, pt, validation_window_steps=0, min_stable_steps=15, meta=meta)

    ax.plot(t_sub, pt_sub, color='lightgray', alpha=0.8, label=r'$p(t)$')
    ax.plot(t_smooth, p_smooth, color='tab:blue', lw=2.0, label=f'Moving avg ($w={w}$)')
    
    if cpp_teq is not None and not np.isnan(cpp_teq):
        ax.axvline(cpp_teq, color='red', linestyle='--', lw=2.5, label=f'C++ online $t_{{eq}}={cpp_teq:.1f}$')
    else:
        ax.plot([], [], ' ', label='C++ online: None (req. h >= L)')
        
    if py_teq is not None:
        ax.axvline(py_teq, color='green', linestyle='-', lw=2.0, label=f'C++ offline algo $t_{{eq}}={py_teq:.1f}$')
    if py_teq_val50 is not None and (py_teq is None or abs(py_teq_val50 - py_teq) > 1):
        ax.axvline(py_teq_val50, color='orange', linestyle='-.', lw=1.8, label=f'Validation=50: $t_{{eq}}={py_teq_val50:.1f}$')
    if py_teq_raw is not None and (py_teq is None or abs(py_teq_raw - py_teq) > 1):
        ax.axvline(py_teq_raw, color='purple', linestyle=':', lw=1.8, label=f'Raw derivative: $t_{{eq}}={py_teq_raw:.1f}$')

    ax.set_title(f'2D | L={L} | $f_T={ft}$ (Zoom t <= {t_zoom})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time step $t$', fontsize=11)
    ax.set_ylabel(r'$p(t)$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('figures/stab_analysis/zoom_2D_cases.png', dpi=150)
plt.close(fig)
print('Saved zoom_2D_cases.png')

# -------------------------------------------------------------
# 3. Zoomed-in 3D Figure (4 subplots)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
cases_3d = [
    ('3D', 64, 0.02, 300),
    ('3D', 64, 0.5, 200),
    ('3D', 2048, 0.02, 200),
    ('3D', 2048, 0.5, 200)
]

for idx, (dim_str, L, ft, t_zoom) in enumerate(cases_3d):
    ax = axes[idx // 2, idx % 2]
    meta, t, pt, nt = loaded_data[(dim_str, L, ft)]
    
    mask = t <= t_zoom
    t_sub = t[mask]
    pt_sub = pt[mask]
    
    w = 15
    p_smooth = np.convolve(pt_sub, np.ones(w)/w, mode='valid')
    t_smooth = t_sub[w//2 : len(pt_sub) - (w - w//2 - 1)]
    
    cpp_teq = meta.get('t_eq')
    py_teq = estimate_t_eq_species(t, pt, meta=meta)
    py_teq_val50 = estimate_t_eq_species(t, pt, validation_window_steps=50, meta=meta)
    py_teq_raw = estimate_t_eq_species(t, pt, validation_window_steps=0, min_stable_steps=15, meta=meta)

    ax.plot(t_sub, pt_sub, color='lightgray', alpha=0.8, label=r'$p(t)$')
    ax.plot(t_smooth, p_smooth, color='tab:blue', lw=2.0, label=f'Moving avg ($w={w}$)')
    
    if cpp_teq is not None and not np.isnan(cpp_teq):
        ax.axvline(cpp_teq, color='red', linestyle='--', lw=2.5, label=f'C++ online $t_{{eq}}={cpp_teq:.1f}$')
    else:
        ax.plot([], [], ' ', label='C++ online: None (req. h >= L)')
        
    if py_teq is not None:
        ax.axvline(py_teq, color='green', linestyle='-', lw=2.0, label=f'C++ offline algo $t_{{eq}}={py_teq:.1f}$')
    if py_teq_val50 is not None and (py_teq is None or abs(py_teq_val50 - py_teq) > 1):
        ax.axvline(py_teq_val50, color='orange', linestyle='-.', lw=1.8, label=f'Validation=50: $t_{{eq}}={py_teq_val50:.1f}$')
    if py_teq_raw is not None and (py_teq is None or abs(py_teq_raw - py_teq) > 1):
        ax.axvline(py_teq_raw, color='purple', linestyle=':', lw=1.8, label=f'Raw derivative: $t_{{eq}}={py_teq_raw:.1f}$')

    ax.set_title(f'3D | L={L} | $f_T={ft}$ (Zoom t <= {t_zoom})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time step $t$', fontsize=11)
    ax.set_ylabel(r'$p(t)$', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

plt.tight_layout()
plt.savefig('figures/stab_analysis/zoom_3D_cases.png', dpi=150)
plt.close(fig)
print('Saved zoom_3D_cases.png')
