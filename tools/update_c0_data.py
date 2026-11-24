#!/usr/bin/env python3
import glob
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path
import numpy as np

def estimate_t_eq_fast(t_arr, p_arr, window_roll=15, window_block=10, min_stable_steps=100, rel_tol=0.025, abs_tol=1e-6, validation_window_steps=200):
    t_arr = np.asarray(t_arr, dtype=np.float64)
    p_arr = np.asarray(p_arr, dtype=np.float64)
    n = len(p_arr)
    if n == 0 or len(t_arr) != n:
        return None
    half_left = window_roll // 2
    half_right = window_roll - half_left - 1
    prefix = np.zeros(n + 1, dtype=np.float64)
    np.cumsum(p_arr, out=prefix[1:])
    idx = np.arange(n)
    a = np.maximum(0, idx - half_left)
    b = np.minimum(n, idx + half_right + 1)
    p_smoothed = (prefix[b] - prefix[a]) / (b - a)
    n_blocks = n // window_block
    if n_blocks < 3:
        return None
    reshaped_t = t_arr[:n_blocks * window_block].reshape(n_blocks, window_block)
    reshaped_p = p_smoothed[:n_blocks * window_block].reshape(n_blocks, window_block)
    t_j = reshaped_t.mean(axis=1)
    j_w = reshaped_p.mean(axis=1)
    validation_links_required = max(1, int(np.ceil(validation_window_steps / max(1, window_block))))
    
    def check_stat(first_pt, last_pt):
        n_pts = last_pt - first_pt + 1
        if n_pts < 3:
            return False
        edge_cnt = max(1, n_pts // 3)
        first_m = j_w[first_pt : first_pt + edge_cnt].mean()
        last_m = j_w[last_pt - edge_cnt + 1 : last_pt + 1].mean()
        drift_scale = max(max(abs(first_m), abs(last_m)), 1e-12)
        drift_thresh = max(abs_tol, rel_tol * drift_scale)
        if abs(last_m - first_m) > drift_thresh:
            return False
        x = t_j[first_pt : last_pt + 1] - t_j[first_pt]
        y = j_w[first_pt : last_pt + 1]
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xx = (x * x).sum()
        sum_xy = (x * y).sum()
        denom = n_pts * sum_xx - sum_x * sum_x
        if denom > 0:
            slope = (n_pts * sum_xy - sum_x * sum_y) / denom
            proj_drift = abs(slope) * (t_j[last_pt] - t_j[first_pt])
            if proj_drift > drift_thresh:
                return False
        return True

    tail_last_pt = n_blocks - 1
    for first_pt in range(n_blocks - validation_links_required):
        last_pt = first_pt + validation_links_required
        if check_stat(first_pt, last_pt) and check_stat(first_pt, tail_last_pt):
            return float(t_j[first_pt])
    return None

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        meta = data.setdefault('meta', {})
        results = data.get('results', {})
        num_colors = int(meta.get('num_colors', 1))
        
        meta['growth_test_t_eq_series'] = 'nt'
        meta['growth_test_t_eq_validation'] = 'global_tail_drift_validation_of_blocked_nt_variation'
        meta['growth_test_t_eq_requires_global_height'] = False
        
        if not results:
            # Dead run
            meta['t_eq'] = None
            meta['t_eq_by_species'] = [None] * num_colors
            meta['growth_test_t_eq_by_species_by_color'] = [None] * num_colors
            orig_status = meta.get('growth_test_species_final_status_by_color', [-1] * num_colors)
            meta['growth_test_species_final_status_by_color'] = orig_status
        else:
            # Supercritical or running
            order_keys = sorted(
                results.keys(),
                key=lambda k: int(k.split()[-1]) if k.split()[-1].isdigit() else 0
            )
            teqs = []
            orig_status = meta.get('growth_test_species_final_status_by_color', [0] * num_colors)
            if not isinstance(orig_status, list) or len(orig_status) != num_colors:
                orig_status = [0] * num_colors
            
            new_status = list(orig_status)
            for idx, k in enumerate(order_keys):
                ord_data = results[k].setdefault('data', {})
                times = ord_data.get('time', [])
                nts = ord_data.get('nt', [])
                teq = estimate_t_eq_fast(times, nts)
                ord_data['t_eq_species'] = teq
                teqs.append(teq)
                
                c_idx = ord_data.get('color')
                color_idx = idx
                if c_idx is not None:
                    try:
                        color_idx = int(c_idx) - 1 # 1-based to 0-based
                    except Exception:
                        pass
                if 0 <= color_idx < num_colors:
                    if teq is not None:
                        new_status[color_idx] = 1
                    elif orig_status[color_idx] != -1:
                        new_status[color_idx] = 0
            
            while len(teqs) < num_colors:
                teqs.append(None)
            
            finite_teqs = [t for t in teqs if t is not None]
            meta['t_eq'] = float(np.mean(finite_teqs)) if finite_teqs else None
            meta['t_eq_by_species'] = teqs
            meta['growth_test_t_eq_by_species_by_color'] = teqs
            meta['growth_test_species_final_status_by_color'] = new_status
            
        # Write back atomically
        tmp_path = filepath + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp_path, filepath)
        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False

def main():
    dirs = [
        'SOP_data/tests_data/linear/bond_percolation/num_colors_2/dim_2/L_4096/fT_constant/fT_0.000000e+00/c_0.000000e+00/rho_5.0000e-01/data',
        'SOP_data/tests_data/linear/node_percolation/num_colors_2/dim_2/L_4096/fT_constant/fT_0.000000e+00/c_0.000000e+00/rho_5.0000e-01/data'
    ]
    
    all_files = []
    for d in dirs:
        p = Path(d)
        if p.exists():
            files = list(p.glob('*.json'))
            print(f"Found {len(files)} files in {d}")
            all_files.extend([str(f) for f in files])
    
    print(f"Total files to process: {len(all_files)}")
    num_workers = min(os.cpu_count() or 4, 12)
    print(f"Running multiprocessing pool with {num_workers} workers...")
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(process_file, all_files, chunksize=100)
    
    success = sum(1 for r in results if r)
    print(f"Successfully processed {success} / {len(all_files)} files.")

if __name__ == '__main__':
    main()
