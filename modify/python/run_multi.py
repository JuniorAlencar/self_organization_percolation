
from tracemalloc import stop
import pandas as pd

try:
	from src.run_multi_functions import (
		run_multi_rho_array,
		get_missing_run_parameters,
		cleanup_logged_output_for_parameters,
		snapshot_existing_outputs_for_parameters,
		write_existing_output_snapshots,
	)
except ModuleNotFoundError:
	from run_multi_functions import (
		run_multi_rho_array,
		get_missing_run_parameters,
		cleanup_logged_output_for_parameters,
		snapshot_existing_outputs_for_parameters,
		write_existing_output_snapshots,
	)
import numpy as np
import pprint

def custom_range(start, stop, n_points, ndigits = 8):
    if n_points <= 0:
        return []
    if n_points == 1:
        return [round(start, ndigits)]

    xs = np.linspace(start, stop, n_points, endpoint=True, dtype=float)
    xs = np.round(xs, ndigits)
    xs[0] = round(start, ndigits)
    xs[-1] = round(stop, ndigits)
    return xs.tolist()

def rho_interval(L, dim, P0, nc):
    base_size = L**(dim - 1)

    rho_min = 1.0 / (P0 * base_size)
    rho_max = 1.0 / nc

    valido = rho_min <= rho_max

    return rho_min, rho_max, valido

# =========================
# Rho values
# =========================

# Parameters
# =========================

#start = 0.001
#stop = 1/num_colors
#n_points = 100
#rho_lst = custom_range(start, stop, n_points)
type_perc = "node"

dim = 2
#NT = 3000
#k = 1.0e-06

seed = -1
#P0 = 0.1
#L_list = [128, 192, 256, 384, 512]
#N_samples_list = [500, 300, 100,75, 50]
# L = 256 => Ns = 150
# L = 320 => Ns = 100
# L = 384 => Ns = 50
# L = 512 => Ns = 25
# L = 640 => Ns = 15
# L = 768 => Ns = 10
# L = 1024 => Ns = 5


#L_lst= [256, 320, 384, 512, 609, 861, 1024]
#N_samples_list = [300, 200, 100, 50, 20, 10, 5]
#max_jobs = [20, 20, 20, 17, 17, 6, 2]
#L_lst = [1024, 2048, 4096, 8192, ]
#N_samples_list = [200, 100, 50, 25]
#max_jobs = [20, 20, 20, 20]
# L_lst = [128, 256, 512,1024]
# max_jobs = [20, 20, 20, 5]
# N_samples_list = [300, 150,  25, 5]
#L_lst = [128]
#max_jobs = [20]
#N_samples_list = [300]
L_lst = [512, 1024, 2048, 4096, 8192, 16384]
num_runs_lst = [700, 500, 400, 200, 100, 50]
ft_min = [0.3178947, 0.2001948, 0.1475368, 0.1238368, 0.1047474, 0.08565789]
#L_lst = [8192, 16384]
#N_samples_list = [100, 50]
#max_jobs = [20, 20, 20, 20, 20]
max_jobs = 20
num_runs_por_L = dict(zip(L_lst, num_runs_lst))
# max_jobs = [20, 20, 4]
#L_lst = [609, 861]
#N_samples_list = [20,10]
#max_jobs = [20, 20]
# P0 = 0.2
# p0 = 0.8
P0 = 0.2
p0 = 0.8
nc = 1
#c_lst = [0.01, 0.05, 0.1, 0.15, 0.2]
#c_lst = [0.11, 0.12, 0.13, 0.14, 0.16, 0.17, 0.18, 0.19, 0.20]
#c_lst = [0.01 * i for i in range(1, 21)]
#c_lst = [0.05, 0.1, 0.15]
#c_lst = np.round(np.arange(0.01, 0.21, 0.01), 2)

#c_lst =  [1.1, 1.2, 1.3, 1.4, 1.6,  1.7,  1.8, 1.9]
equilibration = False
properties = False
run_mode = "growth_test"
initial_layout = "random"
surface_observables = False
save_animation_window_only = False
control_rule = "linear"
control_param = 0.0
log_epsilon = 1.0e-12
# =========================
# Submit jobs
# =========================
SUBMIT_JOBS = True

count = 0
missing_jobs = []
parameter_sets = []
snapshots = []
deleted_dirs = []
#n_points = 40
#ft_lst = np.linspace(0.001, 0.4, 20)
#ft_lst = np.linspace(0.4, 0.8, 20)
#df = pd.read_csv("../SOP_data/ft_min_max_2D.csv")
rho = 1/nc
c = 0.01
for idx, L in enumerate(L_lst):
	
	step = 0.01 * abs(ft_min[idx])

	left_points = ft_min[idx] - step * np.arange(7, 0, -1)
	right_points = ft_min[idx] + step * np.arange(1, 8)

	ft_lst = np.concatenate([
		left_points,
		right_points
	])
	ft_lst
	N_samples = num_runs_por_L[L]
	for ft in ft_lst:
		params = dict(
			L=L,
			p0=p0,
			seed=seed,
			type_perc=type_perc,
			c=c,
			f_T=ft,
			dim=dim,
			num_colors=nc,
			rho=rho,
			N_samples=N_samples,
			P0=P0,
			equilibration=equilibration,
			properties=properties,
			run_mode=run_mode,
			initial_layout=initial_layout,
			surface_observables=surface_observables,
			save_animation_window_only=save_animation_window_only,
			control_rule=control_rule,
			control_param=control_param,
			log_epsilon=log_epsilon,
			max_concurrent=max_jobs,
		)
		parameter_sets.append(params)


if SUBMIT_JOBS:
	for params in parameter_sets:
		snapshots.extend(snapshot_existing_outputs_for_parameters(**params))
		missing_jobs.extend(get_missing_run_parameters(**params))

	for params in parameter_sets:
		deleted_dirs.extend(cleanup_logged_output_for_parameters(**params))

	history_info = write_existing_output_snapshots(snapshots)
	print("Snapshots gravados no histórico:", history_info["snapshots_written"])
	print("Pastas apagadas após registro:", len(deleted_dirs))
	print("Conjuntos faltantes a submeter:", len(missing_jobs))
	for job in missing_jobs:
		run_multi_rho_array(**job)
		count += 1
	print("Total submetido pelo loop:", count)
else:
	for params in parameter_sets:
		missing_jobs.extend(get_missing_run_parameters(**params))

	print("Total de conjuntos faltantes:", len(missing_jobs))
	pprint.pp(missing_jobs)
