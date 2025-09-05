from src.run_samples_functions import *

# num_runs = 20   # number of external repetitions
# rho = [round(i, 2) for i in np.arange(0.05, 0.55, 0.05)]  # rho values
# L = 2000
# NumSamples = 9000
# p0 = 1.0
# seed = -1
# type_perc = "bond"
# k = 1.0e-5
# NT = 200
# dim = 2
# num_colors = 2
# exec_name = "data_2D.sh"

# shell_data(L, NumSamples, type_perc, p0, seed, k, NT, dim,
#            num_colors, num_runs, rho, exec_name)

# num_runs = 20   # number of external repetitions
# rho = [round(i, 2) for i in np.arange(0.03, 0.36, 0.03)]  # rho values
# L = 2000
# NumSamples = 9000
# p0 = 1.0
# seed = -1
# type_perc = "bond"
# k = 1.0e-5
# NT = 200
# dim = 2
# num_colors = 3
# exec_name = "data_2D.sh"

# shell_data(L, NumSamples, type_perc, p0, seed, k, NT, dim,
#            num_colors, num_runs, rho, exec_name)

num_runs = 20   # number of external repetitions

stop = 0.33
start = 0.003
n_points = 20
rho = custom_range(start, stop, n_points)

L = 2000
NumSamples = 12000
p0 = 1.0
seed = -1
type_perc = "bond"
k = 1.0e-4
NT = 200
dim = 2
num_colors = 3
exec_name = f"data_{dim}D.sh"

shell_data(L, NumSamples, type_perc, p0, seed, k, NT, dim,
           num_colors, num_runs, rho, exec_name)