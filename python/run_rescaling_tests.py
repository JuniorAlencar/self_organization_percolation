from src.run_samples_functions import shell_data


seed = -1
dim = 2
type_perc = "bond"
nc = 1

L_lst = [1024, 2048, 4096, 8192, 16384]
num_runs_lst = [500, 400, 200, 100, 50]
num_runs_por_L = dict(zip(L_lst, num_runs_lst))

c_lst = [0.01, 0.05, 0.1, 0.15]
ft_lst = [0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38]

p0 = 0.8
P0 = 0.2
rho = 1 / nc
multi = True
Equilibration = "false"
Properties = "false"
Mode = "growth_test"
InitialLayout = "random"
SurfaceObservables = "false"
SaveAnimationWindowOnly = "false"

tests = [
    {
        "label": "floor_log",
        "control_rule": "floor_log",
        "floor_f0": 0.24,
        "log_epsilon": 1.0e-5,
    },
    {
        "label": "floor_linear",
        "control_rule": "floor_linear",
        "floor_f0": 0.24,
        "log_epsilon": 1.0e-5,
    },
    {
        "label": "log",
        "control_rule": "log",
        "floor_f0": 0.0,
        "log_epsilon": 1.0e-5,
    },
]

for test in tests:
    for c in c_lst:
        for ft in ft_lst:
            for L in L_lst:
                num_runs = num_runs_por_L[L]
                exec_name = (
                    f"test_{test['label']}_L_{L}_ft_{ft:.3f}_c_{c}_"
                    f"f0_{test['floor_f0']:.3f}_eps_{test['log_epsilon']:.0e}_"
                    f"nc_{nc}_dim_{dim}_p0_{p0}_P0_{P0}_type_{type_perc}.sh"
                )
                shell_data(
                    L,
                    type_perc,
                    p0,
                    seed,
                    c,
                    ft,
                    dim,
                    nc,
                    num_runs,
                    [rho],
                    exec_name,
                    P0,
                    Equilibration,
                    multi,
                    properties=Properties,
                    mode=Mode,
                    initial_layout=InitialLayout,
                    surface_observables=SurfaceObservables,
                    save_animation_window_only=SaveAnimationWindowOnly,
                    control_rule=test["control_rule"],
                    floor_f0=test["floor_f0"],
                    log_epsilon=test["log_epsilon"],
                )
