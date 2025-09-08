import os
import stat
import textwrap
import numpy as np

def shell_data(L:int, type_perc:str, p0:float, 
               seed:int, k:float, NT:float, dim:int, num_colors:int, 
               num_runs:int, rho:list, exec_name:str):
    """
    Generate a bash script to run the SOP executable multiple times.

    Parameters:
    -----------
    L : int
        Side of the network
    type_perc : str
        Type of percolation ("bond" or "node")
    p0 : float
        Initial probability (<= 1.0)
    seed : int
        Random seed (-1 means a random seed will be generated inside C++)
    k : float
        Kinetic coefficient (see article)
    NT : int
        Threshold parameter (see article)
    dim : int
        Dimension of the network (2 or 3)
    num_colors : int
        Number of colors in the network
    num_runs : int
        Number of external repetitions
    rho : list of float
        List of density values for colors
    exec_name : str
        Name of the generated shell script file
    """
    if dim not in (2,3):
        return "please, enter with dim = 2 or 3"
    
    print("Creating shell script file in ../shells/")
    
    
    
    script = f"""\
#!/bin/bash

cd .. # Got o project root

num_runs={num_runs}   # number of external repetitions
rho=({" ".join(map(str, rho))})  # list of rho values
L={L}
p0={p0}
seed={seed}
type="{type_perc}"
k={k}
NT={NT}
dim={dim}
num_colors={num_colors}

for ((run=1; run<=num_runs; run++)); do
  echo "=== Run $run ==="
  for idx in "${{!rho[@]}}"; do
    RHO=${{rho[$idx]}}
    echo "./build/SOP $L $p0 $seed $type $k $NT $dim $num_colors $RHO"
    ./build/SOP "$L" " "$p0" "$seed" "$type" "$k" "$NT" "$dim" "$num_colors" "$RHO"
  done
done
"""

    # Clean unwanted indentation
    script = textwrap.dedent(script)

    folder = "../shells/"
    path = os.path.join(folder, exec_name)
    with open(path, "w") as f:
        f.write(script)
    
    # Give execute permission
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC)
    print(f"✅ Shell script created at {path}")


def custom_range(start: float, stop: float, n_points: int):
    return [round(i, 8) for i in np.linspace(start, stop, n_points)]