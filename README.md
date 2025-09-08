# self_organization_percolation

required libraries

* cnpy.h
* zip.h
* zlib.h
* boost.h
* openMP.h

# To install *cnpy.h*

```bash
git clone https://github.com/rogersce/cnpy.git
cd cnpy
mkdir build
cd build
cmake ..
make
sudo make install
```

# To install *zlib* (linux ubuntu)
```bash
sudo apt install zlib1g-dev
```

# To install *boost* (linux ubuntu)
```bash
sudo apt install libboost-all-dev
```

# To install *openMP* (linux ubuntu)
```bash
sudo apt update
sudo apt install build-essential

```


# How to run the program
After all libraries are installed, create the build folder, access it and do

```bash
mkdir build
cd build
cmake ..
make
```

This will generate the <b>SOP</b> executable. Run the code outside the build folder

```bash
./build/SOP <L> <N_samples> <$\boldsymbol{p_0}$> <seed> <type_percolation> <k> <$\boldsymbol{N_t}$> <dim> <num_colors> <$\boldsymbol{\rho}$>
```
where  
<b> L </b>: Side of network;  
<b> N_samples </b>: Number of implementations (time);  
$\boldsymbol{p_0}$: Initial growth probability;  
<b> seed </b>: Seed to implemetantion (If it is <b> -1 </b> it generates a random integer seed);  
<b> type_percolation </b>: Define the type of percolation -> bond or node;  
<b> $k$ </b>: kinetic coefficient (see SOP_paper.pdf in docs);  
$\boldsymbol{N_t}$: threshold parameter (see SOP_paper.pdf in docs);  
<b> dim </b>: dimension of network;    
<b> num_colors </b>: Number of colors in network;  
$\boldsymbol{\rho}$: Density of network for each color;  

<b> VERY IMPORTANT!</b>  
The value of rho must be such that num_colors*rho <= 1. If num_colors*$\rho$ < 1, the gap will be filled with uncolored sites. If num_colors*$\rho$ = 1, all sites in the network will be uniformly and equally colored.

<b> Example </b>: $\boldsymbol{\rho}$ = 0.5 for 2 colors, this gives us num_colors*$\boldsymbol{\rho}$ = 1.0, meaning that half of the network will be composed of inactive sites of each color.  
<b> Example </b>: $\boldsymbol{\rho}$ = 0.4 for 2 colors, this gives us num_colors*$\boldsymbol{\rho}$ = 0.8, meaning that 40% of the network sites will have color 2, 40% will have color 3, and 20% will have no color.  


## Running this way will generate a folder structure

```bash
SELF_ORGANIZATION_PERCOLATION/
├── build/
├── Data/              # simulation results
├── docs/
├── jupyter/
├── src/
├── .gitignore
├── CMakeLists.txt
└── README.md
```

# Initial trails to execute

This algorithm is quite sensitive to the parameters $N_T$ and $k$, which vary greatly depending on the dimension. Below are some suggested parameters, depending on the network size. For dimension $d = 2$, the values used in the article /docs/SOP_paper.pdf are good. For $d = 3$, we have:

We always used $N_t > f_{active}$, where $f_{active}$ representing the number of nodes initial activate. In 2D was used $f_{active} =P0 * L$ and in 3D $f_{active} =(P_0 * L)^2$. 

# How to choice the $Nt$
By performing some tests, I realized that the choice of Nt can be based on the fraction of initially active sites, $f_{active}$. We can adopt $Nt = 1.25 * f_{active}$. For dim = 3, we will have $Nt = 1.25 * (P_0 * L)^2$. For this to be valid, we must pay attention to the value we use for $k$, which varies according to the size of the network. For $L = 128$, we can use $k=1.0e-05$. Below we have some tested values ​​for $num_colors = 4$ and dim$ = 3$. Was  


