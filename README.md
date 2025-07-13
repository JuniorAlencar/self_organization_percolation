# self_organization_percolation

required libraries

* cnpy.h
* zip.h
* zlib.h
* boost.h

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
./build/SOP <L> <N_samples> <p0> <seed>
```
where  
<b> L </b>: Side of network;  
<b> N_samples </b>: Number of implementations (time);  
<b> p0 </b>: Initial growth probability;  
<b> seed </b>: Seed to implemetantion (If it is <b> -1 </b> it generates a random integer seed).

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



