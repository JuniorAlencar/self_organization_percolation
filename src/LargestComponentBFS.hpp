#ifndef LargestComponentBFS_hpp
#define LargestComponentBFS_hpp

#pragma once
#include <vector>
#include <functional>

struct BFSResult {
    int size = 0;
    std::vector<int> nodes;
    bool touches_base = false;
    bool touches_top  = false;
    int zmin =  1e9;
    int zmax = -1e9;
};

struct BiggestComponent {
    // Agora valid_coord pode AJUSTAR (wrap) x,y,z por referência
    BFSResult find_largest_component_for_color(
        int dim, int SX, int SY, int SZ,
        const std::function<int(int,int,int)>& get_val,
        const std::function<bool(int&,int&,int&)>& valid_coord,  // <<<<< aqui
        int active_val
    );

private:
    int  lin_index(int dim, int SX, int SY, int SZ, int x, int y, int z);
    void unlin(int dim, int SX, int SY, int SZ, int id, int& x, int& y, int& z);
    bool is_active_same_color(
        int dim, int SX, int SY, int SZ,
        int x, int y, int z,
        const std::function<int(int,int,int)>& get_val,
        int active_val
    );

    BFSResult bfs_component_from_seed(
        int dim, int SX, int SY, int SZ,
        int seed_id, int active_val,
        const std::function<int(int,int,int)>& get_val,
        const std::function<bool(int&,int&,int&)>& valid_coord,  // <<<<< aqui
        std::vector<char>& visited
    );
};

#endif // LargestComponentBFS_hpp
