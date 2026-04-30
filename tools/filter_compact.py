#!/usr/bin/env python3
import struct
import sys

in_fn = 'results/network_compact_seed_12345_L_256_T_200.bin'
out_fn = 'results/network_compact_active_regen_seed_12345_L_256_T_200.bin'

with open(in_fn,'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    if magic != 0x4E455447:
        raise SystemExit('bad magic')
    N = struct.unpack('<I', f.read(4))[0]
    E = struct.unpack('<Q', f.read(8))[0]

    pos = [struct.unpack('<I', f.read(4))[0] for _ in range(N)]
    species = [struct.unpack('<B', f.read(1))[0] for _ in range(N)]
    activation_time = [struct.unpack('<I', f.read(4))[0] for _ in range(N)]
    edge_offsets = [struct.unpack('<I', f.read(4))[0] for _ in range(N+1)]
    edges = [struct.unpack('<I', f.read(4))[0] for _ in range(E)]

# build remap
remap = [-1]*N
active_idx = []
for i in range(N):
    if species[i] != 0:
        remap[i] = len(active_idx)
        active_idx.append(i)

M = len(active_idx)
# rebuild edge pairs
pairs = []
for u in range(N):
    if remap[u] == -1: continue
    start = edge_offsets[u]
    end = edge_offsets[u+1]
    for k in range(start,end):
        v = edges[k]
        if v<0 or v>=N: continue
        if remap[v] == -1: continue
        pairs.append((remap[u], remap[v]))

# build CSR
offsets = [0]*(M+1)
for p in pairs:
    offsets[p[0]+1] += 1
for i in range(1,M+1): offsets[i]+=offsets[i-1]
E2 = len(pairs)
edges2 = [0]*E2
cursor = offsets.copy()
for u,v in pairs:
    edges2[cursor[u]] = v
    cursor[u]+=1

# write output
with open(out_fn,'wb') as f:
    f.write(struct.pack('<I',0x4E455447))
    f.write(struct.pack('<I', M))
    f.write(struct.pack('<Q', E2))
    for i in active_idx:
        f.write(struct.pack('<I', pos[i]))
    for i in active_idx:
        # normalize species to 1
        f.write(struct.pack('<B', 1))
    for i in active_idx:
        f.write(struct.pack('<I', activation_time[i]))
    for v in offsets:
        f.write(struct.pack('<I', v))
    for e in edges2:
        f.write(struct.pack('<I', e))

print('Wrote', out_fn, 'M=', M, 'E=', E2)
