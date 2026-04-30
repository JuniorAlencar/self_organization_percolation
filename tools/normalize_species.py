#!/usr/bin/env python3
import struct, sys, os
if len(sys.argv) < 4:
    print('Usage: normalize_species.py IN.BIN OUT.BIN NS')
    sys.exit(1)
inf, outf = sys.argv[1], sys.argv[2]
ns = int(sys.argv[3])
with open(inf,'rb') as f:
    magic = struct.unpack('<I', f.read(4))[0]
    if magic != 0x4E455447:
        raise SystemExit('bad magic')
    N = struct.unpack('<I', f.read(4))[0]
    E = struct.unpack('<Q', f.read(8))[0]
    pos = f.read(4*N)
    species = bytearray(f.read(N))
    at = f.read(4*N)
    rest = f.read()
# normalize
for i in range(N):
    species[i] = 0 if species[i] == 0 else (1 if ns==1 else min(species[i], ns))
with open(outf,'wb') as f:
    f.write(struct.pack('<I', magic))
    f.write(struct.pack('<I', N))
    f.write(struct.pack('<Q', E))
    f.write(pos)
    f.write(species)
    f.write(at)
    f.write(rest)
print('Wrote',outf)
