#!/usr/bin/env python3
import json, sys, os
import matplotlib.pyplot as plt

fn = sys.argv[1] if len(sys.argv) > 1 else 'results/percolation_seed_12345_L_128_T_500.json'
if not os.path.exists(fn):
    print('File not found:', fn)
    sys.exit(2)

with open(fn, 'r') as f:
    data = json.load(f)

# pick the first percolation entry
results = data.get('results', {})
if not results:
    print('No results in JSON')
    sys.exit(1)

first_key = next(iter(results.keys()))
entry = results[first_key]['data']

t = entry.get('time', [])
pt = entry.get('pt', [])
nt = entry.get('nt', [])

os.makedirs('results', exist_ok=True)

plt.figure(figsize=(8,4))
plt.plot(t, pt, '-o', markersize=3)
plt.axhline(0.24881182, color='red', linestyle='--', label='p = 0.24881182')
plt.xlabel('t')
plt.ylabel('p(t)')
plt.title('t vs p(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/plot_p_t.png', dpi=150)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(t, nt, '-o', markersize=3)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('t vs f(t)')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/plot_f_t.png', dpi=150)
plt.close()

print('Saved plots: results/plot_p_t.png, results/plot_f_t.png')
