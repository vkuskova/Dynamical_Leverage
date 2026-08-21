# regen_fig_departure_vs_advantage.py
# WSDM 2027 synthetic figure: per-cell pooled departure (x) vs fidelity
# advantage (y) across the 144-cell grid, colored by regime structure.
# column (Spearman(E_j, AC_pooled) per cell) — run the modified phase cell
# first; this script recomputes nothing.
#
# Input  : results/phase_diagram_results.csv (with ej_acp column)
# Output : fig_departure_vs_advantage.pdf (+ .png preview) next to the CSV

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = 'results/phase_diagram_results.csv'
OUT_DIR = os.path.dirname(CSV)
assert os.path.isfile(CSV), f'missing {CSV}'

d = pd.read_csv(CSV)
assert len(d) == 144, f'expected 144 cells, got {len(d)}'
assert 'ej_acp' in d.columns, (
    'ej_acp column missing — this is the OLD grid output; '
    'rerun the modified phase cell first')

d['departure'] = 1.0 - d.ej_acp          # pooled departure, the paper's diagnostic
order = ['static', 'smooth-drift', 'persistent-switch', 'sign-flip-switch']
labels = {'static': 'static', 'smooth-drift': 'smooth drift',
          'persistent-switch': 'persistent reroute',
          'sign-flip-switch': 'sign reversal'}
colors = {'static': '#7f7f7f', 'smooth-drift': '#1f77b4',
          'persistent-switch': '#2ca02c', 'sign-flip-switch': '#b0413e'}
markers = {'static': 'o', 'smooth-drift': 's',
           'persistent-switch': '^', 'sign-flip-switch': 'D'}

fig, ax = plt.subplots(figsize=(3.33, 2.8))
ax.axhline(0, color='0.85', lw=0.8, zorder=1)
for s in order:
    g = d[d.struct == s]
    ax.scatter(g.departure, g.d_pool, s=13, marker=markers[s],
               color=colors[s], alpha=0.75, linewidths=0,
               label=labels[s], zorder=3)
ax.set_xlabel(r'pooled departure $1-\rho(E_j,\,\mathrm{AC_{pooled}})$',
              fontsize=7.5)
ax.set_ylabel(r'$E_j$ fidelity advantage over pooled AC', fontsize=7.5)
ax.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=6, frameon=False, loc='upper left',
          handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_departure_vs_advantage.pdf'))
fig.savefig(os.path.join(OUT_DIR, 'fig_departure_vs_advantage.png'), dpi=200)
plt.show()

# numbers for the paper's prose — report these back for insertion
rho, _ = spearmanr(d.departure, d.d_pool)
print(f'Spearman(departure, advantage) across 144 cells: {rho:+.3f}')
for s in order:
    g = d[d.struct == s]
    print(f'{s:>18s}: departure mean {g.departure.mean():.3f} '
          f'range [{g.departure.min():.3f}, {g.departure.max():.3f}] | '
          f'advantage mean {g.d_pool.mean():+.3f}')
print('written:', os.path.join(OUT_DIR, 'fig_departure_vs_advantage.pdf'))