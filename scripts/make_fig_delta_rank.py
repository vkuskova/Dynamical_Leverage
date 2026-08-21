"""make_fig_delta_rank.py -- volume rank minus E_j rank, diverging bars.
Reads results/clean_rankings.csv and results/clean_validation.csv."""
import os, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
rk = pd.read_csv(os.path.join(R, "clean_rankings.csv"))
val = pd.read_csv(os.path.join(R, "clean_validation.csv"))
vol_rank = val.set_index('nation')['volume'].rank(ascending=False).astype(int)
m = rk.set_index('nation')
delta = (vol_rank - m['mean_rank']).sort_values()
sd = m['sd'].reindex(delta.index)
short = {'United States of America':'United States','China Hong Kong SAR':'Hong Kong SAR',
         'Russian Federation':'Russia','China mainland':'China (mainland)'}
labels = [short.get(n, n) for n in delta.index]
fig, ax = plt.subplots(figsize=(3.5, 3.9))
colors = ['#b2182b' if d > 0 else '#2166ac' for d in delta.values]
ax.barh(range(len(delta)), delta.values, xerr=sd.values, color=colors,
        height=0.72, error_kw=dict(lw=0.7, capsize=1.5, ecolor='0.35'))
ax.set_yticks(range(len(delta))); ax.set_yticklabels(labels, fontsize=6.4)
ax.axvline(0, color='0.2', lw=0.8)
ax.set_xlabel(r'volume rank $-$ $E_j$ rank', fontsize=8)
ax.tick_params(axis='x', labelsize=7)
ax.text(0.97, 0.96, 'leverage above size', transform=ax.transAxes,
        ha='right', va='top', fontsize=6.5, color='#b2182b')
ax.text(0.03, 0.04, 'size above leverage', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=6.5, color='#2166ac')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(R, "fig_delta_rank.pdf"))
print("saved results/fig_delta_rank.pdf")
