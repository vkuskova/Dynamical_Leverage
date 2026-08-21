# regen_fig_trade_scatter.py
# Regenerates fig_trade_scatter.pdf for the WSDM 2027 paper from the verified
#
# Inputs : WSDM_TemporalEj/trade_clean/clean_validation.csv
#          WSDM_TemporalEj/trade_clean/clean_rankings.csv
# Output : WSDM_TemporalEj/trade_clean/fig_trade_scatter.pdf (+ .png preview)

import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = 'results'
assert os.path.isdir(DATA_DIR), f'missing {DATA_DIR}'

v = pd.read_csv(os.path.join(DATA_DIR, 'clean_validation.csv'))
r = pd.read_csv(os.path.join(DATA_DIR, 'clean_rankings.csv'))

v['volume_rank'] = v.volume.rank(ascending=False).astype(int)
df = v.merge(r[['nation', 'mean_rank', 'sd']], on='nation')
assert len(df) == 20, f'expected 20 nations, got {len(df)}'

iso3 = {
    'United States of America': 'USA', 'United Kingdom': 'GBR',
    'Germany': 'DEU', 'France': 'FRA', 'Netherlands': 'NLD',
    'Japan': 'JPN', 'Italy': 'ITA', 'China mainland': 'CHN',
    'Russian Federation': 'RUS', 'India': 'IND', 'Spain': 'ESP',
    'Saudi Arabia': 'SAU', 'Turkey': 'TUR', 'Belgium': 'BEL',
    'Canada': 'CAN', 'Indonesia': 'IDN', 'Portugal': 'PRT',
    'Malaysia': 'MYS', 'Switzerland': 'CHE', 'China Hong Kong SAR': 'HKG',
}
df['lab'] = df.nation.map(iso3)
assert df.lab.notna().all(), 'unmapped nation name'

# manual label offsets (dx, dy, ha) for crowded points; default (0.35, 0, left)
off = {'USA': (-0.4, 0.0, 'right'), 'GBR': (0.35, -0.75, 'left'),
       'FRA': (0.35, 0.65, 'left'), 'PRT': (-0.4, 0.55, 'right'),
       'CHE': (0.0, 1.0, 'center'), 'HKG': (0.4, -0.3, 'left'),
       'BEL': (-0.4, 0.0, 'right'), 'CAN': (0.4, 0.35, 'left'),
       'IDN': (0.4, 0.35, 'left'), 'MYS': (0.4, 0.3, 'left')}

fig, ax = plt.subplots(figsize=(3.33, 3.1))
ax.plot([0.5, 20.5], [0.5, 20.5], color='0.75', lw=0.8, ls='--', zorder=1)
ax.errorbar(df.volume_rank, df.mean_rank, yerr=df.sd, fmt='o', ms=4,
            color='#1f4e79', ecolor='#9db8d2', elinewidth=0.9, zorder=3)
for _, row in df.iterrows():
    dx, dy, ha = off.get(row.lab, (0.35, 0.0, 'left'))
    ax.annotate(row.lab, (row.volume_rank, row.mean_rank),
                xytext=(row.volume_rank + dx, row.mean_rank + dy),
                fontsize=6.5, ha=ha, va='center', color='0.15')
ax.set_xlabel('Attraction-volume rank', fontsize=8)
ax.set_ylabel(r'$E_j$ rank (mean $\pm$ sd, 8 seeds)', fontsize=8)
ax.set_xlim(0.0, 21.8)
ax.set_ylim(21.4, 0.0)
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_yticks([1, 5, 10, 15, 20])
ax.tick_params(labelsize=7)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(DATA_DIR, 'fig_trade_scatter.pdf'))
fig.savefig(os.path.join(DATA_DIR, 'fig_trade_scatter.png'), dpi=200)
plt.show()
print('written:', os.path.join(DATA_DIR, 'fig_trade_scatter.pdf'))