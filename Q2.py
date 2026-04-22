"""
Q2_diagnostic.py

RQ2: What patterns exist in cost changes across household types?
Diagnostic analysis — decomposed financial sub-components (Table 3),
basket weight snapshot, and cross-household correlation heatmap.

RBA cash rate data (Reserve Bank of Australia) integrated as a heterogeneous
source to explain the structural divergence in financial sub-components.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches


#  LOAD & FILTER 

df = pd.read_parquet('cleaned_data_multiindex.parquet')
df = df[df.index >= '2007-06-01']

# Normalise SLCI index to quarter-end so it aligns with RBA parquet
df.index = df.index.to_period('Q').to_timestamp('Q')

LABELS = {
    'Pensioner and beneficiary households':           'Pensioner & Beneficiary',
    'Employee households':                            'Employee',
    'Age pensioner households':                       'Age Pensioner',
    'Other government transfer recipient households': 'Other Govt Transfer',
    'Self-funded retiree households':                 'Self-funded Retiree',
}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

#  LOAD RBA DATA 
# Heterogeneous source: Reserve Bank of Australia cash rate decisions.
# Resampled from irregular meeting dates to quarterly period-end in etl_rba_cashrate.py.

df_rba = pd.read_parquet('cleaned_data_rba.parquet').set_index('Date')
df_rba.index = pd.to_datetime(df_rba.index).to_period('Q').to_timestamp('Q')
df_rba = df_rba[df_rba.index >= '2007-06-01']

# Quarter-end dates where the cash rate changed — for event lines on time-series panels
hike_dates = df_rba[df_rba['RBA_Direction'] == 1].index
cut_dates  = df_rba[df_rba['RBA_Direction'] == -1].index


#  PANEL DATA 

yoy = df['Percentage Change from Corresponding Quarter of Previous Year']
qoq = df['Percentage Change from Previous Period']
pts = df['Points Contribution to All Groups']

# Panel A: Housing annual % change
housing_pct = yoy.xs('Housing', level='Commodity', axis=1).rename(columns=LABELS)

# Panel B: Health quarterly % change (PBS safety net cycle)
health_qtr = qoq.xs('Health', level='Commodity', axis=1).rename(columns=LABELS)

# Panel C: Decomposed financial sub-components (Table 3)
focus_hh = {
    'Employee households':      ('Employee',      '-',  0.95),
    'Age pensioner households': ('Age Pensioner', '--', 0.65),
}
t3_commodities = {
    'Mortgage interest charges': '#FF5722',
    'Gross Insurance':           '#42A5F5',
    'Consumer credit charges':   '#66BB6A',
}

# Panel D: Full basket snapshot — Dec 2025 points contributions
ALL_CATS = [
    'Food and non-alcoholic beverages', 'Housing', 'Health', 'Transport',
    'Education', 'Insurance and financial services',
    'Furnishings, household equipment and services',
    'Recreation and culture', 'Clothing and footwear',
    'Alcohol and tobacco', 'Communication',
]
SHORT_CAT = {
    'Food and non-alcoholic beverages':             'Food',
    'Housing':                                      'Housing',
    'Health':                                       'Health',
    'Transport':                                    'Transport',
    'Education':                                    'Education',
    'Insurance and financial services':             'Insurance\n(blended)',
    'Furnishings, household equipment and services':'Furnishings',
    'Recreation and culture':                       'Recreation',
    'Clothing and footwear':                        'Clothing',
    'Alcohol and tobacco':                          'Alcohol',
    'Communication':                                'Comms',
}

snapshot = {}
for cat in ALL_CATS:
    row = pts.xs(cat, level='Commodity', axis=1).rename(columns=LABELS).dropna()
    if not row.empty:
        snapshot[SHORT_CAT[cat]] = row.iloc[-1]
snapshot_df = pd.DataFrame(snapshot)

# Panel E: Cross-household correlation per cost category
corr_results = {}
for cat in ALL_CATS + ['Mortgage interest charges', 'Gross Insurance']:
    if cat not in yoy.columns.get_level_values('Commodity'):
        continue
    sub = yoy.xs(cat, level='Commodity', axis=1).rename(columns=LABELS).dropna()
    if sub.shape[1] < 2:
        continue
    corr_results[SHORT_CAT.get(cat, cat)] = sub.corr().values[np.triu_indices(5, k=1)].mean()

corr_series = pd.Series(corr_results).sort_values()

# RBA context for Panel E annotation:
# Compute mean cash rate during hike cycle vs cut cycle for the annotation text
rba_hike_mean = df_rba.loc[df_rba['RBA_Direction'] == 1,  'RBA_Cash_Rate_Pct'].mean()
rba_cut_mean  = df_rba.loc[df_rba['RBA_Direction'] == -1, 'RBA_Cash_Rate_Pct'].mean()
n_hike_qtrs   = int((df_rba['RBA_Direction'] == 1).sum())
n_cut_qtrs    = int((df_rba['RBA_Direction'] == -1).sum())


#  PLOT 

fig = plt.figure(figsize=(16, 26))
fig.patch.set_facecolor('#0F1117')
fig.suptitle(
    'RQ2: What Patterns Exist in Cost Changes Across Household Types?\n'
    'Diagnostic Analysis — Jun 2007 – Dec 2025',
    fontsize=14, fontweight='bold', color='white', y=0.99
)

gs = fig.add_gridspec(5, 1, hspace=0.52, left=0.09, right=0.97, top=0.97, bottom=0.04)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3])
ax5 = fig.add_subplot(gs[4])

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_facecolor('#1A1D27')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.tick_params(colors='#aaa')


def shade_events(ax):
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-01'),
               alpha=0.10, color='grey')
    ax.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2023-06-01'),
               alpha=0.10, color='red')
    ax.grid(True, alpha=0.2, color='#555')


def add_rba_events(ax):
    """Dashed vertical lines at every RBA hike (orange) and cut (blue) quarter."""
    for d in hike_dates:
        if d >= pd.Timestamp('2007-06-01'):
            ax.axvline(d, color='#FF5722', alpha=0.30, linewidth=0.8, linestyle='--')
    for d in cut_dates:
        if d >= pd.Timestamp('2007-06-01'):
            ax.axvline(d, color='#42A5F5', alpha=0.30, linewidth=0.8, linestyle='--')


#  Panel A: Housing 
for col, color in zip(housing_pct.columns, COLORS):
    ax1.plot(housing_pct.index, housing_pct[col], label=col, color=color, linewidth=2)
ax1.axhline(0, color='white', linewidth=0.5)
ax1.set_title(
    'A: Housing — Annual % Change\n'
    'Renters (pensioners) experience persistent pressure; '
    'Employee peak moderated by mortgage falls in 2025',
    color='white', fontsize=9, fontweight='bold'
)
ax1.set_ylabel('Annual Change (%)', color='#aaa')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.legend(fontsize=7.5, loc='upper left', framealpha=0.2, facecolor='#222', labelcolor='white')
shade_events(ax1)
add_rba_events(ax1)

#  Panel B: Health (PBS cycle 
for col, color in zip(health_qtr.columns, COLORS):
    ax2.plot(health_qtr.index, health_qtr[col], label=col, color=color, linewidth=2)
ax2.axhline(0, color='white', linewidth=0.5)
ax2.set_title(
    'B: Health — Quarterly % Change\n'
    'Negative spikes = PBS safety net resets; '
    'deeper and more frequent for pensioner/government households',
    color='white', fontsize=9, fontweight='bold'
)
ax2.set_ylabel('Quarterly Change (%)', color='#aaa')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.legend(fontsize=7.5, loc='upper left', framealpha=0.2, facecolor='#222', labelcolor='white')
shade_events(ax2)
add_rba_events(ax2)

#  Panel C: Financial sub-components + RBA cash rate overlay 
# This is the primary integration panel for Q2.
# Left axis  → ABS SLCI Table 3 YoY % change (mortgage, insurance, credit)
# Right axis → RBA Cash Rate Target % (Reserve Bank of Australia)
# Together they show that Employee mortgage costs are mechanically driven by RBA decisions,
# while Age Pensioner insurance costs are structurally independent of the rate cycle.

ax3_r = ax3.twinx()
ax3_r.set_facecolor('#1A1D27')

# Plot SLCI Table 3 series (left axis)
legend_handles = []
for hh_full, (hh_short, ls, alpha) in focus_hh.items():
    for comm, color in t3_commodities.items():
        if (hh_full, comm) in yoy.columns:
            series = yoy[(hh_full, comm)].dropna()
            line, = ax3.plot(series.index, series.values,
                             color=color, linestyle=ls, alpha=alpha,
                             linewidth=2, label=f'{hh_short} — {comm}')
            legend_handles.append(line)

ax3.axhline(0, color='white', linewidth=0.5)
ax3.set_ylabel('Annual Change % (ABS SLCI Table 3)', color='#aaa')
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())

# Plot RBA cash rate (right axis) as a filled step chart
rba_cash = df_rba['RBA_Cash_Rate_Pct'].dropna()
ax3_r.step(rba_cash.index, rba_cash.values, where='post',
           color='#FFCA28', linewidth=2, alpha=0.9, label='RBA Cash Rate Target %')
ax3_r.fill_between(rba_cash.index, rba_cash.values,
                   step='post', alpha=0.08, color='#FFCA28')
ax3_r.set_ylabel('RBA Cash Rate Target % (RBA)', color='#FFCA28', fontsize=8)
ax3_r.tick_params(axis='y', colors='#FFCA28')
ax3_r.yaxis.set_major_formatter(mtick.PercentFormatter())
ax3_r.set_ylim(-2, 20)
for spine in ax3_r.spines.values():
    spine.set_edgecolor('#333')

ax3.set_title(
    'C: Financial Sub-Components Decomposed — Annual % Change  (ABS Table 3)  '
    '+  RBA Cash Rate (amber, right axis)\n'
    'Solid = Employee  |  Dashed = Age Pensioner  —  '
    'Mortgage tracks RBA decisions; Insurance & Credit are rate-independent',
    color='white', fontsize=9, fontweight='bold'
)

# Combined legend: SLCI series + RBA line
rba_line = mpatches.Patch(color='#FFCA28', alpha=0.9, label='RBA Cash Rate Target % (RBA source)')
ax3.legend(handles=legend_handles + [rba_line],
           fontsize=7.5, loc='upper left',
           framealpha=0.2, facecolor='#222', labelcolor='white', ncol=2)

shade_events(ax3)
add_rba_events(ax3)

ax3.annotate('Mortgage peaks\n~+90% (Jun-23)',
             xy=(pd.Timestamp('2023-06-01'), 88),
             xytext=(pd.Timestamp('2021-06-01'), 65),
             color='#FF5722', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.2))
ax3.annotate('RBA cuts → -6.4%\n(Dec-25)',
             xy=(pd.Timestamp('2025-12-01'), -6),
             xytext=(pd.Timestamp('2024-03-01'), -30),
             color='#FF5722', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.2))
ax3.annotate('Insurance: flat ~2–3%/yr\nregardless of rate cycle',
             xy=(pd.Timestamp('2023-06-01'), 5),
             xytext=(pd.Timestamp('2018-06-01'), 25),
             color='#42A5F5', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#42A5F5', lw=1.2))

#  Panel D: Full basket snapshot bar chart 
x      = np.arange(len(snapshot_df.columns))
n_hh   = len(snapshot_df.index)
width  = 0.15
offsets = np.linspace(-(n_hh-1)/2 * width, (n_hh-1)/2 * width, n_hh)

for i, (hh_short, color) in enumerate(zip(snapshot_df.index, COLORS)):
    bars = ax4.bar(x + offsets[i], snapshot_df.loc[hh_short].values,
                   width=width, label=hh_short, color=color, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if abs(h) > 0.5:
            ax4.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                     f'{h:.1f}', ha='center', va='bottom',
                     fontsize=5.5, color='#ccc')

ax4.set_title(
    'D: Full Basket — Points Contribution to All Groups  (Dec 2025)\n'
    'Housing dominates Other Govt Transfer; Insurance/mortgage dominates Employee; '
    'Health elevated for Age Pensioner & Self-funded Retiree',
    color='white', fontsize=9, fontweight='bold'
)
ax4.set_ylabel('Points Contribution', color='#aaa')
ax4.set_xticks(x)
ax4.set_xticklabels(snapshot_df.columns, fontsize=8, rotation=30,
                    ha='right', color='#aaa')
ax4.legend(fontsize=7.5, framealpha=0.2, facecolor='#222', labelcolor='white')
ax4.grid(True, alpha=0.2, axis='y', color='#555')
ax4.axhline(0, color='white', linewidth=0.4)

#  Panel E: Cross-household correlation + RBA annotation 
bar_colors = ['#FF5722' if v < 0.85 else '#42A5F5' for v in corr_series.values]
bars = ax5.barh(corr_series.index, corr_series.values, color=bar_colors)
ax5.set_xlim(0, 1.15)
ax5.axvline(0.85, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

# Annotate the two RBA-sensitive categories with their RBA context
rba_sensitive = ['Mortgage\nInterest', 'Insurance\n(blended)']
for label in rba_sensitive:
    if label in corr_series.index:
        y_pos = list(corr_series.index).index(label)
        val   = corr_series[label]
        note  = ('Rate-cycle driven\n(RBA hike/cut dependent)'
                 if 'Mortgage' in label else
                 'Rate-independent\n(structural divergence)')
        ax5.text(val + 0.03, y_pos, note,
                 va='center', ha='left', fontsize=7,
                 color='#FFCA28',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#1A1D27',
                           edgecolor='#FFCA28', alpha=0.7))

ax5.set_title(
    'E: Mean Pairwise Cross-Household Correlation — Annual % Change per Cost Category\n'
    f'High correlation = universal shocks; Low correlation = structural divergence  '
    f'(RBA drove {n_hike_qtrs} hike qtrs, {n_cut_qtrs} cut qtrs 2007–2025)',
    color='white', fontsize=9, fontweight='bold'
)
ax5.set_xlabel('Mean Pairwise Pearson Correlation', color='#aaa')
ax5.bar_label(bars, fmt='%.2f', color='#ccc', padding=3, fontsize=8)
ax5.grid(True, alpha=0.2, axis='x', color='#555')

diverge_p = mpatches.Patch(color='#FF5722', label='Divergent (<0.85) — structural difference between households')
aligned_p = mpatches.Patch(color='#42A5F5', label='Aligned (≥0.85) — all households move together')
ax5.legend(handles=[diverge_p, aligned_p], framealpha=0.2,
           facecolor='#222', labelcolor='white', fontsize=8)

#  Shared shading legend 
covid_p = mpatches.Patch(color='grey',    alpha=0.4, label='COVID-19 (Mar–Dec 2020)')
rate_p  = mpatches.Patch(color='red',     alpha=0.4, label='RBA rate rise cycle (Mar 2022–Jun 2023)')
hike_p  = mpatches.Patch(color='#FF5722', alpha=0.6, label='RBA hike quarter (dashed, panels A–C)')
cut_p   = mpatches.Patch(color='#42A5F5', alpha=0.6, label='RBA cut quarter (dashed, panels A–C)')
fig.legend(handles=[covid_p, rate_p, hike_p, cut_p],
           loc='lower center', ncol=4, framealpha=0.2,
           facecolor='#222', labelcolor='white',
           fontsize=8, bbox_to_anchor=(0.5, 0.005))

plt.savefig('Q2_diagnostic.png', dpi=150, bbox_inches='tight', facecolor='#0F1117')
print("Saved → Q2_diagnostic.png")