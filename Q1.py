"""
Q1_living_costs_over_time.py

RQ1: How have living costs changed over time?
Descriptive analysis — index levels, annual % change, financial sub-component
breakdown (Table 3: mortgage, insurance, credit), and RBA cash rate overlay
(heterogeneous data source: Reserve Bank of Australia).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches


#  LOAD & SLICE 

df = pd.read_parquet('cleaned_data_multiindex.parquet')
df = df[df.index >= '2007-06-01']

# Normalise SLCI index to period-end (QE) so it aligns with the RBA parquet
df.index = df.index.to_period('Q').to_timestamp('Q')

LABELS = {
    'Pensioner and beneficiary households':             'Pensioner & Beneficiary',
    'Employee households':                              'Employee',
    'Age pensioner households':                         'Age Pensioner',
    'Other government transfer recipient households':   'Other Govt Transfer',
    'Self-funded retiree households':                   'Self-funded Retiree',
}
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

#  LOAD RBA DATA 
# Heterogeneous source: Reserve Bank of Australia cash rate decisions.
# Resampled from irregular meeting dates to quarterly period-end in etl_rba_cashrate.py.

df_rba = pd.read_parquet('cleaned_data_rba.parquet').set_index('Date')
df_rba.index = pd.to_datetime(df_rba.index).to_period('Q').to_timestamp('Q')
df_rba = df_rba[df_rba.index >= '2007-06-01']

# Quarter-end dates where the cash rate actually changed — used for event lines
rate_change_dates = df_rba[df_rba['RBA_Rate_Changed'] == 1].index
hike_dates = df_rba[df_rba['RBA_Direction'] == 1].index
cut_dates  = df_rba[df_rba['RBA_Direction'] == -1].index


#  PANEL DATA 

# Panel 1: All groups index levels
df_index = (df['Index Numbers']
            .xs('All groups', level='Commodity', axis=1)
            .rename(columns=LABELS))

# Panel 2: All groups YoY % change
df_annual = (df['Percentage Change from Corresponding Quarter of Previous Year']
             .xs('All groups', level='Commodity', axis=1)
             .rename(columns=LABELS))

# Panel 3: Table 3 financial sub-components YoY
yoy = df['Percentage Change from Corresponding Quarter of Previous Year']

t3_series = {}
for commodity, linestyle in [
    ('Mortgage interest charges', '-'),
    ('Gross Insurance',           '--'),
    ('Consumer credit charges',   ':'),
]:
    for hh_full, hh_short in LABELS.items():
        if (hh_full, commodity) in yoy.columns:
            series = yoy[(hh_full, commodity)].dropna()
            t3_series[(hh_short, commodity, linestyle)] = series

# Panel 4: Heatmap — mean YoY per household × cost category (post-2020)
recent = df[df.index >= '2020-06-01']
yoy_recent = recent['Percentage Change from Corresponding Quarter of Previous Year']
cost_cats = [
    'Food and non-alcoholic beverages', 'Housing', 'Health', 'Transport',
    'Education', 'Insurance and financial services', 'Recreation and culture',
    'Mortgage interest charges', 'Gross Insurance', 'Consumer credit charges',
]
SHORT_CAT = {
    'Food and non-alcoholic beverages':   'Food',
    'Housing':                            'Housing',
    'Health':                             'Health',
    'Transport':                          'Transport',
    'Education':                          'Education',
    'Insurance and financial services':   'Insurance\n(blended)',
    'Recreation and culture':             'Recreation',
    'Mortgage interest charges':          'Mortgage\nInterest',
    'Gross Insurance':                    'Gross\nInsurance',
    'Consumer credit charges':            'Consumer\nCredit',
}
heat_rows = {}
for hh_full, hh_short in LABELS.items():
    row = {}
    for cat in cost_cats:
        if (hh_full, cat) in yoy_recent.columns:
            row[SHORT_CAT[cat]] = yoy_recent[(hh_full, cat)].dropna().mean()
        else:
            row[SHORT_CAT[cat]] = float('nan')
    heat_rows[hh_short] = row
heat_df = pd.DataFrame(heat_rows).T

# Panel 5: RBA cash rate + Employee mortgage interest overlay
# This is the "heterogeneous data integration" panel — two completely different
# institutions and data types on one chart, demonstrating structural divergence.
mortgage_employee = yoy[('Employee households', 'Mortgage interest charges')].dropna()
# Align RBA to the mortgage series date range
rba_aligned = df_rba['RBA_Cash_Rate_Pct'].reindex(
    df_rba.index[df_rba.index >= mortgage_employee.index.min()]
).dropna()


# PLOT 

fig = plt.figure(figsize=(16, 28))
fig.patch.set_facecolor('#0F1117')
fig.suptitle(
    'RQ1: How Have Living Costs Changed Over Time?\nAll Household Types, Jun 2007 – Dec 2025',
    fontsize=15, fontweight='bold', color='white', y=0.99
)

gs = fig.add_gridspec(5, 1, hspace=0.5, left=0.09, right=0.97, top=0.97, bottom=0.04)
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
    """Shade COVID and RBA rate-rise cycle, add gridlines."""
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-12-01'),
               alpha=0.12, color='grey')
    ax.axvspan(pd.Timestamp('2022-03-01'), pd.Timestamp('2023-06-01'),
               alpha=0.12, color='red')
    ax.grid(True, alpha=0.2, color='#555')


def add_rba_events(ax, y_hike, y_cut, fontsize=6.5):
    """
    Add vertical tick marks at each RBA rate-change quarter.
    Hikes → red ticks at top, cuts → blue ticks at top.
    Keeps panels readable without overplotting.
    """
    ymin, ymax = ax.get_ylim()
    tick_height = (ymax - ymin) * 0.04
    for d in hike_dates:
        if d >= pd.Timestamp('2007-06-01'):
            ax.axvline(d, color='#FF5722', alpha=0.35, linewidth=0.8, linestyle='--')
    for d in cut_dates:
        if d >= pd.Timestamp('2007-06-01'):
            ax.axvline(d, color='#42A5F5', alpha=0.35, linewidth=0.8, linestyle='--')


#  Panel 1: Index levels 
for col, color in zip(df_index.columns, COLORS):
    ax1.plot(df_index.index, df_index[col], label=col, color=color, linewidth=2)
ax1.set_title('All Groups Index Level  (Sep-2025 = 100)', color='white',
              fontsize=11, fontweight='bold')
ax1.set_ylabel('Index Number', color='#aaa')
ax1.legend(fontsize=8, loc='upper left', framealpha=0.2,
           facecolor='#222', labelcolor='white')
shade_events(ax1)
add_rba_events(ax1, None, None)

# ── Panel 2: Annual % change ──────────────────────────────────────────────────
for col, color in zip(df_annual.columns, COLORS):
    ax2.plot(df_annual.index, df_annual[col], label=col, color=color, linewidth=2)
ax2.axhline(0, color='white', linewidth=0.6)
ax2.set_title('Annual % Change — All Groups  (Year-on-Year)', color='white',
              fontsize=11, fontweight='bold')
ax2.set_ylabel('Annual Change (%)', color='#aaa')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.legend(fontsize=8, loc='upper left', framealpha=0.2,
           facecolor='#222', labelcolor='white')
shade_events(ax2)
add_rba_events(ax2, None, None)

#  Panel 3: Table 3 financial sub-components 
focus_hh = ['Employee', 'Age Pensioner']
comm_colors = {
    'Mortgage interest charges': '#FF5722',
    'Gross Insurance':           '#42A5F5',
    'Consumer credit charges':   '#66BB6A',
}
comm_styles = {
    'Mortgage interest charges': '-',
    'Gross Insurance':           '--',
    'Consumer credit charges':   ':',
}
hh_alpha = {'Employee': 1.0, 'Age Pensioner': 0.55}

for (hh_short, commodity, _), series in t3_series.items():
    if hh_short not in focus_hh:
        continue
    ax3.plot(series.index, series.values,
             color=comm_colors[commodity],
             linestyle=comm_styles[commodity],
             alpha=hh_alpha[hh_short],
             linewidth=2,
             label=f'{hh_short} — {commodity}')

ax3.axhline(0, color='white', linewidth=0.6)
ax3.set_title(
    'Financial Sub-Components: Annual % Change\n'
    'Solid = Employee  |  Faded = Age Pensioner  '
    '(Table 3: Mortgage, Insurance, Consumer Credit)',
    color='white', fontsize=11, fontweight='bold'
)
ax3.set_ylabel('Annual Change (%)', color='#aaa')
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
ax3.legend(fontsize=7.5, loc='upper left', framealpha=0.2,
           facecolor='#222', labelcolor='white', ncol=2)
shade_events(ax3)
add_rba_events(ax3, None, None)

ax3.annotate('RBA rate hikes\ndrive mortgage spike\nfor Employee hh',
             xy=(pd.Timestamp('2023-03-01'), 70),
             xytext=(pd.Timestamp('2021-01-01'), 55),
             color='#FF5722', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.2))
ax3.annotate('RBA cuts 2025\nmortgage falls',
             xy=(pd.Timestamp('2025-09-01'), -6),
             xytext=(pd.Timestamp('2024-01-01'), -25),
             color='#FF5722', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#FF5722', lw=1.2))

#  Panel 4: Heatmap 
data = heat_df.values.astype(float)
im   = ax4.imshow(data, cmap='RdYlGn_r', aspect='auto',
                  vmin=np.nanpercentile(data, 5),
                  vmax=np.nanpercentile(data, 95))
ax4.set_xticks(range(len(heat_df.columns)))
ax4.set_xticklabels(heat_df.columns, color='#aaa', fontsize=8, rotation=30, ha='right')
ax4.set_yticks(range(len(heat_df.index)))
ax4.set_yticklabels(heat_df.index, color='#aaa', fontsize=9)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        if not np.isnan(val):
            ax4.text(j, i, f'{val:.1f}%', ha='center', va='center',
                     color='white' if abs(val) > 5 else '#222', fontsize=7.5,
                     fontweight='bold')
cbar = fig.colorbar(im, ax=ax4, shrink=0.8, pad=0.01)
cbar.ax.yaxis.set_tick_params(color='#aaa')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaa')
ax4.set_title(
    'Mean Annual % Change by Household × Cost Category  (Jun 2020 – Dec 2025)\n'
    'Includes Table 3 financial sub-components (right 3 columns)',
    color='white', fontsize=11, fontweight='bold'
)

# ── Panel 5: RBA cash rate + Employee mortgage interest (heterogeneous overlay)
# Left axis  → RBA Cash Rate Target % (RBA source)
# Right axis → Employee mortgage interest annual % change (ABS SLCI Table 3)
# This panel directly demonstrates why the Employee LCI diverged during 2022-2025.

color_rba      = '#FFCA28'   # amber  — RBA cash rate
color_mortgage = '#FF5722'   # red    — mortgage interest YoY

ax5_r = ax5.twinx()

# Plot RBA cash rate (step — rate holds between decisions)
ax5.step(rba_aligned.index, rba_aligned.values,
         where='post', color=color_rba, linewidth=2.5, label='RBA Cash Rate Target %')
ax5.fill_between(rba_aligned.index, rba_aligned.values,
                 step='post', alpha=0.15, color=color_rba)

# Plot Employee mortgage interest YoY on right axis
ax5_r.plot(mortgage_employee.index, mortgage_employee.values,
           color=color_mortgage, linewidth=2, linestyle='-',
           label='Employee — Mortgage Interest YoY %')
ax5_r.axhline(0, color='white', linewidth=0.5, linestyle=':')

# Axis formatting
ax5.set_facecolor('#1A1D27')
ax5.set_ylabel('RBA Cash Rate Target (%)', color=color_rba, fontsize=9)
ax5.tick_params(axis='y', colors=color_rba)
ax5.yaxis.set_major_formatter(mtick.PercentFormatter())
ax5.set_ylim(-0.5, 18)

ax5_r.set_ylabel('Mortgage Interest Annual % Change  (Employee hh)', color=color_mortgage, fontsize=9)
ax5_r.tick_params(axis='y', colors=color_mortgage)
ax5_r.yaxis.set_major_formatter(mtick.PercentFormatter())

ax5.tick_params(axis='x', colors='#aaa')
for spine in ax5.spines.values():
    spine.set_edgecolor('#333')
ax5.grid(True, alpha=0.2, color='#555')

shade_events(ax5)

ax5.set_title(
    'Panel 5 — Heterogeneous Data Integration: RBA Cash Rate vs Employee Mortgage Cost\n'
    'RBA Source (amber, left axis)  ×  ABS SLCI Table 3 (red, right axis)',
    color='white', fontsize=11, fontweight='bold'
)

# Combined legend for both axes
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_r.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2,
           fontsize=8, loc='upper left', framealpha=0.2,
           facecolor='#222', labelcolor='white')

# Annotation: lag between RBA decision and mortgage impact
ax5.annotate(
    'RBA begins hiking\nFeb 2022 → mortgage\ncosts surge +92% YoY',
    xy=(pd.Timestamp('2023-06-30'), 3.85),
    xytext=(pd.Timestamp('2018-01-01'), 10),
    color=color_rba, fontsize=8,
    arrowprops=dict(arrowstyle='->', color=color_rba, lw=1.2)
)
ax5.annotate(
    'RBA cuts Feb, May,\nAug 2025 → mortgage\ncosts fall −6.4% YoY',
    xy=(pd.Timestamp('2025-09-30'), 3.60),
    xytext=(pd.Timestamp('2023-06-01'), 1),
    color=color_rba, fontsize=8,
    arrowprops=dict(arrowstyle='->', color=color_rba, lw=1.2)
)

#  Shared legend for shaded regions 
covid_patch = mpatches.Patch(color='grey',    alpha=0.4, label='COVID-19 (Mar–Dec 2020)')
rate_patch  = mpatches.Patch(color='red',     alpha=0.4, label='RBA rate rise cycle (Mar 2022–Jun 2023)')
hike_patch  = mpatches.Patch(color='#FF5722', alpha=0.6, label='RBA hike quarter (dashed line, panels 1–3)')
cut_patch   = mpatches.Patch(color='#42A5F5', alpha=0.6, label='RBA cut quarter (dashed line, panels 1–3)')
fig.legend(handles=[covid_patch, rate_patch, hike_patch, cut_patch],
           loc='lower center', ncol=4, framealpha=0.2,
           facecolor='#222', labelcolor='white',
           fontsize=8, bbox_to_anchor=(0.5, 0.01))

plt.savefig('Q1_living_costs_over_time.png', dpi=150, bbox_inches='tight',
            facecolor='#0F1117')
print("Saved → Q1_living_costs_over_time.png")