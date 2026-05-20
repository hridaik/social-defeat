"""Generate Figure S1 and Figure S2."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL   = '#2a9d8f'   # pre-defeat
PURPLE = '#7b2d8b'   # post-defeat
GREY   = '#888888'

# ── Name maps ─────────────────────────────────────────────────────────────────
metric_names = {
    't_shelter':       'Time in Shelter',
    't_investigating': 'Time Investigating',
    'heatmap_entropy': 'Spatial Entropy',
    'laziness':        'Immobility',
    'n_sh_co':         'Shelter→Corridor',
    'n_co_sh':         'Corridor→Shelter',
    'n_co_ch':         'Corridor→Chamber',
    'n_ch_co':         'Chamber→Corridor',
}

# figure S1 order (top → bottom on y-axis)
S1_ORDER = ['t_shelter', 't_investigating', 'heatmap_entropy', 'laziness',
            'n_sh_co', 'n_co_sh', 'n_co_ch', 'n_ch_co']

# figure S2 order (left → right across panels)
S2_ORDER = S1_ORDER

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   11,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S1 — Temporal split prediction error by metric
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure S1…")

mae  = pd.read_csv('step5_mae.csv')
pval = pd.read_csv('step4_vs_zero.csv')

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, phase, color, panel_title in [
    (axes[0], 'def1', TEAL,   'Pre-defeat'),
    (axes[1], 'def3', PURPLE, 'Post-defeat'),
]:
    sub_mae  = mae[mae['phase'] == phase].set_index('metric')
    sub_pval = pval[pval['phase'] == phase].set_index('metric')

    # y positions: top metric at top (invert axis later)
    labels = [metric_names[m] for m in S1_ORDER]
    values = [sub_mae.loc[m, 'model_median_abs_err'] for m in S1_ORDER]
    sigs   = [sub_pval.loc[m, 'p_corrected'] < 0.05  for m in S1_ORDER]

    # Reverse order so index 0 is at bottom, last index at top — giving
    # the specified top-to-bottom reading order without needing invert_yaxis.
    plot_order = list(reversed(S1_ORDER))
    labels  = [metric_names[m] for m in plot_order]
    values  = [sub_mae.loc[m, 'model_median_abs_err'] for m in plot_order]
    sigs    = [sub_pval.loc[m, 'p_corrected'] < 0.05  for m in plot_order]

    y = np.arange(len(plot_order))
    bars = ax.barh(y, values, color=color, alpha=0.85, height=0.6, zorder=3)

    # asterisks
    for i, (val, sig) in enumerate(zip(values, sigs)):
        if sig:
            ax.text(val + 0.02, i, '*', va='center', ha='left',
                    fontsize=13, color='#c0392b', fontweight='bold')

    # reference line
    ax.axvline(0.56, color=GREY, linestyle='--', linewidth=1.2, zorder=2)
    # place label above the top bar (y = n_metrics - 0.5)
    ax.text(0.56 + 0.02, len(S1_ORDER) - 0.5, 'Original fit (0.56σ)',
            fontsize=8, color=GREY, va='center', ha='left', style='italic')

    # axes decoration
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Median Absolute Prediction Error (σ)', fontsize=11)
    ax.set_title(panel_title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlim(0, max(values) * 1.22)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color='#e0e0e0', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

# shared y-axis label only on left panel
axes[0].set_ylabel('')   # metric names serve as labels

fig.suptitle('Figure S1. Temporal split prediction error by metric',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()

for ext in ('pdf', 'png'):
    fname = f'figure_S1_prediction_errors.{ext}'
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"  Saved {fname}")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S2 — Within-session behavioral change: paired plots
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure S2…")

half = pd.read_csv('half_trajectory_results.csv')
stats = pd.read_csv('half_trajectory_stats.csv')

# exclude m19
half = half[half['mouse'] != 19].copy()

# pivot to wide: one row per (mouse, phase), cols = metric_first, metric_second
first  = half[half['half'] == 'first'].set_index(['mouse', 'phase'])
second = half[half['half'] == 'second'].set_index(['mouse', 'phase'])

# Count-based metrics (add y=0 reference)
COUNT_METRICS = {'n_sh_co', 'n_co_sh', 'n_co_ch', 'n_ch_co'}

# Build significance lookup: {(phase, metric): bool}
sig_lookup = {}
for _, row in stats.iterrows():
    if row['phase'] in ('def1', 'def3'):
        sig_lookup[(row['phase'], row['metric'])] = row['p_value'] < 0.05

phases = ['def1', 'def3']
phase_labels = {'def1': 'Pre-defeat', 'def3': 'Post-defeat'}
phase_colors = {'def1': TEAL, 'def3': PURPLE}

n_metrics = len(S2_ORDER)
n_phases  = len(phases)

PANEL_W = 3.5   # inches per panel
PANEL_H = 4.5   # inches per row

fig, axes = plt.subplots(
    n_phases, n_metrics,
    figsize=(PANEL_W * n_metrics, PANEL_H * n_phases),
    squeeze=False,
)

for row_i, phase in enumerate(phases):
    for col_i, metric in enumerate(S2_ORDER):
        ax = axes[row_i][col_i]

        y_first  = first.xs(phase, level='phase')[metric]
        y_second = second.xs(phase, level='phase')[metric]

        common = y_first.index.intersection(y_second.index)
        y1 = y_first.loc[common].values
        y2 = y_second.loc[common].values

        for v1, v2 in zip(y1, y2):
            color = TEAL if v2 >= v1 else PURPLE
            ax.plot([0, 1], [v1, v2], color=color, alpha=0.6, linewidth=1.4, zorder=2)
            ax.scatter([0], [v1], color=color, s=30, zorder=3, alpha=0.85)
            ax.scatter([1], [v2], color=color, s=30, zorder=3, alpha=0.85)

        if metric in COUNT_METRICS:
            ax.axhline(0, color=GREY, linestyle='--', linewidth=0.9, zorder=1)

        is_sig = sig_lookup.get((phase, metric), False)
        title_str = metric_names[metric] + (' *' if is_sig else '')
        ax.set_title(title_str, fontsize=10, pad=6,
                     color='#c0392b' if is_sig else 'black', fontweight='bold' if is_sig else 'normal')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['First Half', 'Second Half'], fontsize=9)
        ax.set_xlim(-0.3, 1.3)
        ax.tick_params(axis='y', labelsize=9, pad=2)
        ax.tick_params(axis='x', labelsize=9)

        ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', length=0)

        # Row label: bold phase name as y-axis label on leftmost panel only
        if col_i == 0:
            ax.set_ylabel(phase_labels[phase] + '\n', fontsize=12,
                          fontweight='bold', labelpad=8)
        else:
            ax.set_ylabel('')

# legend
teal_patch   = mpatches.Patch(color=TEAL,   label='2nd half ≥ 1st half')
purple_patch = mpatches.Patch(color=PURPLE, label='2nd half < 1st half')
fig.legend(handles=[teal_patch, purple_patch], loc='lower center',
           ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))

fig.suptitle('Within-session behavioral change by metric',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout(rect=[0, 0.04, 1, 1], h_pad=4.5, w_pad=2.0)

for ext in ('pdf', 'png'):
    fname = f'figure_S2_within_session.{ext}'
    fig.savefig(fname, dpi=250, bbox_inches='tight')
    print(f"  Saved {fname}")
plt.close(fig)

print("\nAll figures generated successfully.")
