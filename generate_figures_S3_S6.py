"""Generate Figures S3–S6."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
from matplotlib.backends.backend_pdf import PdfPages

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL    = '#2a9d8f'   # Control
PURPLE  = '#7b2d8b'   # Defeat
ORANGE  = '#e76f51'   # Susceptible
STEEL   = '#457b9d'   # Resilient
GREY    = '#888888'

# ── Name maps ─────────────────────────────────────────────────────────────────
param_names = {
    'k_threat':           'Threat Aversion',
    'delta_stay':         'Motor Inertia',
    'k_shelter':          'Shelter Preference',
    'id_threshold':       'Identity Threshold',
    'sensory_prec_slope': 'Sensory Precision',
}
PARAM_COLS = ['id_threshold', 'k_shelter', 'k_threat', 'delta_stay', 'sensory_prec_slope']

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


def save_fig(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=220, bbox_inches='tight')
        print(f'  Saved {stem}.{ext}')
    plt.close(fig)


def nan_correct(val, source_db):
    """Apply NaN corrections based on source_db name."""
    return val  # passthrough — call-sites handle corrections explicitly


def confidence_ellipse_cov(x, y, ax, n_std=2.448, **kwargs):
    """Draw a confidence ellipse from x,y data using the covariance approach."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h,
                  angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S3 — Fit quality for independent dataset (strip plots)
# ══════════════════════════════════════════════════════════════════════════════
print('Building Figure S3…')

lib_fr = pd.read_csv('library_matches_frontiers.csv')
lib_fr['mouse_id'] = 'n' + lib_fr['mouse'].astype(str)
lib_fr['Phase']    = lib_fr['phase'].map({'def1': 'Pre-defeat', 'def3': 'Post-defeat'})
# mice 1-3 = Defeat, 4-6 = Control
lib_fr['GroupLabel'] = lib_fr['mouse'].apply(lambda m: 'Defeat' if m <= 3 else 'Control')
lib_fr['Color']      = lib_fr['GroupLabel'].map({'Control': TEAL, 'Defeat': PURPLE})

fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

rng = np.random.default_rng(42)
for ax, phase_label in zip(axes, ['Pre-defeat', 'Post-defeat']):
    sub = lib_fr[lib_fr['Phase'] == phase_label].copy()

    # jitter x within group
    x_map = {'Control': 0, 'Defeat': 1}
    for _, row in sub.iterrows():
        xbase = x_map[row['GroupLabel']]
        xjit  = xbase + rng.uniform(-0.12, 0.12)
        ax.scatter(xjit, row['sigma'], color=row['Color'],
                   s=60, zorder=4, alpha=0.9, edgecolors='white', linewidths=0.5)
        ax.text(xjit + 0.04, row['sigma'], row['mouse_id'],
                fontsize=7.5, va='center', color='#333333')

    # reference lines
    ax.axhline(0.56, color='#555555', linestyle='--', linewidth=1.2, zorder=2)
    ax.axhline(1.00, color=GREY,      linestyle='--', linewidth=1.0, zorder=2)
    ax.text(1.55, 0.56 + 0.02, 'Original cohort median (0.56σ)',
            fontsize=7.5, color='#555555', va='bottom', ha='right')
    ax.text(1.55, 1.00 + 0.02, '1σ threshold',
            fontsize=7.5, color=GREY, va='bottom', ha='right')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Defeat'])
    ax.set_xlim(-0.5, 1.8)
    ax.set_title(phase_label, fontsize=12, fontweight='bold', pad=8)
    ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)

axes[0].set_ylabel('Fit Error (σ)', fontsize=11)

# legend
ctrl_patch = mpatches.Patch(color=TEAL,   label='Control')
def_patch  = mpatches.Patch(color=PURPLE, label='Defeat')
fig.legend(handles=[ctrl_patch, def_patch], loc='upper right',
           fontsize=9, frameon=False, bbox_to_anchor=(0.98, 0.98))

fig.suptitle('Figure S3. Fit quality for independent dataset',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'figure_S3_frontiers_fit')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S4 — Post-defeat parameter effect sizes: susceptible vs resilient
# ══════════════════════════════════════════════════════════════════════════════
print('Building Figure S4…')

mining = pd.read_csv('mining_results_adj.csv')

# Unique def3 assignments (iloc row indices)
UNIQUE_ROWS = {
    'm14': 147, 'm24': 179, 'm26': 128,   # Susceptible
    'm16': 499, 'm17': 507, 'm23': 427,   # Resilient
}

def get_params(row_idx):
    r = mining.iloc[row_idx]
    return {
        'id_threshold':       r['param_id_threshold'],
        'k_shelter':          r['param_k_shelter'],
        'k_threat':           r['param_k_threat'],
        'delta_stay':         r['param_delta_stay'],
        'sensory_prec_slope': r['param_sensory_prec_slope'],
        'source_db':          r['source_db'],
    }

def apply_corrections(d):
    db = d['source_db']
    if np.isnan(d['k_threat']):
        d['k_threat'] = 0.85
    if np.isnan(d['sensory_prec_slope']) and ('Res' in str(db) or 'Resilient' in str(db)):
        d['sensory_prec_slope'] = 0.63
    return d

SUSC_IDS  = ['m14', 'm24', 'm26']
RESIL_IDS = ['m16', 'm17', 'm23']

susc_data  = {mid: apply_corrections(get_params(UNIQUE_ROWS[mid])) for mid in SUSC_IDS}
resil_data = {mid: apply_corrections(get_params(UNIQUE_ROWS[mid])) for mid in RESIL_IDS}

# Cohen's d with pooled SD
def cohen_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1-1)*a.std(ddof=1)**2 + (n2-1)*b.std(ddof=1)**2) / (n1+n2-2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled

# Compute stats per parameter
param_stats = {}
for p in PARAM_COLS:
    sv = np.array([susc_data[m][p]  for m in SUSC_IDS],  dtype=float)
    rv = np.array([resil_data[m][p] for m in RESIL_IDS], dtype=float)
    d  = cohen_d(sv, rv)
    param_stats[p] = {'susc': sv, 'resil': rv, 'd': d, 'absd': abs(d) if not np.isnan(d) else 0}

# Fixed order specified in instructions (|Cohen's d| descending, pre-determined)
ordered_params = ['k_threat', 'delta_stay', 'k_shelter', 'id_threshold', 'sensory_prec_slope']

fig, axes = plt.subplots(1, len(ordered_params), figsize=(3.2 * len(ordered_params), 5))

rng2 = np.random.default_rng(7)
for ax, p in zip(axes, ordered_params):
    sv   = param_stats[p]['susc']
    rv   = param_stats[p]['resil']
    d    = param_stats[p]['d']

    # horizontal mean bar (width = 0.4)
    ax.hlines(sv.mean(), -0.25, 0.25, color=ORANGE, linewidth=2.5, zorder=3)
    ax.hlines(rv.mean(), 0.75,  1.25, color=STEEL,  linewidth=2.5, zorder=3)

    # individual dots
    for v in sv:
        xj = 0 + rng2.uniform(-0.10, 0.10)
        ax.scatter(xj, v, color=ORANGE, s=55, zorder=4, alpha=0.9,
                   edgecolors='white', linewidths=0.5)
    for v in rv:
        xj = 1 + rng2.uniform(-0.10, 0.10)
        ax.scatter(xj, v, color=STEEL, s=55, zorder=4, alpha=0.9,
                   edgecolors='white', linewidths=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Susc.', 'Resil.'], fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(param_names[p], fontsize=11, fontweight='bold', pad=4)

    d_str = f"d = {d:+.2f}" if not np.isnan(d) else "d = n/a"
    ax.text(0.5, -0.11, d_str, transform=ax.transAxes,
            ha='center', va='top', fontsize=9, color='#444444')

    ax.yaxis.grid(True, color='#e8e8e8', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)

axes[0].set_ylabel('Parameter Value', fontsize=11)

# legend
s_patch = mpatches.Patch(color=ORANGE, label='Susceptible (n=3)')
r_patch = mpatches.Patch(color=STEEL,  label='Resilient (n=3)')
fig.legend(handles=[s_patch, r_patch], loc='upper right',
           fontsize=9, frameon=False, bbox_to_anchor=(0.99, 0.99))

fig.suptitle('Figure S4. Post-defeat parameter effect sizes', fontsize=12, fontweight='bold', y=1.02)

footnote = ("Sensory Precision values reflect a fixed parameter rather than "
            "individually fitted values for susceptible mice.")
fig.text(0.5, -0.04, footnote, ha='center', va='top', fontsize=8,
         style='italic', color='#555555')

fig.tight_layout(rect=[0, 0.01, 1, 1])
save_fig(fig, 'figure_S4_effect_sizes')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S5 — PCA of post-defeat parameters: all mice
# ══════════════════════════════════════════════════════════════════════════════
print('Building Figure S5…')

# ── Assemble full parameter table ─────────────────────────────────────────────
lib_orig = pd.read_csv('library_matches.csv')
lib_fr2  = pd.read_csv('library_matches_frontiers.csv')

records = []

# Original cohort defeat mice — unique def3 assignments
for label, mids, row_indices in [
    ('Susceptible', ['m14','m24','m26'], [147,179,128]),
    ('Resilient',   ['m16','m17','m23'], [499,507,427]),
]:
    for mid, ridx in zip(mids, row_indices):
        d = apply_corrections(get_params(ridx))
        records.append({
            'mouse_id': mid, 'cohort': 'original', 'group': label,
            **{p: d[p] for p in PARAM_COLS},
        })

# Original cohort control mice — library_matches def3
ctrl_orig_ids = [13, 15, 18, 20, 21, 22, 25]
for mid in ctrl_orig_ids:
    row = lib_orig[(lib_orig['mouse_id'] == mid) & (lib_orig['phase'] == 'def3')].iloc[0]
    records.append({
        'mouse_id': f'm{mid}', 'cohort': 'original', 'group': 'Control',
        'id_threshold':       row['param_id_threshold'],
        'k_shelter':          row['param_k_shelter'],
        'k_threat':           row['param_k_threat'],
        'delta_stay':         row['param_delta_stay'],
        'sensory_prec_slope': row['param_sensory_prec_slope'],
    })

# New cohort — library_matches_frontiers def3
new_labels = {
    1: 'Susceptible', 2: 'Susceptible', 3: 'Resilient',
    4: 'Control',     5: 'Control',     6: 'Control',
}
for _, row in lib_fr2[lib_fr2['phase'] == 'def3'].iterrows():
    mid_num = int(row['mouse'])
    mid_str = f'n{mid_num}'
    sps     = row['sensory_prec_slope']
    best_r  = int(row['best_row']) if 'best_row' in row.index else None
    # NaN correction: look up source_db
    if np.isnan(sps) and best_r is not None:
        src_db = mining.iloc[best_r]['source_db']
        if 'Res' in str(src_db) or 'Resilient' in str(src_db):
            sps = 0.63
    k_thr = row['k_threat']
    if np.isnan(k_thr):
        k_thr = 0.85
    records.append({
        'mouse_id': mid_str, 'cohort': 'new', 'group': new_labels[mid_num],
        'id_threshold':       row['id_threshold'],
        'k_shelter':          row['k_shelter'],
        'k_threat':           k_thr,
        'delta_stay':         row['delta_stay'],
        'sensory_prec_slope': sps,
    })

pca_df = pd.DataFrame(records).reset_index(drop=True)
print(f'  PCA table: {len(pca_df)} mice')
print(pca_df[['mouse_id','cohort','group'] + PARAM_COLS].to_string())

# ── Run PCA ───────────────────────────────────────────────────────────────────
X = pca_df[PARAM_COLS].values.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
pca_df['PC1'] = pcs[:, 0]
pca_df['PC2'] = pcs[:, 1]
var1, var2 = pca.explained_variance_ratio_ * 100

# ── Plot ──────────────────────────────────────────────────────────────────────
group_color  = {'Control': TEAL, 'Susceptible': ORANGE, 'Resilient': STEEL}
cohort_marker = {'original': 'o', 'new': '^'}
cohort_label  = {'original': 'Original cohort', 'new': 'New cohort'}

fig, ax = plt.subplots(figsize=(8, 6.5))

# 95% confidence ellipses (n_std = sqrt(chi2(2, 0.95)) ≈ 2.448)
n_std_95 = np.sqrt(chi2.ppf(0.95, df=2))
for grp, color in group_color.items():
    sub = pca_df[pca_df['group'] == grp]
    if len(sub) >= 3:
        confidence_ellipse_cov(sub['PC1'].values, sub['PC2'].values, ax,
                               n_std=n_std_95,
                               facecolor=color, alpha=0.10,
                               edgecolor=color, linewidth=1.2, linestyle='--',
                               zorder=1)

# Scatter points
legend_handles = {}
for _, row in pca_df.iterrows():
    color  = group_color[row['group']]
    marker = cohort_marker[row['cohort']]
    sc = ax.scatter(row['PC1'], row['PC2'], color=color, marker=marker,
                    s=80, zorder=4, alpha=0.92,
                    edgecolors='white', linewidths=0.6)
    # label
    ax.annotate(row['mouse_id'], (row['PC1'], row['PC2']),
                textcoords='offset points', xytext=(6, 3),
                fontsize=7.5, color='#333333')

    # legend keys: group × cohort
    key = (row['group'], row['cohort'])
    if key not in legend_handles:
        legend_handles[key] = mpatches.Patch(
            facecolor=color, label=f"{row['group']} — {cohort_label[row['cohort']]}",
            edgecolor='grey', linewidth=0.5)

# Build ordered legend
leg_order = [('Control','original'),('Control','new'),
             ('Susceptible','original'),('Susceptible','new'),
             ('Resilient','original'),('Resilient','new')]

# Use Line2D for marker shape in legend
from matplotlib.lines import Line2D
leg_items = []
for grp, color in group_color.items():
    for coh, mk in cohort_marker.items():
        if (grp, coh) in legend_handles:
            leg_items.append(Line2D([0],[0], marker=mk, color='w',
                                    markerfacecolor=color, markersize=9,
                                    label=f"{grp} — {cohort_label[coh]}"))

ax.legend(handles=leg_items, loc='upper right', fontsize=8.5,
          frameon=True, framealpha=0.9, edgecolor='#cccccc', ncol=2)

ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
ax.set_title('Figure S5. PCA of post-defeat parameters — all mice',
             fontsize=12, fontweight='bold', pad=10)
ax.axhline(0, color='#dddddd', linewidth=0.7, zorder=0)
ax.axvline(0, color='#dddddd', linewidth=0.7, zorder=0)
ax.yaxis.grid(True, color='#ebebeb', linewidth=0.5, zorder=0)
ax.xaxis.grid(True, color='#ebebeb', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
save_fig(fig, 'figure_S5_pca_def3')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S6 — Trauma vectors: susceptible, resilient, control
#   Left panel : 3D arrow plot  (azim=50, elev=20)
#   Right panel: 2D projection  Δk_shelter (x) vs Δk_threat (y)
# ══════════════════════════════════════════════════════════════════════════════
print('Building Figure S6…')

AZIM, ELEV = 50, 20    # final viewing angle for 3D panel

tv   = pd.read_csv('trauma_vectors_corrected.csv')
orig = tv[tv['cohort'] == 'original'].copy()

susc_ids  = [14, 24, 26]
resil_ids = [16, 17, 23]
ctrl_ids2 = [13, 15, 18, 20, 21, 22, 25]

groups_tv = {
    'Susceptible': (susc_ids,  ORANGE),
    'Resilient':   (resil_ids, STEEL),
    'Control':     (ctrl_ids2, TEAL),
}

DELTA_COLS = ['delta_id_threshold', 'delta_k_shelter', 'delta_k_threat']

fig = plt.figure(figsize=(15, 6.5))
fig.patch.set_facecolor('white')

ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

handles = []
all_vecs = []

for grp_name, (mids, color) in groups_tv.items():
    sub = orig[orig['mouse_id'].isin(mids)][DELTA_COLS].values
    all_vecs.append(sub)
    mean_vec = sub.mean(axis=0)
    mag      = np.linalg.norm(mean_vec)

    # ── 3D panel ──────────────────────────────────────────────────────────────
    # individual faded arrows
    for vec in sub:
        ax3d.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                    color=color, alpha=0.18, linewidth=0.9,
                    arrow_length_ratio=0.10)
    # mean arrow
    ax3d.quiver(0, 0, 0, mean_vec[0], mean_vec[1], mean_vec[2],
                color=color, alpha=0.95, linewidth=2.5,
                arrow_length_ratio=0.10)
    # label at tip
    tip = mean_vec * 1.15
    ax3d.text(tip[0], tip[1], tip[2],
              f'{grp_name}\n‖v‖={mag:.2f}',
              fontsize=8.5, color=color, fontweight='bold', ha='center')

    # ── 2D panel (Δk_shelter vs Δk_threat) ───────────────────────────────────
    sh_col = DELTA_COLS.index('delta_k_shelter')   # 1
    th_col = DELTA_COLS.index('delta_k_threat')    # 2

    # individual faded arrows
    for vec in sub:
        ax2d.annotate('', xy=(vec[sh_col], vec[th_col]), xytext=(0, 0),
                      arrowprops=dict(arrowstyle='->', color=color,
                                      lw=1.0, alpha=0.25,
                                      shrinkA=0, shrinkB=0))

    # mean arrow (bold)
    ax2d.annotate('', xy=(mean_vec[sh_col], mean_vec[th_col]), xytext=(0, 0),
                  arrowprops=dict(arrowstyle='->', color=color,
                                  lw=2.2, alpha=0.95,
                                  shrinkA=0, shrinkB=3))
    # label near tip
    offset = np.array([mean_vec[sh_col], mean_vec[th_col]])
    if np.linalg.norm(offset) > 0:
        unit = offset / np.linalg.norm(offset)
    else:
        unit = np.array([0.1, 0.1])
    label_pos = offset + unit * 0.25
    ax2d.text(label_pos[0], label_pos[1], grp_name,
              fontsize=9, color=color, fontweight='bold', ha='center', va='center')

    handles.append(mpatches.Patch(color=color, label=f'{grp_name} (n={len(mids)})'))

# ── 3D axis limits & style ────────────────────────────────────────────────────
all_vecs_np = np.vstack(all_vecs)
pad = 0.15
for col_i, setlim in enumerate([ax3d.set_xlim, ax3d.set_ylim, ax3d.set_zlim]):
    vals = all_vecs_np[:, col_i]
    lo, hi = vals.min(), vals.max()
    span = max(hi - lo, 0.1)
    setlim(lo - pad * span, hi + pad * span)

ax3d.set_xlabel('Δ Identity Threshold', fontsize=9, labelpad=6)
ax3d.set_ylabel('Δ Shelter Preference', fontsize=9, labelpad=6)
ax3d.set_zlabel('Δ Threat Aversion',    fontsize=9, labelpad=6)
ax3d.tick_params(labelsize=7)
ax3d.scatter([0],[0],[0], color='black', s=25, zorder=5)
ax3d.set_title('3D view', fontsize=10, fontweight='bold', pad=8)
ax3d.legend(handles=handles, loc='upper left', fontsize=8,
            frameon=True, framealpha=0.85, edgecolor='#cccccc')
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#dddddd')
ax3d.yaxis.pane.set_edgecolor('#dddddd')
ax3d.zaxis.pane.set_edgecolor('#dddddd')
ax3d.grid(True, color='#eeeeee', linewidth=0.5)
ax3d.view_init(elev=ELEV, azim=AZIM)

# ── 2D axis style ──────────────────────────────────────────────────────────────
sh_vals = all_vecs_np[:, 1]
th_vals = all_vecs_np[:, 2]
sh_pad  = (sh_vals.max() - sh_vals.min()) * 0.18
th_pad  = (th_vals.max() - th_vals.min()) * 0.25

ax2d.set_xlim(sh_vals.min() - sh_pad, sh_vals.max() + sh_pad)
ax2d.set_ylim(th_vals.min() - th_pad, th_vals.max() + th_pad)
ax2d.axhline(0, color='#cccccc', linewidth=0.8, zorder=0)
ax2d.axvline(0, color='#cccccc', linewidth=0.8, zorder=0)
ax2d.scatter([0],[0], color='black', s=25, zorder=5)
ax2d.set_xlabel('Δ Shelter Preference', fontsize=11)
ax2d.set_ylabel('Δ Threat Aversion',    fontsize=11)
ax2d.set_title('2D projection', fontsize=10, fontweight='bold', pad=8)
ax2d.yaxis.grid(True, color='#ebebeb', linewidth=0.5, zorder=0)
ax2d.xaxis.grid(True, color='#ebebeb', linewidth=0.5, zorder=0)
ax2d.set_axisbelow(True)
ax2d.tick_params(labelsize=9)

fig.suptitle('Defeat-induced parameter shifts by phenotype group',
             fontsize=12, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, 'figure_S6_trauma_vectors')

print(f'  3D viewing angle: azim={AZIM}, elev={ELEV}')
print('\nAll figures (S3–S6) generated successfully.')
