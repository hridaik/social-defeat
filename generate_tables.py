"""Generate Tables S1–S4 as CSV and PDF."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# ── Metric / parameter name maps ──────────────────────────────────────────────
metric_names = {
    't_shelter':       'Time in Shelter',
    't_investigating': 'Time Investigating',
    'n_sh_co':         'Shelter→Corridor',
    'n_co_sh':         'Corridor→Shelter',
    'n_co_ch':         'Corridor→Chamber',
    'n_ch_co':         'Chamber→Corridor',
    'heatmap_entropy': 'Spatial Entropy',
    'laziness':        'Immobility',
}

param_names = {
    'id_threshold':       'Identity Threshold',
    'k_shelter':          'Shelter Preference',
    'k_threat':           'Threat Aversion',
    'delta_stay':         'Motor Inertia',
    'sensory_prec_slope': 'Sensory Precision',
}

# Desired row order for S1
PRIMARY_METRICS = ['t_shelter', 't_investigating']
OTHER_METRICS   = ['n_sh_co', 'n_co_sh', 'n_co_ch', 'n_ch_co', 'heatmap_entropy', 'laziness']
METRIC_ORDER    = PRIMARY_METRICS + OTHER_METRICS


# ── helpers ───────────────────────────────────────────────────────────────────
def cohen_d(a, b):
    """Cohen's d with pooled SD (n–1 denominator)."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1)**2 + (n2 - 1) * b.std(ddof=1)**2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled


def save_table_pdf(df_display, filename, footer, title, col_widths=None):
    """Render a DataFrame as a clean PDF table."""
    n_rows, n_cols = df_display.shape
    row_h = 0.45
    header_h = 0.55
    footer_h = 0.60
    fig_h = header_h + n_rows * row_h + footer_h + 0.5
    fig_w = sum(col_widths) if col_widths else n_cols * 2.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    # table
    tbl = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, footer_h / fig_h, 1, 1 - footer_h / fig_h],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # style header
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(header_h / fig_h)

    # alternating row shading; highlight asterisked cells
    for r in range(1, n_rows + 1):
        for c in range(n_cols):
            cell = tbl[r, c]
            bg = '#f0f4f8' if r % 2 == 0 else 'white'
            cell.set_facecolor(bg)
            cell.set_height(row_h / fig_h)
            txt = str(df_display.iloc[r - 1, c])
            if '*' in txt:
                cell.set_text_props(color='#c0392b', fontweight='bold')

    # col widths
    if col_widths:
        total = sum(col_widths)
        for c, w in enumerate(col_widths):
            for r in range(n_rows + 1):
                tbl[r, c].set_width(w / total)

    # title
    fig.text(0.5, 0.98, title, ha='center', va='top', fontsize=11, fontweight='bold')

    # footer
    fig.text(0.02, 0.02, footer, ha='left', va='bottom', fontsize=7.5,
             style='italic', wrap=True,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaf0fb', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE S1 — Temporal split prediction error summary
# ══════════════════════════════════════════════════════════════════════════════
print("Building Table S1…")

mae  = pd.read_csv('step5_mae.csv')
pval = pd.read_csv('step4_vs_zero.csv')

rows = []
for metric in METRIC_ORDER:
    pre_mae  = mae.loc[(mae['phase']=='def1') & (mae['metric']==metric), 'model_median_abs_err'].values[0]
    post_mae = mae.loc[(mae['phase']=='def3') & (mae['metric']==metric), 'model_median_abs_err'].values[0]
    pre_p    = pval.loc[(pval['phase']=='def1') & (pval['metric']==metric), 'p_corrected'].values[0]
    post_p   = pval.loc[(pval['phase']=='def3') & (pval['metric']==metric), 'p_corrected'].values[0]

    pre_p_str  = f"{pre_p:.3f}"  + ('*' if pre_p  < 0.05 else '')
    post_p_str = f"{post_p:.3f}" + ('*' if post_p < 0.05 else '')

    rows.append({
        'Metric':                        metric_names[metric],
        'Pre-defeat Median |Error| (σ)': f"{pre_mae:.2f}",
        'Post-defeat Median |Error| (σ)':f"{post_mae:.2f}",
        'Pre-defeat FDR p':              pre_p_str,
        'Post-defeat FDR p':             post_p_str,
    })

s1 = pd.DataFrame(rows)
s1.to_csv('table_S1_prediction_errors.csv', index=False)
print("  Saved table_S1_prediction_errors.csv")

footer_s1 = (
    "Errors expressed in units of population standard deviation (σ). "
    "P-values from one-sample Wilcoxon signed-rank tests against zero, "
    "Benjamini-Hochberg corrected across 8 metrics within each session. "
    "* p < 0.05 after FDR correction."
)
save_table_pdf(
    s1, 'table_S1_prediction_errors.pdf', footer_s1,
    'Table S1. Temporal split prediction error summary',
    col_widths=[2.2, 2.2, 2.2, 1.8, 1.8],
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE S2 — Fit quality for independent dataset
# ══════════════════════════════════════════════════════════════════════════════
print("Building Table S2…")

lib = pd.read_csv('library_matches_frontiers.csv')
# mouse col is numeric 1-6; remap to n1-n6
lib['mouse_id'] = 'n' + lib['mouse'].astype(str)
lib['Phase']    = lib['phase'].map({'def1': 'Pre-defeat', 'def3': 'Post-defeat'})
lib['Group']    = lib['group']  # already Control/Defeat

# sort: defeat first, then control; within group by mouse id; pre before post
group_order = {'Defeat': 0, 'Control': 1}
lib['_go'] = lib['Group'].map(group_order)
lib = lib.sort_values(['_go', 'mouse', 'phase']).reset_index(drop=True)

data_rows = []
for _, row in lib.iterrows():
    data_rows.append({
        'Mouse ID': row['mouse_id'],
        'Group':    row['Group'],
        'Phase':    row['Phase'],
        'Fit Error (σ)': f"{row['sigma']:.2f}",
    })

# summary rows
all_sig  = lib['sigma']
def_sig  = lib.loc[lib['Group']=='Defeat',  'sigma']
ctrl_sig = lib.loc[lib['Group']=='Control', 'sigma']
pre_sig  = lib.loc[lib['phase']=='def1',    'sigma']
post_sig = lib.loc[lib['phase']=='def3',    'sigma']

for label, vals in [
    ('Overall median',    all_sig),
    ('Control median',    ctrl_sig),
    ('Defeat median',     def_sig),
    ('Pre-defeat median', pre_sig),
    ('Post-defeat median',post_sig),
]:
    data_rows.append({
        'Mouse ID':     label,
        'Group':        '',
        'Phase':        '',
        'Fit Error (σ)': f"{vals.median():.2f}",
    })

s2 = pd.DataFrame(data_rows)
s2.to_csv('table_S2_frontiers_fit.csv', index=False)
print("  Saved table_S2_frontiers_fit.csv")

footer_s2 = (
    "Fit error expressed in σ units (σ = √(loss / Σweights)). "
    "Mice 1–3 (n1, n2, n3) = Defeat; Mice 4–6 (n4, n5, n6) = Control. "
    "Original cohort benchmark: 0.56σ."
)
save_table_pdf(
    s2, 'table_S2_frontiers_fit.pdf', footer_s2,
    'Table S2. Fit quality for independent dataset',
    col_widths=[1.8, 1.5, 1.8, 1.8],
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE S3 — Post-defeat parameter effect sizes (susceptible vs resilient)
# ══════════════════════════════════════════════════════════════════════════════
print("Building Table S3…")

lib_orig = pd.read_csv('library_matches.csv')
def3_params = lib_orig[lib_orig['phase'] == 'def3'].copy()
def3_params = def3_params.rename(columns={
    'param_id_threshold':       'id_threshold',
    'param_k_shelter':          'k_shelter',
    'param_k_threat':           'k_threat',
    'param_delta_stay':         'delta_stay',
    'param_sensory_prec_slope': 'sensory_prec_slope',
})

SUSC_IDS  = [14, 24, 26]
RESIL_IDS = [16, 17, 23]

susc_df  = def3_params[def3_params['mouse_id'].isin(SUSC_IDS)].set_index('mouse_id')
resil_df = def3_params[def3_params['mouse_id'].isin(RESIL_IDS)].set_index('mouse_id')

# Apply NaN corrections
# k_threat NaN → 0.85 (Resilient_def source_db, applies to m17 def3)
# sensory_prec_slope NaN → 0.63 (applies to m24, m26 def3)
for mid in RESIL_IDS:
    if pd.isna(resil_df.loc[mid, 'k_threat']):
        print(f"  Correcting m{mid} k_threat NaN → 0.85")
        resil_df.loc[mid, 'k_threat'] = 0.85
    if pd.isna(resil_df.loc[mid, 'sensory_prec_slope']):
        print(f"  Correcting m{mid} sensory_prec_slope NaN → 0.63")
        resil_df.loc[mid, 'sensory_prec_slope'] = 0.63

for mid in SUSC_IDS:
    if pd.isna(susc_df.loc[mid, 'k_threat']):
        print(f"  Correcting m{mid} k_threat NaN → 0.85")
        susc_df.loc[mid, 'k_threat'] = 0.85
    if pd.isna(susc_df.loc[mid, 'sensory_prec_slope']):
        print(f"  Correcting m{mid} sensory_prec_slope NaN → 0.63")
        susc_df.loc[mid, 'sensory_prec_slope'] = 0.63

params = list(param_names.keys())
rows_s3 = []
for p in params:
    sv = susc_df[p].values.astype(float)
    rv = resil_df[p].values.astype(float)
    s_mean, s_sd = sv.mean(), sv.std(ddof=1)
    r_mean, r_sd = rv.mean(), rv.std(ddof=1)
    d = cohen_d(sv, rv)
    rows_s3.append({
        'Parameter':               param_names[p],
        'Susceptible Mean (SD)':   f"{s_mean:.3f} ({s_sd:.3f})",
        'Resilient Mean (SD)':     f"{r_mean:.3f} ({r_sd:.3f})",
        "Cohen's d":               f"{d:.2f}",
        '_abs_d':                  abs(d),
    })

s3 = pd.DataFrame(rows_s3).sort_values('_abs_d', ascending=False).drop(columns='_abs_d').reset_index(drop=True)
s3.to_csv('table_S3_effect_sizes.csv', index=False)
print("  Saved table_S3_effect_sizes.csv")

footer_s3 = (
    "Cohen's d computed with pooled SD. "
    "Susceptible: n=3 (m14, m24, m26); Resilient: n=3 (m16, m17, m23). "
    "NaN corrections applied: k_threat → 0.85 (m17 def3), sensory_prec_slope → 0.63 (m24, m26 def3)."
)
save_table_pdf(
    s3, 'table_S3_effect_sizes.pdf', footer_s3,
    'Table S3. Post-defeat parameter effect sizes (susceptible vs resilient)',
    col_widths=[2.3, 2.5, 2.5, 1.5],
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE S4 — Prospective classification of new cohort defeat mice
# ══════════════════════════════════════════════════════════════════════════════
print("Building Table S4…")

lib_fr = pd.read_csv('library_matches_frontiers.csv')

# New defeat mice are mouse 1,2,3 in Defeat group, post-defeat (def3)
new_def_post = lib_fr[(lib_fr['group'] == 'Defeat') & (lib_fr['phase'] == 'def3')].copy()
new_def_post = new_def_post.sort_values('mouse').reset_index(drop=True)

THRESHOLD = 0.883

# Empirical scores (Δt_shelter − Δt_investigating) provided by user
emp_scores = {1: +0.463, 2: +0.380, 3: -0.002}
emp_labels  = {1: 'Susceptible', 2: 'Susceptible', 3: 'Resilient'}

rows_s4 = []
for _, row in new_def_post.iterrows():
    mid   = int(row['mouse'])
    kt    = row['k_threat']
    # NaN correction
    if pd.isna(kt):
        # determine which db – instruction: 0.85 if 'calibration', 0.63 if 'Resilient'
        # No explicit source_db column; apply 0.85 as default NaN k_threat correction
        kt = 0.85
        note = 'corrected'
    else:
        note = ''

    pred_label = 'Susceptible' if kt > THRESHOLD else 'Resilient'
    emp_score  = emp_scores[mid]
    emp_label  = emp_labels[mid]
    correct    = 'Yes' if pred_label == emp_label else 'No'

    kt_str = f"{kt:.3f}" + (f" ({note})" if note else '')
    rows_s4.append({
        'Mouse':                         f"n{mid}",
        'Post-defeat k_threat':          kt_str,
        'Predicted Label':               pred_label,
        'Empirical Score (Δ)':           f"{emp_score:+.3f}",
        'Empirical Label':               emp_label,
        'Correct':                       correct,
    })

s4 = pd.DataFrame(rows_s4)
s4.to_csv('table_S4_classification.csv', index=False)
print("  Saved table_S4_classification.csv")

footer_s4 = (
    "k_threat threshold = 0.883 (derived from original cohort). "
    "Empirical susceptibility score = Δt_shelter − Δt_investigating (post-defeat minus pre-defeat). "
    "Top 2 empirical scores → Susceptible; bottom 1 → Resilient. "
    "Library coverage note: n2's k_threat confirmed by direct optimisation (> 0.883)."
)
save_table_pdf(
    s4, 'table_S4_classification.pdf', footer_s4,
    'Table S4. Prospective classification of new cohort defeat mice',
    col_widths=[1.2, 2.0, 2.0, 1.8, 2.0, 1.2],
)

print("\nAll tables generated successfully.")
