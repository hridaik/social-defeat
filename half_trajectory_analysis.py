"""
Half-trajectory behavioral metric analysis.
Splits each mouse's 2000-step trajectory at timestep 1000 and computes
all ethological metrics independently on each half.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import wilcoxon, norm

# ---------------------------------------------------------------------------
# Grid / bin machinery (identical to notebook)
# ---------------------------------------------------------------------------

final_mask = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
])

active_cells = np.argwhere(final_mask == 1)
bin_id_map = {tuple(pos): i for i, pos in enumerate(active_cells)}


def get_closest_bin(r, c):
    if final_mask[r, c] == 1:
        return bin_id_map[(r, c)]
    distances = np.sum((active_cells - np.array([r, c])) ** 2, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_r, nearest_c = active_cells[nearest_idx]
    return bin_id_map[(nearest_r, nearest_c)]


def map_position(row, min_x, min_y, max_x, max_y, n_rows=5, n_cols=15, flipped=False):
    x_norm = np.clip((row["Center_x"] - min_x) / (max_x - min_x), 0, 0.999)
    y_norm = np.clip((row["Center_y"] - min_y) / (max_y - min_y), 0, 0.999)
    if flipped:
        x_norm = 0.999 - x_norm
        y_norm = 0.999 - y_norm
    r = int(np.floor(y_norm * n_rows))
    c = int(np.floor(x_norm * n_cols))
    return get_closest_bin(r, c)


def get_trajectory(df, n_steps=2000, flipped=False):
    df_filtered = df.copy()
    df_filtered["step_number"] = np.linspace(0, n_steps, len(df_filtered), endpoint=False).astype(int)
    coarse_data = df_filtered.groupby("step_number").agg(
        Center_x=("Center_x", "mean"),
        Center_y=("Center_y", "mean"),
    ).reset_index()

    min_x, max_x = df_filtered["Center_x"].min(), df_filtered["Center_x"].max()
    min_y, max_y = df_filtered["Center_y"].min(), df_filtered["Center_y"].max()

    coarse_data["location"] = coarse_data.apply(
        map_position, axis=1, args=(min_x, min_y, max_x, max_y), flipped=flipped
    )
    return coarse_data[["step_number", "location"]]


# ---------------------------------------------------------------------------
# Metric functions (identical to notebook)
# ---------------------------------------------------------------------------

def frac_time_spent_in_shelter(df):
    shelter_indices = {6, 7, 21, 22, 36, 37}
    total_t = len(df)
    shelter_t = sum(1 for _, row in df.iterrows() if row["location"] in shelter_indices)
    return shelter_t / total_t


def frac_time_spent_investigating(df):
    investigation_indices = {33, 34, 35, 48, 54, 17, 18, 19, 20, 32, 47, 53}
    total_t = len(df)
    locs = df["location"].tolist()
    in_zone = [loc in investigation_indices for loc in locs]
    inv_t = sum(
        1 for t in range(1, total_t - 1)
        if in_zone[t] and in_zone[t - 1] and in_zone[t + 1]
    )
    return inv_t / total_t


def calculate_zone_transitions(df, active_cells_mapping=active_cells):
    def get_zone(bin_id):
        r, c = active_cells_mapping[bin_id]
        if c >= 9:
            return "Chamber"
        elif c == 0:
            return "Shelter"
        elif 1 <= r <= 3:
            return "Corridor"

    temp_df = df.copy()
    temp_df["zone"] = temp_df["location"].apply(get_zone)
    temp_df["prev_zone"] = temp_df["zone"].shift(1)
    transitions = temp_df[temp_df["zone"] != temp_df["prev_zone"]].dropna(subset=["prev_zone"])
    transitions = transitions.copy()
    transitions["path"] = transitions["prev_zone"] + " -> " + transitions["zone"]
    counts = transitions["path"].value_counts()
    return {
        "Shelter to Corridor": counts.get("Shelter -> Corridor", 0),
        "Corridor to Shelter": counts.get("Corridor -> Shelter", 0),
        "Corridor to Chamber": counts.get("Corridor -> Chamber", 0),
        "Chamber to Corridor": counts.get("Chamber -> Corridor", 0),
    }


def calculate_heatmap_entropy(df, num_active_bins=57):
    from scipy.stats import entropy as scipy_entropy
    counts = df["location"].value_counts()
    prob_dist = np.zeros(num_active_bins)
    for bin_id, count in counts.items():
        if 0 <= bin_id < num_active_bins:
            prob_dist[bin_id] = count
    if np.sum(prob_dist) == 0:
        return 0.0
    prob_dist = prob_dist / np.sum(prob_dist)
    return scipy_entropy(prob_dist, base=2)


def calculate_laziness(df):
    temp_df = df.copy()
    temp_df["prev_location"] = temp_df["location"].shift(1)
    lazy_rows = temp_df[temp_df["location"] == temp_df["prev_location"]]
    if len(temp_df) == 0:
        return 0.0
    return len(lazy_rows) / len(temp_df)


def compute_all_metrics(traj):
    """Compute all metrics for a trajectory DataFrame."""
    t_shelter = frac_time_spent_in_shelter(traj)
    t_investigating = frac_time_spent_investigating(traj)
    tr = calculate_zone_transitions(traj)
    heatmap_entropy = calculate_heatmap_entropy(traj)
    laziness = calculate_laziness(traj)
    return {
        "t_shelter": t_shelter,
        "t_investigating": t_investigating,
        "n_sh_co": tr["Shelter to Corridor"],
        "n_co_sh": tr["Corridor to Shelter"],
        "n_co_ch": tr["Corridor to Chamber"],
        "n_ch_co": tr["Chamber to Corridor"],
        "heatmap_entropy": heatmap_entropy,
        "laziness": laziness,
    }


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

MICE = list(range(13, 27))
PHASES = ["def1", "def3"]
CONFIDENCE_THRESHOLD = 0.8
SPLIT = 1000  # split point (timestep index)

records = []

for mouse_num in MICE:
    for phase in PHASES:
        path = f"./dlc/DLC_all_batches/m{mouse_num}_investigation_{phase}_cropped.csv"
        df = pd.read_csv(path, index_col=0)
        df = df[df["Center_p"] > CONFIDENCE_THRESHOLD]

        traj = get_trajectory(df, n_steps=2000, flipped=True)

        # Split at timestep 1000 by positional index
        traj_h1 = traj.iloc[:SPLIT].reset_index(drop=True)
        traj_h2 = traj.iloc[SPLIT:].reset_index(drop=True)

        m1 = compute_all_metrics(traj_h1)
        m2 = compute_all_metrics(traj_h2)

        for half_label, metrics in [("first", m1), ("second", m2)]:
            records.append({"mouse": mouse_num, "phase": phase, "half": half_label, **metrics})

        print(f"  m{mouse_num} {phase} done")

results = pd.DataFrame(records)
results.to_csv("half_trajectory_results.csv", index=False)
print("\nSaved half_trajectory_results.csv")
print(results.head(8))


# ---------------------------------------------------------------------------
# Paired plots + Wilcoxon tests
# ---------------------------------------------------------------------------

METRICS = [
    "t_shelter", "t_investigating",
    "n_sh_co", "n_co_sh", "n_co_ch", "n_ch_co",
    "heatmap_entropy", "laziness",
]
METRIC_LABELS = {
    "t_shelter": "Frac. time in shelter",
    "t_investigating": "Frac. time investigating",
    "n_sh_co": "Shelter→Corridor transitions",
    "n_co_sh": "Corridor→Shelter transitions",
    "n_co_ch": "Corridor→Chamber transitions",
    "n_ch_co": "Chamber→Corridor transitions",
    "heatmap_entropy": "Spatial entropy (bits)",
    "laziness": "Laziness (frac. immobile)",
}

stat_rows = []

# Compute stats per phase separately, and pooled
for phase in PHASES + ["pooled"]:
    if phase == "pooled":
        subset = results
    else:
        subset = results[results["phase"] == phase]

    h1 = subset[subset["half"] == "first"].set_index(["mouse", "phase"])
    h2 = subset[subset["half"] == "second"].set_index(["mouse", "phase"])

    for metric in METRICS:
        vals1 = h1[metric].values.astype(float)
        vals2 = h2[metric].values.astype(float)
        diffs = vals2 - vals1
        n = len(diffs)

        if n < 4 or np.all(diffs == 0):
            stat_rows.append(dict(phase=phase, metric=metric, n=n, stat=np.nan,
                                  p_value=np.nan, effect_size_r=np.nan))
            continue

        stat, pval = wilcoxon(vals1, vals2, alternative="two-sided")
        # Effect size: rank-biserial correlation via Z-score approximation
        z = norm.ppf(1 - pval / 2) if pval < 1.0 else 0.0
        r = z / np.sqrt(n)
        stat_rows.append(dict(phase=phase, metric=metric, n=n, statistic=stat,
                              p_value=pval, effect_size_r=r))

stats_df = pd.DataFrame(stat_rows)
stats_df.to_csv("half_trajectory_stats.csv", index=False)
print("\nWilcoxon results (pooled across phases):")
print(stats_df[stats_df["phase"] == "pooled"][
    ["metric", "n", "statistic", "p_value", "effect_size_r"]
].to_string(index=False))


# ---------------------------------------------------------------------------
# Paired plots  – one figure per metric, faceted by phase
# ---------------------------------------------------------------------------

# Use a clean palette
PHASE_PALETTE = {"def1": "#4C72B0", "def3": "#DD8452"}

n_metrics = len(METRICS)
n_cols = 4
n_rows = int(np.ceil(n_metrics / n_cols))

for phase in PHASES:
    phase_data = results[results["phase"] == phase]
    h1 = phase_data[phase_data["half"] == "first"].set_index("mouse")
    h2 = phase_data[phase_data["half"] == "second"].set_index("mouse")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        mice_common = sorted(set(h1.index) & set(h2.index))
        v1 = h1.loc[mice_common, metric].values.astype(float)
        v2 = h2.loc[mice_common, metric].values.astype(float)

        # Draw connecting lines first
        for x1, x2 in zip(v1, v2):
            ax.plot([0, 1], [x1, x2], color="grey", alpha=0.35, lw=1)

        ax.scatter(np.zeros(len(v1)), v1, color=PHASE_PALETTE[phase], s=55,
                   zorder=5, label="First half")
        ax.scatter(np.ones(len(v2)), v2, color="black", s=55, zorder=5,
                   label="Second half", marker="D")

        # Add mean ± SE bars
        for xi, vals in [(0, v1), (1, v2)]:
            m, se = np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(xi, m, yerr=se, fmt="none", color="black", capsize=6,
                        lw=2, zorder=6)

        # Annotate p-value
        row = stats_df[(stats_df["phase"] == phase) & (stats_df["metric"] == metric)]
        if len(row) and not np.isnan(row.iloc[0]["p_value"]):
            pval = row.iloc[0]["p_value"]
            r_eff = row.iloc[0]["effect_size_r"]
            p_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
            ax.set_title(f"{METRIC_LABELS[metric]}\n{p_str}, r={r_eff:.2f}", fontsize=9)
        else:
            ax.set_title(METRIC_LABELS[metric], fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["First\nhalf", "Second\nhalf"])
        ax.set_xlim(-0.4, 1.4)
        sns.despine(ax=ax)

    # Hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(f"Phase: {phase}  –  Behavioral metrics: first vs second half (n={len(mice_common)} mice)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"half_trajectory_paired_{phase}.pdf", bbox_inches="tight")
    plt.savefig(f"half_trajectory_paired_{phase}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved half_trajectory_paired_{phase}.png/.pdf")

# ---------------------------------------------------------------------------
# Pooled paired plot (both phases overlaid, mice color-coded by phase)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = axes.flatten()

for ax, metric in zip(axes, METRICS):
    for phase, color in PHASE_PALETTE.items():
        phase_data = results[results["phase"] == phase]
        h1 = phase_data[phase_data["half"] == "first"].set_index("mouse")
        h2 = phase_data[phase_data["half"] == "second"].set_index("mouse")
        mice_common = sorted(set(h1.index) & set(h2.index))
        v1 = h1.loc[mice_common, metric].values.astype(float)
        v2 = h2.loc[mice_common, metric].values.astype(float)

        for x1, x2 in zip(v1, v2):
            ax.plot([0, 1], [x1, x2], color=color, alpha=0.3, lw=1)
        ax.scatter(np.zeros(len(v1)), v1, color=color, s=45, zorder=5,
                   alpha=0.8, label=phase if metric == METRICS[0] else "")
        ax.scatter(np.ones(len(v2)), v2, color=color, s=45, zorder=5,
                   alpha=0.8, marker="D")

    row = stats_df[(stats_df["phase"] == "pooled") & (stats_df["metric"] == metric)]
    if len(row) and not np.isnan(row.iloc[0]["p_value"]):
        pval = row.iloc[0]["p_value"]
        r_eff = row.iloc[0]["effect_size_r"]
        p_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
        ax.set_title(f"{METRIC_LABELS[metric]}\n{p_str}, r={r_eff:.2f}", fontsize=9)
    else:
        ax.set_title(METRIC_LABELS[metric], fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["First\nhalf", "Second\nhalf"])
    ax.set_xlim(-0.4, 1.4)
    sns.despine(ax=ax)

for ax in axes[n_metrics:]:
    ax.set_visible(False)

axes[0].legend(title="Phase", loc="best", fontsize=8)
fig.suptitle("All phases – Behavioral metrics: first vs second half (pooled)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("half_trajectory_paired_pooled.pdf", bbox_inches="tight")
plt.savefig("half_trajectory_paired_pooled.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved half_trajectory_paired_pooled.png/.pdf")

print("\nDone.")


# ---------------------------------------------------------------------------
# Full-session vs first-half deviation analysis (def1, excl. m19)
# ---------------------------------------------------------------------------

TARGET_STD = {
    't_shelter': 0.195, 't_investigating': 0.15,
    'n_sh_co': 6.48,    'n_co_sh': 6.59,
    'n_co_ch': 3.62,    'n_ch_co': 3.64,
    'heatmap_entropy': 0.68, 'laziness': 0.045,
}

MICE_DEV = [m for m in range(13, 27) if m != 19]

full_records = []
for mouse_num in MICE_DEV:
    path = f"./dlc/DLC_all_batches/m{mouse_num}_investigation_def1_cropped.csv"
    df = pd.read_csv(path, index_col=0)
    df = df[df["Center_p"] > CONFIDENCE_THRESHOLD]
    traj_full = get_trajectory(df, n_steps=2000, flipped=True)
    m_full = compute_all_metrics(traj_full)
    full_records.append({"mouse": mouse_num, **m_full})
    print(f"  full def1 m{mouse_num} done")

full_df = pd.DataFrame(full_records).set_index("mouse")

# Pull first-half def1 metrics from already-computed results
h1_def1 = (
    results[(results["phase"] == "def1") & (results["half"] == "first")]
    .set_index("mouse")[METRICS]
)
h1_def1 = h1_def1.loc[h1_def1.index.isin(MICE_DEV)]

# Compute raw deviation (full minus first-half)
dev_raw = full_df.loc[MICE_DEV, METRICS] - h1_def1.loc[MICE_DEV, METRICS]

# Normalize by target_std
dev_norm = dev_raw.copy()
for metric in METRICS:
    dev_norm[metric] = dev_raw[metric] / TARGET_STD[metric]

# Per-mouse table: raw and normalized side by side
dev_raw.columns    = [f"{m}_raw"  for m in METRICS]
dev_norm.columns   = [f"{m}_norm" for m in METRICS]
dev_table = pd.concat([dev_raw, dev_norm], axis=1)
# Reorder: metric pairs together
col_order = [col for m in METRICS for col in (f"{m}_raw", f"{m}_norm")]
dev_table = dev_table[col_order]
dev_table.to_csv("full_vs_firsthalf_deviation.csv")
print("\nSaved full_vs_firsthalf_deviation.csv")

# Summary: mean |deviation| and max |deviation| in normalized units (re-derive norm from original)
dev_norm_orig = (full_df.loc[MICE_DEV, METRICS] - h1_def1.loc[MICE_DEV, METRICS]).copy()
for metric in METRICS:
    dev_norm_orig[metric] = dev_norm_orig[metric] / TARGET_STD[metric]

summary = pd.DataFrame({
    "mean_abs_norm": dev_norm_orig.abs().mean(),
    "max_abs_norm":  dev_norm_orig.abs().max(),
})

print("\n=== Per-mouse, per-metric deviation (raw | normalized) ===")
# Pretty-print: show raw and norm interleaved
display_rows = []
for mouse in MICE_DEV:
    row = {"mouse": mouse}
    for metric in METRICS:
        row[f"{metric}_raw"]  = round(full_df.loc[mouse, metric] - h1_def1.loc[mouse, metric], 4)
        row[f"{metric}_norm"] = round(row[f"{metric}_raw"] / TARGET_STD[metric], 3)
    display_rows.append(row)
display_df = pd.DataFrame(display_rows).set_index("mouse")
print(display_df.to_string())

print("\n=== Mean and max |normalized deviation| per metric ===")
print(summary.round(3).to_string())

exceeding = summary[summary["mean_abs_norm"] > 0.5].index.tolist()
print(f"\n=== Metrics with mean |normalized deviation| > 0.5 ===")
if exceeding:
    for m in exceeding:
        print(f"  {m:20s}  mean={summary.loc[m,'mean_abs_norm']:.3f}  max={summary.loc[m,'max_abs_norm']:.3f}")
else:
    print("  None")


# ---------------------------------------------------------------------------
# Same deviation analysis for def3 (excl. m19)
# ---------------------------------------------------------------------------

full_records_def3 = []
for mouse_num in MICE_DEV:
    path = f"./dlc/DLC_all_batches/m{mouse_num}_investigation_def3_cropped.csv"
    df = pd.read_csv(path, index_col=0)
    df = df[df["Center_p"] > CONFIDENCE_THRESHOLD]
    traj_full = get_trajectory(df, n_steps=2000, flipped=True)
    m_full = compute_all_metrics(traj_full)
    full_records_def3.append({"mouse": mouse_num, **m_full})
    print(f"  full def3 m{mouse_num} done")

full_df3 = pd.DataFrame(full_records_def3).set_index("mouse")

h1_def3 = (
    results[(results["phase"] == "def3") & (results["half"] == "first")]
    .set_index("mouse")[METRICS]
)
h1_def3 = h1_def3.loc[h1_def3.index.isin(MICE_DEV)]

dev_raw3  = full_df3.loc[MICE_DEV, METRICS] - h1_def3.loc[MICE_DEV, METRICS]
dev_norm3 = dev_raw3.copy()
for metric in METRICS:
    dev_norm3[metric] = dev_raw3[metric] / TARGET_STD[metric]

out3 = pd.concat([dev_raw3.add_suffix("_raw"), dev_norm3.add_suffix("_norm")], axis=1)
out3 = out3[[col for m in METRICS for col in (f"{m}_raw", f"{m}_norm")]]
out3.to_csv("full_vs_firsthalf_deviation_def3.csv")
print("\nSaved full_vs_firsthalf_deviation_def3.csv")

summary3 = pd.DataFrame({
    "mean_abs_norm": dev_norm3.abs().mean(),
    "max_abs_norm":  dev_norm3.abs().max(),
})

print("\n=== def3 | Per-mouse deviation (raw | normalized) ===")
for mouse in MICE_DEV:
    print(f"  m{mouse}")
    for metric in METRICS:
        print(f"    {metric:20s}  raw={dev_raw3.loc[mouse, metric]:+.4f}  norm={dev_norm3.loc[mouse, metric]:+.3f}")

print("\n=== def3 | Mean and max |normalized deviation| per metric ===")
print(summary3.round(3).to_string())

exceeding3 = summary3[summary3["mean_abs_norm"] > 0.5].index.tolist()
print("\n=== def3 | Metrics with mean |norm dev| > 0.5 ===")
for m in exceeding3:
    print(f"  {m:20s}  mean={summary3.loc[m,'mean_abs_norm']:.3f}  max={summary3.loc[m,'max_abs_norm']:.3f}")
if not exceeding3:
    print("  None")


# ---------------------------------------------------------------------------
# Library matching: best parameter set per mouse × phase (first-half targets)
# ---------------------------------------------------------------------------

WEIGHTS = {
    't_shelter':       2.0,
    't_investigating': 2.0,
    'n_sh_co':         0.5,
    'n_co_sh':         0.5,
    'n_co_ch':         0.5,
    'n_ch_co':         0.5,
    'heatmap_entropy': 0.5,
    'laziness':        1.0,
}

TARGET_STD_MATCH = {
    't_shelter': 0.195, 't_investigating': 0.15,
    'n_sh_co': 6.48,    'n_co_sh': 6.59,
    'n_co_ch': 3.62,    'n_ch_co': 3.64,
    'heatmap_entropy': 0.68, 'laziness': 0.045,
}

# Map from our metric name → library column name (entropy key differs)
LIB_COL = {
    't_shelter':       'metric_t_shelter',
    't_investigating': 'metric_t_investigating',
    'n_sh_co':         'metric_n_sh_co',
    'n_co_sh':         'metric_n_co_sh',
    'n_co_ch':         'metric_n_co_ch',
    'n_ch_co':         'metric_n_ch_co',
    'heatmap_entropy': 'metric_entropy',
    'laziness':        'metric_laziness',
}

# Transition metrics that need ×2 scaling (first-half counts → full-session scale)
SCALE2 = {'n_sh_co', 'n_co_sh', 'n_co_ch', 'n_ch_co'}

library = pd.read_csv("mining_results_adj.csv").iloc[:536].reset_index(drop=True)
library.index.name = "param_set_id"

param_cols = [c for c in library.columns if c.startswith("param_")]

half_results = pd.read_csv("half_trajectory_results.csv")
h1 = half_results[half_results["half"] == "first"].copy()

MICE_MATCH = [m for m in range(13, 27) if m != 19]

# Pre-build library metric matrix for vectorised loss (n_lib × n_metrics)
lib_vals = np.stack([library[LIB_COL[k]].values for k in WEIGHTS], axis=1).astype(float)
std_arr  = np.array([TARGET_STD_MATCH[k] + 1e-6 for k in WEIGHTS])
w_arr    = np.array([WEIGHTS[k] for k in WEIGHTS])

match_records = []

for mouse in MICE_MATCH:
    for phase in ["def1", "def3"]:
        row = h1[(h1["mouse"] == mouse) & (h1["phase"] == phase)]
        if len(row) == 0:
            print(f"  WARNING: no first-half data for m{mouse} {phase}, skipping")
            continue
        row = row.iloc[0]

        # Build target vector (same order as WEIGHTS keys)
        target = np.array([
            row[k] * (2.0 if k in SCALE2 else 1.0)
            for k in WEIGHTS
        ], dtype=float)

        # Vectorised squared-weighted loss over all 536 library entries
        diff  = (lib_vals - target[np.newaxis, :]) / std_arr[np.newaxis, :]
        loss  = (w_arr[np.newaxis, :] * diff ** 2).sum(axis=1)

        best_idx  = int(np.argmin(loss))
        best_loss = float(loss[best_idx])
        best_params = library.iloc[best_idx][param_cols].to_dict()

        match_records.append({
            "mouse_id":          mouse,
            "phase":             phase,
            "matched_param_set_id": best_idx,
            "loss_value":        round(best_loss, 6),
            **{k: round(float(v), 8) for k, v in best_params.items()},
        })

match_df = pd.DataFrame(match_records)
match_df.to_csv("library_matches.csv", index=False)
print("\nSaved library_matches.csv")
print(match_df[["mouse_id", "phase", "matched_param_set_id", "loss_value"]].to_string(index=False))

# Summary statistics on loss values
print("\n=== Loss distribution summary ===")
print(match_df.groupby("phase")["loss_value"].describe().round(3).to_string())
print("\nOverall:")
print(match_df["loss_value"].describe().round(3).to_string())

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# (a) Stripplot per phase
ax = axes[0]
for i, phase in enumerate(["def1", "def3"]):
    vals = match_df[match_df["phase"] == phase]["loss_value"].values
    ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(vals)),
               vals, s=55, alpha=0.8, color=PHASE_PALETTE[phase], zorder=5)
    ax.errorbar(i, vals.mean(), yerr=vals.std(ddof=1)/np.sqrt(len(vals)),
                fmt="none", color="black", capsize=6, lw=2, zorder=6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["def1", "def3"])
ax.set_ylabel("Loss value"); ax.set_title("Loss by phase")
sns.despine(ax=ax)

# (b) Per-mouse loss, both phases
ax = axes[1]
x = np.arange(len(MICE_MATCH))
for phase, offset, color in [("def1", -0.18, PHASE_PALETTE["def1"]),
                               ("def3",  0.18, PHASE_PALETTE["def3"])]:
    vals = [match_df[(match_df["mouse_id"] == m) & (match_df["phase"] == phase)]["loss_value"].values[0]
            for m in MICE_MATCH]
    ax.bar(x + offset, vals, width=0.35, color=color, alpha=0.8, label=phase)
ax.set_xticks(x); ax.set_xticklabels([f"m{m}" for m in MICE_MATCH], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Loss value"); ax.set_title("Per-mouse loss")
ax.legend(fontsize=8); sns.despine(ax=ax)

# (c) Histogram of all loss values
ax = axes[2]
all_losses = match_df["loss_value"].values
ax.hist(all_losses, bins=15, color="#888", edgecolor="white")
ax.axvline(np.median(all_losses), color="crimson", lw=1.5, linestyle="--", label=f"median={np.median(all_losses):.1f}")
ax.set_xlabel("Loss value"); ax.set_ylabel("Count")
ax.set_title("Loss distribution (all mouse×phase)")
ax.legend(fontsize=8); sns.despine(ax=ax)

plt.tight_layout()
plt.savefig("library_matches_loss_distribution.pdf", bbox_inches="tight")
plt.savefig("library_matches_loss_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved library_matches_loss_distribution.png/.pdf")
