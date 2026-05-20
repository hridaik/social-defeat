import optuna
import pandas as pd
import numpy as np

WEIGHTS = {
    't_shelter':       2.0,
    't_investigating': 2.0,
    'n_sh_co':         0.5,
    'n_co_sh':         0.5,
    'n_co_ch':         0.5,
    'n_ch_co':         0.5,
    'entropy':         0.5,
    'laziness':        1.0,
}

target_avg = {'t_shelter': 0.3215, 't_investigating': 0.256, 'n_sh_co': 11, 
              'n_co_sh': 11, 'n_co_ch': 12, 'n_ch_co': 12, 'entropy': 4.166, 'laziness': 0.826}
target_std = {'t_shelter': 0.195, 't_investigating': 0.15, 'n_sh_co': 6.48,
              'n_co_sh': 6.59, 'n_co_ch': 3.62, 'n_ch_co': 3.64, 'entropy': 0.68, 'laziness': 0.045}

models = ["full", "noD", "noT", "M_only"]
mouse = "avg"

rows = []
for model in models:
    db_url = f"sqlite:///{model}_{mouse}.db"
    study = optuna.load_study(study_name=f"{model}_{mouse}", storage=db_url)
    best = study.best_trial
    
    row = {"model": model, "total_loss": best.value}
    for metric, w in WEIGHTS.items():
        val = best.user_attrs.get(metric, None)
        if val is not None:
            z = (val - target_avg[metric]) / (target_std[metric] + 1e-6)
            weighted_contrib = w * z**2
            row[f"{metric}_val"] = round(val, 4)
            row[f"{metric}_z"] = round(z, 3)
            row[f"{metric}_contrib"] = round(weighted_contrib, 3)
    rows.append(row)

df = pd.DataFrame(rows)

# Summary: just the weighted contributions per metric
contrib_cols = [f"{m}_contrib" for m in WEIGHTS]
summary = df[["model", "total_loss"] + contrib_cols].copy()
summary.columns = ["model", "total_loss"] + list(WEIGHTS.keys())
print(summary.to_string(index=False))

# Also print actual values vs target for context
print("\n--- Actual metric values at best trial ---")
for model in models:
    print(f"\n{model}:")
    row = df[df.model == model].iloc[0]
    for metric in WEIGHTS:
        print(f"  {metric:20s} val={row[f'{metric}_val']:.4f}  "
              f"target={target_avg[metric]:.4f}  z={row[f'{metric}_z']:+.3f}")