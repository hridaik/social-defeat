import optuna
import pandas as pd
import glob
import os

# target_std = {'t_shelter': 0.145, 't_investigating': 0.08, 'n_sh_co': 5.68, 'n_co_sh': 5.61, 'n_co_ch': 3.56, 'n_ch_co': 3.43, 'entropy': 0.58, 'laziness': 0.035}

# 0.194947581	0.149524265	6.477787793	6.587131859	3.617206612	3.643907412	0.678171382	0.045547246
target_std = {'t_shelter': 0.19, 't_investigating': 0.15, 'n_sh_co': 6.48, 'n_co_sh': 6.59, 'n_co_ch': 3.62, 'n_ch_co': 3.65, 'entropy': 0.68, 'laziness': 0.0455}

TARGETS = {
    '13_hab3': {
        'avg': {'t_shelter': 0.6485, 't_investigating': 0.1405, 'n_sh_co': 23, 'n_co_sh': 23, 'n_co_ch': 6, 'n_ch_co': 6, 'heatmap_entropy': 4.132956915, 'laziness': 0.8295},
        'std': target_std
    },
    '13_def2': {
        'avg': {'t_shelter': 0.312, 't_investigating': 0.101, 'n_sh_co': 13, 'n_co_sh': 13, 'n_co_ch': 15, 'n_ch_co': 14, 'heatmap_entropy': 4.721737822, 'laziness': 0.7955},
        'std': target_std
    },
    '14_hab3': {
        'avg': {'t_shelter': 0.562, 't_investigating': 0.044, 'n_sh_co': 18, 'n_co_sh': 18, 'n_co_ch': 12, 'n_ch_co': 12, 'heatmap_entropy': 3.909343409, 'laziness': 0.856},
        'std': target_std
    },
    '14_def2': {
        'avg': {'t_shelter': 0.5385, 't_investigating': 0.053, 'n_sh_co': 14, 'n_co_sh': 14, 'n_co_ch': 8, 'n_ch_co': 8, 'heatmap_entropy': 3.818612692, 'laziness': 0.853},
        'std': target_std
    },
    '15_hab3': {
        'avg': {'t_shelter': 0.5395, 't_investigating': 0.1345, 'n_sh_co': 22, 'n_co_sh': 22, 'n_co_ch': 3, 'n_ch_co': 3, 'heatmap_entropy': 3.864439473, 'laziness': 0.846},
        'std': target_std
    },
    '15_def2': {
        'avg': {'t_shelter': 0.4255, 't_investigating': 0.1075, 'n_sh_co': 17, 'n_co_sh': 17, 'n_co_ch': 8, 'n_ch_co': 7, 'heatmap_entropy': 4.179584098, 'laziness': 0.847},
        'std': target_std
    },
    '16_hab3': {
        'avg': {'t_shelter': 0.359, 't_investigating': 0.1925, 'n_sh_co': 8, 'n_co_sh': 8, 'n_co_ch': 7, 'n_ch_co': 7, 'heatmap_entropy': 3.919248356, 'laziness': 0.871},
        'std': target_std
    },
    '16_def2': {
        'avg': {'t_shelter': 0.6935, 't_investigating': 0.0455, 'n_sh_co': 18, 'n_co_sh': 17, 'n_co_ch': 10, 'n_ch_co': 9, 'heatmap_entropy': 2.939803933, 'laziness': 0.858},
        'std': target_std
    },
    '17_hab3': {
        'avg': {'t_shelter': 0.404, 't_investigating': 0.279, 'n_sh_co': 19, 'n_co_sh': 19, 'n_co_ch': 9, 'n_ch_co': 8, 'heatmap_entropy': 4.35692005, 'laziness': 0.8315},
        'std': target_std
    },
    '17_def2': {
        'avg': {'t_shelter': 0.664, 't_investigating': 0.062, 'n_sh_co': 15, 'n_co_sh': 15, 'n_co_ch': 5, 'n_ch_co': 5, 'heatmap_entropy': 3.364352495, 'laziness': 0.879},
        'std': target_std
    },
    '18_hab3': {
        'avg': {'t_shelter': 0.306, 't_investigating': 0.218, 'n_sh_co': 13, 'n_co_sh': 14, 'n_co_ch': 14, 'n_ch_co': 14, 'heatmap_entropy': 4.994446817, 'laziness': 0.7835},
        'std': target_std
    },
    '18_def2': {
        'avg': {'t_shelter': 0.407, 't_investigating': 0.1615, 'n_sh_co': 15, 'n_co_sh': 15, 'n_co_ch': 11, 'n_ch_co': 10, 'heatmap_entropy': 4.643901027, 'laziness': 0.814},
        'std': target_std
    },
    '19_hab3': {
        'avg': {'t_shelter': 0.363, 't_investigating': 0.14, 'n_sh_co': 7, 'n_co_sh': 8, 'n_co_ch': 13, 'n_ch_co': 13, 'heatmap_entropy': 4.720152234, 'laziness': 0.811},
        'std': target_std
    },
    '19_def2': {
        'avg': {'t_shelter': 0.474, 't_investigating': 0.1875, 'n_sh_co': 18, 'n_co_sh': 19, 'n_co_ch': 12, 'n_ch_co': 12, 'heatmap_entropy': 4.61703006, 'laziness': 0.8035},
        'std': target_std
    },
    '20_hab3': {
        'avg': {'t_shelter': 0.293, 't_investigating': 0.1745, 'n_sh_co': 3, 'n_co_sh': 3, 'n_co_ch': 16, 'n_ch_co': 15, 'heatmap_entropy': 4.73309349, 'laziness': 0.753},
        'std': target_std
    },
    '20_def2': {
        'avg': {'t_shelter': 0.1705, 't_investigating': 0.3455, 'n_sh_co': 5, 'n_co_sh': 5, 'n_co_ch': 10, 'n_ch_co': 9, 'heatmap_entropy': 5.151318048, 'laziness': 0.781},
        'std': target_std
    },
}

TARGETS = {
    '13_def1': {
        'avg': {'t_shelter': 0.5015, 't_investigating': 0.126, 'n_sh_co': 9, 'n_co_sh': 9, 'n_co_ch': 8, 'n_ch_co': 8, 'heatmap_entropy': 4.157401889, 'laziness': 0.8345},
        'std': target_std
    },
    '13_def3': {
        'avg': {'t_shelter': 0.2165, 't_investigating': 0.2755, 'n_sh_co': 14, 'n_co_sh': 14, 'n_co_ch': 13, 'n_ch_co': 14, 'heatmap_entropy': 4.679772779, 'laziness': 0.7895},
        'std': target_std
    },
    '14_def1': {
        'avg': {'t_shelter': 0.4895, 't_investigating': 0.104, 'n_sh_co': 13, 'n_co_sh': 13, 'n_co_ch': 9, 'n_ch_co': 9, 'heatmap_entropy': 3.746474719, 'laziness': 0.8495},
        'std': target_std
    },
    '14_def3': {
        'avg': {'t_shelter': 0.3215, 't_investigating': 0.2595, 'n_sh_co': 11, 'n_co_sh': 11, 'n_co_ch': 12, 'n_ch_co': 12, 'heatmap_entropy': 4.166432251, 'laziness': 0.826},
        'std': target_std
    },
    '15_def1': {
        'avg': {'t_shelter': 0.688, 't_investigating': 0.0825, 'n_sh_co': 10, 'n_co_sh': 10, 'n_co_ch': 4, 'n_ch_co': 4, 'heatmap_entropy': 2.652491439, 'laziness': 0.9215},
        'std': target_std
    },
    '15_def3': {
        'avg': {'t_shelter': 0.354, 't_investigating': 0.1315, 'n_sh_co': 15, 'n_co_sh': 16, 'n_co_ch': 7, 'n_ch_co': 7, 'heatmap_entropy': 4.345699479, 'laziness': 0.8325},
        'std': target_std
    },
    '16_def1': {
        'avg': {'t_shelter': 0.1445, 't_investigating': 0.387, 'n_sh_co': 7, 'n_co_sh': 6, 'n_co_ch': 7, 'n_ch_co': 6, 'heatmap_entropy': 4.487010226, 'laziness': 0.811},
        'std': target_std
    },
    '16_def3': {
        'avg': {'t_shelter': 0.6055, 't_investigating': 0.1115, 'n_sh_co': 11, 'n_co_sh': 10, 'n_co_ch': 11, 'n_ch_co': 10, 'heatmap_entropy': 2.828770281, 'laziness': 0.8585},
        'std': target_std
    },
    '17_def1': {
        'avg': {'t_shelter': 0.4405, 't_investigating': 0.2235, 'n_sh_co': 7, 'n_co_sh': 7, 'n_co_ch': 8, 'n_ch_co': 9, 'heatmap_entropy': 4.068416593, 'laziness': 0.8595},
        'std': target_std
    },
    '17_def3': {
        'avg': {'t_shelter': 0.426, 't_investigating': 0.2915, 'n_sh_co': 8, 'n_co_sh': 8, 'n_co_ch': 5, 'n_ch_co': 6, 'heatmap_entropy': 3.74524117, 'laziness': 0.8805},
        'std': target_std
    },
    '18_def1': {
        'avg': {'t_shelter': 0.4175, 't_investigating': 0.1405, 'n_sh_co': 22, 'n_co_sh': 22, 'n_co_ch': 13, 'n_ch_co': 13, 'heatmap_entropy': 4.564578899, 'laziness': 0.8215},
        'std': target_std
    },
    '18_def3': {
        'avg': {'t_shelter': 0.317, 't_investigating': 0.206, 'n_sh_co': 22, 'n_co_sh': 22, 'n_co_ch': 12, 'n_ch_co': 11, 'heatmap_entropy': 5.024379947, 'laziness': 0.7955},
        'std': target_std
    },
    '19_def1': {
        'avg': {'t_shelter': 0.2555, 't_investigating': 0.21, 'n_sh_co': 10, 'n_co_sh': 10, 'n_co_ch': 18, 'n_ch_co': 18, 'heatmap_entropy': 5.211984137, 'laziness': 0.755},
        'std': target_std
    },
    '19_def3': {
        'avg': {'t_shelter': 0.5945, 't_investigating': 0.046, 'n_sh_co': 39, 'n_co_sh': 39, 'n_co_ch': 14, 'n_ch_co': 13, 'heatmap_entropy': 4.006918493, 'laziness': 0.824},
        'std': target_std
    },
    '20_def1': {
        'avg': {'t_shelter': 0.1435, 't_investigating': 0.3655, 'n_sh_co': 12, 'n_co_sh': 12, 'n_co_ch': 13, 'n_ch_co': 13, 'heatmap_entropy': 4.824145152, 'laziness': 0.7645},
        'std': target_std
    },
    '20_def3': {
        'avg': {'t_shelter': 0.1, 't_investigating': 0.391, 'n_sh_co': 7, 'n_co_sh': 7, 'n_co_ch': 8, 'n_ch_co': 8, 'heatmap_entropy': 4.857023003, 'laziness': 0.755},
        'std': target_std
    },
    '21_def1': {
        'avg': {'t_shelter': 0.241, 't_investigating': 0.483, 'n_sh_co': 11, 'n_co_sh': 11, 'n_co_ch': 10, 'n_ch_co': 10, 'heatmap_entropy': 4.167765163, 'laziness': 0.8145},
        'std': target_std
    },
    '21_def3': {
        'avg': {'t_shelter': 0.135, 't_investigating': 0.434, 'n_sh_co': 12, 'n_co_sh': 12, 'n_co_ch': 11, 'n_ch_co': 10, 'heatmap_entropy': 4.72916643, 'laziness': 0.7305},
        'std': target_std
    },
    '22_def1': {
        'avg': {'t_shelter': 0.2155, 't_investigating': 0.139, 'n_sh_co': 12, 'n_co_sh': 13, 'n_co_ch': 15, 'n_ch_co': 15, 'heatmap_entropy': 5.05205629, 'laziness': 0.7885},
        'std': target_std
    },
    '22_def3': {
        'avg': {'t_shelter': 0.142, 't_investigating': 0.2575, 'n_sh_co': 13, 'n_co_sh': 13, 'n_co_ch': 10, 'n_ch_co': 10, 'heatmap_entropy': 5.006154148, 'laziness': 0.79223},
        'std': target_std
    },
    '23_def1': {
        'avg': {'t_shelter': 0.1945, 't_investigating': 0.3365, 'n_sh_co': 12, 'n_co_sh': 12, 'n_co_ch': 12, 'n_ch_co': 12, 'heatmap_entropy': 4.903375069, 'laziness': 0.78623},
        'std': target_std
    },
    '23_def3': {
        'avg': {'t_shelter': 0.3255, 't_investigating': 0.3865, 'n_sh_co': 18, 'n_co_sh': 17, 'n_co_ch': 13, 'n_ch_co': 12, 'heatmap_entropy': 4.532798358, 'laziness': 0.839524},
        'std': target_std
    },
    '24_def1': {
        'avg': {'t_shelter': 0.09, 't_investigating': 0.559, 'n_sh_co': 10, 'n_co_sh': 9, 'n_co_ch': 7, 'n_ch_co': 6, 'heatmap_entropy': 3.979085758, 'laziness': 0.833524},
        'std': target_std
    },
    '24_def3': {
        'avg': {'t_shelter': 0.6385, 't_investigating': 0.1175, 'n_sh_co': 22, 'n_co_sh': 22, 'n_co_ch': 9, 'n_ch_co': 9, 'heatmap_entropy': 4.007712686, 'laziness': 0.83925},
        'std': target_std
    },
    '25_def1': {
        'avg': {'t_shelter': 0.127, 't_investigating': 0.5135, 'n_sh_co': 5, 'n_co_sh': 4, 'n_co_ch': 5, 'n_ch_co': 4, 'heatmap_entropy': 4.585052569, 'laziness': 0.860525},
        'std': target_std
    },
    '25_def3': {
        'avg': {'t_shelter': 0.127, 't_investigating': 0.517, 'n_sh_co': 4, 'n_co_sh': 4, 'n_co_ch': 6, 'n_ch_co': 5, 'heatmap_entropy': 4.527635349, 'laziness': 0.8326},
        'std': target_std
    },
    '26_def1': {
        'avg': {'t_shelter': 0.291, 't_investigating': 0.3945, 'n_sh_co': 14, 'n_co_sh': 15, 'n_co_ch': 7, 'n_ch_co': 7, 'heatmap_entropy': 4.48366893, 'laziness': 0.82526},
        'std': target_std
    },
    '26_def3': {
        'avg': {'t_shelter': 0.824, 't_investigating': 0.057, 'n_sh_co': 15, 'n_co_sh': 15, 'n_co_ch': 2, 'n_ch_co': 2, 'heatmap_entropy': 2.535852912, 'laziness': 0.9355},
        'std': target_std
    }
}

WEIGHTS = {
    't_shelter': 2.0, 
    't_investigating': 2.0, 
    'n_sh_co': 0.5, 
    'n_co_sh': 0.5, 
    'n_co_ch': 0.5, 
    'n_ch_co': 0.5, 
    'entropy': 0.5,
    'laziness': 1.0
}

def calculate_raw_loss(trial_metrics, target_name):
    t_avg = TARGETS[target_name]['avg']
    t_std = TARGETS[target_name]['std']
    
    loss = 0.0
    
    for k, weight in WEIGHTS.items():
        if k not in t_avg: continue
        if k not in trial_metrics: continue # Skip if metric missing in DB
            
        val = trial_metrics[k]
        target = t_avg[k]
        std = t_std.get(k, 1.0)
        
        term = ((val - target) / (std + 1e-6)) ** 2
        loss += weight * term
        
    return loss

def mine_data(skip_no_lazy=False):
    db_files = glob.glob("*.db")
    print(f"Found databases: {db_files}")
    
    rows = []

    for db_path in db_files:
        try:
            # Connect to DB
            storage = optuna.storages.RDBStorage(url=f"sqlite:///{db_path}")
            
            # Iterate over all studies in the DB
            for summary in optuna.study.get_all_study_summaries(storage):
                study = optuna.load_study(study_name=summary.study_name, storage=storage)
                
                print(f"  -> Mining {db_path} ({len(study.trials)} trials)...")
                
                for trial in study.trials:
                    if trial.state != optuna.trial.TrialState.COMPLETE:
                        continue
                    
                    # 1. Check for laziness if skip_no_lazy is True
                    metrics = trial.user_attrs
                    has_laziness = 'laziness' in metrics
                    
                    if skip_no_lazy and not has_laziness:
                        continue

                    # Basic Info
                    row = {
                        "source_db": db_path,
                        "trial_id": trial.number,
                    }
                    
                    # 2. Add Params
                    for p_name, p_val in trial.params.items():
                        row[f"param_{p_name}"] = p_val
                        
                    # 3. Add Metrics info
                    row['has_laziness'] = has_laziness
                    
                    for m_name, m_val in metrics.items():
                        row[f"metric_{m_name}"] = m_val
                        
                    # 4. Calculate Loss for EACH Target Profile
                    for profile in TARGETS.keys():
                        loss = calculate_raw_loss(metrics, profile)
                        row[f"LOSS_{profile}"] = loss
                        
                    rows.append(row)
                    
        except Exception as e:
            print(f"  [!] Error reading {db_path}: {e}")

    # Convert to DataFrame
    if not rows:
        print("No data found matching the criteria.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    
    # Reorder columns for easier reading
    cols = ['source_db', 'trial_id'] + \
           [c for c in df.columns if "LOSS_" in c] + \
           ['has_laziness'] + \
           [c for c in df.columns if "param_" in c] + \
           [c for c in df.columns if "metric_" in c]
           
    df = df[cols]
    
    # Save
    output_file = "mining_results_adj.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDone! Saved {len(df)} trials to {output_file}")
    return df


if __name__ == "__main__":
    mine_data(skip_no_lazy=True)