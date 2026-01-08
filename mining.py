import optuna
import pandas as pd
import glob
import os

target_std = {'t_shelter': 0.145, 't_investigating': 0.08, 'n_sh_co': 5.68, 'n_co_sh': 5.61, 'n_co_ch': 3.56, 'n_ch_co': 3.43, 'entropy': 0.58, 'laziness': 0.035}

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

def mine_data():
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
                    
                    # Basic Info
                    row = {
                        "source_db": db_path,
                        "trial_id": trial.number,
                    }
                    
                    # 1. Add Params
                    for p_name, p_val in trial.params.items():
                        row[f"param_{p_name}"] = p_val
                        
                    # 2. Add Metrics (and check for laziness)
                    metrics = trial.user_attrs
                    row['has_laziness'] = 'laziness' in metrics
                    
                    for m_name, m_val in metrics.items():
                        row[f"metric_{m_name}"] = m_val
                        
                    # 3. Calculate Loss for EACH Target Profile
                    for profile in TARGETS.keys():
                        loss = calculate_raw_loss(metrics, profile)
                        row[f"LOSS_{profile}"] = loss
                        
                    rows.append(row)
                    
        except Exception as e:
            print(f"  [!] Error reading {db_path}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns for easier reading (Source -> Loss -> Params -> Metrics)
    cols = ['source_db', 'trial_id'] + \
           [c for c in df.columns if "LOSS_" in c] + \
           ['has_laziness'] + \
           [c for c in df.columns if "param_" in c] + \
           [c for c in df.columns if "metric_" in c]
           
    df = df[cols]
    
    # Save
    output_file = "mining_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDone! Saved {len(df)} trials to {output_file}")

if __name__ == "__main__":
    mine_data()