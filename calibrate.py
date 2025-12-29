import optuna
import pandas as pd
import numpy as np
from sim import run_sim
import os
import datetime

final_mask = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
])

active_cells = np.argwhere(final_mask == 1) 
bin_id_map = {tuple(pos): i for i, pos in enumerate(active_cells)}


# Metrics

def frac_time_spent_in_shelter(df):
    loc_table = df.copy()
    shelter_indices = [6, 21, 36]
    total_t = len(loc_table)
    shelter_t = 0
    for timestep in range(total_t):
        loc = loc_table.iloc[timestep]['location']
        if loc in shelter_indices:
            shelter_t += 1

    frac = shelter_t / total_t
    return frac

def frac_time_spent_investigating(df):
    loc_table = df.copy()
    investigation_indices = [33, 34, 35, 48, 54, 17, 18, 19, 20, 32, 47, 53] # cells at a distance of <= 2 from threat cluster
    total_t = len(loc_table)
    inv_t = 0
    in_zone = [loc_table.iloc[t]['location'] in investigation_indices for t in range(total_t)]
    for t in range(1, total_t - 1): # Skip first and last frame to avoid index errors
        if in_zone[t] and in_zone[t-1] and in_zone[t+1]:
            inv_t += 1

    frac = inv_t / total_t
    return frac


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
    temp_df['zone'] = temp_df['location'].apply(get_zone)
    temp_df['prev_zone'] = temp_df['zone'].shift(1)
    
    # Filter for rows where the zone actually changed
    transitions = temp_df[temp_df['zone'] != temp_df['prev_zone']].dropna(subset=['prev_zone'])

    transitions['path'] = transitions['prev_zone'] + " -> " + transitions['zone']
    counts = transitions['path'].value_counts()

    results = {
        "Shelter to Corridor": counts.get("Shelter -> Corridor", 0),
        "Corridor to Shelter": counts.get("Corridor -> Shelter", 0),
        "Corridor to Chamber": counts.get("Corridor -> Chamber", 0),
        "Chamber to Corridor": counts.get("Chamber -> Corridor", 0)
    }
    
    return results

from scipy.stats import entropy

def calculate_heatmap_entropy(df, num_active_bins=57):
    counts = df['location'].value_counts()
    prob_dist = np.zeros(num_active_bins)
    
    for bin_id, count in counts.items():
        if 0 <= bin_id < num_active_bins:
            prob_dist[bin_id] = count
            
    if np.sum(prob_dist) == 0:
        return 0.0
    
    prob_dist = prob_dist / np.sum(prob_dist)

    return entropy(prob_dist, base=2)

def calculate_laziness(df):
    temp_df = df.copy()
    temp_df['prev_location'] = temp_df['location'].shift(1)
    lazy_rows = temp_df[temp_df['location'] == temp_df['prev_location']]

    if len(temp_df) == 0:
        return 0.0
    
    return len(lazy_rows)/len(temp_df)

def calculate_metrics(df):
    results = {}
    t_shelter = frac_time_spent_in_shelter(df)
    t_invest = frac_time_spent_investigating(df)
    transition_dict = calculate_zone_transitions(df)
    n_sh_co = transition_dict['Shelter to Corridor']
    n_co_sh = transition_dict["Corridor to Shelter"]
    n_co_ch = transition_dict["Corridor to Chamber"]
    n_ch_co = transition_dict["Chamber to Corridor"]
    entropy = calculate_heatmap_entropy(df)
    laziness = calculate_laziness(df)

    results['t_shelter'] = t_shelter
    results['t_investigating'] = t_invest
    results['n_sh_co'] = n_sh_co
    results['n_co_sh'] = n_co_sh
    results['n_co_ch'] = n_co_ch
    results['n_ch_co'] = n_ch_co
    results['entropy'] = entropy
    results['laziness'] = laziness

    return results

FIXED_K_THREAT = 0.45

metric_names = ['t_shelter', 't_investigating', 'n_sh_co', 'n_co_sh', 'n_co_ch', 'n_ch_co', 'entropy', 'laziness']
target_avg = {'t_shelter': 0.4475, 't_investigating': 0.149, 'n_sh_co': 14, 'n_co_sh': 14, 'n_co_ch': 10, 'n_ch_co': 9, 'entropy': 4.25, 'laziness': 0.825}
target_std = {'t_shelter': 0.145, 't_investigating': 0.08, 'n_sh_co': 5.68, 'n_co_sh': 5.61, 'n_co_ch': 3.56, 'n_ch_co': 3.43, 'entropy': 0.58, 'laziness': 0.035}

OUTPUT_DIR = "calibration_histories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def objective(trial):
    id_threshold = trial.suggest_float("id_threshold", 0.55, 0.75)       # Trial 30 was 0.66
    sensory_prec_slope = trial.suggest_float("sensory_prec_slope", 0.55, 0.75) # Trial 30 was 0.67
    k_shelter = trial.suggest_float("k_shelter", 0.15, 0.60)             # Trial 30 was 0.25
    
    delta_stay = trial.suggest_float("delta_stay", 0.15, 5.0)

    weights = {'t_shelter': 2.0, 't_investigating': 2.0, 'n_sh_co': 0.5, 
               'n_co_sh': 0.5, 'n_co_ch': 0.5, 'n_ch_co': 0.5, 'entropy': 0.5, 'laziness': 1.0}
    accumulated_metrics = {k: [] for k in metric_names}
    
    n_sims = 1
    for step in range(n_sims):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Trial {trial.number}, Sim {step+1}/{n_sims}: STARTING...", flush=True)

        try:
            # Run Sim
            history = run_sim(
                id_threshold=id_threshold,
                sensory_imprecision=sensory_prec_slope,
                k_shelter=k_shelter,
                k_threat=FIXED_K_THREAT,
                delta_stay=delta_stay
            )
            
            if history is None or len(history['agent_loc']) < 2: return 1000.0
            
            traj = pd.DataFrame({'location': history['agent_loc']})
            save_path = os.path.join(OUTPUT_DIR, f"trial_{trial.number}_sim{step}.csv")
            traj.to_csv(save_path, index=False)
            
            m = calculate_metrics(traj)

        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Trial {trial.number} CRASHED: {e}. penalizing...", flush=True)
            return 1000.0
        
        current_loss = 0.0
        
        # Update our history of metrics
        for k in metric_names:
            accumulated_metrics[k].append(m[k])

        # Calculate the running mean for all metrics so far
        running_means = {k: np.mean(v) for k, v in accumulated_metrics.items()}

        for k in weights:
            metric_val = running_means.get(k)             
            diff = metric_val - target_avg[k]
            std = target_std[k] + 1e-6
            current_loss += weights[k] * ((diff / std) ** 2)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Trial {trial.number}, Sim {step+1}/{n_sims}: FINISHED. Current Loss: {current_loss:.4f}", flush=True)

        for k, v in running_means.items():
            trial.set_user_attr(k, v)
            
        trial.report(current_loss, step=step)

        if step > 0 and trial.should_prune():
            raise optuna.TrialPruned()

    return current_loss

if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=2)
    study = optuna.create_study(direction="minimize", 
                                sampler=sampler,
                                pruner=pruner,
                                study_name="calibration",
                                storage="sqlite:///calibration.db", 
                                load_if_exists=True)
    
    study.optimize(objective, n_trials=5) 
    
    print("Best params:", study.best_params)