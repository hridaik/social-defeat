from sim import run_sim
from utils import render_grid_frame_arena, make_two_rooms_with_corridor, grid
import numpy as np
import pandas as pd
import tqdm
import itertools
import os
import optuna
import sys
import datetime

final_mask = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
])

active_cells = np.argwhere(final_mask == 1) 
# Create a dictionary to map (row, col) to a zero-indexed bin ID (0-56)
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

def calculate_num_flights(df, active_cells_mapping=active_cells, dist_multiplier=1.2):
    social_investigation_bins = {33, 34, 35, 48, 54, 17, 18, 19, 20, 32, 47, 53}
    
    def get_zone_info(bin_id):
        r, c = active_cells_mapping[bin_id]
        if bin_id in social_investigation_bins:
            return "Social", c
        elif c == 0:
            return "Shelter", c
        else:
            return "Other", c

    df = df.copy()
    zone_data = df['location'].apply(get_zone_info)
    df['zone'] = [x[0] for x in zone_data]
    df['col'] = [x[1] for x in zone_data]
    
    flights = 0
    is_investigating = False
    start_step = 0
    start_col = 0
    
    for i in range(len(df)):
        current_zone = df['zone'].iloc[i]
        current_step = df['step_number'].iloc[i]
        current_col = df['col'].iloc[i]
        
        if current_zone == "Social":
            is_investigating = True
            start_step = current_step
            start_col = current_col
            
        elif current_zone == "Shelter" and is_investigating:
            duration = current_step - start_step
            # Most direct path is the number of columns to cross (except cell (4,12) but okay)
            distance_to_shelter = start_col 
            # dist_multiplier: Tolerance for 'directness' => 1.0 = perfect direct path, 1.2 = slight deviation allowed
            if duration <= (distance_to_shelter * dist_multiplier):
                flights += 1
            is_investigating = False
            
        elif current_zone == "Other" and is_investigating:
            if (current_step - start_step) > (start_col * dist_multiplier):
                is_investigating = False
                
    return flights





# Fitting starts here
metric_names = ['t_shelter', 't_investigating', 'n_sh_co', 'n_co_sh', 'n_co_ch', 'n_ch_co', 'entropy']
real_avg = [0.218409091, 0.185818182, 13.04545455, 13, 9.590909091, 9.409090909, 4.121289926]
real_std	= [0.18642642, 0.114025271, 5.191577731, 5.308655026, 3.393296747, 3.33928532, 0.76394649]

target_avg = {}
target_std = {}
for i in range(len(metric_names)):
    target_avg[metric_names[i]] = real_avg[i]
    target_std[metric_names[i]] = real_std[i]

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

    results['t_shelter'] = t_shelter
    results['t_investigating'] = t_invest
    results['n_sh_co'] = n_sh_co
    results['n_co_sh'] = n_co_sh
    results['n_co_ch'] = n_co_ch
    results['n_ch_co'] = n_ch_co
    results['entropy'] = entropy

    return results

def objective(trial, target_avg, target_std):
    # 1. Suggest Parameters
    id_threshold = trial.suggest_float("id_threshold", 0.55, 0.95)
    sensory_prec_slope = trial.suggest_float("sensory_prec_slope", 0.1, 2.0)
    k_shelter = trial.suggest_float("k_shelter", 0.2, 2.0)
    k_threat = trial.suggest_float("k_threat", 0.2, 2.0)

    # 2. Dynamic Simulation Count (Optional tweak)
    # You can start with fewer sims and increase if the trial looks promising,
    # but standard pruning is usually cleaner.
    n_sims = 5 
    
    # Store results to calculate running mean
    accumulated_metrics = {k: [] for k in metric_names}
    
    weights = {'t_shelter': 1.0, 't_investigating': 1.0, 'n_sh_co': 1.0, 
               'n_co_sh': 1.0, 'n_co_ch': 1.0, 'n_ch_co': 1.0, 'entropy': 0.5}

    for step in range(n_sims):
        # --- Run Simulation ---
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Trial {trial.number}, Sim {step+1}/{n_sims}: STARTING...", flush=True)
        history = run_sim(id_threshold=id_threshold, 
                          sensory_imprecision=sensory_prec_slope, 
                          k_shelter=k_shelter, 
                          k_threat=k_threat,
                          max_steps=2000)
        
        # Create DataFrame
        trajectory = pd.DataFrame({
            'timestep': range(len(history['agent_loc'])),
            'location': history['agent_loc']
        })
        
        # Calculate metrics for this specific run
        m = calculate_metrics(df=trajectory) 
        
        # --- ACCUMULATE & CHECK LOSS ---
        current_loss = 0.0
        
        # Update our history of metrics
        for k in metric_names:
            accumulated_metrics[k].append(m[k])
            
        # Calculate the running mean for all metrics so far
        running_means = {k: np.mean(v) for k, v in accumulated_metrics.items()}

        # Calculate Loss based on the running mean
        for k in weights:
            # Use .get() or ensure calculate_metrics returns exact keys
            metric_val = running_means.get(k)             
            diff = metric_val - target_avg[k]
            std = target_std[k] + 1e-6
            current_loss += weights[k] * ((diff / std) ** 2)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Trial {trial.number}, Sim {step+1}/{n_sims}: FINISHED. Current Loss: {current_loss:.4f}", flush=True)

        # --- PRUNING STEP ---
        # We report the loss to Optuna. 
        # Optuna decides if this current_loss is worse than the median of other trials 
        # at this specific step (sim number).
        for k, v in running_means.items():
            trial.set_user_attr(k, v)
            
        trial.report(current_loss, step=step)

        # If we are performing poorly compared to other trials, stop early.
        # We skip pruning on step 0 (first sim) because it might be a noisy outlier.
        if step > 0 and trial.should_prune():
            raise optuna.TrialPruned()
        

    return current_loss

# --- OPTIMIZATION SETUP ---

# Use the Multivariate sampler for correlated parameters
sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)

# Use a Pruner. Hyperband is excellent for resource allocation.
# It effectively runs many trials with few sims, and only the best get the full 5 sims.
pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=5, reduction_factor=3)

study = optuna.create_study(
    direction="minimize", 
    sampler=sampler, 
    pruner=pruner,
    study_name="avg_mouse_fit_v1",
    storage="sqlite:///avg_mouse_fit.db",
    load_if_exists=True
)

study.optimize(lambda t: objective(t, target_avg, target_std), n_trials=40)

print(f"Best parameters: {study.best_params}")