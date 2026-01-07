import optuna
import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
import datetime
from sim import run_sim
from scipy.stats import entropy
# from utils import calculate_metrics


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

metric_names = ['t_shelter', 't_investigating', 'n_sh_co', 'n_co_sh', 'n_co_ch', 'n_ch_co', 'entropy', 'laziness']
# target_avg = {'t_shelter': 0.4475, 't_investigating': 0.149, 'n_sh_co': 14, 'n_co_sh': 14, 'n_co_ch': 10, 'n_ch_co': 9, 'entropy': 4.25, 'laziness': 0.825}
target_std = {'t_shelter': 0.145, 't_investigating': 0.08, 'n_sh_co': 5.68, 'n_co_sh': 5.61, 'n_co_ch': 3.56, 'n_ch_co': 3.43, 'entropy': 0.58, 'laziness': 0.035}


# DEF 2 only
REAL_MOUSE_DATA = {
    "Resilient": {
        # 0.5385	0.053	14	14	8	8	3.818612692	0.853
        "avg": {'t_shelter': 0.54, 't_investigating': 0.053, 'n_sh_co': 14, 'n_co_sh': 14, 'n_co_ch': 8, 'n_ch_co': 8, 'entropy': 3.82, 'laziness': 0.853},
        "std": target_std
    },
    "Susceptible": {
        # 0.3815	0.23575	13.5	13.5	8	7.5	4.138084203	0.85075
        # 0.67875	0.05375	16.5	16	7.5	7	3.152078214	0.8685
        "avg": {'t_shelter': 0.67875, 't_investigating': 0.054, 'n_sh_co': 16.5, 'n_co_sh': 16, 'n_co_ch': 7.5, 'n_ch_co': 7, 'entropy': 3.152, 'laziness': 0.8685},
        "std": target_std
    },
    # "Control": {
        # 0.44675	0.166875	15.3	15.5	9.75	9.5	4.431234174	0.803
        # "avg": {'t_shelter': 0.447, 't_investigating': 0.167, 'n_sh_co': 15.3, 'n_co_sh': 15.5, 'n_co_ch': 9.75, 'n_ch_co': 9.5, 'entropy': 4.43, 'laziness': 0.803},
        # "std": target_std
    # }
}

pre_params = {
        "id_threshold": 0.102,
        "sensory_prec_slope": 0.055,
        "k_shelter": 4.95,
        "k_threat": 0.115,
        "delta_stay": 3.154
        }

def objective(trial, target_avg, target_std, output_dir):
    id_threshold = trial.suggest_float("id_threshold", 0.01, 0.45) # Directional lock  
    sensory_prec_slope = pre_params['sensory_prec_slope'] # Locked - mouse hardware 
    k_shelter = trial.suggest_float("k_shelter", 2.0, 8.0)
    k_threat = trial.suggest_float("k_threat", 0.05, 1.5)
    delta_stay = trial.suggest_float("delta_stay", 2.0, 7.0)

    n_sims = 1
    
    for step in range(n_sims):
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Trial {trial.number} Running...", flush=True)

            history = run_sim(
                id_threshold=id_threshold,
                sensory_imprecision=sensory_prec_slope,
                k_shelter=k_shelter,
                k_threat=k_threat,
                delta_stay=delta_stay
            )
            
            if history is None or len(history['agent_loc']) < 2: return 1000.0
            
            traj = pd.DataFrame({'location': history['agent_loc']})
            
            save_path = os.path.join(output_dir, f"trial_{trial.number}.csv")
            traj.to_csv(save_path, index=False)
            
            m = calculate_metrics(traj)
            
        except Exception as e:
            print(f"Crash in Trial {trial.number}: {e}")
            return 1000.0
        
        loss = 0.0
        
        for k, weight in WEIGHTS.items():
            t_avg = target_avg.get(k)
            t_std = target_std.get(k)
            
            val = m[k]
            
            z_sq = ((val - t_avg) / (t_std + 1e-6)) ** 2
            loss += weight * z_sq

        for k, v in m.items():
            trial.set_user_attr(k, float(v))
            
        return loss

if __name__ == "__main__":
    # python fit_after.py --mouse Mouse_A
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse", type=str, default='Susceptible')
    args = parser.parse_args()

    mouse_name = args.mouse
    print(f"--- FITTING INDIVIDUAL: {mouse_name} ---")
    
    target_avg = REAL_MOUSE_DATA[mouse_name]['avg']
    target_std = REAL_MOUSE_DATA[mouse_name]['std']
    phase = 'def'
    db_url = f"sqlite:///{mouse_name}_{phase}.db"
    output_dir = f"{mouse_name}_{phase}_history"
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=f"{mouse_name}_post",
        storage=db_url,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True)
    )

    # seed_params = {
    #     "id_threshold": 0.3,
    #     "sensory_prec_slope": 0.63,
    #     "k_shelter": 1.5,
    #     "k_threat": 1.0,
    #     "delta_stay": 1.41
    #     } # == pre-defeat params

    seed_params = {
        "id_threshold": 0.05,
        "sensory_prec_slope": 0.63,
        "k_shelter": 5.0,
        "k_threat": 0.1,
        "delta_stay": 3.5
        }
    
    if len(study.trials) == 0:
        study.enqueue_trial(seed_params)

    print(f"Launching optimization for {mouse_name}...")
    study.optimize(lambda t: objective(t, target_avg, target_std, output_dir), n_trials=100)
    
    print("Best params:", study.best_params)

# nice -n 10 python fit_individual.py --mouse Resilient > log_Res_1.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse Resilient > log_Res_2.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse Resilient > log_Res_3.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse Susceptible > log_Sus_1.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse Susceptible > log_Sus_2.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse Susceptible > log_Sus_3.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse Control > log_Con_1.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse Control > log_Con_2.txt 2>&1 &

