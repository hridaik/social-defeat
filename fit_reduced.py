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
from reduced_models import run_sim_M_only, run_sim_no_D, run_sim_no_T

optuna.logging.set_verbosity(optuna.logging.WARNING)

final_mask = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
])

active_cells = np.argwhere(final_mask == 1)
bin_id_map = {tuple(pos): i for i, pos in enumerate(active_cells)}

# ---------------------------------------------------------------------------
# Model registry
# Each entry defines which params are *active* for that model variant.
# Inactive params are passed as their seed/default values (they exist in the
# sim signature but have no effect in the reduced model).
# n_params drives trial-budget scaling.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "full": {
        "active_params": ["id_threshold", "sensory_prec_slope", "k_shelter", "k_threat", "delta_stay"],
        "n_params": 5,
        "run_fn": lambda p: run_sim(
            id_threshold=p["id_threshold"],
            sensory_imprecision=p["sensory_prec_slope"],
            k_shelter=p["k_shelter"],
            k_threat=p["k_threat"],
            delta_stay=p["delta_stay"],
        ),
    },
    "noD": {
        # Removes id_threshold — 4 active params
        "active_params": ["sensory_prec_slope", "k_shelter", "k_threat", "delta_stay"],
        "n_params": 4,
        "run_fn": lambda p: run_sim_no_D(
            id_threshold=p["id_threshold"],        # passed but unused by sim
            sensory_imprecision=p["sensory_prec_slope"],
            k_shelter=p["k_shelter"],
            k_threat=p["k_threat"],
            delta_stay=p["delta_stay"],
        ),
    },
    "noT": {
        # Removes id_threshold + sensory_prec_slope — 3 active params
        "active_params": ["k_shelter", "k_threat", "delta_stay"],
        "n_params": 3,
        "run_fn": lambda p: run_sim_no_T(
            id_threshold=p["id_threshold"],        # passed but unused by sim
            sensory_imprecision=p["sensory_prec_slope"],  # passed but unused
            k_shelter=p["k_shelter"],
            k_threat=p["k_threat"],
            delta_stay=p["delta_stay"],
        ),
    },
    "M_only": {
        # Minimal model — 3 active params (same active set as noT based on comments;
        # adjust active_params here if M_only further restricts anything)
        "active_params": ["k_shelter", "k_threat", "delta_stay"],
        "n_params": 3,
        "run_fn": lambda p: run_sim_M_only(
            id_threshold=p["id_threshold"],
            sensory_imprecision=p["sensory_prec_slope"],
            k_shelter=p["k_shelter"],
            k_threat=p["k_threat"],
            delta_stay=p["delta_stay"],
        ),
    },
}

# Fixed values injected for inactive params so the sim signature is always satisfied
INACTIVE_PARAM_DEFAULTS = {
    "id_threshold": 0.15,
    "sensory_prec_slope": 0.15,
}

PARAM_BOUNDS = {
    "id_threshold":      (0.05, 0.95),
    "sensory_prec_slope": (0.01, 1.0),
    "k_shelter":         (-6.0, 6.0),
    "k_threat":          (-3.0, 3.0),
    "delta_stay":        (1.5, 8.0),
}

# Seed used only for the full model (already well-characterised).
# Reduced models start without a seed so Phase 1 is unbiased.
# 0.134363192	5.178042798	1.382055521	2.284639358	0
FULL_MODEL_SEED = {
    "id_threshold": 0.5,
    "sensory_prec_slope": 0.5,
    "k_shelter": 0.0,
    "k_threat": 0.0,
    "delta_stay": 3.0,
}


# ---------------------------------------------------------------------------
# Trial budget
# Base = 50 trials (matching original for the full model).
# Scaled down proportionally for reduced models, with a floor of 20.
# Split 20% exploratory (random) / 80% TPE.
# ---------------------------------------------------------------------------
BASE_TRIALS = 100
FULL_MODEL_N_PARAMS = MODEL_REGISTRY["full"]["n_params"]
PHASE1_FRAC = 0.0


def compute_budget(n_params: int) -> tuple[int, int]:
    """Return (phase1_trials, phase2_trials) for a model with n_params active params."""
    # total = max(20, int(BASE_TRIALS * (n_params / FULL_MODEL_N_PARAMS)))
    total = max(20, int(BASE_TRIALS))
    phase1 = max(5, int(total * PHASE1_FRAC))
    phase2 = total - phase1
    return phase1, phase2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def frac_time_spent_in_shelter(df):
    shelter_indices = [6, 21, 36]
    return df['location'].isin(shelter_indices).sum() / len(df)


def frac_time_spent_investigating(df):
    investigation_indices = [33, 34, 35, 48, 54, 17, 18, 19, 20, 32, 47, 53]
    in_zone = df['location'].isin(investigation_indices).values
    inv_t = sum(
        1 for t in range(1, len(in_zone) - 1)
        if in_zone[t] and in_zone[t - 1] and in_zone[t + 1]
    )
    return inv_t / len(df)


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
    transitions = temp_df[temp_df['zone'] != temp_df['prev_zone']].dropna(subset=['prev_zone'])
    transitions = transitions.copy()
    transitions['path'] = transitions['prev_zone'] + " -> " + transitions['zone']
    counts = transitions['path'].value_counts()
    return {
        "Shelter to Corridor":  counts.get("Shelter -> Corridor", 0),
        "Corridor to Shelter":  counts.get("Corridor -> Shelter", 0),
        "Corridor to Chamber":  counts.get("Corridor -> Chamber", 0),
        "Chamber to Corridor":  counts.get("Chamber -> Corridor", 0),
    }


def calculate_heatmap_entropy(df, num_active_bins=57):
    counts = df['location'].value_counts()
    prob_dist = np.zeros(num_active_bins)
    for bin_id, count in counts.items():
        if 0 <= bin_id < num_active_bins:
            prob_dist[bin_id] = count
    if np.sum(prob_dist) == 0:
        return 0.0
    prob_dist /= prob_dist.sum()
    return entropy(prob_dist, base=2)


def calculate_laziness(df):
    prev_loc = df['location'].shift(1)
    return (df['location'] == prev_loc).sum() / len(df)


def calculate_metrics(df):
    transition_dict = calculate_zone_transitions(df)
    return {
        't_shelter':       frac_time_spent_in_shelter(df),
        't_investigating': frac_time_spent_investigating(df),
        'n_sh_co':         transition_dict['Shelter to Corridor'],
        'n_co_sh':         transition_dict['Corridor to Shelter'],
        'n_co_ch':         transition_dict['Corridor to Chamber'],
        'n_ch_co':         transition_dict['Chamber to Corridor'],
        'entropy':         calculate_heatmap_entropy(df),
        'laziness':        calculate_laziness(df),
    }


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

target_std = {
    't_shelter': 0.195, 't_investigating': 0.15,
    'n_sh_co': 6.48,    'n_co_sh': 6.59,
    'n_co_ch': 3.62,    'n_ch_co': 3.64,
    'entropy': 0.68,    'laziness': 0.045,
}

# Representative mice - scared, avg, curious
REAL_MOUSE_DATA = {
    "puc": {
        # 26-post 0.824	0.057	15	15	2	2	2.535852912	0.9355
        "avg": {'t_shelter': 0.824, 't_investigating': 0.057, 'n_sh_co': 15, 'n_co_sh': 15, 'n_co_ch': 2, 'n_ch_co': 2, 'entropy': 2.54, 'laziness': 0.94},
        "std": target_std
    },
    "avg": {
        # 14-post 0.3215	0.2595	11	11	12	12	4.166432251	0.826
        "avg": {'t_shelter': 0.3215, 't_investigating': 0.256, 'n_sh_co': 11, 'n_co_sh': 11, 'n_co_ch': 12, 'n_ch_co': 12, 'entropy': 4.166, 'laziness': 0.826},
        "std": target_std
    },
    "chd": {
        #24-pre 0.09	0.559	10	9	7	6	3.979085758	0.8335
        "avg": {'t_shelter': 0.09, 't_investigating': 0.56, 'n_sh_co': 10, 'n_co_sh': 9, 'n_co_ch': 7, 'n_ch_co': 6, 'entropy': 3.98, 'laziness': 0.834},
        "std": target_std
    }
}

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def suggest_params(trial, active_params: list[str]) -> dict:
    """Suggest only the active params; fill inactive ones with defaults."""
    params = dict(INACTIVE_PARAM_DEFAULTS)  # start with defaults for inactive
    for name in active_params:
        lo, hi = PARAM_BOUNDS[name]
        params[name] = trial.suggest_float(name, lo, hi)
    return params


def objective(trial, target_avg, target_std, output_dir, model: str = "full"):
    registry = MODEL_REGISTRY[model]
    params = suggest_params(trial, registry["active_params"])

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Trial {trial.number} ({model}) Running...", flush=True)

    try:
        history = registry["run_fn"](params)

        if history is None or len(history['agent_loc']) < 2:
            return 1000.0

        traj = pd.DataFrame({'location': history['agent_loc']})
        traj.to_csv(os.path.join(output_dir, f"trial_{trial.number}.csv"), index=False)

        m = calculate_metrics(traj)

    except Exception as e:
        print(f"Crash in Trial {trial.number}: {e}")
        return 1000.0

    loss = sum(
        WEIGHTS[k] * ((m[k] - target_avg[k]) / (target_std[k] + 1e-6)) ** 2
        for k in WEIGHTS
    )

    for k, v in m.items():
        trial.set_user_attr(k, float(v))

    return loss


# ---------------------------------------------------------------------------
# Two-phase optimisation
# ---------------------------------------------------------------------------

def run_two_phase(study, objective_fn, phase1_trials: int, phase2_trials: int,
                  model: str, seed_params: dict | None = None):
    """
    Phase 1 — random space-filling (unbiased exploration).
    Phase 2 — TPE guided search seeded from Phase 1's best.

    For the full model only, a known-good seed_params is enqueued at the
    start of Phase 1 (it occupies one of the phase1_trials slots).
    For reduced models no seed is enqueued, keeping Phase 1 neutral.
    """
    active_params = MODEL_REGISTRY[model]["active_params"]

    # --- Phase 1: random sampler ---
    print(f"\n=== Phase 1: {phase1_trials} exploratory trials (RandomSampler) ===")
    random_sampler = optuna.samplers.RandomSampler(seed=42)
    study.sampler = random_sampler

    # Enqueue known-good seed only for the full model
    if seed_params is not None and model == "full" and len(study.trials) == 0:
        # Only enqueue the subset of params that are active; Optuna ignores extras
        active_seed = {k: v for k, v in seed_params.items() if k in active_params}
        study.enqueue_trial(active_seed)

    study.optimize(objective_fn, n_trials=phase1_trials)

    # Diagnostic summary after Phase 1
    phase1_losses = [t.value for t in study.trials if t.value is not None and t.value < 1000]
    if phase1_losses:
        print(f"Phase 1 diagnostics — best: {min(phase1_losses):.3f} | "
              f"median: {np.median(phase1_losses):.3f} | "
              f"std: {np.std(phase1_losses):.3f}")
    else:
        print("Phase 1: no successful trials — check your sim.")

    # --- Phase 2: TPE guided by Phase 1 ---
    print(f"\n=== Phase 2: {phase2_trials} guided trials (TPE) ===")
    tpe_sampler = optuna.samplers.TPESampler(
        multivariate=True,
        seed=42,
        n_startup_trials=0,   # Phase 1 already provides the warm-up corpus
    )
    study.sampler = tpe_sampler
    study.optimize(objective_fn, n_trials=phase2_trials)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse", type=str, required=True,
                        choices=list(REAL_MOUSE_DATA.keys()))
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()

    mouse_name = args.mouse
    model = args.model
    registry = MODEL_REGISTRY[model]

    phase1_trials, phase2_trials = compute_budget(registry["n_params"])
    total_trials = phase1_trials + phase2_trials

    print(f"--- MODEL: {model} ({registry['n_params']} active params) | "
          f"MOUSE: {mouse_name} ---")
    print(f"Budget: {total_trials} total ({phase1_trials} exploratory + "
          f"{phase2_trials} TPE)")
    print(f"Active params: {registry['active_params']}")

    target_avg = REAL_MOUSE_DATA[mouse_name]['avg']
    target_std  = REAL_MOUSE_DATA[mouse_name]['std']

    db_url = f"sqlite:///{model}_{mouse_name}.db"
    output_dir = f"model_comparison/{model}_{mouse_name}_history"
    os.makedirs(output_dir, exist_ok=True)

    # Sampler is overridden per-phase inside run_two_phase; this is a placeholder
    study = optuna.create_study(
        study_name=f"{model}_{mouse_name}",
        storage=db_url,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler(seed=42),
    )

    objective_fn = lambda t: objective(t, target_avg, target_std, output_dir, model)

    run_two_phase(
        study=study,
        objective_fn=objective_fn,
        phase1_trials=phase1_trials,
        phase2_trials=phase2_trials,
        model=model,
        seed_params=FULL_MODEL_SEED,
    )

    print(f"\nBest loss:   {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

# Parallel runs (same pattern as before, now with --model arg):
# nice -n 10 python fit_individual.py --mouse puc --model full    > log_puc_full.txt 2>&1 &
# nice -n 10 python fit_individual.py --mouse puc --model noD     > log_puc_noD.txt  2>&1 &
# nice -n 10 python fit_individual.py --mouse puc --model noT     > log_puc_noT.txt  2>&1 &
# nice -n 10 python fit_individual.py --mouse puc --model M_only  > log_puc_Monly.txt 2>&1 &