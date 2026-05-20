import optuna
import pandas as pd
import numpy as np
import os
import datetime

from utils import grid, world_env, make_two_rooms_with_corridor
from levy_walk import LevyWalkAgent, simulate_and_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Arena — built once; world_env is re-instantiated fresh inside each trial
# ---------------------------------------------------------------------------
_mask, _ = make_two_rooms_with_corridor(
    rows=5, left_cols=0, corridor_cols=9, right_cols=6,
    corridor_rows=(1, 2, 3), prefer_total_cols=None,
)
_arena = grid(mask=_mask)
_cols_with_passable = np.where(_mask.any(axis=0))[0]
_leftcol_states  = _arena.collect_states_in_column(int(_cols_with_passable.min()))
_rightcol_states = _arena.collect_states_in_column(int(_cols_with_passable.max()))


def _make_env():
    return world_env(
        arena=_arena,
        true_agent_pos=(1, 0),
        true_threat_pos=_rightcol_states[0],
        true_shelter_pos=np.array(_leftcol_states, dtype=int),
    )


# ---------------------------------------------------------------------------
# Loss components — identical to fit_reduced.py
# ---------------------------------------------------------------------------
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

REAL_MOUSE_DATA = {
    "puc": {
        # m26-post
        "avg": {'t_shelter': 0.824, 't_investigating': 0.057, 'n_sh_co': 15, 'n_co_sh': 15,
                'n_co_ch': 2, 'n_ch_co': 2, 'entropy': 2.54, 'laziness': 0.94},
        "std": target_std,
    },
    "avg": {
        # m14-post
        "avg": {'t_shelter': 0.3215, 't_investigating': 0.256, 'n_sh_co': 11, 'n_co_sh': 11,
                'n_co_ch': 12, 'n_ch_co': 12, 'entropy': 4.166, 'laziness': 0.826},
        "std": target_std,
    },
    "chd": {
        # m24-pre
        "avg": {'t_shelter': 0.09, 't_investigating': 0.56, 'n_sh_co': 10, 'n_co_sh': 9,
                'n_co_ch': 7, 'n_ch_co': 6, 'entropy': 3.98, 'laziness': 0.834},
        "std": target_std,
    },
}

SEED_TRIAL = {"mu": 2.0, "kappa": 1.5, "p_stay": 0.3}
N_STEPS = 2000


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial, target_avg, target_std, output_dir):
    mu      = trial.suggest_float("mu",     1.0, 3.0)
    kappa   = trial.suggest_float("kappa",  0.0, 3.0)
    p_stay  = trial.suggest_float("p_stay", 0.0, 0.6)

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Trial {trial.number}  mu={mu:.3f}  kappa={kappa:.3f}  p_stay={p_stay:.3f}",
          flush=True)

    try:
        env   = _make_env()
        agent = LevyWalkAgent(mu=mu, kappa=kappa, p_stay=p_stay)
        states, m = simulate_and_score(agent, env, n_steps=N_STEPS)

        if len(states) < N_STEPS:
            return 1000.0

        traj = pd.DataFrame({'timestep': np.arange(N_STEPS), 'location': states})
        traj.to_csv(os.path.join(output_dir, f"trial_{trial.number}.csv"), index=False)

    except Exception as e:
        print(f"  Crash in trial {trial.number}: {e}")
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
def run_two_phase(study, objective_fn, phase1_trials=5, phase2_trials=95):
    # Phase 1 — random exploration
    print(f"\n=== Phase 1: {phase1_trials} exploratory trials (RandomSampler) ===")
    study.sampler = optuna.samplers.RandomSampler(seed=42)
    study.optimize(objective_fn, n_trials=phase1_trials)

    phase1_losses = [t.value for t in study.trials if t.value is not None and t.value < 1000]
    if phase1_losses:
        print(f"Phase 1 diagnostics — best: {min(phase1_losses):.3f} | "
              f"median: {np.median(phase1_losses):.3f} | "
              f"std: {np.std(phase1_losses):.3f}")
    else:
        print("Phase 1: no successful trials.")

    # Phase 2 — TPE guided, seeded trial enqueued here
    print(f"\n=== Phase 2: {phase2_trials} guided trials (TPE) ===")
    study.sampler = optuna.samplers.TPESampler(multivariate=True, seed=42, n_startup_trials=0)
    study.enqueue_trial(SEED_TRIAL)
    study.optimize(objective_fn, n_trials=phase2_trials)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results_summary = {}

    for mouse_name, mouse_data in REAL_MOUSE_DATA.items():
        print(f"\n{'='*60}")
        print(f"  Fitting Lévy walk to mouse: {mouse_name}")
        print(f"{'='*60}")

        target_avg = mouse_data["avg"]
        target_std_m = mouse_data["std"]

        output_dir = f"levy_comparison/{mouse_name}_history"
        os.makedirs(output_dir, exist_ok=True)

        db_url = f"sqlite:///levy_{mouse_name}_study.db"

        study = optuna.create_study(
            study_name=f"levy_{mouse_name}",
            storage=db_url,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.RandomSampler(seed=42),
        )

        obj_fn = lambda t, ta=target_avg, ts=target_std_m, od=output_dir: objective(t, ta, ts, od)

        run_two_phase(study, obj_fn, phase1_trials=5, phase2_trials=95)

        best = study.best_trial
        results_summary[mouse_name] = {
            "loss":   best.value,
            "mu":     best.params["mu"],
            "kappa":  best.params["kappa"],
            "p_stay": best.params["p_stay"],
        }

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY — best fits per mouse")
    print(f"{'='*60}")
    print(f"{'Mouse':<8}  {'Loss':>8}  {'mu':>6}  {'kappa':>6}  {'p_stay':>7}")
    print("-" * 44)
    for mouse_name, r in results_summary.items():
        print(f"{mouse_name:<8}  {r['loss']:>8.4f}  {r['mu']:>6.3f}  {r['kappa']:>6.3f}  {r['p_stay']:>7.3f}")
