import numpy as np
import pandas as pd
import sys
import warnings
from hmmlearn.hmm import CategoricalHMM

sys.path.insert(0, '/home/hridai/embl')
from utils import grid, calculate_metrics

# ── arena + pipeline (verbatim) ───────────────────────────────────────────────
final_mask = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
])
active_cells = np.argwhere(final_mask == 1)
bin_id_map = {tuple(pos): i for i, pos in enumerate(active_cells)}
arena = grid(final_mask)

def get_closest_bin(r, c):
    if final_mask[r, c] == 1:
        return bin_id_map[(r, c)]
    distances = np.sum((active_cells - np.array([r, c]))**2, axis=1)
    nearest_idx = np.argmin(distances)
    nearest_r, nearest_c = active_cells[nearest_idx]
    return bin_id_map[(nearest_r, nearest_c)]

def map_position(row, min_x, min_y, max_x, max_y, n_rows=5, n_cols=15, flipped=False):
    x_norm = np.clip((row['Center_x'] - min_x) / (max_x - min_x), 0, 0.999)
    y_norm = np.clip((row['Center_y'] - min_y) / (max_y - min_y), 0, 0.999)
    if flipped:
        x_norm = 0.999 - x_norm
        y_norm = 0.999 - y_norm
    r = int(np.floor(y_norm * n_rows))
    c = int(np.floor(x_norm * n_cols))
    return get_closest_bin(r, c)

def get_trajectory(df, n_steps=2000, flipped=False):
    df_filtered = df.copy()
    df_filtered['step_number'] = np.linspace(0, n_steps, len(df_filtered), endpoint=False).astype(int)
    coarse_data = df_filtered.groupby('step_number').agg(
        {'Center_x': 'mean', 'Center_y': 'mean'}).reset_index()
    min_x, max_x = df_filtered['Center_x'].min(), df_filtered['Center_x'].max()
    min_y, max_y = df_filtered['Center_y'].min(), df_filtered['Center_y'].max()
    coarse_data['location'] = coarse_data.apply(
        map_position, axis=1, args=(min_x, min_y, max_x, max_y), flipped=flipped)
    return coarse_data[['step_number', 'location']]

def trajectory_to_actions(state_sequence, arena):
    states = list(state_sequence)
    actions = np.empty(len(states) - 1, dtype=int)
    for t, (s_curr, s_next) in enumerate(zip(states[:-1], states[1:])):
        matching = [a for a in range(5) if arena.step_from_state(s_curr, a) == s_next]
        if not matching:
            raise ValueError(f"t={t}: {s_curr}->{s_next}")
        if s_curr == s_next and 4 in matching:
            actions[t] = 4
        else:
            actions[t] = matching[0]
    return actions

def clean_trajectory(state_sequence, arena, min_chunk=10):
    states = list(state_sequence)
    resolved = [states[0]]
    split_at = []
    for i in range(len(states) - 1):
        s, sn = states[i], states[i + 1]
        if any(arena.step_from_state(s, a) == sn for a in range(5)):
            resolved.append(sn)
            continue
        r0, c0 = arena.state_idx_to_rc(s)
        r1, c1 = arena.state_idx_to_rc(sn)
        dr, dc = r1 - r0, c1 - c0
        if abs(dr) == 1 and abs(dc) == 1:
            row_first = (r0 + dr, c0)
            col_first = (r0, c0 + dc)
            if (0 <= row_first[0] < arena.rows and 0 <= row_first[1] < arena.cols
                    and arena.mask[row_first[0], row_first[1]]):
                mid = arena.rc_to_state_idx(*row_first)
            elif (0 <= col_first[0] < arena.rows and 0 <= col_first[1] < arena.cols
                    and arena.mask[col_first[0], col_first[1]]):
                mid = arena.rc_to_state_idx(*col_first)
            else:
                split_at.append(len(resolved))
                resolved.append(sn)
                continue
            resolved.append(mid)
            resolved.append(sn)
        elif abs(dr) == 0 and abs(dc) == 2:
            mid = arena.rc_to_state_idx(r0, c0 + (1 if dc > 0 else -1))
            resolved.append(mid)
            resolved.append(sn)
        else:
            split_at.append(len(resolved))
            resolved.append(sn)
    boundaries = [0] + split_at + [len(resolved)]
    chunks = [resolved[boundaries[k]:boundaries[k + 1]] for k in range(len(boundaries) - 1)]
    chunks = [c for c in chunks if len(c) >= 2]
    action_seqs = [trajectory_to_actions(c, arena) for c in chunks]
    return [a for a in action_seqs if len(a) >= min_chunk]

# ── HMM fitting ───────────────────────────────────────────────────────────────
N_STATES    = 3
N_EMISSIONS = 5
N_RESTARTS  = 30
N_ITER      = 200
TOL         = 1e-4
ACTION_LABELS = ['U', 'D', 'L', 'R', 'S']

rng = np.random.default_rng(42)

def dirichlet_row(alpha, size, rng):
    row = rng.gamma(alpha, 1.0, size=size)
    return row / row.sum()

def random_init(n_states, n_emissions, rng):
    startprob    = dirichlet_row(1.0, n_states, rng)
    transmat     = np.array([dirichlet_row(1.0, n_states, rng) for _ in range(n_states)])
    emissionprob = np.array([dirichlet_row(0.5, n_emissions, rng) for _ in range(n_states)])
    return startprob, transmat, emissionprob

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

def fit_hmm(action_seqs, n_restarts, rng):
    obs     = np.concatenate(action_seqs).reshape(-1, 1)
    lengths = [len(a) for a in action_seqs]
    best_ll, best_model = -np.inf, None
    for _ in range(n_restarts):
        sp, tm, em = random_init(N_STATES, N_EMISSIONS, rng)
        model = CategoricalHMM(n_components=N_STATES, n_iter=N_ITER, tol=TOL,
                               init_params='', params='ste')
        model.startprob_    = sp
        model.transmat_     = tm
        model.emissionprob_ = em
        model.n_features    = N_EMISSIONS
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                model.fit(obs, lengths)
                ll = model.score(obs, lengths)
            except Exception:
                continue
        if ll > best_ll:
            best_ll, best_model = ll, model
    return best_model, best_ll

# ── Step 1: simulation ────────────────────────────────────────────────────────
def simulate_hmm(model, arena, start_state, n_steps=2000):
    """
    Sample an action sequence from the HMM and replay it through the arena.
    Returns a dataframe with columns step_number and location.
    """
    obs, _ = model.sample(n_steps)
    actions = obs[:, 0]
    locations = np.empty(n_steps, dtype=int)
    state = start_state
    for t, action in enumerate(actions):
        state = arena.step_from_state(state, int(action))
        locations[t] = state
    return pd.DataFrame({'step_number': np.arange(n_steps), 'location': locations})

# ── Steps 2 & 3: metrics and loss ─────────────────────────────────────────────
WEIGHTS = {
    't_shelter': 2.0, 't_investigating': 2.0,
    'n_sh_co': 0.5, 'n_co_sh': 0.5,
    'n_co_ch': 0.5, 'n_ch_co': 0.5,
    'entropy': 0.5, 'laziness': 1.0,
}
target_std = {
    't_shelter': 0.195, 't_investigating': 0.15,
    'n_sh_co': 6.48, 'n_co_sh': 6.59,
    'n_co_ch': 3.62, 'n_ch_co': 3.64,
    'entropy': 0.68, 'laziness': 0.045,
}
REAL_MOUSE_DATA = {
    "puc": {"avg": {'t_shelter': 0.824, 't_investigating': 0.057, 'n_sh_co': 15, 'n_co_sh': 15,
                    'n_co_ch': 2, 'n_ch_co': 2, 'entropy': 2.54, 'laziness': 0.94}},
    "avg": {"avg": {'t_shelter': 0.3215, 't_investigating': 0.256, 'n_sh_co': 11, 'n_co_sh': 11,
                    'n_co_ch': 12, 'n_ch_co': 12, 'entropy': 4.166, 'laziness': 0.826}},
    "chd": {"avg": {'t_shelter': 0.09, 't_investigating': 0.56, 'n_sh_co': 10, 'n_co_sh': 9,
                    'n_co_ch': 7, 'n_ch_co': 6, 'entropy': 3.98, 'laziness': 0.834}},
}

def compute_loss(metrics, target_avg):
    return sum(
        WEIGHTS[k] * ((metrics[k] - target_avg[k]) / (target_std[k] + 1e-6)) ** 2
        for k in WEIGHTS
    )

# ── main ──────────────────────────────────────────────────────────────────────
mice   = list(range(21, 27))
phases = ['def1', 'def3']
START_STATE = 6
N_SIMS = 50

# ── fit ───────────────────────────────────────────────────────────────────────
print("Fitting HMMs …")
fitted = {}
for mouse_num in mice:
    for phase in phases:
        path = f'./dlc/DLC_all_batches/m{mouse_num}_investigation_{phase}_cropped.csv'
        df = pd.read_csv(path, index_col=0)
        df = df[df['Center_p'] > 0.8]
        traj = get_trajectory(df, n_steps=2000, flipped=True)
        action_seqs = clean_trajectory(traj['location'].values, arena, min_chunk=10)
        model, ll = fit_hmm(action_seqs, N_RESTARTS, rng)
        fitted[(mouse_num, phase)] = (model, ll)
        print(f"  m{mouse_num} {phase}  LL={ll:.2f}")

# ── print emission matrices + collapse check ───────────────────────────────────
COLLAPSE_THRESH = 0.99
print("\n" + "=" * 65)
print("FITTED EMISSION MATRICES")
print("=" * 65)
for mouse_num in mice:
    for phase in phases:
        model, ll = fitted[(mouse_num, phase)]
        B = model.emissionprob_
        pairs = [(i, j) for i in range(N_STATES) for j in range(i+1, N_STATES)]
        sims  = {(i, j): cosine_sim(B[i], B[j]) for i, j in pairs}
        collapsed = [p for p, s in sims.items() if s > COLLAPSE_THRESH]
        print(f"\n── m{mouse_num} {phase}  (LL={ll:.2f}) ──")
        print(f"  {'State':<7}" + "".join(f"{lb:>7}" for lb in ACTION_LABELS))
        print("  " + "-" * (7 + 7 * N_EMISSIONS))
        for s in range(N_STATES):
            print(f"  S{s:<6}" + "".join(f"{B[s,a]:>7.3f}" for a in range(N_EMISSIONS)))
        sim_str = "  ".join(f"S{i}·S{j}={v:.3f}" for (i,j),v in sims.items())
        print(f"  cosine: {sim_str}", end="")
        if collapsed:
            print(f"  *** COLLAPSE: {collapsed}")
        else:
            print(f"  [max={max(sims.values()):.3f}]")

# ── simulate + evaluate ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"SIMULATION + EVALUATION  ({N_SIMS} runs per fitted model)")
print("=" * 65)

sim_rng = np.random.default_rng(0)

# results[session_key][target] = list of losses
results = {}
for key in fitted:
    results[key] = {t: [] for t in REAL_MOUSE_DATA}

# metric lists for breakdown: results_metrics[session_key][target] = list of metric dicts
results_metrics = {}
for key in fitted:
    results_metrics[key] = {t: [] for t in REAL_MOUSE_DATA}

for key, (model, ll) in fitted.items():
    for _ in range(N_SIMS):
        sim_df = simulate_hmm(model, arena, START_STATE, n_steps=2000)
        metrics = calculate_metrics(sim_df)
        for target_name, target_data in REAL_MOUSE_DATA.items():
            loss = compute_loss(metrics, target_data["avg"])
            results[key][target_name].append(loss)
            results_metrics[key][target_name].append(metrics)

# ── Step 4: report best session per target ────────────────────────────────────
METRIC_KEYS = list(WEIGHTS.keys())

for target_name in REAL_MOUSE_DATA:
    target_avg = REAL_MOUSE_DATA[target_name]["avg"]
    print(f"\n{'='*65}")
    print(f"TARGET: {target_name.upper()}")
    print(f"{'='*65}")

    # rank sessions by mean loss
    ranked = sorted(fitted.keys(),
                    key=lambda k: np.mean(results[k][target_name]))

    best_key = ranked[0]
    best_losses = results[best_key][target_name]
    best_metrics_list = results_metrics[best_key][target_name]

    print(f"\nBest session: m{best_key[0]} {best_key[1]}  "
          f"mean loss = {np.mean(best_losses):.3f}  std = {np.std(best_losses):.3f}")

    print(f"\nAll sessions ranked by mean loss (target={target_name}):")
    print(f"  {'Session':<12} {'Mean loss':>10} {'Std':>8}")
    print("  " + "-" * 32)
    for k in ranked:
        losses = results[k][target_name]
        marker = " <-- best" if k == best_key else ""
        print(f"  m{k[0]} {k[1]:<6}  {np.mean(losses):>10.3f} {np.std(losses):>8.3f}{marker}")

    # per-metric breakdown for best session
    mean_sim = {mk: np.mean([m[mk] for m in best_metrics_list]) for mk in METRIC_KEYS}
    std_sim  = {mk: np.std( [m[mk] for m in best_metrics_list]) for mk in METRIC_KEYS}

    print(f"\nPer-metric breakdown — best session (m{best_key[0]} {best_key[1]}) vs target {target_name}:")
    print(f"  {'Metric':<20} {'Target':>10} {'Sim mean':>10} {'Sim std':>9} {'Wt err²':>9}")
    print("  " + "-" * 62)
    for mk in METRIC_KEYS:
        t_val = target_avg[mk]
        s_val = mean_sim[mk]
        s_std = std_sim[mk]
        norm_err2 = WEIGHTS[mk] * ((s_val - t_val) / (target_std[mk] + 1e-6)) ** 2
        print(f"  {mk:<20} {t_val:>10.3f} {s_val:>10.3f} {s_std:>9.3f} {norm_err2:>9.3f}")

print("\nDone.")
