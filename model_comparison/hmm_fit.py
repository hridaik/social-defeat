import numpy as np
import pandas as pd
import sys
import warnings
from hmmlearn.hmm import CategoricalHMM

sys.path.insert(0, '/home/hridai/embl')
from utils import grid

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
    # apply min-chunk filter
    action_seqs = [a for a in action_seqs if len(a) >= min_chunk]
    return action_seqs

# ── HMM fitting ───────────────────────────────────────────────────────────────
N_STATES    = 3
N_EMISSIONS = 5       # actions 0-4
N_RESTARTS  = 30
N_ITER      = 200
TOL         = 1e-4
ACTION_LABELS = ['U', 'D', 'L', 'R', 'S']

rng = np.random.default_rng(42)

def dirichlet_row(alpha, size, rng):
    """Draw a single normalised row from Dirichlet(alpha * ones)."""
    row = rng.gamma(alpha, 1.0, size=size)
    return row / row.sum()

def random_init(n_states, n_emissions, rng):
    startprob = dirichlet_row(1.0, n_states, rng)
    transmat  = np.array([dirichlet_row(1.0, n_states, rng) for _ in range(n_states)])
    emissionprob = np.array([dirichlet_row(0.5, n_emissions, rng) for _ in range(n_states)])
    return startprob, transmat, emissionprob

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15))

def fit_hmm(action_seqs, n_states, n_restarts, rng):
    """
    Fit a MultinomialHMM with n_restarts random initialisations.
    Returns the best model and its log-likelihood.
    """
    obs    = np.concatenate(action_seqs).reshape(-1, 1)
    lengths = [len(a) for a in action_seqs]

    best_ll  = -np.inf
    best_model = None

    for _ in range(n_restarts):
        sp, tm, em = random_init(n_states, N_EMISSIONS, rng)
        model = CategoricalHMM(
            n_components=n_states,
            n_iter=N_ITER,
            tol=TOL,
            init_params='',   # we set all params manually
            params='ste',     # optimise startprob, transmat, emissionprob
        )
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
            best_ll    = ll
            best_model = model

    return best_model, best_ll

# ── main loop ─────────────────────────────────────────────────────────────────
mice   = list(range(21, 27))
phases = ['def1', 'def3']

fitted = {}   # (mouse, phase) -> (model, ll)

for mouse_num in mice:
    for phase in phases:
        path = f'./dlc/DLC_all_batches/m{mouse_num}_investigation_{phase}_cropped.csv'
        df = pd.read_csv(path, index_col=0)
        df = df[df['Center_p'] > 0.8]
        traj = get_trajectory(df, n_steps=2000, flipped=True)
        state_seq = traj['location'].values
        action_seqs = clean_trajectory(state_seq, arena, min_chunk=10)

        model, ll = fit_hmm(action_seqs, N_STATES, N_RESTARTS, rng)
        fitted[(mouse_num, phase)] = (model, ll)
        print(f"m{mouse_num} {phase}  LL={ll:.2f}  n_chunks={len(action_seqs)}")

# ── report ────────────────────────────────────────────────────────────────────
COLLAPSE_THRESH = 0.99

print("\n" + "=" * 70)
for mouse_num in mice:
    for phase in phases:
        model, ll = fitted[(mouse_num, phase)]
        B = model.emissionprob_          # shape (3, 5)

        # cosine similarities between all pairs of rows
        pairs = [(i, j) for i in range(N_STATES) for j in range(i+1, N_STATES)]
        sims  = {(i, j): cosine_sim(B[i], B[j]) for i, j in pairs}
        collapsed = [p for p, s in sims.items() if s > COLLAPSE_THRESH]

        print(f"\n── m{mouse_num} {phase}  (LL = {ll:.2f}) ──")
        header = f"  {'State':<8}" + "".join(f"{lb:>7}" for lb in ACTION_LABELS)
        print(header)
        print("  " + "-" * (8 + 7 * N_EMISSIONS))
        for s in range(N_STATES):
            row = f"  S{s:<7}" + "".join(f"{B[s, a]:>7.3f}" for a in range(N_EMISSIONS))
            print(row)

        if collapsed:
            for (i, j) in collapsed:
                print(f"  *** STATE COLLAPSE: S{i} & S{j} cosine sim = {sims[(i,j)]:.4f} > {COLLAPSE_THRESH}")
        else:
            max_sim = max(sims.values())
            sim_str = "  ".join(f"S{i}·S{j}={v:.3f}" for (i,j),v in sims.items())
            print(f"  Pairwise cosine sims: {sim_str}   [max={max_sim:.3f}]")

print("\n" + "=" * 70)
print("Fitting complete.")
