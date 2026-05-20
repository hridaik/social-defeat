import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/hridai/embl')
from utils import grid

# ── arena ────────────────────────────────────────────────────────────────────
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

# ── trajectory pipeline (verbatim) ───────────────────────────────────────────
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
    coarse_data = df_filtered.groupby('step_number').agg({
        'Center_x': 'mean', 'Center_y': 'mean'
    }).reset_index()
    min_x, max_x = df_filtered['Center_x'].min(), df_filtered['Center_x'].max()
    min_y, max_y = df_filtered['Center_y'].min(), df_filtered['Center_y'].max()
    coarse_data['location'] = coarse_data.apply(
        map_position, axis=1, args=(min_x, min_y, max_x, max_y), flipped=flipped
    )
    return coarse_data[['step_number', 'location']]

# ── action inference ──────────────────────────────────────────────────────────
ACTION_NAMES = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}

def trajectory_to_actions(state_sequence, arena):
    """
    Returns an integer array of length len(state_sequence)-1.
    Prefers Stay (4) for same-state transitions when multiple actions match.
    Raises ValueError if no action explains a transition.
    """
    states = list(state_sequence)
    actions = np.empty(len(states) - 1, dtype=int)

    for t, (s_curr, s_next) in enumerate(zip(states[:-1], states[1:])):
        matching = [a for a in range(5) if arena.step_from_state(s_curr, a) == s_next]

        if not matching:
            raise ValueError(
                f"No action explains transition at t={t}: "
                f"state {s_curr} -> {s_next} "
                f"(rc {arena.state_idx_to_rc(s_curr)} -> {arena.state_idx_to_rc(s_next)})"
            )

        if s_curr == s_next and 4 in matching:
            actions[t] = 4
        else:
            actions[t] = matching[0]

    return actions

# ── trajectory cleaning ───────────────────────────────────────────────────────
def clean_trajectory(state_sequence, arena):
    """
    Resolves diagonal (|Δr|=1, |Δc|=1) and pure-horizontal (|Δr|=0, |Δc|=2)
    bad transitions by inserting one intermediate state each.

    Diagonal: intermediate placed at (r+dr, c) — row axis first, then column.
    (0,2):    intermediate placed at (r, c+sign(dc)) — midpoint column.

    If the row-first intermediate cell for a diagonal is outside the mask,
    falls back to column-first; if both are invalid, the transition is treated
    as unresolvable.

    All remaining bad transitions cause the sequence to be split at that point;
    the jump itself is discarded (destination state starts the next chunk).

    Returns
    -------
    action_seqs : list of np.ndarray
        One action array per contiguous chunk (via trajectory_to_actions).
    n_insertions : int
    n_splits : int
    chunk_lengths : list of int  (number of actions, i.e. states - 1)
    """
    states = list(state_sequence)
    resolved = [states[0]]
    split_at = []       # indices in resolved[] where a new chunk begins
    n_insertions = 0
    n_splits = 0

    for i in range(len(states) - 1):
        s, sn = states[i], states[i + 1]

        # Already a valid single-step transition — nothing to do
        if any(arena.step_from_state(s, a) == sn for a in range(5)):
            resolved.append(sn)
            continue

        r0, c0 = arena.state_idx_to_rc(s)
        r1, c1 = arena.state_idx_to_rc(sn)
        dr, dc = r1 - r0, c1 - c0

        if abs(dr) == 1 and abs(dc) == 1:
            # Diagonal: try row-first intermediate, fall back to col-first
            row_first = (r0 + dr, c0)
            col_first = (r0, c0 + dc)
            if (0 <= row_first[0] < arena.rows and
                    0 <= row_first[1] < arena.cols and
                    arena.mask[row_first[0], row_first[1]]):
                mid = arena.rc_to_state_idx(*row_first)
            elif (0 <= col_first[0] < arena.rows and
                    0 <= col_first[1] < arena.cols and
                    arena.mask[col_first[0], col_first[1]]):
                mid = arena.rc_to_state_idx(*col_first)
            else:
                # Neither intermediate is valid — treat as split
                split_at.append(len(resolved))
                resolved.append(sn)
                n_splits += 1
                continue
            resolved.append(mid)
            resolved.append(sn)
            n_insertions += 1

        elif abs(dr) == 0 and abs(dc) == 2:
            # Pure horizontal 2-cell: midpoint column
            mid_c = c0 + (1 if dc > 0 else -1)
            mid = arena.rc_to_state_idx(r0, mid_c)
            resolved.append(mid)
            resolved.append(sn)
            n_insertions += 1

        else:
            # Unresolvable — split; destination starts the next chunk
            split_at.append(len(resolved))
            resolved.append(sn)
            n_splits += 1

    # Partition resolved sequence into contiguous chunks
    boundaries = [0] + split_at + [len(resolved)]
    chunks = [resolved[boundaries[k]:boundaries[k + 1]]
              for k in range(len(boundaries) - 1)]
    chunks = [c for c in chunks if len(c) >= 2]    # drop singletons

    action_seqs = [trajectory_to_actions(c, arena) for c in chunks]
    chunk_lengths = [len(a) for a in action_seqs]

    return action_seqs, n_insertions, n_splits, chunk_lengths

# ── main ──────────────────────────────────────────────────────────────────────
mice = list(range(21, 27))
phases = ['def1', 'def3']

all_action_seqs = {}   # (mouse, phase) -> list of action arrays
summary_rows = []

for mouse_num in mice:
    for phase in phases:
        path = f'./dlc/DLC_all_batches/m{mouse_num}_investigation_{phase}_cropped.csv'
        df = pd.read_csv(path, index_col=0)
        df = df[df['Center_p'] > 0.8]
        traj = get_trajectory(df, n_steps=2000, flipped=True)
        state_seq = traj['location'].values

        action_seqs, n_ins, n_spl, chunk_lens = clean_trajectory(state_seq, arena)
        all_action_seqs[(mouse_num, phase)] = action_seqs

        summary_rows.append({
            'mouse':       mouse_num,
            'phase':       phase,
            'insertions':  n_ins,
            'splits':      n_spl,
            'n_chunks':    len(action_seqs),
            'chunk_lens':  chunk_lens,
            'total_acts':  sum(chunk_lens),
        })

# ── per-session summary table ─────────────────────────────────────────────────
print(f"{'Mouse':>6} {'Phase':>5}  {'Ins':>5} {'Splits':>6} "
      f"{'Chunks':>7} {'TotalActs':>10}  Chunk lengths")
print("-" * 90)
for r in summary_rows:
    lens_str = str(r['chunk_lens'])
    print(f"m{r['mouse']:>4} {r['phase']:>5}  {r['insertions']:>5} {r['splits']:>6} "
          f"{r['n_chunks']:>7} {r['total_acts']:>10}  {lens_str}")

grand_ins   = sum(r['insertions'] for r in summary_rows)
grand_spl   = sum(r['splits']     for r in summary_rows)
grand_acts  = sum(r['total_acts'] for r in summary_rows)
grand_chunks = sum(r['n_chunks']  for r in summary_rows)
print("-" * 90)
print(f"{'TOTAL':>11}  {grand_ins:>5} {grand_spl:>6} {grand_chunks:>7} {grand_acts:>10}")

# ── aggregate action distribution across all cleaned sequences ────────────────
action_counts = {a: 0 for a in range(5)}
for seqs in all_action_seqs.values():
    for acts in seqs:
        for a in range(5):
            action_counts[a] += int((acts == a).sum())

total = sum(action_counts.values())
print(f"\n=== Aggregate action distribution (all mice, sessions, chunks) ===")
print(f"{'Action':<8} {'Count':>8} {'Pct':>8}")
print("-" * 26)
for a in range(5):
    pct = 100 * action_counts[a] / total if total > 0 else 0
    print(f"{ACTION_NAMES[a]:<8} {action_counts[a]:>8,} {pct:>7.2f}%")
print(f"{'TOTAL':<8} {total:>8,} {'100.00%':>8}")
