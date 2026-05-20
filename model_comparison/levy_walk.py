"""
Lévy walk agent for use as a statistical baseline against the active inference model.
"""

import numpy as np
import pandas as pd

# Direction index → angle in radians (standard unit circle, CCW from East)
_DIR_ANGLES = {
    0: np.pi / 2,   # Up
    1: 3 * np.pi / 2,  # Down
    2: np.pi,       # Left
    3: 0.0,         # Right
}
_L_MAX = 50


class LevyWalkAgent:
    """
    Blind agent that generates movement via a correlated Lévy walk.

    Parameters
    ----------
    mu : float
        Lévy tail exponent (1 < mu <= 3). Governs run-length distribution
        P(l) ∝ l^(-mu) over integers {1 … L_max}. Smaller mu → heavier tail
        → longer runs on average.
    kappa : float
        Directional persistence (>= 0). New run direction is sampled from a
        discrete von Mises-like distribution: P(d) ∝ exp(kappa * cos(θ_d − θ_prev)).
        kappa=0 gives uniform random direction; large kappa biases toward the
        previous heading.
    p_stay : float
        Probability in [0, 1) of issuing a Stay action (action=4) at any
        individual timestep instead of the current run's direction. Injects
        random pauses without ending the run.
    """

    def __init__(self, mu: float = 2.0, kappa: float = 1.0, p_stay: float = 0.1):
        if not (1 < mu <= 3):
            raise ValueError(f"mu must be in (1, 3], got {mu}")
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        if not (0 <= p_stay < 1):
            raise ValueError(f"p_stay must be in [0, 1), got {p_stay}")

        self.mu = mu
        self.kappa = kappa
        self.p_stay = p_stay

        # Precompute power-law weights once
        lengths = np.arange(1, _L_MAX + 1, dtype=float)
        weights = lengths ** (-mu)
        self._run_length_probs = weights / weights.sum()

        # Internal run state
        self._steps_remaining = 0
        self._current_dir = None  # None until first run is sampled
        self._run_dir_history: list[int] = []  # direction chosen at start of each run

    # ------------------------------------------------------------------
    def _sample_run_length(self) -> int:
        return int(np.random.choice(_L_MAX, p=self._run_length_probs) + 1)

    def _sample_direction(self) -> int:
        dirs = list(_DIR_ANGLES.keys())  # [0, 1, 2, 3]
        if self._current_dir is None or self.kappa == 0:
            return int(np.random.choice(dirs))

        prev_angle = _DIR_ANGLES[self._current_dir]
        logits = np.array([
            self.kappa * np.cos(_DIR_ANGLES[d] - prev_angle)
            for d in dirs
        ])
        # Subtract max for numerical stability before exp
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(np.random.choice(dirs, p=probs))

    def _start_new_run(self):
        self._current_dir = self._sample_direction()
        self._steps_remaining = self._sample_run_length()
        self._run_dir_history.append(self._current_dir)

    # ------------------------------------------------------------------
    def step(self) -> int:
        """Return the next action integer (0–4). No observations required."""
        if self._steps_remaining <= 0:
            self._start_new_run()

        self._steps_remaining -= 1

        if np.random.random() < self.p_stay:
            return 4  # Stay

        return self._current_dir


def run_simulation(agent, env, n_steps=2000):
    """Run agent in env for n_steps; return state-index trajectory of shape (n_steps,)."""
    env.start()
    states = np.empty(n_steps, dtype=int)
    for t in range(n_steps):
        action = agent.step()
        agent_obs, _threat_obs, _shelter_obs = env.step(action)
        states[t] = agent_obs
    return states


def simulate_and_score(agent, env, n_steps=2000):
    """
    Run agent in env and compute the standard behavioural metrics.

    Returns
    -------
    states : np.ndarray, shape (n_steps,)
    metrics : dict  — output of calculate_metrics (8 scalar values)
    """
    from utils import calculate_metrics

    states = run_simulation(agent, env, n_steps)
    df = pd.DataFrame({'timestep': np.arange(n_steps), 'location': states})
    metrics = calculate_metrics(df)
    return states, metrics


# ---------------------------------------------------------------------------
# Helpers for sanity checks (measure run structure directly from action stream)
# ---------------------------------------------------------------------------

def _collect_actions(agent, n_steps):
    """Step an agent n_steps times without an environment; return action array."""
    actions = np.empty(n_steps, dtype=int)
    for t in range(n_steps):
        actions[t] = agent.step()
    return actions


def _run_lengths_from_actions(actions):
    """
    Return list of run lengths from an action stream.
    A run is a maximal consecutive block of the same non-Stay action.
    Stay steps (action==4) are ignored (skipped over) so they don't split runs.
    """
    directional = actions[actions != 4]
    if len(directional) == 0:
        return []
    runs = []
    current_len = 1
    for i in range(1, len(directional)):
        if directional[i] == directional[i - 1]:
            current_len += 1
        else:
            runs.append(current_len)
            current_len = 1
    runs.append(current_len)
    return runs


def _same_dir_fraction(actions):
    """
    Extract the sequence of run directions (one per run), then return the
    fraction of consecutive run-pairs where the new run repeated the previous
    run's direction.  kappa=0 → ~0.25 (uniform); large kappa → near 1.
    """
    directional = actions[actions != 4]
    if len(directional) < 2:
        return float("nan")
    # Keep only the first step of each run → gives one direction per run
    run_starts = np.concatenate([[True], np.diff(directional) != 0])
    run_dirs = directional[run_starts]
    if len(run_dirs) < 2:
        return float("nan")
    return float(np.mean(run_dirs[:-1] == run_dirs[1:]))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils import grid, world_env, make_two_rooms_with_corridor

    np.random.seed(42)

    mask, _ = make_two_rooms_with_corridor(
        rows=5, left_cols=0, corridor_cols=9, right_cols=6,
        corridor_rows=(1, 2, 3), prefer_total_cols=None
    )
    arena = grid(mask=mask)
    cols_with_passable = np.where(mask.any(axis=0))[0]
    leftcol_states  = arena.collect_states_in_column(int(cols_with_passable.min()))
    rightcol_states = arena.collect_states_in_column(int(cols_with_passable.max()))

    env = world_env(
        arena=arena,
        true_agent_pos=(1, 0),
        true_threat_pos=rightcol_states[0],
        true_shelter_pos=np.array(leftcol_states, dtype=int),
    )

    agent = LevyWalkAgent(mu=2.0, kappa=1.0, p_stay=0.2)
    _, metrics = simulate_and_score(agent, env, n_steps=2000)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
