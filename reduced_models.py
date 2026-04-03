import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import entropy as scipy_entropy
import pickle

from sim import run_sim          # Full model (M+T+D)
from setup import setup
from utils import world_env, decay_qs, calculate_metrics, final_mask, active_cells

# ─── No-D model (M + T only) ────────────────────────────────────────────────
# T still fires every T_ticks and can switch M to APPROACH mode.
# There is no danger/flee switching — D's cooldown and DANGER C-scaling are absent.
# M's preferences are modulated only by T (approach or default), never by a flee signal.

def run_sim_no_D(
    max_steps=2000, id_threshold=0.8,
    M_fr=0.1, T_fr=0.2,
    T_ticks=16, D_ticks=48,           # D_ticks kept for signature parity but unused
    k_shelter=0.6, k_threat=0.8,
    threat_grad=[-0.10,-0.10,-0.10,-0.10,-0.10,-0.12,-0.15,-0.18,-0.21,-0.25],
    shelter_grad=[-0.0, 3.0],
    delta_stay=0.15, epistemic_drive=1.0,
    T_scale=(-3.0, -3.0), D_scale=(5.0, 20.0),
    sensory_imprecision=0.75, printing=False, pkl_path=None
):
    arena, build_scaled_C, M_agent, _D_agent, T_agent, D_control_scales, T_control_scales, \
        *_ , rightcol_states, leftcol_states = setup(
            k_shelter=k_shelter, k_threat=k_threat,
            threat_grad=threat_grad, shelter_grad=shelter_grad,
            delta_stay=delta_stay, epistemic_drive=epistemic_drive,
            T_action=T_scale, D_action=D_scale,
            sensory_imprecision=sensory_imprecision, printing=False,
        )

    world = world_env(arena=arena, true_agent_pos=(1, 0),
                      true_threat_pos=rightcol_states[0],
                      true_shelter_pos=np.array(leftcol_states, dtype=int))

    agent_obs, threat_obs, shelter_obs = world.start()
    obs = [threat_obs, shelter_obs, agent_obs]

    base_scale   = D_control_scales[0]
    current_scale = base_scale
    T_ticker     = T_ticks

    history = {'agent_loc': [], 'M_action': [], 'context': [],
               'T_beliefs': [], 'T_action': [], 'T_act_t': []}

    for t in range(max_steps):
        M_qs = M_agent.infer_states(obs)
        M_agent.infer_policies()
        M_action = M_agent.sample_action()[0]

        current_state = world.agent_pos
        agent_obs, threat_obs, shelter_obs = world.step(M_action)
        obs = [threat_obs, shelter_obs, agent_obs]

        # T level: threat identification
        T_qs = T_agent.infer_states([obs[0]])

        # T fires every T_ticks — can only switch M to APPROACH or back to DEFAULT
        if T_ticker == 0:
            T_ticker += T_ticks + 1
            T_qpi, T_G = T_agent.infer_policies()
            T_action = int(T_agent.sample_action()[0])
            history['T_act_t'].append(t)
            history['T_action'].append(T_action)
            updated_scale = T_control_scales[T_action]
            if updated_scale != current_scale:
                label = 'DEFAULT' if updated_scale == base_scale else 'APPROACH'
                if printing: print(f't={t} | T → {label}')
                M_agent.C = build_scaled_C(updated_scale)
                current_scale = updated_scale

        T_ticker -= 1

        # Decay
        M_agent.reset(init_qs=np.array(
            [M_qs[0], decay_qs(M_qs[1], M_fr), M_qs[2]], dtype=object))
        T_agent.reset(init_qs=np.array(
            [decay_qs(T_qs[0], T_fr), T_qs[1]], dtype=object))

        ctx = 'DEFAULT' if current_scale == base_scale else 'APPROACH'
        history['agent_loc'].append(current_state)
        history['M_action'].append(int(M_action))
        history['context'].append(ctx)
        history['T_beliefs'].append(T_qs)

    if pkl_path is not None:
        with open(pkl_path, 'wb') as f:
            pickle.dump(history, f)

    return history

# ─── No-T model (M + D only) ────────────────────────────────────────────────
# T's threat-identity inference is removed. D now uses raw spatial proximity
# (from M's beliefs about threat location) and always treats the object as a
# threat (D_id_obs = 1 always → D_obs in range 10-19 of the joint obs space).
# D fires periodically (D_ticks) or immediately when d_to_cage < distance_threshold.

def run_sim_no_T(
    max_steps=2000, id_threshold=0.8,
    M_fr=0.1, D_fr=0.2,
    T_ticks=16, D_ticks=48,           # T_ticks kept for signature parity but unused
    k_shelter=0.6, k_threat=0.8,
    threat_grad=[-0.10,-0.10,-0.10,-0.10,-0.10,-0.12,-0.15,-0.18,-0.21,-0.25],
    shelter_grad=[-0.0, 3.0],
    delta_stay=0.15, epistemic_drive=1.0,
    T_scale=(-3.0, -3.0), D_scale=(5.0, 20.0),
    sensory_imprecision=0.75, printing=False, pkl_path=None
):
    arena, build_scaled_C, M_agent, D_agent, _T_agent, D_control_scales, T_control_scales, \
        *_ , rightcol_states, leftcol_states = setup(
            k_shelter=k_shelter, k_threat=k_threat,
            threat_grad=threat_grad, shelter_grad=shelter_grad,
            delta_stay=delta_stay, epistemic_drive=epistemic_drive,
            T_action=T_scale, D_action=D_scale,
            sensory_imprecision=sensory_imprecision, printing=False,
        )

    world = world_env(arena=arena, true_agent_pos=(1, 0),
                      true_threat_pos=rightcol_states[0],
                      true_shelter_pos=np.array(leftcol_states, dtype=int))

    agent_obs, threat_obs, shelter_obs = world.start()
    obs = [threat_obs, shelter_obs, agent_obs]

    base_scale    = D_control_scales[0]
    current_scale = base_scale
    D_ticker      = D_ticks
    D_cooldown    = False
    D_DISTANCE_THRESHOLD = 3  # mirrors setup.py distance_threshold

    history = {'agent_loc': [], 'M_action': [], 'context': [],
               'D_beliefs': [], 'D_action': [], 'D_act_t': []}

    for t in range(max_steps):
        M_qs = M_agent.infer_states(obs)
        M_agent.infer_policies()
        M_action = M_agent.sample_action()[0]

        current_state = world.agent_pos
        agent_obs, threat_obs, shelter_obs = world.step(M_action)
        obs = [threat_obs, shelter_obs, agent_obs]

        # D level: compute distance from M's belief about agent and threat positions
        mp_A_rc = arena.state_idx_to_rc(np.argmax(M_qs[0]))
        mp_T_rc = arena.state_idx_to_rc(np.argmax(M_qs[1]))
        threat_cluster = [
            (mp_T_rc[0],   mp_T_rc[1]),
            (mp_T_rc[0],   mp_T_rc[1]-1),
            (mp_T_rc[0]-1, mp_T_rc[1]),
            (mp_T_rc[0]-1, mp_T_rc[1]-1),
        ]
        d_to_cage = min(
            abs(mp_A_rc[0]-r) + abs(mp_A_rc[1]-c)
            for r, c in threat_cluster
            if 0 <= r < arena.rows and 0 <= c < arena.cols
        )

        # No T agent: always assume threat is real (D_id_obs = 1)
        D_obs = min(d_to_cage, 9) + 10
        D_qs  = D_agent.infer_states([D_obs])

        # D fires periodically OR when proximity < threshold (raw proximity trigger)
        proximity_trigger = (d_to_cage < D_DISTANCE_THRESHOLD) and not D_cooldown
        if (D_ticker == 0) or proximity_trigger:
            D_qpi, D_G = D_agent.infer_policies()
            D_action = int(D_agent.sample_action()[0])
            history['D_act_t'].append(t)
            history['D_action'].append(D_action)
            updated_scale = D_control_scales[D_action]
            if updated_scale != current_scale:
                D_scale_name = 'SAFE' if updated_scale == base_scale else 'DANGER'
                if D_scale_name == 'DANGER':
                    if printing: print(f't={t} | D → DANGER')
                    M_agent.C = build_scaled_C(updated_scale)
                    current_scale = updated_scale
                    D_ticker += D_ticks // 2
                    D_cooldown = True
            if D_ticker == 0:
                D_ticker += D_ticks + 1
                if D_cooldown:
                    D_cooldown = False

        D_ticker -= 1

        # Decay
        M_agent.reset(init_qs=np.array(
            [M_qs[0], decay_qs(M_qs[1], M_fr), M_qs[2]], dtype=object))
        D_agent.reset(init_qs=np.array(
            [decay_qs(D_qs[0], D_fr)], dtype=object))

        ctx = 'SAFE' if current_scale == base_scale else 'DANGER'
        history['agent_loc'].append(current_state)
        history['M_action'].append(int(M_action))
        history['context'].append(ctx)
        history['D_beliefs'].append(D_qs)

    if pkl_path is not None:
        with open(pkl_path, 'wb') as f:
            pickle.dump(history, f)

    return history

# ─── M-only model ────────────────────────────────────────────────────────────
# Motor agent alone with fixed preferences throughout.
# No T, no D — no higher-level context switching of any kind.

def run_sim_M_only(
    max_steps=2000, id_threshold=0.8,
    M_fr=0.1, D_fr=0.2, T_fr=0.2,
    T_ticks=16, D_ticks=48,
    k_shelter=0.6, k_threat=0.8,
    threat_grad=[-0.10,-0.10,-0.10,-0.10,-0.10,-0.12,-0.15,-0.18,-0.21,-0.25],
    shelter_grad=[-0.0, 3.0],
    delta_stay=0.15, epistemic_drive=1.0,
    T_scale=(-3.0, -3.0), D_scale=(5.0, 20.0),
    sensory_imprecision=0.75, printing=False, pkl_path=None
):
    arena, build_scaled_C, M_agent, *_, rightcol_states, leftcol_states = setup(
        k_shelter=k_shelter, k_threat=k_threat,
        threat_grad=threat_grad, shelter_grad=shelter_grad,
        delta_stay=delta_stay, epistemic_drive=epistemic_drive,
        T_action=T_scale, D_action=D_scale,
        sensory_imprecision=sensory_imprecision, printing=False, 
    )

    world = world_env(arena=arena, true_agent_pos=(1, 0),
                      true_threat_pos=rightcol_states[0],
                      true_shelter_pos=np.array(leftcol_states, dtype=int))

    agent_obs, threat_obs, shelter_obs = world.start()
    obs = [threat_obs, shelter_obs, agent_obs]

    history = {'agent_loc': [], 'M_action': [], 'context': []}

    for t in range(max_steps):
        M_qs = M_agent.infer_states(obs)
        M_agent.infer_policies()
        M_action = M_agent.sample_action()[0]

        current_state = world.agent_pos
        agent_obs, threat_obs, shelter_obs = world.step(M_action)
        obs = [threat_obs, shelter_obs, agent_obs]

        # Decay M beliefs only
        M_agent.reset(init_qs=np.array(
            [M_qs[0], decay_qs(M_qs[1], M_fr), M_qs[2]], dtype=object))

        history['agent_loc'].append(current_state)
        history['M_action'].append(int(M_action))
        history['context'].append('FIXED')

    if pkl_path is not None:
        with open(pkl_path, 'wb') as f:
            pickle.dump(history, f)

    return history