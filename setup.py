from utils import grid, softmax, make_two_rooms_with_corridor
import numpy as np
from pymdp.agent import Agent
from itertools import product

def setup(M_policy_precision = 4.0, k_threat=0.8, k_shelter=0.6, threat_grad = [-0.10, -0.10, -0.10, -0.10, -0.10, -0.12, -0.15, -0.18, -0.21, -0.25], shelter_grad = [-0.0, 3.0], delta_stay = 0.15, epistemic_drive = 1.0, T_action = (-0.3, -0.3), D_action = (0.5, 0.7)):
    # "T" maze with the perpendicular arm on the right
    rows = 5
    left_cols = 0
    corridor_cols = 6   # corridor width
    right_cols = 9
    corridor_rows = (1,2,3)  # allow rows 1 and 2 (middle two rows), top(0) and bottom(3) are blocked in corridor
    mask, regions = make_two_rooms_with_corridor(rows=rows, left_cols=left_cols,
                                                corridor_cols=corridor_cols, right_cols=right_cols,
                                                corridor_rows=corridor_rows, prefer_total_cols=None)

    print("Arena mask shape:", mask.shape)
    print("Passable cells count:", mask.sum())

    # setup arena env
    arena = grid(mask=mask)
    n_states = arena.n_states


    # SETUP M (Motor) agent

    # observation modalities -> threat intensity/dist, shelter, agent location
    n_threat_obs = 10                         # intensities 0..3 with higher intensity is closer => can make this modular in the future
    n_shelter_obs = 13 # [NOT AT, AT]
    n_agent_obs = n_states

    actions = ["up", "down", "left", "right", "stay"]
    n_actions = len(actions)

    # Build B transition matrix

    n_agent = n_states
    n_threat = n_states
    n_shelter = n_states

    n_controls_agent = n_actions
    n_controls_threat = 1
    n_controls_shelter = 1

    # build B_agent: shape (S_next, S_current, n_controls_agent)
    B_agent = np.zeros((n_agent, n_agent, n_controls_agent), dtype=float)

    for s in range(n_agent):
        for a in range(n_actions):
            ns = arena.step_from_state(s, a)
            # make deterministic-ish
            B_agent[ns, s, a] += 0.99 # [next_state, state, action]
            # distribute small noise among other reachable neighbors (including stay)
            # find reachable neighbors from s
            neighbors = set(arena.step_from_state(s, aa) for aa in range(n_actions))
            neighbors = sorted(neighbors)
            # distribute 0.01 equally (excluding ns since we already gave 0.99)
            residue = 0.01
            for nb in neighbors:
                if nb == ns:
                    continue
                B_agent[nb, s, a] += residue / max(1, (len(neighbors)-1))

    # normalize columns for numerical safety
    for a in range(n_controls_agent):
        col_sums = B_agent[:, :, a].sum(axis=0)
        for s in range(n_agent):
            if col_sums[s] <= 0:
                B_agent[:, s, a] = 1.0 / n_agent
            else:
                B_agent[:, s, a] /= col_sums[s]

    # B for static threat and shelter (identity)
    B_threat = np.zeros((n_threat, n_threat, n_controls_threat))
    for s in range(n_threat):
        B_threat[s, s, 0] = 1.0

    B_shelter = np.zeros((n_shelter, n_shelter, n_controls_shelter))
    for s in range(n_shelter):
        B_shelter[s, s, 0] = 1.0

    B = np.empty(3, dtype=object)
    B[0] = B_agent
    B[1] = B_threat
    B[2] = B_shelter

    print("Built B with shapes:")
    print(" B_agent:", B_agent.shape, "n_states:", n_agent)


    # Build A state-observation map likelihoods

    # observation sizes
    n_threat_obs = 10
    n_shelter_obs = 13
    n_agent_obs = n_agent

    A_threat = np.zeros((n_threat_obs, n_agent, n_threat, n_shelter))
    for a in range(n_agent):
        for t in range(n_threat):
            for s in range(n_shelter):
                d = arena.manhattan_states(a, t)
                if d == 0:
                    A_threat[3, a, t, s] = 1.0
                    # A_threat[2, a, t, s] = 0.25
                elif d == 1:
                    A_threat[2, a, t, s] = 0.75
                    A_threat[1, a, t, s] = 0.25
                elif d == 2:
                    A_threat[1, a, t, s] = 0.75
                    A_threat[0, a, t, s] = 0.25
                else:
                    A_threat[0, a, t, s] = 1.0
    A_threat /= A_threat.sum(axis=0, keepdims=True)

    max_t_obs_idx = n_threat_obs - 1
    A_threat = np.zeros((n_threat_obs, n_agent, n_threat, n_shelter))
    for a in range(n_agent):
        for t in range(n_threat):
            for s in range(n_shelter):
                d = arena.manhattan_states(a, t)

                if d == 0:
                    A_threat[max_t_obs_idx, a, t, s] = 1.0
                    
                elif d < max_t_obs_idx:
                    obs_high = max_t_obs_idx - d
                    obs_low = max_t_obs_idx - d - 1
                    
                    A_threat[obs_high, a, t, s] = 0.75
                    A_threat[obs_low, a, t, s] = 0.25
                    
                else:
                    A_threat[0, a, t, s] = 1.0

    A_threat /= A_threat.sum(axis=0, keepdims=True)

    # shelter modality (is agent at shelter index?)
    max_s_obs_idx = n_shelter_obs - 1

    A_shelter = np.zeros((n_shelter_obs, n_agent, n_threat, n_shelter))

    for a in range(n_agent):
        for t in range(n_threat):
            for s in range(n_shelter):
                d = arena.manhattan_states(a, s)

                if d == 0:
                    A_shelter[max_s_obs_idx, a, t, s] = 1.0

                elif d == 1:
                    A_shelter[max_s_obs_idx - 1, a, t, s] = 0.8
                    A_shelter[max_s_obs_idx - 2, a, t, s] = 0.2

                elif d == 2:
                    A_shelter[max_s_obs_idx - 2, a, t, s] = 0.95
                    A_shelter[max_s_obs_idx - 3, a, t, s] = 0.05

                elif d < max_s_obs_idx:
                    obs_idx = max_s_obs_idx - d
                    A_shelter[obs_idx, a, t, s] = 1.0

                else:
                    A_shelter[0, a, t, s] = 1.0
    A_shelter /= A_shelter.sum(axis=0, keepdims=True)

    # agent self-location
    A_agent = np.zeros((n_agent_obs, n_agent, n_threat, n_shelter))
    for a in range(n_agent):
        A_agent[a, a, :, :] = 1.0
        # small uniform noise for others
        # A_agent[:, a, :, :] += (0.05 / (n_agent_obs - 1))
    A_agent /= A_agent.sum(axis=0, keepdims=True)

    # pack into object array
    A = np.empty(3, dtype=object)
    A[0] = A_threat
    A[1] = A_shelter
    A[2] = A_agent

    print("Built A with shapes:", [A[i].shape for i in range(3)])

    # D priors — agent & threat location probabilities spread over whole arena (no idea), shelter known
    # find columns that contain any passable cells
    cols_with_passable = np.where(mask.any(axis=0))[0]
    if cols_with_passable.size == 0:
        raise RuntimeError("Arena mask has no passable cells.")

    leftmost_col = int(cols_with_passable.min())
    rightmost_col = int(cols_with_passable.max())

    # collect states in leftmost and rightmost columns
    leftcol_states = arena.collect_states_in_column(leftmost_col)
    rightcol_states = arena.collect_states_in_column(rightmost_col)

    print(f"Leftmost column index: {leftmost_col}, states count: {len(leftcol_states)}")
    print(f"Rightmost column index: {rightmost_col}, states count: {len(rightcol_states)}")


    D_agent = np.ones(n_agent, dtype=float) / float(n_agent)
    D_threat = np.ones(n_threat, dtype=float) / float(n_threat)


    D_shelter = np.zeros(n_shelter, dtype=float)
    if len(leftcol_states) > 0:
        D_shelter[leftcol_states] = 1.0 / len(leftcol_states)


    # pack into object array for pymdp
    D = np.empty(3, dtype=object)
    D[0] = D_agent
    D[1] = D_threat
    D[2] = D_shelter

    print("D shapes:", [D[i].shape for i in range(3)])
    print("Agent prior sum:", D_agent.sum(), "Threat prior sum:", D_threat.sum(),
        "Shelter-prior mass on leftcol:", D_shelter.sum())

    # create preferences (C matrix)
    # threat_obs outcomes: [none, far, near, at] -> negative increases with closeness
    U_threat = np.array(threat_grad)  # more negative -> avoid
    U_threat *= k_threat
    C_threat = np.log(softmax(U_threat))
    # shelter
    positive_gradient = np.linspace(shelter_grad[0], shelter_grad[1], n_shelter_obs - 1)
    U_shelter = np.concatenate(([-0.1], positive_gradient))
    U_shelter *= k_shelter
    C_shelter = np.log(softmax(U_shelter))
    # agent self-location: no utility, keep zeros
    U_agent = np.zeros(n_agent_obs)
    C_agent = np.log(softmax(U_agent))


    # make C arrays of shape (n_outcomes, T) - here broadcast across T
    T = policy_len = 2
    C = [C_threat[:,None].repeat(T,axis=1),
        C_shelter[:,None].repeat(T,axis=1),
        C_agent[:,None].repeat(T,axis=1)]

    print(f"C (log pref): {C}")

    # Convert C (list of arrays shape (n_outcomes, T)) to object array with one entry per modality
    C_obj = np.empty(3, dtype=object)
    C_obj[0] = C[0]   # threat modality log-pref (shape (4,T))
    C_obj[1] = C[1]   # shelter modality log-pref (shape (2,T))
    C_obj[2] = C[2]   # agent proprio (shape (9,T))

    C = C_obj

    D_obj = np.empty(3, dtype=object)
    D_obj[0] = D_agent
    D_obj[1] = D_threat
    D_obj[2] = D_shelter

    D = D_obj

    print("C and D converted to object-arrays for pymdp.")
    print("C shapes:", [C[i].shape for i in range(len(C))])
    print("D shapes:", [D[i].shape for i in range(len(D))])


    # Generate policies and habits
    # n_control_factors = number of hidden-state factors = 3 (agent, threat dummy, shelter dummy)
    n_control_factors = 3
    num_controls = [n_controls_agent, n_controls_threat, n_controls_shelter]  # [5,1,1] (includes 1 control => nothing)

    def generate_policies(n_actions, n_control_factors, policy_len):
        policies = []
        for seq in product(range(n_actions), repeat=policy_len):
            pol = np.zeros((policy_len, n_control_factors), dtype=int)
            pol[:, 0] = seq   # agent control
            # other factors remain zeros (no control)
            policies.append(pol)
        return policies # (policy_len, n_control_factors),

    policies = generate_policies(n_actions, n_control_factors, policy_len)
    n_policies = len(policies)

    print(f"Generated {n_policies} policies (h={policy_len}, n_actions={n_actions}).")
    print("policy[0] shape:", policies[0].shape)

    # habit over stay (action 4)
    E_single = np.array([0.15, 0.15, 0.15, 0.15, (0.15 + delta_stay)])

    policies = list(product(range(n_actions), repeat=policy_len))
    E_policy = np.array([np.prod([E_single[a] for a in p]) for p in policies])
    E = E_policy/E_policy.sum()

    # Create the low-level pymdp Agent (movement controller)
    # control facets: [n_actions, 1, 1]

    M_agent = Agent(
        A = A,
        B = B,
        C = C,
        D = D,
        E = E,
        policy_len = policy_len,
        num_controls = [n_controls_agent, n_controls_threat, n_controls_shelter],
        control_fac_idx = [0],   # only the agent factor is controllable
        gamma = M_policy_precision,
        alpha = 8.0,
        action_selection='stochastic',
        use_utility = True,
        use_states_info_gain = True,
        k_ig = epistemic_drive # scalar for epistemic drive
    )

    print("M agent constructed.")


    # Create D agent (danger/safety context)

    n_danger = 2 # Danger states (NO DANGER, DANGER)
    n_obs_joint = 2 * n_threat_obs # [0-19] (no threat/threat)*(distance (0-9))

    # A observation model (likelihood): P(obs | danger)
    p_correct = 0.99
    distance_threshold = 2

    A_D = np.zeros((n_obs_joint, n_danger), dtype=float)

    for i in range(n_obs_joint // 2): # no threat * dist
        A_D[i, 0] = p_correct # no danger
        A_D[i, 1] = 1 - p_correct

    for i in range(n_obs_joint // 2, (n_obs_joint // 2) + distance_threshold):
        A_D[i, 0] = 1 - p_correct
        A_D[i, 1] = p_correct # in danger when distance_threshold steps away (0, 1)

    for i in range((n_obs_joint // 2) + distance_threshold, n_obs_joint):
        A_D[i, 0] = p_correct
        A_D[i, 1] = 1 - p_correct

    A_D = A_D / A_D.sum(axis=0, keepdims=True)

    A_D_obj = np.empty(1, dtype=object); A_D_obj[0] = A_D

    # control 0 = no-scale, control 1 = apply-scale (aims to push system toward safe)
    D_control_scales = {
        0: (1.0, 1.0),   # no scaling (threat_scale, shelter_scale)
        1: D_action,   # danger scaling: stronger negative utility for threat (but we model causal effect below)
    }
    n_controls_D = len(D_control_scales)

    # Build B transition model with shape (S_next, S_current, n_controls_D)
    # B_high[s_next, s_current, c] = P(s_next | s_current, control=c)
    B_D = np.zeros((n_danger, n_danger, n_controls_D), dtype=float)

    # Control 0: identity (no intervention)
    for s in range(n_danger):
        B_D[s, s, 0] = 1.0

    # Control 1: intervention that tends to move the system to SAFE in the next step
    #  P(next=safe | current=safe, control=1)   = 0.99
    #  P(next=danger | current=safe, control=1) = 0.01
    #  P(next=safe | current=danger, control=1) = 0.95
    #  P(next=danger | current=danger, control=1) = 0.05
    B_D[0, 0, 1] = 0.99   # safe -> safe
    B_D[1, 0, 1] = 0.01
    B_D[0, 1, 1] = 0.95   # danger -> safe (intervention helps resolve danger)
    B_D[1, 1, 1] = 0.05

    B_D_obj = np.empty(1, dtype=object)
    B_D_obj[0] = B_D

    # D_high prior (uniform)
    D_D = np.array([0.5, 0.5], dtype=float)
    D_D_obj = np.empty(1, dtype=object); D_D_obj[0] = D_D

    # utility should not matter in this case????????????
    T_D = 1
    # U_D = np.ones(n_obs_joint) / n_obs_joint
    U_D = np.zeros(n_obs_joint)
    for i in range(n_obs_joint // 2, (n_obs_joint // 2) + distance_threshold):
        U_D[i] = -10
    C_D_vec = np.log(softmax(U_D))
    # shape (n_obs_high, T_high), using T_high=1
    C_D = C_D_vec[:, None].repeat(T_D, axis=1)
    C_D_obj = np.empty(1, dtype=object); C_D_obj[0] = C_D

    # High-level agent action helper - scale low-level C from base
    U_threat_base = U_threat.copy()
    U_shelter_base = U_shelter.copy()
    U_agent_base = U_agent.copy()

    def build_scaled_C(scale):
        if isinstance(scale, dict):
            threat_s = float(scale.get('threat', 1.0))
            shelter_s = float(scale.get('shelter', 1.0))
        elif isinstance(scale, (tuple, list, np.ndarray)):
            threat_s = float(scale[0])
            shelter_s = float(scale[1])
        else:
            # scalar
            threat_s = shelter_s = float(scale)

        # scale base utilities
        U_threat_scaled = np.array(U_threat_base, dtype=float) * threat_s
        U_shelter_scaled = np.array(U_shelter_base, dtype=float) * shelter_s
        U_agent_scaled = np.array(U_agent_base, dtype=float).copy()

        # convert to log-preferences for each modality (softmax -> log)
        C_th = np.log(softmax(U_threat_scaled))[:, None].repeat(policy_len, axis=1)
        C_sh = np.log(softmax(U_shelter_scaled))[:, None].repeat(policy_len, axis=1)
        C_ag = np.log(softmax(U_agent_scaled))[:, None].repeat(policy_len, axis=1)

        C_obj_new = np.empty(3, dtype=object)
        C_obj_new[0] = C_th
        C_obj_new[1] = C_sh
        C_obj_new[2] = C_ag
        return C_obj_new

    D_agent = Agent(
        A = A_D_obj,
        B = B_D_obj,
        C = C_D_obj,
        D = D_D_obj,
        policy_len = 1,
        num_controls = [n_controls_D],
        control_fac_idx = [0],    # the only hidden factor is also controllable????????????
        gamma = 8.0,
        alpha = 8.0,
        use_utility = True,          # allow it to use preferences
        use_states_info_gain = False  # don't allow random shit
    )

    print("Danger controller constructed with controls:", list(D_control_scales.items()))


    # Create threat inference controller agent
    # hidden state factors - identity [NOT THREAT, THREAT], distance [0, 1, 2, 3]
    # observation modalities - smell [0, 1, 2, 3]
    # actions - [nothing, approach (scale down C_movement)]

    n_identity = 2
    n_dist = 10

    n_obs_smell = 10

    # A observation model (likelihood): P(obs | danger)
    A_T = np.zeros((n_obs_smell, n_identity, n_dist), dtype=float)

    # If it is a threat, smell should inversely correspond to distance
    A_T_main_w  = 0.70
    A_T_noise_w = 0.10

    M = np.zeros((n_obs_smell, n_dist), dtype=float)

    for col in range(n_dist):
        diag_row = (n_obs_smell - 1) - col # inverse diagonal position
        M[diag_row, col] = A_T_main_w

        # noise at +- 1
        if diag_row - 1 >= 0:
            M[diag_row - 1, col] = A_T_noise_w
        if diag_row + 1 < n_obs_smell:
            M[diag_row + 1, col] = A_T_noise_w

    # For not threat, high smell is confusing/contains less information
    M0 = M.copy()
    likelihood_lower_T = 0.95
    M0[:, :2] = (1 - likelihood_lower_T) * M[:, :2] + likelihood_lower_T * np.ones((n_obs_smell, 1)) / n_obs_smell


    A_T[:, 0, :] = M0
    A_T[:, 1, :] = M

    A_T /= A_T.sum(axis=0, keepdims=True)

    A_T_obj = np.empty(1, dtype=object); A_T_obj[0] = A_T

    # # A observation model (likelihood): P(obs | identity, distance)
# This scales the signal strength from 0.2 (Far/Blurry) to 0.9 (Close/Clear)
# A_T = np.zeros((n_obs_smell, n_identity, n_dist), dtype=float)

# # DYNAMIC PRECISION: 
# # We want high confidence at dist 0, low confidence at dist 9.
# # safe_weight is roughly 1/10 = 0.1
# # At dist 9: threat_weight ~ 0.15 (barely distinguishable from safe) -> Belief ~55%
# # At dist 0: threat_weight ~ 0.90 (very distinct) -> Belief ~90%

# for col in range(n_dist):
#     # Calculate weight based on distance (Linear decay)
#     # col 0 (Close) -> w ~ 0.9
#     # col 9 (Far)   -> w ~ 0.15
# slope = 0.75
# Low Slope (e.g., 0.1): "Eagle Eye." The mouse sees the threat clearly from across the room.
# High Slope (e.g., 1.5): "Myopic." The mouse sees a blurry mess until it is right next to the object.
#     w_main = 0.9 - (slope* (col / (n_dist - 1)))
    
#     # Fill Threat Column (Identity 1)
#     diag_row = (n_obs_smell - 1) - col # Inverse diagonal
    
#     # Distribute the weight
#     noise_rem = 1.0 - w_main
    
#     # Fill the column with uniform noise first
#     A_T[:, 1, col] = noise_rem / (n_obs_smell - 1)
    
#     # Set the peak (main diagonal)
#     A_T[diag_row, 1, col] = w_main

# # Fill Not Threat Column (Identity 0) - Uniform / Flat
# # A safe object smells 'random' or 'ambiguous' everywhere
# A_T[:, 0, :] = 1.0 / n_obs_smell

    # control 0 = no-scale, control 1 = apply-scale (approach/investigate)
    T_control_scales = {
        0: (1.0, 1.0),   # no scaling (threat_scale, shelter_scale)
        1: T_action,   # approach scaling
    }
    n_controls_T = len(T_control_scales)

    # Build B_T transition model with shape (n_hs, S_next, S_current, n_controls_T)
    B_T_identity = np.zeros((n_identity, n_identity, n_controls_T), dtype=float)
    B_T_dist = np.zeros((n_dist, n_dist, n_controls_T), dtype=float)

    # actions don't change identity of threat
    B_T_identity[:, :, 0] = np.eye(n_identity, n_identity)
    B_T_identity[:, :, 1] = np.eye(n_identity, n_identity)

    # null can do whatever
    B_T_dist[:, :, 0] = np.ones((n_dist, n_dist), dtype=float) / n_dist

    # approach reduces distance
    M_dist = np.zeros((n_dist, n_dist), dtype=float)
    B_T_main_w  = 0.30 # Lower timescale -> more main weight (here this is weight to reduce dist by 1)
    B_T_noise_w = 0.20 # Higher timescale -> more noise

    for col in range(n_dist):
        if col == 0:
            row = 0
        else:
            row = col - 1
        for i in range(n_dist):
            if row - i >= 0:
                M_dist[row - i, col] = B_T_noise_w
        # Add noise below (row+1), only if row+1 <= col (dist can only reduce => upper triangular)
        if row + 1 <= col and row + 1 < n_dist:
            M_dist[row + 1, col] = B_T_noise_w
        
        M_dist[row, col] = B_T_main_w

    B_T_dist[:, :, 1] = M_dist / M_dist.sum(axis=0, keepdims=True)

    B_T_obj = np.empty(2, dtype=object)
    B_T_obj[0] = B_T_identity
    B_T_obj[1] = B_T_dist

    # priors (identity uniform, dist = 3)
    D_T_identity = np.ones(n_identity) / n_identity
    # D_T_dist = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    D_T_dist = np.zeros(n_dist)
    D_T_dist[-1] = 1.0
    D_T_obj = np.empty(2, dtype=object)
    D_T_obj[0] = D_T_identity
    D_T_obj[1] = D_T_dist

    # C_high prefer safe
    T_T = 1
    U_T = np.ones(n_obs_smell) / n_obs_smell
    C_T_vec = np.log(softmax(U_T))
    C_T = C_T_vec[:, None].repeat(T_T, axis=1)
    C_T_obj = np.empty(1, dtype=object); C_T_obj[0] = C_T


    T_agent = Agent(
        A = A_T_obj,
        B = B_T_obj,
        C = C_T_obj,
        D = D_T_obj,
        policy_len = 1,
        num_controls = [n_controls_T, n_controls_T],
        control_fac_idx = [0],    # the only hidden factor is also controllable
        gamma = 8.0,
        alpha = 8.0,
        use_utility = True,          # allow it to use preferences
        use_states_info_gain = False  # allow EFE-driven choice
    )

    print("Identifier controller constructed with controls:", list(T_control_scales.items()))

    # Calculate implicit M agent params

    M_num_states = [B[f].shape[0] for f in range(len(B))]
    M_num_factors = len(M_num_states)
    M_num_obs = [A[m].shape[0] for m in range(len(A))]
    M_num_modalities = len(M_num_obs)

    A_factor_list = M_num_modalities * [list(range(M_num_factors))]
    B_factor_list = [[f] for f in range(M_num_factors)]

    M_policies = M_agent._construct_policies()

    return arena, build_scaled_C, M_agent, D_agent, T_agent, D_control_scales, T_control_scales, U_agent_base, U_shelter_base, U_threat_base, U_T, U_D, E_single, rightcol_states, leftcol_states
