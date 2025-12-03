import numpy as np
from utils import world_env, render_grid_frame_arena, decay_qs
from setup import setup
import imageio
import pickle

def run_sim(gif_path=None, pkl_path=None, M_fr=0.1, D_fr=0.1, T_fr=0.2, max_steps=1000, id_threshold=0.8, \
            T_ticks=4, D_ticks=16, k_shelter=0.6, k_threat=0.8, threat_grad=[-0.1, -0.1, -0.2, -0.25], shelter_grad=[-0.1, 0.1, 0.15, 0.2, 0.3], \
            delta_stay=0.15, epistemic_drive=1.0, T_scale=(-0.3, -0.3), D_scale=(0.5, 0.7)):
    
    arena, build_scaled_C, M_agent, D_agent, T_agent, D_control_scales, T_control_scales, U_agent_base, U_shelter_base, U_threat_base, U_T, U_D, E_single, rightcol_states, leftcol_states = setup(k_shelter=k_shelter, k_threat=k_threat, \
                                                                                                                                                                                                   threat_grad=threat_grad, shelter_grad=shelter_grad, \
                                                                                                                                                                                                    delta_stay=delta_stay, epistemic_drive=epistemic_drive, \
                                                                                                                                                                                                    T_action=T_scale, D_action=D_scale)
    
    world = world_env(arena=arena, true_agent_pos=(1, 0), true_threat_pos=rightcol_states[0], true_shelter_pos=np.array(leftcol_states, dtype=int))

    MAX_STEPS = max_steps

    agent_obs, threat_obs, shelter_obs = world.start()

    obs = [threat_obs, shelter_obs, agent_obs]

    actions = ["Up", "Down", "Left", "Right", "Stay"]

    visited = {world.starting_pos}

    M_forgetting_rate = M_fr
    T_forgetting_rate = T_fr
    D_forgetting_rate = D_fr

    danger_detection_threshold = id_threshold

    T_ticks = 4
    D_ticks = 16

    T_ticker = T_ticks
    D_ticker = D_ticks

    base_scale = D_control_scales[0]
    current_scale = base_scale
    D_scale_name = 'SAFE'
    T_scale_name = 'DEFAULT'
    
    history = {'init': {'forgetting_rates': {'M_fr' : M_forgetting_rate, 'T_fr': T_forgetting_rate, 'D_fr': D_forgetting_rate}, 
                    'ticks': {'T_ticks': T_ticks, 'D_ticks': D_ticks},
                    'base_scale': base_scale,
                    'T_act_1': T_control_scales[1],
                    'D_act_1': D_control_scales[1],
                    'detection_threshold': danger_detection_threshold,
                    'utils': {'M': {'agent': U_agent_base, 'shelter': U_shelter_base, 'threat': U_threat_base},
                              'T': U_T,
                              'D': U_D},
                    'M_act_habits_single': E_single,
                    'threat_loc': rightcol_states[0],
                    'shelter_loc': np.array(leftcol_states),
                    }, 
           'agent_loc': [], 
                            'M_beliefs': [], 'M_util': [],  'M_info_gain': [], 'M_q_pi': [], 'M_neg_efe': [], 'M_action': [], 
                            'T_beliefs': [], 'T_util': [], 'T_info_gain': [], 'T_q_pi': [], 'T_neg_efe': [], 'T_act_t': [], 'T_action': [],
                            'D_beliefs': [], 'D_util': [], 'D_info_gain': [], 'D_q_pi': [], 'D_neg_efe': [], 'D_act_t': [], 'D_action': [],}
    # T_history = {}
    # D_history = {}

    frames = []

    for t in range(MAX_STEPS):

        # L1 inference & action selection
        M_qs = M_agent.infer_states(obs)    
        M_qpi, M_G = M_agent.infer_policies()
        M_action = M_agent.sample_action()[0]

        current_state = world.agent_pos
        next_state = arena.step_from_state(current_state, M_action)
        visited.add(next_state)

        agent_obs, threat_obs, shelter_obs = world.step(M_action) # In M gen model: Obs - T,S,A ; HS - A,T,S 
        obs = [threat_obs, shelter_obs, agent_obs]

        # Inference for T level
        T_obs = obs[0] # Threat smell
        T_qs = T_agent.infer_states([obs[0]])

        mp_A_loc = np.argmax(M_qs[0])
        mp_T_loc = np.argmax(M_qs[1])
        D_dist_obs = arena.manhattan_states(mp_T_loc, mp_A_loc)
        D_id_obs = np.argmax(T_qs[0])

        # Inference for D level    
        D_obs = min(D_dist_obs, 3) + 4*D_id_obs
        D_qs = D_agent.infer_states([D_obs])

        if (T_ticker == 0):
            T_qpi, T_G = T_agent.infer_policies()
            T_action = int(T_agent.sample_action()[0])
            history['T_act_t'].append('t')
            history['T_action'].append(T_action)
            history['T_neg_efe'].append(T_G)
            history['T_q_pi'].append(T_qpi)
            updated_scale = T_control_scales[T_action]
            # Take action => 0 = reset C(M) to default, 1 = set C(M) to approach
            if updated_scale != current_scale:
                T_scale_name = 'DEFAULT' if (updated_scale == base_scale) else 'APPROACH'
                print(f't = {t} | T changed behavior to {T_scale_name}')
                M_agent.C = build_scaled_C(updated_scale)
                current_scale = updated_scale

            T_ticker += T_ticks + 1

        if (D_ticker == 0) or (T_qs[0][0] > danger_detection_threshold):
            D_qpi, D_G = D_agent.infer_policies()
            D_action = int(D_agent.sample_action()[0])
            history['D_act_t'].append('t')
            history['D_action'].append(D_action)
            history['D_neg_efe'].append(D_G)
            history['D_q_pi'].append(D_qpi)
            updated_scale = D_control_scales[D_action]
            # Take action => 0 = reset C(M) to default, 1 = set C(M) to run        
            if updated_scale != current_scale: # T & D don't both say default/base scale
                D_scale_name = 'SAFE' if (updated_scale == base_scale) else 'DANGER'
                if D_scale_name == 'DANGER':
                    # run
                    print(f't = {t} | D changed context to {D_scale_name}')
                    M_agent.C = build_scaled_C(updated_scale)
                    current_scale = updated_scale
                else:
                    # T says approach, D thinks safe => approach
                    print(f't = {t} | D set context to {D_scale_name}')
                    # don't update current scale
            
            if D_ticker == 0:
                D_ticker += D_ticks + 1
            else:
                D_ticker += D_ticks // 2

        T_ticker -= 1
        D_ticker -= 1

        # Decay 
        M_decayed_qs = np.array([M_qs[0], decay_qs(M_qs[1], M_forgetting_rate), M_qs[2]], dtype=object)
        M_agent.reset(init_qs=M_decayed_qs)

        T_decayed_qs = np.array([decay_qs(T_qs[0], T_forgetting_rate), T_qs[1]], dtype=object)
        T_agent.reset(init_qs=T_decayed_qs)

        D_decayed_qs = np.array([decay_qs(D_qs[0], D_forgetting_rate)], dtype=object)
        D_agent.reset(init_qs=D_decayed_qs)

        if gif_path is not None:
            frame_img = render_grid_frame_arena(world.agent_pos, world.threat_pos, world.shelter_pos, visited, t,
                                    threat_posterior=M_qs[1],
                                    high_level_mode=D_scale_name,
                                    cell_size=48, arena=arena)
        
            frames.append(frame_img)

        history['agent_loc'].append(current_state)
        
        history['M_beliefs'].append(M_qs)
        history['M_neg_efe'].append(M_G)
        history['M_q_pi'].append(M_qpi)

        history['M_action'].append(actions[int(M_action)])
        
        history['T_beliefs'].append(T_qs)

        history['D_beliefs'].append(D_qs)


        # history['util'].append(M_utils)
        # history['info_gain'].append(M_igs)

    print('Simulation finished')

    if gif_path is not None:
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Saved {gif_path}")

    if pkl_path is not None:
        with open(pkl_path, 'wb') as f:
            pickle.dump(history, f)

    return history

if __name__ == "__main__":
    print("Running Simulation in Standalone Mode")
    
    GIF_PATH = "test_run.gif"
    LOG_PATH = "test_run.pkl"
    
    history = run_sim(
        gif_path=GIF_PATH,
        pkl_path=LOG_PATH,
        max_steps=100
    )
    
    print(f"Test simulation complete, saved to {GIF_PATH} and {LOG_PATH}")