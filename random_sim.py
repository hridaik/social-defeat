import numpy as np
from utils import world_env, calculate_metrics
from setup import setup
import time
import pandas as pd
from random import randint

def run_random_sim(n_sims=1, gif_path=None, pkl_path=None, M_fr=0.1, D_fr=0.2, T_fr=0.2, max_steps=2000, id_threshold=0.8, \
            T_ticks=16, D_ticks=48, k_shelter=0.6, k_threat=0.8, threat_grad=[-0.10, -0.10, -0.10, -0.10, -0.10, -0.12, -0.15, -0.18, -0.21, -0.25], shelter_grad = [-0.0, 3.0], \
            delta_stay=0.15, epistemic_drive=1.0, T_scale=(-3.0, -3.0), D_scale=(5.0, 20.0), sensory_imprecision=0.75, printing=False):
    
    arena, build_scaled_C, M_agent, D_agent, T_agent, D_control_scales, T_control_scales, U_agent_base, U_shelter_base, U_threat_base, U_T, U_D, E_single, rightcol_states, leftcol_states = setup(k_shelter=k_shelter, k_threat=k_threat, \
                                                                                                                                                                                                   threat_grad=threat_grad, shelter_grad=shelter_grad, \
                                                                                                                                                                                                    delta_stay=delta_stay, epistemic_drive=epistemic_drive, \
                                                                                                                                                                                                    T_action=T_scale, D_action=D_scale, sensory_imprecision=sensory_imprecision, printing=printing)
    

    history = []

    for i in range(n_sims):
        world = world_env(arena=arena, true_agent_pos=(1, 0), true_threat_pos=rightcol_states[0], true_shelter_pos=np.array(leftcol_states, dtype=int))

        MAX_STEPS = max_steps

        agent_obs, threat_obs, shelter_obs = world.start()

        obs = [threat_obs, shelter_obs, agent_obs]

        actions = ["Up", "Down", "Left", "Right", "Stay"]

        visited = {world.starting_pos}

        locs = []
        for t in range(MAX_STEPS):

            M_action = randint(0,4)

            current_state = world.agent_pos
            locs.append(current_state)
            next_state = arena.step_from_state(current_state, M_action)
            world.agent_pos = next_state
            visited.add(next_state)

            

        history.append(locs)

        # print(f'Simulation {i} finished')


    return history

def random_metrics(n_sims):
    
    history = run_random_sim(
        n_sims=n_sims,
        max_steps=2000,
    )

    results = {'t_shelter': [], 't_investigating': [], 'n_sh_co': [], 'n_co_sh': [], 'n_ch_co': [], 'n_co_ch': [], 'entropy': [], 'laziness': []}

    for sim_locs in history:
        trajectory = pd.DataFrame({
            'timestep': range(len(sim_locs)),
            'location': sim_locs
        })

        metrics = calculate_metrics(trajectory)
        results['t_shelter'].append(metrics['t_shelter'])
        results['t_investigating'].append(metrics['t_investigating'])
        results['n_sh_co'].append(metrics['n_sh_co'])
        results['n_co_sh'].append(metrics['n_co_sh'])
        results['n_co_ch'].append(metrics['n_co_ch'])
        results['n_ch_co'].append(metrics['n_ch_co'])
        results['entropy'].append(metrics['entropy'])
        results['laziness'].append(metrics['laziness'])
    
    return results


if __name__ == "__main__":
    print("Running Simulations in Standalone Mode")
    
    total_sims = 1000
    start_time = time.time()
    history = run_random_sim(
        n_sims=total_sims,
        max_steps=2000,
    )


    print(f"Test simulation complete, time taken: {time.time() - start_time}s")

    results = {'t_shelter': [], 't_investigating': [], 'n_sh_co': [], 'n_co_sh': [], 'n_ch_co': [], 'n_co_ch': [], 'entropy': [], 'laziness': []}
    avg_results = {'t_shelter': 0, 't_investigating': 0, 'n_sh_co': 0, 'n_co_sh': 0, 'n_ch_co': 0, 'n_co_ch': 0, 'entropy': 0, 'laziness': 0}
    for sim_locs in history:
        trajectory = pd.DataFrame({
            'timestep': range(len(sim_locs)),
            'location': sim_locs
        })

        metrics = calculate_metrics(trajectory)
        results['t_shelter'].append(metrics['t_shelter'])
        results['t_investigating'].append(metrics['t_investigating'])
        results['n_sh_co'].append(metrics['n_sh_co'])
        results['n_co_sh'].append(metrics['n_co_sh'])
        results['n_co_ch'].append(metrics['n_co_ch'])
        results['n_ch_co'].append(metrics['n_ch_co'])
        results['entropy'].append(metrics['entropy'])
        results['laziness'].append(metrics['laziness'])
    
    for key, val in results.items():
        print(f'{key} : {sum(val)/total_sims}')