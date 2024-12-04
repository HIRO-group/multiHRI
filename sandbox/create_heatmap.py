from pathlib import Path
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3.common.utils import obs_as_tensor

from oai_agents.agents.agent_utils import DummyAgent, load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI


def get_value_function(args, observation):
    obs_tensor = obs_as_tensor(observation, args.device)
    visual_obs = obs_tensor['visual_obs'].clone().detach()
    repeated_obs = visual_obs.unsqueeze(0).repeat(args.n_envs, 1, 1, 1)
    obs_tensor['visual_obs'] = repeated_obs
    with th.no_grad():
        values = agent.policy.predict_values(obs_tensor)
    return values[0].item()


def get_tile_map(args, trajectory):
    observations = trajectory['observations']
    joint_trajectory = trajectory['positions']
    agent1_trajectory = [tr[0] for tr in joint_trajectory]

    tiles_v = np.zeros((5,  7)) # value function
    tiles_p = np.zeros((5,  7)) # position counter
    for i in range(0, len(agent1_trajectory)):
        y, x = agent1_trajectory[i]
        value = get_value_function(args, observations[i])
        tiles_v[x, y] += value
        tiles_p[x, y] += 1
    
    # normalize tiles_v
    tiles_v = tiles_v / tiles_p
    tiles_v = np.nan_to_num(tiles_v)

    return tiles_v, tiles_p


def plot_heatmap(tiles_v, tiles_p, title=''):
    plt.figure(figsize=(20, 8))  # Wider figure to accommodate two plots

    # First subplot
    plt.subplot(1, 2, 1)
    sns.heatmap(tiles_v, annot=True, cmap='YlOrRd', fmt='.0f', cbar_kws={'label': 'Value Function'})
    plt.title('value function normalized')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Second subplot
    plt.subplot(1, 2, 2)
    sns.heatmap(tiles_p, annot=True, cmap='YlOrRd', fmt='.0f', cbar_kws={'label': 'Visit Counter'})
    plt.title('visit counter')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.tight_layout()
    plt.savefig(f'data/plots/heatmap_{title}.png')
    plt.show()

if __name__ == "__main__":
    args = get_arguments()
    args.num_players = 2
    args.layout = f'{args.num_players}_chefs_counter_circuit_adv'
    args.p_idx = 0
    args.n_envs = 200
    
    path = 'agent_models/DummyADV/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM]_ran_originaler/best'
    agent = load_agent(Path(path), args) # blue

    teammates_path = [
        'agent_models/DummyADV/2/SP_hd64_seed14/ck_0', # green
        'agent_models/DummyADV/2/SP_hd64_seed14/ck_0', # orange
        'agent_models/DummyADV/2/SP_hd64_seed14/ck_0'
    ]
    teammates = [load_agent(Path(tm_path), args) for tm_path in teammates_path[:args.num_players - 1]]

    dc = OvercookedGUI(args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=1000, horizon=400)
    dc.on_execute()

    tiles_v, tiles_p = get_tile_map(args, dc.trajectory)
    plot_heatmap(tiles_v, tiles_p, title='dlmh')