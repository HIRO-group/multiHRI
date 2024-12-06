from pathlib import Path
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from oai_agents.common.heatmap import get_tile_map
from oai_agents.agents.agent_utils import DummyAgent, load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI
from oai_agents.common.overcooked_simulation import OvercookedSimulation


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
    args.p_idx = 1
    args.n_envs = 200
    
    path = 'agent_models/DummyADV/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM]_ran_originaler/best'
    agent = load_agent(Path(path), args) # blue

    teammates_path = [
        'agent_models/DummyADV/2/SP_hd64_seed14/ck_0', # green
        'agent_models/DummyADV/2/SP_hd64_seed14/ck_0', # orange
        'agent_models/DummyADV/2/SP_hd64_seed14/ck_0'
    ]
    teammates = [load_agent(Path(tm_path), args) for tm_path in teammates_path[:args.num_players - 1]]

    # If you want to see the agent play then the heatmap
    # dc = OvercookedGUI(args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=1000, horizon=400)
    # dc.on_execute()
    # trajectory = dc.trajectory

    # if you just care about the heatmap
    simulation = OvercookedSimulation(args=args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, horizon=400)
    trajectory = simulation.run_simulation()

    tiles_v, tiles_p = get_tile_map(args=args, trajectory=trajectory, agent=agent)
    plot_heatmap(tiles_v, tiles_p, title='dlmh')