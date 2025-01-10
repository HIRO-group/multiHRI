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
    sns.heatmap(tiles_v.T, annot=True, cmap='YlOrRd', fmt='.0f', cbar_kws={'label': 'Value Function'})
    plt.title('accumulated value function')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Second subplot
    plt.subplot(1, 2, 2)
    sns.heatmap(tiles_p.T, annot=True, cmap='YlOrRd', fmt='.0f', cbar_kws={'label': 'Visit Counter'})
    plt.title('visit counter')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.tight_layout()
    plt.savefig(f'data/plots/heatmap_{title}.png')
    plt.show()

if __name__ == "__main__":
    args = get_arguments()
    args.num_players = 2
    args.layout = f'forced_coordination'
    args.p_idx = 0
    args.n_envs = 200
    args.layout_names = [args.layout]
    
    path = 'agent_models/Classic/2/SP_hd256_seed13/best'
    agent = load_agent(Path(path), args)

    high_perf_teammates = [agent for _ in range(args.num_players - 1)]
    low_perf_teammates = [DummyAgent(action='random') for _ in range(args.num_players - 1)]

    # If you want to see the agent play then the heatmap
    # dc = OvercookedGUI(args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=1000, horizon=400)
    # dc.on_execute()
    # trajectories = [dc.trajectory]

    # If you just care about the heatmap
    final_tiles_v = np.zeros((20, 20))
    final_tiles_p = np.zeros((20, 20))
    for p_idx in range(args.num_players):
        for teammates in [low_perf_teammates, high_perf_teammates]:
            simulation = OvercookedSimulation(args=args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=p_idx, horizon=400)
            trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_gen)
            tile = get_tile_map(args=args, agent=agent, p_idx=p_idx, trajectories=trajectories, interact_actions_only=False)
            final_tiles_p += tile['P']
            final_tiles_v += tile['V']

    plot_heatmap(tiles_v=final_tiles_v, tiles_p=final_tiles_p, title=f'tile')