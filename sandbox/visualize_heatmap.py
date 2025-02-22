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


def extract_layout_features(grid):
    """
    Extracts layout features such as counters, pots, onions, and player starting positions.
    Returns a dictionary with their coordinates and the grid shape.
    """
    layout_features = {
        "P": [],
        "O": [],
        "D": [],
        "X": [],
        "S": []
    }
    feature_positions = set()  # Store all feature coordinates for masking

    grid_lines = [line.strip() for line in grid.strip().split("\n")]
    grid_height = len(grid_lines)
    grid_width = max(len(line) for line in grid_lines)  # Accounts for irregular widths

    for y, row in enumerate(grid_lines):
        for x, char in enumerate(row):
            if char == "P":
                layout_features["P"].append((x, y))
                feature_positions.add((x, y))
            elif char == "O":
                layout_features["O"].append((x, y))
                feature_positions.add((x, y))
            elif char == "D":
                layout_features["D"].append((x, y))
                feature_positions.add((x, y))
            elif char == "X":
                layout_features["X"].append((x, y))
                feature_positions.add((x, y))
            elif char == "S":
                layout_features["S"].append((x, y))
                feature_positions.add((x, y))

    return layout_features, feature_positions, (grid_width, grid_height)


def plot_heatmap(tiles_v, layout_features, feature_positions, title=''):
    plt.figure(figsize=(12, 6))

    # Create a custom colormap with white for zero values
    # Create more distinct separation between zero and non-zero values
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under('white')  # Color for values below vmin

    # Find the minimum non-zero value and maximum value
    non_zero_min = np.min(tiles_v[tiles_v > 0]) if np.any(tiles_v > 0) else 0.1
    max_val = np.max(tiles_v)

    # Create annotation array that matches the shape of tiles_v.T
    annot_data = np.zeros_like(tiles_v.T, dtype='<U20')
    for y in range(tiles_v.shape[1]):
        for x in range(tiles_v.shape[0]):
            if (x, y) in feature_positions:
                annot_data[y, x] = ''
            else:
                value = tiles_v[x, y]
                if value == 0:
                    annot_data[y, x] = '0'  # Explicitly show zeros
                else:
                    annot_data[y, x] = f'{value:.0f}'

    # Create the heatmap with custom normalization
    ax = sns.heatmap(
        tiles_v.T,
        annot=annot_data,
        cmap=cmap,
        fmt='',
        cbar_kws={'label': 'Value Function'},
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": 20},
        vmin=non_zero_min * 0.9,  # Set minimum slightly below smallest non-zero value
        center=None,
        robust=True
    )

    # Overlay layout features
    for feature, positions in layout_features.items():
        for (x, y) in positions:
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color="lightgrey", zorder=2))
            # if feature[0] != 'X':  # Don't add text for 'X' features
            color = 'black' if feature[0] != 'X' else 'gray'
            weight = 'bold' if feature[0] != 'X' else 'normal'
            ax.text(x + 0.5, y + 0.5, feature[0], fontsize=28, color=color,
                       ha='center', va='center', weight=weight)

    # Remove x and y ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # plt.title('Accumulated Value Function with Layout Features')
    plt.tight_layout()
    plt.savefig(f'data/plots/heatmap_{title}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    args = get_arguments()
    args.num_players = 2
    args.layout = 'storage_room'

    # grid_layout = """XXXPPXXX
    #                  X   2  X
    #                  D XXXX S
    #                  X   1  X
    #                  XXXOOXXX"""

    grid_layout = """XPXXXXXXXXPX
                     S   XODX   S
                     X    12    X
                     X   XDOX   X
                     XXXXXXXXXXXX"""

    # grid_layout = """XODSXXXXSDXX
    #                  X          X
    #                  S PP XX    X
    #                  D PP OX 1  X
    #                  O PP DX 2  X
    #                  X    SX    X
    #                  XSDOXXXXOPXX"""

    # grid_layout = """XXXPPXXX
    #                  X  2 4 X
    #                  S XXXX5S
    #                  X  1 3 X
    #                  XXDOODXX"""

    args.p_idx = 0
    args.n_envs = 200
    args.layout_names = [args.layout]

    # path = 'agent_models/Complex/5/FCP_s1010_h256_tr[AMX]_ran/last'
    # path = 'agent_models/Complex/5/SP_hd256_seed2602/last'
    # path = 'agent_models/Complex/5/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack0/last'
    # path = 'agent_models/Complex/5/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack1/last'
    # path = 'agent_models/Complex/5/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack5/last'
    path = 'agent_models/Complex_Ori/2/SP_hd256_seed13/best'
    # path = 'agent_models/Complex_Ori/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack2/best'
    # path = 'agent_models/Complex_Ori/2/N-1-SP_s2020_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack2/best'
    # path = 'agent_models/Complex_Ori/2/N-1-SP_s2602_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack2/best'

    agent = load_agent(Path(path), args)
    title = f'{args.layout}_{path.split("/")[-2]}'

    high_perf_teammates = [agent for _ in range(args.num_players - 1)]
    low_perf_teammates = [DummyAgent(action='random') for _ in range(args.num_players - 1)]

    # Define the environment grid layout (modify this based on the actual layout)

    # Extract layout features, feature positions, and shape dynamically
    layout_features, feature_positions, shape = extract_layout_features(grid_layout)

    # Initialize heatmap matrices dynamically based on extracted shape
    final_tiles_v = np.zeros(shape)
    final_tiles_p = np.zeros(shape)

    for p_idx in range(args.num_players):
        for teammates in [low_perf_teammates, high_perf_teammates]:
            simulation = OvercookedSimulation(args=args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=p_idx, horizon=400)
            trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_gen)
            tile = get_tile_map(args=args, shape=shape, agent=agent, p_idx=p_idx, trajectories=trajectories, interact_actions_only=False)
            final_tiles_v += tile['V']
            final_tiles_p += tile['P']
    final_tiles = final_tiles_v/final_tiles_p

    plot_heatmap(tiles_v=final_tiles_p, layout_features=layout_features, feature_positions=feature_positions, title=title)
