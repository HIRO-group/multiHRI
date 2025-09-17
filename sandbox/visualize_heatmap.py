from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from oai_agents.common.heatmap import get_tile_map
from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_simulation import OvercookedSimulation


def extract_layout_features(args):
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    grid = mdp.terrain_mtx

    layout_features = {
        "P": [],
        "O": [],
        "D": [],
        "X": [],
        "S": []
    }
    feature_positions = set()  # Store all feature coordinates for masking

    # grid_lines = [line.strip() for line in grid.strip().split("\n")]
    grid_height = len(grid)
    grid_width = max(len(line) for line in grid)  # Accounts for irregular widths

    for y, row in enumerate(grid):
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
    args.layout = 'c4'

    args.p_idx = 0
    args.n_envs = 200
    args.layout_names = [args.layout]

    path = 'agent_models/c4_best_EGO_with_CAP/best_c4_adv/best'
    # path = 'agent_models/c4_best_EGO/best_c4/best'

    agent = load_agent(Path(path), args)
    title = f'{args.layout}_{path.split("/")[-2]}'


    high_perf_paths = [
        'agent_models/c4_v4/SP_s1010_h256_tr[SP]_ran/best',
        'agent_models/c4_v3/SP_s1010_h256_tr[SP]_ran/best',
        'agent_models/c4_v2/SP_s1010_h256_tr[SP]_ran/best',
        'agent_models/c4_v1/SP_s1010_h256_tr[SP]_ran/best',
    ]
    high_perf_teammates = [[load_agent(Path(tm_path), args)] for tm_path in high_perf_paths[:args.num_players - 1]]
    # high_perf_teammates = [agent for _ in range(args.num_players - 1)]
    # low_perf_teammates = [DummyAgent(action='random') for _ in range(args.num_players - 1)]
    low_perf_teammates = []


    layout_features, feature_positions, shape = extract_layout_features(args)

    final_tiles_v = np.zeros(shape)
    for p_idx in range(args.num_players):
        for teammates in high_perf_teammates:
            simulation = OvercookedSimulation(args=args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=p_idx, horizon=400)
            trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_gen)
            tile = get_tile_map(args=args, shape=shape, agent=agent, p_idx=p_idx, trajectories=trajectories, interact_actions_only=False)
            final_tiles_v += tile['P']

    # final_tiles_v = not_used_function_get_tile_v_using_all_states(args=args, agent=agent, layout=args.layout, shape=shape)


    plot_heatmap(tiles_v=final_tiles_v, layout_features=layout_features, feature_positions=feature_positions, title=title)
