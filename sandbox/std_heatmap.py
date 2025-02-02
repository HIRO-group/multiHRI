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


def combine_all_stds(all_stds, new_std, new_std2):
    # Create a combined dictionary
    combined_stds = {}
    
    # Get all unique layouts
    layouts = set(list(all_stds.keys()) + list(new_std.keys()) + list(new_std2.keys()))
    
    # Combine the data
    for layout in layouts:
        combined_stds[layout] = {}
        
        # Add data from all_stds
        if layout in all_stds:
            # Clean up SP path if it's a full path
            for key, value in all_stds[layout].items():
                if isinstance(key, str) and 'agent_models' in key:
                    combined_stds[layout]['SP'] = value
                else:
                    combined_stds[layout][key] = value
        
        # Add data from new_std
        if layout in new_std:
            combined_stds[layout]['FCP'] = new_std[layout]['FCP']
            
        # Add data from new_std2
        if layout in new_std2:
            combined_stds[layout]['FCP'] = new_std2[layout]['FCP']
            
    return combined_stds


def plot_stds_by_layout(all_stds, title='Standard Deviations by Layout'):
    # Get number of layouts for subplot grid
    n_layouts = len(all_stds)
    
    # Calculate subplot grid dimensions
    n_cols = 4  # You can adjust this for different grid layouts
    n_rows = 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration if multiple rows
    axes_flat = axes.flatten()
    
    # Create a plot for each layout
    for idx, (layout_name, path_stds) in enumerate(all_stds.items()):
        ax = axes_flat[idx]
        
        # Extract paths and their corresponding STDs
        paths = list(path_stds.keys())
        stds = list(path_stds.values())
        
        # Create x-coordinates for the paths
        x = range(len(paths))
        
        # Plot the line
        ax.plot(x, stds, marker='o', linewidth=2, markersize=4)
        
        # Customize the plot
        ax.set_title(f'{layout_name}', fontsize=16)
        # ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis with actual path names
        ax.set_xticks(x)
        
        # Extract just the relevant part of the path (assuming paths are like 'agent_models/...')
        path_labels = [path.split('/')[-2] if '/' in path else path for path in paths]
        ax.set_xticklabels(path_labels, rotation=0, ha='center', fontsize=14)
        

        # set fontsize for y labels
        ax.tick_params(axis='y', labelsize=14)
        
        # Add value labels on top of points
        # for i, std in enumerate(stds):
        #     ax.annotate(f'{std:.2f}', 
        #                (i, std), 
        #                textcoords="offset points", 
        #                xytext=(0,10), 
        #                ha='center')
    
    # Remove empty subplots if any
    for idx in range(len(all_stds), len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    # Adjust layout to prevent overlap
    # plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig('data/plots/stds_by_layout.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # args = get_arguments()
    # args.num_players = 2

    # layouts = [
    #     'coordination_ring',
    #     'counter_circuit',
    #     'cramped_room',
    #     'asymmetric_advantages',
    #     'forced_coordination',

    # # 'resource_corridor',
    # # 'secret_resources'
    # ]

    # paths = [         
    #       'FCP'
    #         # 'agent_models/Complex/2/SP_hd256_seed13/last',
    #         # 'CAP 1d 1s',
    #         # 'CAP 2d 2s',
    #         # 'CAP 3d 3s'
    # ]

    # all_stds = {
    #     layout: {path: 0 for path in paths} for layout in layouts
    # }

    # for layout in layouts:

    #     args.layout = layout
    #     args.p_idx = 0
    #     args.n_envs = 200
    #     args.layout_names = [args.layout]

    #     for path in paths:
    #         agent = load_agent(Path(path), args)
    #         title = f'{args.layout}_{path.split("/")[-2]}'

    #         high_perf_teammates = [agent for _ in range(args.num_players - 1)]
    #         low_perf_teammates = [DummyAgent(action='random') for _ in range(args.num_players - 1)]


    #         shape = (20, 20)
    #         final_tiles_v = np.zeros(shape)

    #         for p_idx in range(args.num_players):
    #             for teammates in [low_perf_teammates, high_perf_teammates]:
    #                 simulation = OvercookedSimulation(args=args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=p_idx, horizon=400)
    #                 trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_gen)
    #                 tile = get_tile_map(args=args, shape=shape, agent=agent, p_idx=p_idx, trajectories=trajectories, interact_actions_only=False)
    #                 final_tiles_v += tile['V']
        
    #         std = np.std(final_tiles_v)

    #         all_stds[layout][path] = std


    # print(all_stds)


    
    all_stds = {'coordination_ring': {'SP': np.float64(2597.9749973860494), 'CAP 1d 1s': np.float64(1874.773584372634), 'CAP 2d 2s': np.float64(1703.146197683143), 'CAP 3d 3s': np.float64(1940.9894198561815)}, 'counter_circuit': {'SP': np.float64(1176.0687867763402), 'CAP 1d 1s': np.float64(876.5761172965472), 'CAP 2d 2s': np.float64(906.195828395413), 'CAP 3d 3s': np.float64(975.9509894714745)}, 'cramped_room': {'SP': np.float64(3101.058302061381), 'CAP 1d 1s': np.float64(2935.8603610460614), 'CAP 2d 2s': np.float64(2852.008231661217), 'CAP 3d 3s': np.float64(2843.650547597651)}, 'asymmetric_advantages': {'SP': np.float64(6727.689342997774), 'CAP 1d 1s': np.float64(4043.748436635012), 'CAP 2d 2s': np.float64(4290.791989126182), 'CAP 3d 3s': np.float64(4553.397944010883)}, 'forced_coordination': {'SP': np.float64(2422.2924200005555), 'CAP 1d 1s': np.float64(1465.6519257280636), 'CAP 2d 2s': np.float64(1285.0913681855034), 'CAP 3d 3s': np.float64(1276.0141657846073)}, 'resource_corridor': {'SP': np.float64(2153.2496144299603), 'CAP 1d 1s': np.float64(1432.8369943934683), 'CAP 2d 2s': np.float64(1416.5385703725715), 'CAP 3d 3s': np.float64(1431.3089462971318)}, 'secret_resources': {'SP': np.float64(7523.491741197232), 'CAP 1d 1s': np.float64(6140.533841626895), 'CAP 2d 2s': np.float64(6238.754735513586), 'CAP 3d 3s': np.float64(6366.827451955678)}}


    # filtered_
    # new_std = {'resource_corridor': {'FCP': np.float64(452.2329026050015)}, 'secret_resources': {'FCP': np.float64(3833.180683234071)}}
    # new_std2 = {'coordination_ring': {'FCP': np.float64(1962.4887767929624)}, 'counter_circuit': {'FCP': np.float64(1038.3859335477055)}, 'cramped_room': {'FCP': np.float64(2882.3580011765175)}, 'asymmetric_advantages': {'FCP': np.float64(4256.901812752819)}, 'forced_coordination': {'FCP': np.float64(1629.7905161329734)}}

    # combined_stds = combine_all_stds(all_stds, new_std, new_std2)   

    filtered_stds = {}
    for layout in all_stds.keys():
        if layout in ['counter_circuit', 'asymmetric_advantages', 'secret_resources', 'resource_corridor']:
            filtered_stds[layout] = all_stds[layout]


    plot_stds_by_layout(filtered_stds, title='Standard Deviations by Layout')