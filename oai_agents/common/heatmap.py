import random
import torch as th
import numpy as np

from stable_baselines3.common.utils import obs_as_tensor
from overcooked_ai_py.mdp.overcooked_mdp import Action
from oai_agents.common.overcooked_simulation import OvercookedSimulation
from oai_agents.common.tags import TeammatesCollection, TeamType
from oai_agents.agents.agent_utils import CustomAgent


def get_value_function(args, agent, observation):
    obs_tensor = obs_as_tensor(observation, args.device)
    visual_obs = obs_tensor['visual_obs'].clone().detach()
    repeated_obs = visual_obs.unsqueeze(0).repeat(args.n_envs, 1, 1, 1)
    obs_tensor['visual_obs'] = repeated_obs
    with th.no_grad():
        values = agent.policy.predict_values(obs_tensor)
    return values[0].item()


def get_tile_map(args, agent, trajectories, p_idx, interact_actions_only=True):
    if interact_actions_only:
        raise NotImplementedError

    tiles_v = np.zeros((20,  20)) # value function
    tiles_p = np.zeros((20,  20)) # position counter

    for trajectory in trajectories:
        observations = trajectory['observations']
        joint_trajectory = trajectory['positions']
        agent1_trajectory = [tr[p_idx] for tr in joint_trajectory]
        for i in range(0, len(agent1_trajectory)):
            x, y = agent1_trajectory[i]
            value = get_value_function(args=args, agent=agent, observation=observations[i])
            tiles_v[x, y] += value
            tiles_p[x, y] += 1
    
    tiles_v = tiles_v / tiles_p
    tiles_v = np.nan_to_num(tiles_v)
    return tiles_v, tiles_p


def generate_static_adversaries(args, all_tiles):
    mode = 'V' if args.use_val_func_for_heatmap_gen else 'P'
    heatmap_xy_coords = {layout: [] for layout in args.layout_names}
    for layout in args.layout_names:
        layout_heatmap_top_xy_coords = []
        for tiles in all_tiles[layout][mode]:
            top_n_indices = np.argsort(tiles.ravel())[-args.num_static_advs_per_heatmap:][::-1]
            top_n_coords = np.column_stack(np.unravel_index(top_n_indices, tiles.shape))
            layout_heatmap_top_xy_coords.extend(top_n_coords)
        heatmap_xy_coords[layout] = random.choices(layout_heatmap_top_xy_coords, k=args.num_static_advs_per_heatmap)
    agents = []
    for adv_idx in range(args.num_static_advs_per_heatmap):
        start_position = {layout: (-1, -1) for layout in args.layout_names}
        for layout in args.layout_names:
            start_position[layout] = [tuple(map(int, heatmap_xy_coords[layout][adv_idx]))]

        agents.append(CustomAgent(args=args, name=f'SA{adv_idx}', trajectories=start_position))
    return agents



def generate_dynamic_adversaries(args, all_tiles):
    '''
    The goal of dynamic adversary, is to enable the ego agents to learn a solution outside of their comfort zone.

    Static adversaries are helpful:
    - They render scenarios impossible thus forcing the ego agent to learn new solutions
    - But they can make the game impossible as well

    Dynamic adversaries exist to make the game harder, but not impossible. How can they do this?
    - Given a heatmap of the layout we know what are the states the ego agents are likely to visit
    - If we make it harder to visit those states, the ego agent will have to learn a new solution
    - Assumptions: we only have access to policies and value functions
    - Given: heatmap of the layout, trajectories of the ego agent
    - Output: dynamic adversaries that occupy the states the ego agent is likely to visit
    

    - Use trajectories of the agent that were used to create the heatmap to construct states s0 -> s1 -> ... -> sK (But what if such a trajectory that goes through these positions does not exist?)
    - Now, randomly sample actions that can reach these states a0 -> a1 -> ... -> aK-1
    - This way we have:
        - a dynamic adversary that occupies a certain region of the heatmap
        - by having the dynamic adversary randomly sampling actions it
    - This is the dynamic adversary

    Using the heatmap -> sample start and end positions
    Using the trajectories, we should now find a set of actions that takes us from start to end: 
    There are many trajectories that takes us from start to end, to find a trajectory easier:
    Find the highest action given the start position, then find a trajectory from that location that reaches end position
    - Why should we only find the highest action for the first state?
    - There is no clear motivation for this approach

    Only sample start positions and follow the policy for a few steps
    - How many steps should we follow? Does that matter?
    - This is just partial SP though
    - But, what if it ends up serving soups?
    - Or, what if it ends up putting onions in the pot and the other agent learns to rely on that agent?

    - Dynamic adversary:
    - p0: start position: the hottest spot in the heatmap
    - p1: out of all the positions connected to p0, find the next hottest spot in the heatmap thats not p0
    - p2: out of all the positions connected to p1, find the next hottest spot in the heatmap thats not p0, p1
    - pN: continue doing this until we have N connected positions
    - Given positions p0 -> p1 -> ... -> pN:
    - Sample all actions and see which ones take us from p0 -> p1 -> ... -> pN
    '''

    mode = 'V' if args.use_val_func_for_heatmap_gen else 'P'
    heatmap_trajectories = {layout: [] for layout in args.layout_names}
    for layout in args.layout_names:
        layout_trajectories = []
        for tiles in all_tiles[layout][mode]:
            top_1_indices = np.argsort(tiles.ravel())[-1:][::-1]
            top_1_coords = np.column_stack(np.unravel_index(top_1_indices, tiles.shape))
            trajectory = create_trajectory_from_heatmap(args=args, start_pos=top_1_coords[0], heatmap=tiles)
            layout_trajectories.append(trajectory)
        heatmap_trajectories[layout] = random.choices(layout_trajectories, k=args.num_dynamic_advs_per_heatmap)
    agents = []
    for adv_idx in range(args.num_dynamic_advs_per_heatmap):
        trajectories = {layout: [tuple(map(int, step)) for step in heatmap_trajectories[layout][adv_idx]] for layout in args.layout_names}
        agents.append(CustomAgent(args=args, name=f'DA{adv_idx}', trajectories=trajectories))
    return agents


def get_connected_positions(heatmap, start_pos):
    connected_positions = []
    rows, cols = heatmap.shape
    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1)
    ]
    cur_step_x, cur_step_y = start_pos
    for dx, dy in neighbor_offsets:
        new_x, new_y = cur_step_x + dx, cur_step_y + dy
        if 0 <= new_x < rows and 0 <= new_y < cols:
            connected_positions.append(np.array([new_x, new_y]))
    return connected_positions


def create_trajectory_from_heatmap(args, start_pos, heatmap):
    trajectory = [start_pos]
    for _ in range(args.num_steps_in_traj_for_dyn_adv):
        connected_positions = get_connected_positions(heatmap=heatmap, start_pos=trajectory[-1])
        next_connected_hottest_value, next_connected_hottest_pos = -1, (-1, -1)
        for pos in connected_positions:
            if not any(np.array_equal(pos, traj) for traj in trajectory):
                if heatmap[pos[0], pos[1]] > next_connected_hottest_value:
                    next_connected_hottest_value = heatmap[pos[0], pos[1]]
                    next_connected_hottest_pos = pos
        if next_connected_hottest_value not in [-1, 0]:
            trajectory.append(next_connected_hottest_pos)
    return trajectory


def generate_adversaries_based_on_heatmap(args, heatmap_source, teammates_collection, train_types):
    print('Heatmap source:', heatmap_source.name)
    all_tiles = {layout: {'V': [], 'P': []} for layout in args.layout_names}

    for layout in args.layout_names:
        for p_idx in range(args.num_players):
            for train_type_for_teammate in train_types:
                if train_type_for_teammate not in [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MIDDLE, TeamType.SELF_PLAY_HIGH]:
                    continue

                all_teammates_for_train_type = teammates_collection[TeammatesCollection.TRAIN][layout][train_type_for_teammate]
                selected_teammates = random.choice(all_teammates_for_train_type)

                simulation = OvercookedSimulation(args=args, agent=heatmap_source, teammates=selected_teammates, layout_name=layout, p_idx=p_idx, horizon=400)
                trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_gen)
                tiles_v, tiles_p = get_tile_map(args=args, agent=heatmap_source, p_idx=p_idx, trajectories=trajectories, interact_actions_only=False)
                
                all_tiles[layout]['V'].append(tiles_v)
                all_tiles[layout]['P'].append(tiles_p)


    adversaries = {}
    if TeamType.SELF_PLAY_STATIC_ADV in train_types:
        static_advs = generate_static_adversaries(args, all_tiles)
        adversaries[TeamType.SELF_PLAY_STATIC_ADV] = static_advs

    if TeamType.SELF_PLAY_DYNAMIC_ADV in train_types:
        dynamic_advs = generate_dynamic_adversaries(args, all_tiles)
        adversaries[TeamType.SELF_PLAY_DYNAMIC_ADV] = dynamic_advs 
    
    return adversaries
