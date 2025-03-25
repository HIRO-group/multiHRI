import random
import torch as th
import numpy as np

from stable_baselines3.common.utils import obs_as_tensor
from oai_agents.common.tags import TeamType
from oai_agents.agents.agent_utils import CustomAgent, DummyAgent


def not_used_function_get_tile_v_using_all_states(args, agent, layout, shape):
    '''
    This function is currently NOT used in the codebase.
    Get the value function for all possible states in the layout
    '''
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, SoupState, ObjectState
    from oai_agents.common.state_encodings import OAI_egocentric_encode_state
    import numpy as np
    from itertools import product

    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=400)

    tiles_v = np.zeros(shape=shape)
    all_valid_joint_pos = env.mdp.get_valid_joint_player_positions()

    possible_objects = [None, "dish", "onion", "soup"]
    player_object_combinations = list(product(possible_objects, repeat=args.num_players))

    for pos in all_valid_joint_pos:
        env.reset()
        for i in range(args.num_players):
            env.state.players[i].position = pos[i]

        pots = env.mdp.get_pot_states(env.state)["empty"]
        for pot_loc in pots:
            for n in range(4):
                if n == 3:
                    cooking_ticks = range(20)
                else:
                    cooking_ticks = [-1]
                for cooking_tick in cooking_ticks:
                    env.state.objects[pot_loc] = SoupState.get_soup(pot_loc, num_onions=n, cooking_tick=cooking_tick)

                    for objects_combination in player_object_combinations:
                        for player_idx, obj in enumerate(objects_combination):
                            if env.state.players[player_idx].has_object():
                                env.state.players[player_idx].remove_object()
                            if obj is None:
                                continue
                            elif obj == "soup":
                                held_obj = SoupState.get_soup(pos[player_idx], num_onions=3, finished=True)
                                env.state.players[player_idx].set_object(held_obj)
                            else:
                                held_obj = ObjectState(obj, pos[player_idx])
                                env.state.players[player_idx].set_object(held_obj)

                            obs = OAI_egocentric_encode_state(env.mdp, env.state, (7, 7), 400)
                            value = get_value_function(args=args, agent=agent, observation=obs)
                            tiles_v[pos[0][0], pos[0][1]] += value

    return tiles_v


def get_value_function(args, agent, observation):
    obs_tensor = obs_as_tensor(observation, args.device)
    visual_obs = obs_tensor['visual_obs'].clone().detach()
    # repeated_obs = visual_obs
    repeated_obs = visual_obs.unsqueeze(0).repeat(args.n_envs, 1, 1, 1)
    obs_tensor['visual_obs'] = repeated_obs
    with th.no_grad():
        values = agent.policy.predict_values(obs_tensor)
    return values[0].item()


def get_tile_map(args, agent, trajectories, p_idx, shape=(20, 20), interact_actions_only=True):
    if interact_actions_only:
        raise NotImplementedError
    tiles_p = np.zeros(shape) # position counter
    tiles_v = np.zeros(shape) # value counter
    for trajectory in trajectories:
        observations = trajectory['observations']
        joint_trajectory = trajectory['positions']
        agent1_trajectory = [tr[p_idx] for tr in joint_trajectory]
        for i in range(0, len(agent1_trajectory)):
            x, y = agent1_trajectory[i]
            tiles_p[x, y] += 1
            value = get_value_function(args=args, agent=agent, observation=observations[i])
            tiles_v[x, y] += value
    return {'P': tiles_p, 'V': tiles_v}


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
        start_position = dict.fromkeys(args.layout_names, (-1, -1))
        for layout in args.layout_names:
            start_position[layout] = [tuple(map(int, heatmap_xy_coords[layout][adv_idx]))]
        agents.append(CustomAgent(args=args, name=f'SA{adv_idx}', trajectories=start_position))
    return agents


def generate_dynamic_adversaries(args, all_tiles):
    '''
    Dynamic adversary:
    Given a heatmap, create a chain of connected tiles (positions) and stay in that region by sampling appropriate actions
    - p0: start position: the hottest spot in the heatmap
    - p1: out of all the positions connected to p0, find the next hottest spot in the heatmap thats not p0
    - p2: out of all the positions connected to p1, find the next hottest spot in the heatmap thats not p0, p1
    - pN: continue doing this until we have N connected positions
    - Given positions p0 -> p1 -> ... -> pN:
    - Randomly sample actions that enables the agent to go from p0 -> ... -> pN or pN -> .. -> p0
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


def generate_adversaries_based_on_heatmap(args, heatmap_source, teammates_collection, train_types, current_adversaries):
    from oai_agents.common.overcooked_simulation import OvercookedSimulation
    print('Heatmap source:', heatmap_source.name)
    all_tiles = {layout: {'V': [np.zeros((20, 20))], 'P': [np.zeros((20, 20))]} for layout in args.layout_names}

    for layout in args.layout_names:
        for p_idx in range(args.num_players):
            for teammates in [
                [DummyAgent(action='random') for _ in range(args.num_players - 1)], # lowest performance teammates
                [heatmap_source for _ in range(args.num_players - 1)] # highest performance teammates
            ]:
                simulation = OvercookedSimulation(args=args, agent=heatmap_source, teammates=teammates, layout_name=layout, p_idx=p_idx, horizon=400)
                trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_gen)
                tile = get_tile_map(args=args, agent=heatmap_source, p_idx=p_idx, trajectories=trajectories, interact_actions_only=False)
                all_tiles[layout]['V'][0] += tile['V']
                all_tiles[layout]['P'][0] += tile['P']

        for adversary in [item for sublist in current_adversaries.values() for item in sublist]:
            start_pos = adversary.get_start_position(layout, u_env_idx=0)
            all_tiles[layout]['V'][0][start_pos[0], start_pos[1]] = 0
            all_tiles[layout]['P'][0][start_pos[0], start_pos[1]] = 0

    adversaries = {}
    if TeamType.SELF_PLAY_STATIC_ADV in train_types:
        static_advs = generate_static_adversaries(args, all_tiles)
        adversaries[TeamType.SELF_PLAY_STATIC_ADV] = static_advs

    if TeamType.SELF_PLAY_DYNAMIC_ADV in train_types:
        dynamic_advs = generate_dynamic_adversaries(args, all_tiles)
        adversaries[TeamType.SELF_PLAY_DYNAMIC_ADV] = dynamic_advs

    return adversaries
