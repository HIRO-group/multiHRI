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
        heatmap_xy_coords[layout] = layout_heatmap_top_xy_coords

    agents = []
    for adv_idx in range(args.num_static_advs_per_heatmap):
        start_position = {layout: (-1, -1) for layout in args.layout_names}
        for layout in args.layout_names:
            start_position[layout] = tuple(map(int, heatmap_xy_coords[layout][adv_idx]))
        agents.append(CustomAgent(args=args, name=f'SA{adv_idx}', start_position=start_position, action=Action.STAY, encoding_fn=args.specialized_agent_encoding_fn))
    return agents


def generate_dynamic_adversaries(args, all_tiles):
    raise NotImplementedError


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

                simulation = OvercookedSimulation(
                    args=args, agent=heatmap_source, teammates=selected_teammates,
                    layout_name=layout, p_idx=p_idx, team_type=train_type_for_teammate, horizon=400)
                trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_gen)
                tiles_v, tiles_p = get_tile_map(args=args, agent=heatmap_source, p_idx=p_idx, trajectories=trajectories, interact_actions_only=False)

                all_tiles[layout]['V'].append(tiles_v)
                all_tiles[layout]['P'].append(tiles_p)


    adversaries = {}
    if TeamType.SELF_PLAY_STATIC_ADV in train_types:
        static_advs = generate_static_adversaries(args, all_tiles)
        adversaries[TeamType.SELF_PLAY_STATIC_ADV] = static_advs

    if TeamType.SELF_PLAY_DYNAMIC_ADV in train_types:
        raise NotImplementedError
        dynamic_advs = genereate_dynamic_adversaries(args, all_tiles)
        adversaries[TeamType.SELF_PLAY_DYNAMIC_ADV] = dynamic_advs

    return adversaries
