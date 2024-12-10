import random
import torch as th
import numpy as np

from stable_baselines3.common.utils import obs_as_tensor

from oai_agents.common.overcooked_simulation import OvercookedSimulation
from oai_agents.common.tags import TeammatesCollection, TeamType
from oai_agents.agents.agent_utils import CustomAgent

from overcooked_ai_py.mdp.overcooked_mdp import Action

from gym import spaces
import numpy as np
import os
from pathlib import Path
import torch as th


def get_value_function(args, agent, observation):
    obs_tensor = obs_as_tensor(observation, args.device)
    visual_obs = obs_tensor['visual_obs'].clone().detach()
    repeated_obs = visual_obs.unsqueeze(0).repeat(args.n_envs, 1, 1, 1)
    obs_tensor['visual_obs'] = repeated_obs
    with th.no_grad():
        values = agent.policy.predict_values(obs_tensor)
    return values[0].item()


def get_tile_map(args, agent, trajectories, interact_actions_only=True):
    # TODO: Implement interact actions only
    
    tiles_v = np.zeros((100,  100)) # value function
    tiles_p = np.zeros((100,  100)) # position counter

    for trajectory in trajectories:
        observations = trajectory['observations']
        joint_trajectory = trajectory['positions']
        agent1_trajectory = [tr[0] for tr in joint_trajectory]
        for i in range(0, len(agent1_trajectory)):
            x, y = agent1_trajectory[i]
            value = get_value_function(args=args, agent=agent, observation=observations[i])
            tiles_v[x, y] += value
            tiles_p[x, y] += 1
    
    tiles_v = tiles_v / tiles_p
    tiles_v = np.nan_to_num(tiles_v)
    return tiles_v, tiles_p


def generate_static_adversaries(args, tiles_p, tiles_v):
    pass

def genereate_dynamic_adversaries(args, tiles_p, tiles_v):
    pass


    # args.use_value_function_for_heatmap = False
    # args.num_static_advs_per_heatmap = 2
    # args.num_dynamic_advs_per_heatmap = 2
    # args.num_eval_for_heatmap_generation = 2
    

def generate_adversaries_based_on_heatmap(args, heatmap_source, teammates_collection, train_types):
    num_adversaries_per_heatmap = 1
    p_idxes = [i for i in range(args.num_players)]
    heatmap_xy_coords = {layout: [] for layout in args.layout_names}
    for layout in args.layout_names:
        layout_heatmap_top_xy_coords = []
        for p_idx in p_idxes:
            for train_type in train_types:
                if train_type not in [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MIDDLE, TeamType.SELF_PLAY_HIGH]:
                    continue
                all_teammates = teammates_collection[TeammatesCollection.TRAIN][layout][train_type]
                selected_teammates = random.choice(all_teammates)
                
                
                simulation = OvercookedSimulation(args=args, agent=heatmap_source, teammates=selected_teammates, layout_name=layout, p_idx=p_idx, horizon=400)
                trajectories = simulation.run_simulation(how_many_times=args.num_eval_for_heatmap_generation)
                tiles_v, tiles_p = get_tile_map(args=args, agent=heatmap_source, trajectories=trajectories, interact_actions_only=True)



                top_n_indices = np.argsort(tiles_p.ravel())[-num_adversaries_per_heatmap:][::-1]
                top_n_coords = np.column_stack(np.unravel_index(top_n_indices, tiles_p.shape))
                layout_heatmap_top_xy_coords.extend(top_n_coords)
        heatmap_xy_coords[layout] = layout_heatmap_top_xy_coords

    adversaries = {
        TeamType.SELF_PLAY_STATIC_ADV: [],
        TeamType.SELF_PLAY_DYNAMIC_ADV: [],
    }
    for adv_idx in range(num_adversaries_per_heatmap):
        start_position = {layout: (-1, -1) for layout in args.layout_names}
        for layout in args.layout_names:
            start_position[layout] = heatmap_xy_coords[layout][adv_idx]

        static_blocker_agent = CustomAgent(args=args, start_position=start_position)
        adversaries[TeamType.SELF_PLAY_STATIC_ADV].append(static_blocker_agent)
    
    return adversaries