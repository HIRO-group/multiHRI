from oai_agents.common.arguments import get_arguments
from overcooked_ai_py.mdp.overcooked_mdp import Action
from oai_agents.common.tags import AgentPerformance

from gym import spaces
import numpy as np
import os
from pathlib import Path
import torch as th
import random


# Load any agent
def load_agent(agent_path, args=None):
    args = args or get_arguments()
    agent_path = Path(agent_path)
    try:
        load_dict = th.load(agent_path / 'agent_file', map_location=args.device, weights_only=False)
    except FileNotFoundError:
        agent_path = agent_path / 'agents_dir' / 'agent_0'
        load_dict = th.load(agent_path / 'agent_file', map_location=args.device, weights_only=False)
    agent = load_dict['agent_type'].load(agent_path, args)
    return agent

def is_held_obj(player, object):
    '''Returns True if the object that the "player" picked up / put down is the same as the "object"'''
    x, y = player.position[0] + player.orientation[0], player.position[1] + player.orientation[1]
    return player.held_object is not None and \
           ((object.name == player.held_object.name) or
            (object.name == 'soup' and player.held_object.name == 'onion'))\
           and object.position == (x, y)

class DummyPolicy:
    def __init__(self, obs_space):
        self.observation_space = obs_space

class DummyAgent():
    def __init__(self, action=Action.STAY):
        self.name = f'{action}_agent'
        self.action = action if 'random' in action else Action.ACTION_TO_INDEX[action]
        self.policy = DummyPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,))}))
        self.encoding_fn = lambda *args, **kwargs: {}
        self.use_hrl_obs = False

    def get_start_position(self, layout_name, u_env_idx):
        return None

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        add_dim = len(obs) == 1
        if self.action == 'random':
            action = np.random.randint(0, Action.NUM_ACTIONS)
        elif self.action == 'random_dir':
            action = np.random.randint(0, 4)
        else:
            action = self.action
        if add_dim:
            action = np.array([action])
        return action, None

    def set_encoding_params(self, *args, **kwargs):
        pass

    def set_obs_closure_fn(self, obs_closure_fn):
        pass


class CustomPolicy:
    def __init__(self, obs_space):
        self.observation_space = obs_space

class CustomAgent():
    def __init__(self, args, name, trajectories):
        self.args = args
        self.name = f'CA_{name}'
        self.policy = CustomPolicy(spaces.Dict({'visual_obs': spaces.Box(0,1,(1,))})) # It's only purpose is to avoid getting errors from sb3
        self.encoding_fn = lambda *args, **kwargs: {}
        self.trajectories = trajectories
        self.is_dynamic = len(self.trajectories[args.layout_names[0]]) > 1
        self.current_position = {
            layout_name: {u_env_idx: self.trajectories[layout_name][0] for u_env_idx in range(0, args.n_envs + len(args.layout_names))}
                for layout_name in args.layout_names}
        self.heading_to_end = {
            layout_name: {u_env_idx: True for u_env_idx in range(0, args.n_envs+len(args.layout_names))}
                for layout_name in args.layout_names
        } # Defines whether the agent is heading to the end of the trajectory or going back to the start
        self.layout_scores = {layout_name: -1 for layout_name in args.layout_names}
        self.layout_performance_tags = {layout_name: AgentPerformance.NOTSET for layout_name in args.layout_names}

    def get_start_position(self, layout_name, u_env_idx):
        return self.trajectories[layout_name][0]

    def reset(self):
        self.current_position = {
            layout_name: {u_env_idx: self.trajectories[layout_name][0] for u_env_idx in range(0, self.args.n_envs+len(self.args.layout_names))}
                for layout_name in self.args.layout_names}
        self.heading_to_end = {
            layout_name: {u_env_idx: True for u_env_idx in range(0, self.args.n_envs+len(self.args.layout_names))}
                for layout_name in self.args.layout_names
        }
    
    def update_current_position(self, layout_name, new_position, u_env_idx):
        self.current_position[layout_name][u_env_idx] = new_position

    def predict(self, obs, info=None, state=None, episode_start=None, deterministic=False):
        if self.is_dynamic:
            layout_name = info['layout_name']
            u_env_idx = info['u_env_idx']
            if self.current_position[layout_name][u_env_idx] == self.trajectories[layout_name][-1]:
                self.heading_to_end[layout_name][u_env_idx] = False
            elif self.current_position[layout_name][u_env_idx] == self.trajectories[layout_name][0]:
                self.heading_to_end[layout_name][u_env_idx] = True
            if self.heading_to_end[layout_name][u_env_idx]:
                next_position_idx_dx = 1
            else:
                next_position_idx_dx = -1

            cur_pos_idx = self.trajectories[layout_name].index(self.current_position[layout_name][u_env_idx])
            next_position = self.trajectories[layout_name][cur_pos_idx + next_position_idx_dx]
            action_to_move_forward = (next_position[0] - self.current_position[layout_name][u_env_idx][0], next_position[1] - self.current_position[layout_name][u_env_idx][1])
            action = random.choice([action_to_move_forward, Action.INTERACT])
        else: 
            action = Action.STAY
        
        action_idx = Action.ACTION_TO_INDEX[action]
        return action_idx, None

    def set_encoding_params(self, *args, **kwargs):
        pass

    def set_obs_closure_fn(self, obs_closure_fn):
        pass