from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_args_to_save, set_args_from_load, get_arguments
from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.subtasks import calculate_completed_subtask, get_doable_subtasks, Subtasks
from oai_agents.common.tags import AgentPerformance, TeamType, KeyCheckpoints
from oai_agents.gym_environments.base_overcooked_env import USEABLE_COUNTERS

from overcooked_ai_py.mdp.overcooked_mdp import Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager

from abc import ABC, abstractmethod
import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import numpy as np
import torch as th
import torch.nn as nn
from typing import List, Tuple, Union
import stable_baselines3.common.distributions as sb3_distributions
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
import wandb
import os
import random
import re

class OAIAgent(nn.Module, ABC):
    """
    A smaller version of stable baselines Base algorithm with some small changes for my new agents
    https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm
    Ensures that all agents play nicely with the environment
    """

    def __init__(self, name, args):
        super(OAIAgent, self).__init__()
        self.name = name
        # Player index and Teammate index
        self.encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
        self.args = args
        # Must define a policy. The policy must implement a get_distribution(obs) that returns the action distribution
        self.policy = None
        # Used in overcooked-demo code
        self.p_idx = None
        self.mdp = None
        self.horizon = None
        self.prev_subtask = Subtasks.SUBTASKS_TO_IDS['unknown']
        self.use_hrl_obs = False
        self.on_reset = True

        self.layout_scores = {
            layout_name: -1 for layout_name in args.layout_names
        }
        self.layout_performance_tags = {
            layout_name: AgentPerformance.NOTSET for layout_name in args.layout_names
        }

    @abstractmethod
    def predict(self, obs: th.Tensor, state=None, episode_start=None, deterministic: bool = False) -> Tuple[
        int, Union[th.Tensor, None]]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    @abstractmethod
    def get_distribution(self, obs: th.Tensor) -> Union[th.distributions.Distribution, sb3_distributions.Distribution]:
        """
        Given an observation return the index of the action and the agent state if the agent is recurrent.
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """

    def get_obs(self, p_idx, done=False, enc_fn=None, on_reset=False, goal_objects=None):
        obs = self.encoding_fn(self.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx, goal_objects=goal_objects)

        if self.stack_frames:
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if on_reset:  # On reset
                obs['visual_obs'] = self.stackedobs.reset(obs['visual_obs'])
            else:
                obs['visual_obs'], _ = self.stackedobs.update(obs['visual_obs'], np.array([done]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()
        if 'subtask_mask' in self.policy.observation_space.keys():
            obs['subtask_mask'] = get_doable_subtasks(self.state, self.prev_subtask, self.layout_name, self.terrain, p_idx, self.valid_counters, USEABLE_COUNTERS.get(self.layout_name, 2)).astype(bool)

        obs = {k: v for k, v in obs.items() if k in self.policy.observation_space.keys()}
        return obs

    def set_encoding_params(self, p_idx, horizon, env=None, mdp=None, is_haha=False, output_message=False, tune_subtasks=False):
        self.p_idx = p_idx
        self.horizon = horizon
        if env is None:
            print(mdp, flush=True)
            assert mdp is not None
            self.mdp = mdp
            self.layout_name = mdp.layout_name
            self.obs_fn = self.get_obs
            all_counters = self.mdp.get_counter_locations()
            COUNTERS_PARAMS = {
                'start_orientations': False,
                'wait_allowed': False,
                'counter_goals': all_counters,
                'counter_drop': all_counters,
                'counter_pickup': all_counters,
                'same_motion_goals': True
            }
            self.mlam = MediumLevelActionManager.from_pickle_or_compute(mdp, COUNTERS_PARAMS, force_compute=False)
            self.valid_counters = [self.mdp.find_free_counters_valid_for_player(mdp.get_standard_start_state(), self.mlam, i)
                                   for i in range(2)]
        else:
            self.mdp = env.mdp
            self.layout_name = env.layout_name
            self.obs_fn = env.get_obs
            self.valid_counters = env.valid_counters
        self.terrain = self.mdp.terrain_mtx
        self.stack_frames = self.policy.observation_space['visual_obs'].shape[0] == (27 * self.args.num_stack)
        self.stackedobs = StackedObservations(1, self.args.num_stack, self.policy.observation_space['visual_obs'], 'first')
        if is_haha:
            self.set_play_params(output_message, tune_subtasks)
        self.on_reset = True
        self.grid_shape = (7, 7)

    def action(self, state, deterministic=False):
        if self.p_idx is None or self.mdp is None or self.horizon is None:
            raise ValueError('Please call set_encoding_params() before action. '
                             'Or, call predict with agent specific obs')

        self.state = state
        obs = self.get_obs(self.p_idx, on_reset=self.on_reset)
        self.on_reset = False

        try:
            agent_msg = self.get_agent_output()
        except AttributeError as e:
            agent_msg = ' '

        action, _ = self.predict(obs, deterministic=deterministic)
        return Action.INDEX_TO_ACTION[int(action)], agent_msg

    def _get_constructor_parameters(self):
        return dict(name=self.name, args=self.args)

    def step(self):
        pass

    def reset(self):
        self.on_reset = True

    def save(self, path: Path) -> None:
        """
        Save model to a given location.
        :param path:
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        save_path = path / 'agent_file'
        args = get_args_to_save(self.args)
        th.save({'agent_type': type(self), 'state_dict': self.state_dict(),
                 'const_params': self._get_constructor_parameters(), 'args': args}, save_path)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> 'OAIAgent':
        """
        Load model from path.
        :param path: path to save to
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = args.device
        load_path = path / 'agent_file'
        saved_variables = th.load(load_path, map_location=device)
        set_args_from_load(saved_variables['args'], args)
        saved_variables['const_params']['args'] = args
        # Create agent object
        model = cls(**saved_variables['const_params'])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables['state_dict'])
        model.to(device)
        return model


class SB3Wrapper(OAIAgent):

    def __init__(self, agent, name, args):
        super(SB3Wrapper, self).__init__(name, args)
        self.agent = agent
        self.policy = self.agent.policy
        self.num_timesteps = 0


    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        # Based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L305
        # Updated to include action masking
        obs = {k: v for k, v in obs.items() if k in self.policy.observation_space.keys()}
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.agent.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)

            actions = dist.get_actions(deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1,) + self.agent.action_space.shape)
        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def get_distribution(self, obs: th.Tensor):
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.policy.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)
        return dist

    def learn(self, epoch_timesteps):
        self.agent.learn(total_timesteps=epoch_timesteps, reset_num_timesteps=False)
        self.num_timesteps = self.agent.num_timesteps

    def save(self, path: Path) -> None:
        """
        Save model to a given location.
        :param path:
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        save_path = path / 'agent_file'
        args = get_args_to_save(self.args)
        th.save({'agent_type': type(self),
                 'sb3_model_type': type(self.agent),
                 'const_params': self._get_constructor_parameters(),
                 'args': args,
                 'layout_scores': self.layout_scores
                 }, save_path)
        self.agent.save(str(save_path) + '_sb3_agent')

    @classmethod
    def load(cls, path: Path, args: argparse.Namespace, **kwargs) -> 'SB3Wrapper':
        """
        Load model from path.
        :param path: path to save to
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = args.device
        load_path = path / 'agent_file'
        saved_variables = th.load(load_path)
        set_args_from_load(saved_variables['args'], args)
        saved_variables['const_params']['args'] = args
        # Create agent object
        agent = saved_variables['sb3_model_type'].load(str(load_path) + '_sb3_agent')
        # Create wrapper object
        model = cls(agent=agent, **saved_variables['const_params'], **kwargs)  # pytype: disable=not-instantiable
        model.layout_scores = saved_variables['layout_scores']
        model.to(device)
        return model


class SB3LSTMWrapper(SB3Wrapper):
    ''' A wrapper for a stable baselines 3 agents that uses an lstm and controls a single player '''
    def __init__(self, agent, name, args):
        super(SB3LSTMWrapper, self).__init__(agent, name, args)
        self.lstm_states = None

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        episode_start = episode_start or np.ones((1,), dtype=bool)
        action, self.lstm_states = self.agent.predict(obs, state=state, episode_start=episode_start,
                                                      deterministic=deterministic)
        return action, self.lstm_states

    def get_distribution(self, obs: th.Tensor, state=None, episode_start=None):
        # TODO I think i need to store lstm states here
        episode_start = episode_start or np.ones((1,), dtype=bool)
        return self.agent.get_distribution(obs, lstm_states=state, episode_start=episode_start)

class PolicyClone(OAIAgent):
    """
    Policy Clones are copies of other agents policies (and nothing else). They can only play the game.
    They do not support training, saving, or loading, just playing.
    """
    def __init__(self, source_agent, args, device=None, name=None):
        """
        Given a source agent, create a new agent that plays identically.
        WARNING: This just copies the replica's policy, not all the necessary training code
        """
        name = name or 'policy_clone'
        super(PolicyClone, self).__init__(name, args)
        device = device or args.device
        # Create policy object
        policy_cls = type(source_agent.policy)
        const_params = deepcopy(source_agent.policy._get_constructor_parameters())
        self.policy = policy_cls(**const_params)  # pytype: disable=not-instantiable
        # Load weights
        state_dict = deepcopy(source_agent.policy.state_dict())
        self.policy.load_state_dict(state_dict)
        self.policy.to(device)

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        # Based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L305
        # Updated to include action masking
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.policy.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)

            actions = dist.get_actions(deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1,) + self.policy.action_space.shape)
        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def get_distribution(self, obs: th.Tensor):
        self.policy.set_training_mode(False)
        obs, vectorized_env = self.policy.obs_to_tensor(obs)
        with th.no_grad():
            if 'subtask_mask' in obs and np.prod(obs['subtask_mask'].shape) == np.prod(self.policy.action_space.n):
                dist = self.policy.get_distribution(obs, action_masks=obs['subtask_mask'])
            else:
                dist = self.policy.get_distribution(obs)
        return dist

    def update_policy(self, source_agent):
        """
        Update the current agents policy using the source agents weights
        WARNING: This just copies the replica's policy fully, not all the necessary training code
        """
        state_dict = deepcopy(source_agent.policy.state_dict())
        self.policy.load_state_dict(state_dict)

    def learn(self):
        raise NotImplementedError('Learning is not supported for cloned policies')

    def save(self, path):
        raise NotImplementedError('Saving is not supported for cloned policies')

    def load(self, path, args):
        raise NotImplementedError('Loading is not supported for cloned policies')

class OAITrainer(ABC):
    """
    An abstract base class for trainer classes.
    Trainer classes must have two agents that they can train using some paradigm
    """
    def __init__(self, name, args, seed=None):
        super(OAITrainer, self).__init__()
        self.name = name
        self.args = args
        self.ck_list = []
        if seed is not None:
            os.environ['PYTHONASHSEED'] = str(seed)
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if th.cuda.is_available():
                th.cuda.manual_seed_all(seed)
            th.backends.cudnn.deterministic = True

        self.eval_teammates_collection = {}
        self.teammates_collection = {}

        # For environment splits while training
        self.n_layouts = len(self.args.layout_names)
        self.splits = []
        self.curr_split = 0
        for split_size in range(self.n_layouts):
            for split in combinations(range(self.n_layouts), split_size + 1):
                self.splits.append(split)
        self.env_setup_idx, self.weighted_ratio = 0, 0.9
        # TODO: Claim eval_envs

    def _get_constructor_parameters(self):
        return dict(name=self.name, args=self.args)

    def get_linear_schedule(self, start=1e-3, end=1e-4, end_fraction=0.8, name='lr'):
        def linear_anneal(progress_remaining: float) -> float:
            training_completed = self.agents_timesteps[0] / 2e7
            if training_completed > end_fraction:
                lr = end
            else:
                lr = start + training_completed * (end - start) / end_fraction
            if self.agents_timesteps[0] > 0:
                wandb.log({'timestep': self.agents_timesteps[0], name: lr})
            return lr
        return linear_anneal


    def evaluate(self, eval_agent, num_eps_per_layout_per_tm=5, visualize=False, timestep=None, log_wandb=True,
                 deterministic=False):

        timestep = timestep if timestep is not None else eval_agent.num_timesteps
        tot_mean_reward = []
        rew_per_layout_per_teamtype = {}
        rew_per_layout = {}

        # To reduce evaluation time: instead of evaluating all players, we randomly select three of player positions for evaluation
        # This is outside of the for loop, meaning that each time we evaluate the same player positions across all layouts for a fair comparison
        selected_p_indexes = random.sample(range(self.args.num_players), min(3, self.args.num_players))

        for _, env in enumerate(self.eval_envs):
            rew_per_layout_per_teamtype[env.layout_name] = {
                teamtype: [] for teamtype in self.eval_teammates_collection[env.layout_name]
            }
            rew_per_layout[env.layout_name] = 0

            teamtypes_population = self.eval_teammates_collection[env.layout_name]

            for teamtype in teamtypes_population:
                teammates = teamtypes_population[teamtype][np.random.randint(len(teamtypes_population[teamtype]))]
                env.set_teammates(teammates)

                for p_idx in selected_p_indexes:
                    env.set_reset_p_idx(p_idx)
                    mean_reward, std_reward = evaluate_policy(eval_agent, env, n_eval_episodes=num_eps_per_layout_per_tm,
                                                              deterministic=deterministic, warn=False, render=visualize)
                    tot_mean_reward.append(mean_reward)
                    rew_per_layout_per_teamtype[env.layout_name][teamtype].append(mean_reward)


            rew_per_layout_per_teamtype[env.layout_name] = {teamtype: np.mean(rew_per_layout_per_teamtype[env.layout_name][teamtype]) for teamtype in rew_per_layout_per_teamtype[env.layout_name]}
            rew_per_layout[env.layout_name] = np.mean([rew_per_layout_per_teamtype[env.layout_name][teamtype] for teamtype in rew_per_layout_per_teamtype[env.layout_name]])

            if log_wandb:
                wandb.log({f'eval_mean_reward_{env.layout_name}': rew_per_layout[env.layout_name], 'timestep': timestep})
                for teamtype in rew_per_layout_per_teamtype[env.layout_name]:
                    wandb.log({f'eval_mean_reward_{env.layout_name}_teamtype_{teamtype}': rew_per_layout_per_teamtype[env.layout_name][teamtype], 'timestep': timestep})

        if log_wandb:
            wandb.log({f'eval_mean_reward': np.mean(tot_mean_reward), 'timestep': timestep})
        return np.mean(tot_mean_reward), rew_per_layout


    def set_new_teammates(self, curriculum):
        for i in range(self.args.n_envs):
            layout_name = self.env.env_method('get_layout_name', indices=i)[0]
            population_teamtypes = self.teammates_collection[layout_name]

            teammates = curriculum.select_teammates(population_teamtypes=population_teamtypes)

            assert len(teammates) == self.args.teammates_len
            assert type(teammates) == list

            for teammate in teammates:
                assert isinstance(teammate, SB3Wrapper)

            self.env.env_method('set_teammates', teammates, indices=i)


    def get_agents(self) -> List[OAIAgent]:
        """
        Structure should be the same as agents created using stable baselines:
        https://stable-baselines3.readthedocs.io/en/master/modules/base.html#stable_baselines3.common.base_class.BaseAlgorithm.predict
        """
        return self.agents

    def save_agents(self, path: Union[Path, None] = None, tag: Union[str, None] = None):
        ''' Saves each agent that the trainer is training '''
        if not path:
            if self.args.exp_dir:
                path = self.args.base_dir / 'agent_models' / self.args.exp_dir / self.name
            else:
                path = self.args.base_dir / 'agent_models'/ self.name

        tag = tag or self.args.exp_name
        save_path = path / tag / 'trainer_file'
        agent_path = path / tag / 'agents_dir'
        Path(agent_path).mkdir(parents=True, exist_ok=True)
        save_dict = {'agent_fns': []}
        for i, agent in enumerate(self.agents):
            agent_path_i = agent_path / f'agent_{i}'
            agent.save(agent_path_i)
            save_dict['agent_fns'].append(f'agent_{i}')
        th.save(save_dict, save_path)
        return path, tag

    @staticmethod
    def load_agents(args, tag, name: str=None, path: Union[Path, None] = None):
        ''' Loads each agent that the trainer is training '''
        if not path:
            if args.exp_dir:
                path = args.base_dir / 'agent_models' / args.exp_dir / name
            else:
                path = args.base_dir / 'agent_models'/ name

        tag = tag or args.exp_name
        load_path = path / tag / 'trainer_file'
        agent_path = path / tag / 'agents_dir'
        device = args.device
        saved_variables = th.load(load_path, map_location=device)

        # Load weights
        agents = []
        for agent_fn in saved_variables['agent_fns']:
            agent = load_agent(agent_path / agent_fn, args)
            agent.to(device)
            agents.append(agent)
        return agents

    @staticmethod
    def list_agent_checked_tags(args, name: str=None, path: Union[Path, None] = None) -> List[str]:
        '''
        Lists only tags that start with CheckedPoints.CHECKED_MODEL_PREFIX, followed by an integer.
        If the integer is greater than 0, it must be followed by CheckedPoints.REWARD_SUBSTR and a floating-point number.

        Parameters:
        - args: Experiment arguments containing base directory info and experiment directory info.
        - name: The name of the agent for which tags should be listed.
        - path: Optional. If provided, it overrides the default path to the agents directory.

        Returns:
        - A list of tags (directories) that match the specified pattern.
        '''
        if not path:
            if args.exp_dir:
                path = args.base_dir / 'agent_models' / args.exp_dir / name
            else:
                path = args.base_dir / 'agent_models' / name

        # Ensure the directory exists
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Agent directory not found: {path}")

        # Define the prefix and the regular expression to match the pattern
        prefix = KeyCheckpoints.CHECKED_MODEL_PREFIX
        reward_substr = KeyCheckpoints.REWARD_SUBSTR
        pattern = re.compile(f"^{re.escape(prefix)}(\\d+)(?:{re.escape(reward_substr)}[\\d.]+)?$")

        # List all subdirectories (tags) that match the pattern
        tags = []
        for tag in path.iterdir():
            if tag.is_dir() and pattern.match(tag.name):
                match = pattern.match(tag.name)
                integer_part = int(match.group(1))
                # Only add tags that either have no reward substring for integer 0, or have it when integer > 0
                if integer_part == 0 or (integer_part > 0 and reward_substr in tag.name):
                    tags.append(tag.name)

        return tags
