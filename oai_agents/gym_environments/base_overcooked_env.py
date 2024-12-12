from oai_agents.common.state_encodings import ENCODING_SCHEMES
from oai_agents.common.subtasks import Subtasks, calculate_completed_subtask, get_doable_subtasks
from oai_agents.common.learner import LearnerType, Learner

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager
from overcooked_ai_py.utils import read_layout_dict
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from copy import deepcopy
from gym import Env, spaces, register
import numpy as np
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
import torch as th
import random


# DEPRECATED NOTE: For counter circuit, trained workers with 8, but trained manager with 4. Only 4 spots are useful add
# more during subtask worker training for robustness
# Max number of counters the agents should use
USEABLE_COUNTERS = {'counter_circuit_o_1order': 2, 'forced_coordination': 2, 'asymmetric_advantages': 2,
                    'cramped_room': 2, 'coordination_ring': 2}  # FOR WORKER TRAINING


# USEABLE_COUNTERS = {'counter_circuit_o_1order': 4, 'forced_coordination': 3, 'asymmetric_advantages': 2, 'cramped_room': 3, 'coordination_ring': 3} # FOR MANAGER TRAINING
# USEABLE_COUNTERS = {'counter_circuit_o_1order': 2, 'forced_coordination': 4, 'asymmetric_advantages': 4, 'cramped_room': 3, 'coordination_ring': 3}  # FOR EVALUATION AND SP TRAINING


class OvercookedGymEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, learner_type, grid_shape=None, ret_completed_subtasks=False, stack_frames=False, is_eval_env=False,
                 shape_rewards=False, enc_fn=None, full_init=True, args=None, num_enc_channels=27, deterministic=False, start_timestep: int = 0,
                 **kwargs):
        self.is_eval_env = is_eval_env
        self.args = args
        self.device = args.device
        # Observation encoding setup
        enc_fn = enc_fn or args.encoding_fn
        self.encoding_fn = ENCODING_SCHEMES[enc_fn]
        if enc_fn == 'OAI_egocentric':
            # Override grid shape to make it egocentric
            assert grid_shape is None, 'Grid shape cannot be used when egocentric encodings are used!'
            self.grid_shape = (7, 7)
        elif grid_shape is None:
            base_layout_params = read_layout_dict(args.layout_names[0])
            grid = [layout_row.strip() for layout_row in base_layout_params['grid'].split("\n")]
            self.grid_shape = (len(grid[0]), len(grid))
        else:
            self.grid_shape = grid_shape

        # Set Sp Observation Space
        # Currently 20 is the default value for recipe time (which I believe is the largest value used in encoding)
        self.num_enc_channels = num_enc_channels  # Default channels of OAI_Lossless encoding
        self.obs_dict = {}
        if enc_fn == 'OAI_feats':
            self.obs_dict['agent_obs'] = spaces.Box(0, 400, (96,), dtype=int)
        else:
            self.obs_dict['visual_obs'] = spaces.Box(0, 20, (self.num_enc_channels, *self.grid_shape), dtype=int)
            # Stacked obs for players
            # num_envs, num_stack: 3, observation_space, :param channels_order: If "first", stack on first image dimension.
            self.stackedobs = [StackedObservations(1, args.num_stack, self.obs_dict['visual_obs'], 'first') \
                               for _ in range(self.args.num_players)]
        if stack_frames:
            self.obs_dict['visual_obs'] = self.stackedobs[0].stack_observation_space(self.obs_dict['visual_obs'])

        if ret_completed_subtasks:
            self.obs_dict['subtask_mask'] = spaces.MultiBinary(Subtasks.NUM_SUBTASKS)
        # self.obs_dict['layout_idx'] = spaces.MultiBinary(5)
        # self.obs_dict['p_idx'] = spaces.MultiBinary(2)
        self.observation_space = spaces.Dict(self.obs_dict)
        self.return_completed_subtasks = ret_completed_subtasks
        # Default stack frames to false since we don't currently know who is playing what - properly set in reset
        self.main_agent_stack_frames = stack_frames
        self.stack_frames_need_reset = [True, True]
        self.action_space = spaces.Discrete(len(Action.ALL_ACTIONS))

        self.shape_rewards = shape_rewards
        self.visualization_enabled = False
        self.step_count = start_timestep
        self.reset_p_idx = None

        self.learner = Learner(learner_type, args.reward_magnifier)

        self.dynamic_reward = args.dynamic_reward
        self.final_sparse_r_ratio = args.final_sparse_r_ratio

        self.p_idx = None
        self.teammates = []
        self.joint_action = []
        self.deterministic = deterministic
        self.reset_info = {}
        if full_init:
            self.set_env_layout(**kwargs)

    def set_env_layout(self, env_index=None, layout_name=None, base_env=None, horizon=None):
        '''
        Required to play nicely with sb3 make_vec_env. make_vec_env doesn't allow different arguments for each env,
        so to specify the layouts, they must first be created then each this is called.
        :param env_index: int. Used to index the layouts form self.layout_names
        :param layout_name: str, directly pass in layout name
        :param base_env: Base overcooked environment. If None, create env from layout name. Useful if special parameters
                         are required when creating the environment
        :param horizon: horizon for environment. Will default to args.horizon if not provided
        '''
        assert env_index is not None or layout_name is not None or base_env is not None



        if base_env is None:
            self.env_idx = env_index
            self.layout_name = layout_name or self.args.layout_names[env_index]
            self.mdp = OvercookedGridworld.from_layout_name(self.layout_name)
            # print("num players in base_env", self.mdp.num_players)
            all_counters = self.mdp.get_counter_locations()
            COUNTERS_PARAMS = {
                'start_orientations': False,
                'wait_allowed': False,
                'counter_goals': all_counters,
                'counter_drop': all_counters,
                'counter_pickup': all_counters,
                'same_motion_goals': True
            }
            self.mlam = MediumLevelActionManager.from_pickle_or_compute(self.mdp, COUNTERS_PARAMS, force_compute=False, info=self.args.overcooked_verbose)
            self.env = OvercookedEnv.from_mdp(self.mdp, horizon=(horizon or self.args.horizon))  # , **self.get_overcooked_from_mdp_kwargs(horizon=horizon))
        else:
            self.env = base_env
            self.layout_name = self.env.mdp.layout_name
            self.env_idx = self.args.layout_names.index(self.layout_name)

        self.terrain = self.mdp.terrain_mtx


        self.prev_subtask = [Subtasks.SUBTASKS_TO_IDS['unknown'] for _ in range(self.mdp.num_players)]
        self.env.reset(reset_info=self.reset_info)
        self.valid_counters = [self.env.mdp.find_free_counters_valid_for_player(self.env.state, self.mlam, i) for i in
                               range(2)]
        self.reset()

    # def get_overcooked_from_mdp_kwargs(self, horizon=None):
    #     horizon = horizon or self.args.horizon
    #     return {'start_state_fn': self.mdp.get_fully_random_start_state_fn(self.mlam), 'horizon': horizon}

    def get_layout_name(self):
        return self.layout_name

    def get_joint_action(self):
        return self.joint_action

    def set_teammates(self, teammates):
        assert isinstance(teammates, list)
        self.teammates = teammates

        self.reset_info['start_position'] = {}
        for idx, tm in enumerate(self.teammates):
            if tm.get_start_position(self.layout_name) is not None:
                self.reset_info['start_position'][idx] = tm.get_start_position(self.layout_name)

        assert self.mdp.num_players == len(self.teammates) + 1, f"MDP num players: {self.mdp.num_players} != " \
                                                                    f"num teammates: {len(self.teammates)} + main agent: 1"
        self.stack_frames_need_reset = [True for i in range(self.mdp.num_players)]


    def stack_frames(self, c_idx):
        if c_idx == self.p_idx:
            return self.main_agent_stack_frames

        elif len(self.teammates) != 0:
            for t_idx in self.t_idxes:
                if c_idx == t_idx:
                    teammate = self.get_teammate_from_idx(t_idx)
                    return teammate.policy.observation_space['visual_obs'].shape[0] == (27 * self.args.num_stack)
        return False

    def setup_visualization(self):
        self.visualization_enabled = True
        pygame.init()
        surface = StateVisualizer().render_state(self.state, grid=self.env.mdp.terrain_mtx)
        self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.window.blit(surface, (0, 0))
        pygame.display.flip()

    def action_masks(self, p_idx):
        return get_doable_subtasks(self.state, self.prev_subtask[p_idx], self.layout_name, self.terrain, p_idx,
                                   self.valid_counters, USEABLE_COUNTERS.get(self.layout_name, 5)).astype(bool)

    def get_obs(self, c_idx, done=False, enc_fn=None, on_reset=False, goal_objects=None):
        enc_fn = enc_fn or self.encoding_fn
        obs = enc_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=c_idx,
                     goal_objects=goal_objects)

        if self.stack_frames(c_idx):
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if self.stack_frames_need_reset[c_idx]:  # On reset
                obs['visual_obs'] = self.stackedobs[c_idx].reset(obs['visual_obs'])
                self.stack_frames_need_reset[c_idx] = False
            else:
                obs['visual_obs'], _ = self.stackedobs[c_idx].update(obs['visual_obs'], np.array([done]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()

        if self.return_completed_subtasks:
            obs['subtask_mask'] = self.action_masks(c_idx)

        elif self.teammates is not None:
            for t_idx in self.t_idxes:
                if c_idx == t_idx:
                    teammate = self.get_teammate_from_idx(c_idx)
                    if 'subtask_mask' in teammate.policy.observation_space.keys():
                        obs['subtask_mask'] = self.action_masks(c_idx)
                        break

        for t_idx in self.t_idxes:
            if c_idx == t_idx:
                teammate = self.get_teammate_from_idx(t_idx)
                obs = {k: v for k, v in obs.items() if k in teammate.policy.observation_space.keys()}
                break

        return obs

    def get_teammate_from_idx(self, idx):
        assert idx in self.t_idxes
        id = self.t_idxes.index(idx)
        return self.teammates[id]

    def step(self, action):
        if len(self.teammates) == 0:
            raise ValueError('set_teammates must be set called before starting game.')

        joint_action = [None for _ in range(self.mdp.num_players)]
        joint_action[self.p_idx] = action
        with th.no_grad():
            for t_idx in self.t_idxes:
                teammate = self.get_teammate_from_idx(t_idx)
                tm_obs = self.get_obs(c_idx=t_idx, enc_fn=teammate.encoding_fn)
                joint_action[t_idx] = teammate.predict(tm_obs, deterministic=self.deterministic)[0]

        joint_action = [Action.INDEX_TO_ACTION[(a.squeeze() if type(a) != int else a)] for a in joint_action]
        self.joint_action = joint_action

        # If the state didn't change from the previous timestep and the agent is choosing the same action
        # then play a random action instead. Prevents agents from getting stuck
        if self.is_eval_env:
            if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == tuple(
                    self.prev_actions):
                joint_action = [Action.STAY for _ in range(self.mdp.num_players)]
                for t_idx in self.t_idxes:
                    joint_action[t_idx] = Direction.INDEX_TO_DIRECTION[self.step_count % 4]

            self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(joint_action)

        self.state, reward, done, info = self.env.step(joint_action)
        if self.shape_rewards and not self.is_eval_env:
            if self.dynamic_reward:
                ratio = min(self.step_count * self.args.n_envs / 1e7, self.final_sparse_r_ratio)
            else:
                ratio = self.final_sparse_r_ratio
            reward = self.learner.calculate_reward(p_idx=self.p_idx, env_info=info, ratio=ratio, num_players=self.mdp.num_players)
        self.step_count += 1
        return self.get_obs(self.p_idx, done=done), reward, done, info

    def set_reset_p_idx(self, p_idx):
        self.reset_p_idx = p_idx

    def reset(self, p_idx=None):
        if p_idx is not None:
            self.p_idx = p_idx
        elif self.reset_p_idx is not None:
            self.p_idx = self.reset_p_idx
        else:
            self.p_idx = random.randint(0, self.mdp.num_players - 1)

        teammates_ids = list(range(self.mdp.num_players))
        teammates_ids.remove(self.p_idx)

        if not self.is_eval_env: # To have consistent teammates for evaluation
            random.shuffle(teammates_ids)

        self.t_idxes = teammates_ids
        self.stack_frames_need_reset = [True for _ in range(self.mdp.num_players)]
        self.env.reset(reset_info=self.reset_info)

        self.prev_state = None
        self.state = self.env.state

        # Reset subtask counts
        self.completed_tasks = [np.zeros(Subtasks.NUM_SUBTASKS), np.zeros(Subtasks.NUM_SUBTASKS)]
        return self.get_obs(self.p_idx, on_reset=True)


    def render(self, mode='human', close=False):
        if self.visualization_enabled:
            surface = StateVisualizer().render_state(self.state, grid=self.env.mdp.terrain_mtx)
            self.window = pygame.display.set_mode(surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            pygame.time.wait(200)

    def close(self):
        pygame.quit()


register(
    id='OvercookedGymEnv-v0',
    entry_point='OvercookedGymEnv'
)

if __name__ == '__main__':
    from oai_agents.common.arguments import get_arguments

    args = get_arguments()
    env = OvercookedGymEnv(p1=DummyAgent(),
                           args=args)  # make('overcooked_ai.agents:OvercookedGymEnv-v0', layout='asymmetric_advantages', encoding_fn=encode_state, args=args)
    print(check_env(env))
    env.setup_visualization()
    env.reset()
    env.render()
    done = False
    while not done:
        obs, reward, done, info = env.step(Action.ACTION_TO_INDEX[np.random.choice(Action.ALL_ACTIONS)])
        env.render()
