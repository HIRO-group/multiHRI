from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv, USEABLE_COUNTERS, SCALING_FACTORS
from oai_agents.common.subtasks import Subtasks, get_doable_subtasks, calculate_completed_subtask

from overcooked_ai_py.mdp.overcooked_mdp import Action, Direction

from copy import deepcopy
from gym import spaces
import numpy as np
import torch as th


class OvercookedManagerGymEnv(OvercookedGymEnv):
    def __init__(self, worker=None, **kwargs):
        kwargs['ret_completed_subtasks'] = True
        super(OvercookedManagerGymEnv, self).__init__(**kwargs)
        self.action_space = spaces.Discrete(Subtasks.NUM_SUBTASKS)
        self.worker = worker
        self.worker_failures = {k: 0 for k in range(Subtasks.NUM_SUBTASKS)}
        self.base_env_timesteps = 0

    def get_base_env_timesteps(self):
        return self.base_env_timesteps

    def get_worker_failures(self):
        failures = self.worker_failures
        self.worker_failures = {k: 0 for k in range(Subtasks.NUM_SUBTASKS)}
        return (self.layout_name, failures)

    def get_low_level_obs(self, p_idx, done=False, enc_fn=None):
        enc_fn = enc_fn or self.encoding_fn
        obs = enc_fn(self.env.mdp, self.state, self.grid_shape, self.args.horizon, p_idx=p_idx)
        if p_idx == self.p_idx:
            obs['curr_subtask'] = self.curr_subtask
        if self.stack_frames[p_idx]:
            obs['visual_obs'] = np.expand_dims(obs['visual_obs'], 0)
            if self.stack_frames_need_reset[p_idx]: # On reset
                obs['visual_obs'] = self.stackedobs[p_idx].reset(obs['visual_obs'])
                self.stack_frames_need_reset[p_idx] = False
            else:
                obs['visual_obs'], _ = self.stackedobs[p_idx].update(obs['visual_obs'], np.array([done]), [{}])
            obs['visual_obs'] = obs['visual_obs'].squeeze()
        return obs

    def action_masks(self, p_idx=None):
        p_idx = p_idx or self.p_idx
        return get_doable_subtasks(self.state, self.prev_subtask[p_idx], self.layout_name, self.terrain, p_idx, USEABLE_COUNTERS[self.layout_name]).astype(bool)

    def step(self, action):
        # Action is the subtask for subtask agent to perform
        self.curr_subtask = action.cpu() if type(action) == th.tensor else action
        joint_action = [Action.STAY, Action.STAY]
        reward, done, info = 0, False, None
        ready_for_next_subtask = False
        worker_steps = 0
        while (not ready_for_next_subtask and not done):
            joint_action[self.p_idx] = self.worker.predict(self.get_low_level_obs(self.p_idx), deterministic=False)[0]
            tm_obs = self.get_obs(self.t_idx, enc_fn=self.teammate.encoding_fn) if self.teammate.use_hrl_obs else \
                     self.get_low_level_obs(self.t_idx, enc_fn=self.teammate.encoding_fn)
            joint_action[self.t_idx] = self.teammate.predict(tm_obs, deterministic=False)[0] # self.is_eval_env
            joint_action = [Action.INDEX_TO_ACTION[a] for a in joint_action]

            # If the state didn't change from the previous timestep and the agent is choosing the same action
            # then play a random action instead. Prevents agents from getting stuck
            if self.prev_state and self.state.time_independent_equal(self.prev_state) and tuple(joint_action) == self.prev_actions:
                joint_action = [np.random.choice(Direction.ALL_DIRECTIONS), np.random.choice(Direction.ALL_DIRECTIONS)]

            self.prev_state, self.prev_actions = deepcopy(self.state), deepcopy(joint_action)
            next_state, r, done, info = self.env.step(joint_action)
            self.base_env_timesteps += 1
            self.state = deepcopy(next_state)
            if False and self.shape_rewards and not self.is_eval_env:
                ratio = min(self.step_count * self.args.n_envs / 5e5, 1)
                sparse_r = sum(info['sparse_r_by_agent'])
                shaped_r = info['shaped_r_by_agent'][self.p_idx] if self.p_idx else sum(info['shaped_r_by_agent'])
                reward += sparse_r * ratio + shaped_r * (1 - ratio)
            else:
                if False and not self.is_eval_env and r > 0:
                    r *= SCALING_FACTORS[self.layout_name]
                reward += r

            self.step_count += 1
            worker_steps += 1

            if worker_steps % 5 == 0:
                if not get_doable_subtasks(self.state, self.prev_subtask[self.p_idx], self.layout_name, self.terrain, self.p_idx, USEABLE_COUNTERS[self.layout_name])[self.curr_subtask]:
                    ready_for_next_subtask = True
            if worker_steps > 25: # longest task is getting soup if soup needs to cook for the full 20 timsteps. Add some extra lewway
                ready_for_next_subtask = True
            # If subtask equals unknown, HRL agent will just STAY. This essentially forces a recheck every timestep
            # to see if any other task is possible
            if self.curr_subtask == Subtasks.SUBTASKS_TO_IDS['unknown'] and worker_steps >= 2:
                ready_for_next_subtask = True
                self.prev_subtask[self.p_idx] = Subtasks.SUBTASKS_TO_IDS['unknown']
                self.unknowns_in_a_row += 1
                # If no new subtask becomes available after 25 timesteps, end round
                if self.unknowns_in_a_row > 10 and not self.is_eval_env:
                    done = True
            else:
                self.unknowns_in_a_row = 0

            if joint_action[self.p_idx] == Action.INTERACT:
                completed_subtask = calculate_completed_subtask(self.terrain, self.prev_state, self.state, self.p_idx)
                if completed_subtask != self.curr_subtask:
                    self.worker_failures[self.curr_subtask] += 1
                ready_for_next_subtask = True

        return self.get_obs(self.p_idx, done=done), reward, done, info

    def reset(self, p_idx=None):
        if self.is_eval_env:
            ss_kwargs = {'random_pos': False, 'random_dir': False, 'max_random_objs': 0}
        else:
            ss_kwargs = {'random_pos': True, 'random_dir': True, 'max_random_objs': USEABLE_COUNTERS[self.layout_name]}
        self.env.reset(start_state_kwargs=ss_kwargs)
        self.state = self.env.state
        self.prev_state = None
        self.prev_subtask = [Subtasks.SUBTASKS_TO_IDS['unknown'], Subtasks.SUBTASKS_TO_IDS['unknown']]

        if p_idx is not None:
            self.p_idx = p_idx
        elif self.reset_p_idx is not None:
            self.p_idx = self.reset_p_idx
        elif self.p_idx is not None:
            self.p_idx = 1 - self.p_idx
        else:
            self.p_idx = np.random.randint(2)
        self.t_idx = 1 - self.p_idx
        # Setup correct agent observation stacking for agents that need it
        self.stack_frames[self.p_idx] = self.main_agent_stack_frames
        if self.teammate is not None:
            self.stack_frames[self.t_idx] = self.teammate.policy.observation_space['visual_obs'].shape[0] == \
                                            (self.enc_num_channels * self.args.num_stack)
        self.stack_frames_need_reset = [True, True]
        self.curr_subtask = 0
        self.unknowns_in_a_row = 0
        # Reset subtask counts
        self.completed_tasks = [np.zeros(Subtasks.NUM_SUBTASKS), np.zeros(Subtasks.NUM_SUBTASKS)]
        return self.get_obs(self.p_idx, on_reset=True)
