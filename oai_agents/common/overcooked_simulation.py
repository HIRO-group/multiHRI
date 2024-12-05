import json
import numpy as np
import pandas as pd
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
import time
from oai_agents.agents.hrl import HierarchicalRL

class OvercookedSimulation:
    """
    A class to run an Overcooked Gridworld simulation and collect trajectory data.
    Removes GUI and human player dependencies, focuses on agent interaction and data collection.
    """
    def __init__(self, args, agent, teammates, layout_name, p_idx, horizon=400):
        self.args = args
        self.layout_name = layout_name
        
        self.env = OvercookedGymEnv(args=args, 
                                    layout_name=self.layout_name,
                                    ret_completed_subtasks=False,
                                    is_eval_env=True, 
                                    horizon=horizon, 
                                    learner_type='originaler')
        
        self.agent = agent
        self.p_idx = p_idx
        self.env.set_teammates(teammates)
        self.env.reset(p_idx=self.p_idx)
        

        assert self.agent is not 'human'
        self.agent.set_encoding_params(self.p_idx, self.args.horizon, 
                                        env=self.env, 
                                        is_haha=isinstance(self.agent, HierarchicalRL), 
                                        tune_subtasks=False)
        self.env.encoding_fn = self.agent.encoding_fn

        for t_idx, teammate in enumerate(self.env.teammates):
            teammate.set_encoding_params(t_idx+1, self.args.horizon, 
                                         env=self.env, 
                                         is_haha=isinstance(teammate, HierarchicalRL), 
                                         tune_subtasks=True)

        self.env.deterministic = False
    
        self.score = 0
        self.curr_tick = 0

        self.trajectory = {
            'positions': [],
            'actions': [],
            'observations': [],
            'rewards': [],
            'dones': []
        }

    def run_simulation(self):
        """
        Run the Overcooked simulation and collect trajectory data.
        
        Returns:
            dict: Collected trajectory data
        """
        done = False
        on_reset = True

        while not done and self.curr_tick <= self.env.env.horizon:
            obs = self.env.get_obs(self.env.p_idx, on_reset=on_reset)
            action = self.agent.predict(obs, state=self.env.state, 
                                        deterministic=self.env.deterministic)[0]

            obs, reward, done, info = self.env.step(action)
            
            player_positions = [p.position for p in self.env.state.players]
            obs_copy = {k: np.copy(v) for k, v in obs.items()}
            
            self.trajectory['positions'].append(player_positions)
            self.trajectory['actions'].append(self.env.get_joint_action())
            self.trajectory['observations'].append(obs_copy)
            self.trajectory['rewards'].append(reward)
            self.trajectory['dones'].append(done)
            
            curr_reward = sum(info['sparse_r_by_agent'])
            self.score += curr_reward
            
            self.curr_tick += 1
            on_reset = False

        print(f'Simulation finished in {self.curr_tick} steps with total reward {self.score}')
        return self.trajectory

    def save_trajectory(self, data_path):
        df = pd.DataFrame(self.trajectory)
        df.to_pickle(data_path / f'{self.layout_name}.{self.trial_id}.pickle')
        return df