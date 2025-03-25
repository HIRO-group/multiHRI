import numpy as np
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

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

        assert self.agent != 'human'
        self.agent.set_encoding_params(self.p_idx, self.args.horizon,
                                        env=self.env,
                                        is_haha=False,
                                        tune_subtasks=False)
        self.env.encoding_fn = self.agent.encoding_fn

        for t_idx, teammate in enumerate(self.env.teammates):
            teammate.set_encoding_params(t_idx+1, self.args.horizon,
                                         env=self.env,
                                         is_haha=False,
                                         tune_subtasks=True)

        self.env.deterministic = False

    def _run_simulation(self):
        self.env.reset(p_idx=self.p_idx)
        curr_tick = 0
        done = False
        trajectory = {
            'positions': [],
            'actions': [],
            'observations': [],
            'rewards': [],
            'dones': []
        }

        while not done and curr_tick <= self.env.env.horizon:
            obs = self.env.get_obs(self.env.p_idx)
            action = self.agent.predict(obs, state=self.env.state, deterministic=self.env.deterministic)[0]
            obs, reward, done, info = self.env.step(action)

            player_positions = [p.position for p in self.env.state.players]
            obs_copy = {k: np.copy(v) for k, v in obs.items()}

            trajectory['positions'].append(player_positions)
            trajectory['actions'].append(self.env.get_joint_action())
            trajectory['observations'].append(obs_copy)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)

            curr_tick += 1

        return trajectory

    def get_mdp(self):
        return self.env.env.mdp


    def run_simulation(self, how_many_times):
        """
        Run the Overcooked simulation and collect trajectory data.
        Returns:
            dict: Collected trajectory data
        """
        trajectories = []
        for _ in range(how_many_times):
            trajectory = self._run_simulation()
            trajectories.append(trajectory)
        return trajectories
