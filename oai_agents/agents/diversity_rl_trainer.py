from oai_agents.agents.rl import RLAgentTrainer
from stable_baselines3.common.env_util import make_vec_env
# Import both classes from base_overcooked_env.py
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv, BonusOvercookedGymEnv
from stable_baselines3.common.vec_env import DummyVecEnv as VEC_ENV_CLS

class DiversityRLAgentTrainer(RLAgentTrainer):
    """
    A subclass of RLAgentTrainer that creates the training environment using BonusOvercoookedGymEnv
    (to integrate bonus logic) while creating the evaluation environments as plain OvercookedGymEnv,
    exactly as in RLAgentTrainer.
    """
    def get_envs(self, _env, _eval_envs, deterministic, learner_type, start_timestep: int = 0):
        if _env is None:
            # --- Create the training environment using BonusOvercoookedGymEnv ---
            env_kwargs = {
                'shape_rewards': True,
                'full_init': False,
                'stack_frames': self.use_frame_stack,
                'deterministic': deterministic,
                'args': self.args,
                'learner_type': learner_type,
                'start_timestep': start_timestep
            }
            # Note: BonusOvercoookedGymEnv is used here for training.
            base_env = make_vec_env(BonusOvercookedGymEnv,
                                    n_envs=self.args.n_envs,
                                    seed=self.seed,
                                    vec_env_cls=VEC_ENV_CLS,
                                    env_kwargs=env_kwargs)
            env = base_env

            # --- Create evaluation environments as plain OvercookedGymEnv ---
            eval_envs_kwargs = {
                'is_eval_env': True,
                'horizon': 400,
                'stack_frames': self.use_frame_stack,
                'deterministic': deterministic,
                'args': self.args,
                'learner_type': learner_type
            }
            # Here we use the original OvercookedGymEnv so that evaluation is the same as in RLAgentTrainer.
            eval_envs = [
                OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs, 'unique_env_idx': self.args.n_envs + i})
                for i in range(self.n_layouts)
            ]
        else:
            env = _env
            eval_envs = _eval_envs

        # --- Configure each training environment's layout if needed ---
        for i in range(self.n_envs):
            env.env_method('set_env_layout', indices=i, env_index=i % self.n_layouts, unique_env_idx=i)
        return env, eval_envs
