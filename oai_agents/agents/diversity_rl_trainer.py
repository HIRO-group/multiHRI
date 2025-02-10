from oai_agents.agents.rl import RLAgentTrainer
from stable_baselines3.common.env_util import make_vec_env
from overcooked_ai_py.mdp.overcooked_env import OvercookedGymEnv
from oai_agents.gym_wrappers.bonus_wrapper import BonusWrapper
from stable_baselines3.common.vec_env import DummyVecEnv as VEC_ENV_CLS

class DiversityRLAgentTrainer(RLAgentTrainer):
    """
    Subclass of RLAgentTrainer that wraps the training OvercookedGymEnv with
    BonusWrapper (to inject a bonus into rewards)
    while keeping evaluation environments as plain OvercookedGymEnv.
    """
    def get_envs(self, _env, _eval_envs, deterministic, learner_type, start_timestep: int = 0):
        if _env is None:
            # --- Create the base training environment ---
            env_kwargs = {
                'shape_rewards': True,
                'full_init': False,
                'stack_frames': self.use_frame_stack,
                'deterministic': deterministic,
                'args': self.args,
                'learner_type': learner_type,
                'start_timestep': start_timestep
            }
            base_env = make_vec_env(OvercookedGymEnv,
                                    n_envs=self.args.n_envs,
                                    seed=self.seed,
                                    vec_env_cls=VEC_ENV_CLS,
                                    env_kwargs=env_kwargs)
            # --- Wrap the training environment with BonusWrapper ---
            bonus_getter = lambda env_state, action: 0.0  # Dummy bonus initially.
            wrapped_env = BonusWrapper(base_env, bonus_getter)
            env = wrapped_env

            # --- Create evaluation environments as plain OvercookedGymEnv ---
            eval_envs_kwargs = {
                'is_eval_env': True,
                'horizon': 400,
                'stack_frames': self.use_frame_stack,
                'deterministic': deterministic,
                'args': self.args,
                'learner_type': learner_type
            }
            eval_envs = [
                OvercookedGymEnv(**{'env_index': i, **eval_envs_kwargs, 'unique_env_idx': self.args.n_envs + i})
                for i in range(self.n_layouts)
            ]
        else:
            env = _env
            eval_envs = _eval_envs

        # --- Setup each training environment's layout if needed ---
        for i in range(self.n_envs):
            env.env_method('set_env_layout', indices=i, env_index=i % self.n_layouts, unique_env_idx=i)
        return env, eval_envs
