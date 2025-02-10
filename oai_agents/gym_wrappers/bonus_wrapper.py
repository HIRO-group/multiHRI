import gym

class BonusWrapper(gym.Wrapper):
    def __init__(self, env, bonus_getter):
        """
        Wrap an environment so that a bonus is added to the reward.

        bonus_getter: a callable taking (env_state, action) and returning a float bonus.
                      (This will be updated later by the population manager.)
        """
        super().__init__(env)
        self.bonus_getter = bonus_getter

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Retrieve the underlying env's state (adjust if your env uses a different interface).
        bonus = self.bonus_getter(self.env.state, action)
        return obs, reward + bonus, done, info

    def set_bonus_getter(self, bonus_getter):
        """Allow external code to update the bonus_getter."""
        self.bonus_getter = bonus_getter

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying environment.
        This ensures that all functions and attributes of OvercookedGymEnv are accessible.
        """
        return getattr(self.env, name)


# ---------------------------- TEST SCRIPT ----------------------------
if __name__ == "__main__":
    import random
    import time
    from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

    # Create a dummy arguments class with the minimum properties needed.
    class DummyArgs:
        n_envs = 1
        n_layouts = 1
        seed = 0
        layout_names = ["asymmetric_advantages"]
        horizon = 400
        overcooked_verbose = False
        device = "cpu"
        epoch_timesteps = 1000
        num_stack = 4
        num_players = 2
        encoding_fn = "OAI_feats"  # or another valid encoding key
        dynamic_reward = False
        final_sparse_r_ratio = 1.0
        # You can add more dummy properties if required by OvercookedGymEnv.

    args = DummyArgs()

    # Instantiate an OvercookedGymEnv.
    # OvercookedGymEnv requires at least a learner_type and args.
    # For testing, we pass a dummy learner type.
    env = OvercookedGymEnv(learner_type="dummy_learner", args=args)

    # Define a simple bonus_getter that always returns a bonus of 5.0.
    def test_bonus_getter(state, action):
        return 5.0

    # Wrap the environment with BonusWrapper.
    wrapped_env = BonusWrapper(env, test_bonus_getter)

    # Reset the environment.
    obs = wrapped_env.reset()
    done = False
    total_reward = 0.0

    print("Starting test episode on BonusWrapper-wrapped OvercookedGymEnv...")
    while not done:
        # Sample a random action.
        action = wrapped_env.action_space.sample()
        obs, reward, done, info = wrapped_env.step(action)
        total_reward += reward
        print(f"Action: {action}, Step Reward: {reward}, Total Reward: {total_reward}")
        # Optional: if your environment supports rendering, you can call wrapped_env.render()
        time.sleep(0.1)  # Slow down the loop for observation

    print("Episode finished. Total reward:", total_reward)
