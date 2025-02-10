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
