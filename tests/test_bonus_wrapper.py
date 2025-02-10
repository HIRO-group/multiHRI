#!/usr/bin/env python
import time

# Import BonusWrapper and OvercookedGymEnv from their respective modules.
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
# from oai_agents.gym_wrappers.bonus_wrapper import BonusWrapper

# # Define a minimal dummy arguments class that provides the necessary properties for OvercookedGymEnv.
# class DummyArgs:
#     n_envs = 1
#     n_layouts = 1
#     seed = 0
#     layout_names = ["asymmetric_advantages"]
#     horizon = 400
#     overcooked_verbose = False
#     device = "cpu"
#     epoch_timesteps = 1000
#     num_stack = 4
#     num_players = 2
#     encoding_fn = "OAI_feats"  # or another valid encoding key
#     dynamic_reward = False
#     final_sparse_r_ratio = 1.0
#     # Add any additional properties that your OvercookedGymEnv might require.

# def bonus_getter(state, action):
#     """
#     A simple bonus_getter function that always returns 5.0.
#     """
#     return 5.0

# def test_bonus_wrapper():
#     # Create a dummy args instance.
#     args = DummyArgs()

#     # Instantiate the OvercookedGymEnv.
#     env = OvercookedGymEnv(learner_type="dummy_learner", args=args)

#     # Wrap the environment with BonusWrapper using the bonus_getter defined above.
#     wrapped_env = BonusWrapper(env, bonus_getter)

#     # Reset the environment.
#     obs = wrapped_env.reset()
#     done = False
#     total_reward = 0.0

#     print("Starting test episode on BonusWrapper-wrapped OvercookedGymEnv...")
#     while not done:
#         # Sample a random action.
#         action = wrapped_env.action_space.sample()
#         obs, reward, done, info = wrapped_env.step(action)
#         total_reward += reward
#         print(f"Action: {action}, Step Reward: {reward}, Total Reward: {total_reward}")
#         # Optional: Slow down the loop for observation.
#         time.sleep(0.1)

#     print("Episode finished. Total reward:", total_reward)

# if __name__ == "__main__":
#     test_bonus_wrapper()
