from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
import numpy as np

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv



def get_all_teammates_for_evaluation(args, primary_agent, num_players):
    all_teammates = []
    path_prefixes = ['sp-vs-spwsp/3-chefs-all-layouts']
    all_agents = []
    for path_prefix in path_prefixes:
        file_names = [file.name + '/best' for file in Path('agent_models/'+path_prefix).iterdir() if not file.name.startswith('fcp_pop')]
        for file_name in file_names:
            agent = load_agent(Path('agent_models/'+path_prefix+'/'+file_name), args)
            all_agents.append(agent)

    for agent in all_agents:
        teammates = [agent for _ in range(num_players-2)]
        teammates.append(primary_agent)
        all_teammates.append(teammates)

    return all_teammates


def evaluate_agent(args,
                   p_idxes,
                   layout_names,
                   all_teammates,
                   deterministic,
                   number_of_eps):
    all_mean_rewards = np.array([])
    for layout_name in layout_names:
        env = OvercookedGymEnv(args=args, layout_name=layout_name, ret_completed_subtasks=False, is_eval_env=True, horizon=400)
        for teammates in all_teammates:
            env.set_teammates(teammates)
            for p_idx in p_idxes:
                env.reset(p_idx=p_idx)
                mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=number_of_eps, deterministic=deterministic, warn=False, render=False)
                np.append(all_mean_rewards, mean_reward)
                print(mean_reward, std_reward)
    
    print(np.mean(all_mean_rewards), np.std(all_mean_rewards))

if __name__ == "__main__":
    args = get_arguments()
    args.num_players = 3
    layout_names = ['3_chefs_small_kitchen']
    p_idxes = [0]
    deterministic = True
    number_of_eps = 3

    agent_path = 'agent_models/eval/3_chefs/fcp_hd256_seed52/best'
    agent = load_agent(Path(agent_path), args)

    all_teammates = get_all_teammates_for_evaluation(args, agent, args.num_players)

    evaluate_agent(args = args,
                   p_idxes = p_idxes,
                   layout_names = layout_names,
                   all_teammates = all_teammates,
                   deterministic = deterministic,
                   number_of_eps = 10)


