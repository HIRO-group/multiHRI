from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

MAX_NUM_TEAMS_PER_LAYOUT_PER_X = 3

LAYOUT_NAMES_PATHs = {
    '3_chefs_small_kitchen': [
        'agent_models/sp-vs-spwsp/3-chefs-all-layouts/'
    ],
    '3_chefs_forced_coordination': [
        'agent_models/sp-vs-spwsp/3-chefs-all-layouts/'
    ],
    '3_chefs_asymmetric_advantages': [
        'agent_models/sp-vs-spwsp/3-chefs-all-layouts/'
    ],
    '3_chefs_forced_coordination_3OP2S1D': [
        'agent_models/sp-vs-spwsp/3-chefs-all-layouts/'
    ],
    '3_chefs_counter_circuit': [
        'agent_models/sp-vs-spwsp/3-chefs-all-layouts/'
    ],
    '5_chefs_storage_room_lots_resources': [
        'agent_models/sp-vs-spwsp/5-chefs-all-layouts'
    ],
    '5_chefs_clustered_kitchen' :[
        'agent_models/sp-vs-spwsp/5-chefs-all-layouts'
    ],
    '5_chefs_coordination_ring': [
        'agent_models/sp-vs-spwsp/5-chefs-all-layouts'
    ],
    '5_chefs_storage_room_lots_resources': [
        'agent_models/sp-vs-spwsp/5-chefs-all-layouts'
    ],
    '5_chefs_cramped_room': [
        'agent_models/sp-vs-spwsp/5-chefs-all-layouts'
    ],
    
    '5_chefs_counter_circuit': [],
    '5_chefs_asymmetric_advantages': [],
}

def print_all_teammates(all_teammates):
    for layout_name in all_teammates:
        print('Layout:', layout_name)
        for teammates in all_teammates[layout_name]:
            print([agent.name for agent in teammates])
        print()

def get_all_teammates_for_evaluation(args, primary_agent, num_players, layout_names, deterministic):
    N = num_players
    X = list(range(N))
    # x = 0 means all N-1 teammates are primary_agent
    # x = 1 means 1 teammate out of N-1 is unseen agent
    # x = 2 means 2 teammates out of N-1- are unseen agents

    # Contains all the agents which are later used to create all_teammates
    all_agents = {layout_name: [] for layout_name in layout_names}
    # Containts teams for each layout and each x up to MAX_NUM_TEAMS_PER_LAYOUT_PER_X
    all_teammates = {
        layout_name: {
            unseen_count: [] for unseen_count in X} 
        for layout_name in layout_names} 

    for layout_name in layout_names:
        for path in LAYOUT_NAMES_PATHs[layout_name]:
            file_names = [path+'/'+file.name + '/best' for file in Path(path).iterdir() if not file.name.startswith('fcp_pop')]

            for file_name in file_names:
                agent = load_agent(Path(file_name), args)
                agent.deterministic = deterministic
                all_agents[layout_name].append(agent)

    for layout_name in layout_names:
        agents = all_agents[layout_name]

        for unseen_count in X:
            teammates_list = []
            for num_teams in range(MAX_NUM_TEAMS_PER_LAYOUT_PER_X):
                teammates = [primary_agent] * (N-1-unseen_count)
                for i in range(unseen_count):
                    try: 
                        teammates.append(agents[i + (num_teams * unseen_count)])
                    except:
                        continue
                if len(teammates) == N-1:
                    teammates_list.append(teammates)
            all_teammates[layout_name][unseen_count] = teammates_list
    return all_teammates


def plot_evaluation_results(primary_agent_name, all_mean_rewards, all_std_rewards, layout_names, num_players):
    for layout_name in layout_names:
        plt.figure(figsize=(10, 6))
        x_values = np.arange(num_players)

        mean_values = []
        std_values = []
        
        for unseen_count in range(num_players):
            mean_rewards = all_mean_rewards[layout_name][unseen_count]
            std_rewards = all_std_rewards[layout_name][unseen_count]

            mean_values.append(np.mean(mean_rewards))
            std_values.append(np.mean(std_rewards))

        plt.errorbar(x_values, mean_values, yerr=std_values, fmt='-o', label=f'Layout: {layout_name}', capsize=5)

        plt.title(f'Evaluation Results for {layout_name} for agent {primary_agent_name}')
        plt.xlabel('Number of Unseen Teammates')
        plt.ylabel('Mean Reward')
        plt.xticks(x_values)
        plt.legend()

        plt.show()


def evaluate_agent(args,
                   primary_agent,
                   p_idxes,
                   layout_names,
                   all_teammates,
                   deterministic,
                   number_of_eps):

    all_mean_rewards = {
        layout_name: {unseen_count: [] for unseen_count in range(args.num_players)}
        for layout_name in layout_names
    }
    all_std_rewards = {
        layout_name: {unseen_count: [] for unseen_count in range(args.num_players)}
        for layout_name in layout_names
    }

    for layout_name in layout_names:
        for unseen_count in range(args.num_players):
            for teammates in all_teammates[layout_name][unseen_count]:
                env = OvercookedGymEnv(args=args,
                                       layout_name=layout_name,
                                       ret_completed_subtasks=False,
                                       is_eval_env=True,
                                       horizon=400,
                                       deterministic=deterministic)
                env.set_teammates(teammates)
                for p_idx in p_idxes:
                    env.reset(p_idx=p_idx)
                    mean_reward, std_reward = evaluate_policy(primary_agent, env,
                                                              n_eval_episodes=number_of_eps,
                                                              deterministic=deterministic,
                                                              warn=False,
                                                              render=False)
                    all_mean_rewards[layout_name][unseen_count].append(mean_reward)
                    all_std_rewards[layout_name][unseen_count].append(std_reward)

    return all_mean_rewards, all_std_rewards


if __name__ == "__main__":
    args = get_arguments()
    args.num_players = 3
    layout_names = ['3_chefs_small_kitchen']
    p_idxes = [0]
    deterministic = True
    number_of_eps = 5

    agent_path = 'agent_models/eval/3_chefs/fcp_hd256_seed52/best'
    agent = load_agent(Path(agent_path), args)
    agent.deterministic = deterministic

    all_teammates = get_all_teammates_for_evaluation(args=args,
                                                     primary_agent=agent,
                                                     num_players=args.num_players,
                                                     layout_names=layout_names,
                                                     deterministic=deterministic)
    all_mean_rewards, all_std_rewards = evaluate_agent(args = args,
                                                        primary_agent = agent,
                                                        p_idxes = p_idxes,
                                                        layout_names = layout_names,
                                                        all_teammates = all_teammates,
                                                        deterministic = deterministic,
                                                        number_of_eps = number_of_eps)

    plot_evaluation_results(all_mean_rewards=all_mean_rewards,
                            all_std_rewards=all_std_rewards,
                            layout_names=layout_names,
                            num_players=args.num_players,
                            primary_agent_name='fcp_hd256_seed52'
                            )
