from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

LAYOUT_NAMES_PATHs = {
    '3_chefs_small_kitchen': [
        'agent_models/sp-vs-spwsp/3-chefs-all-layouts/'
    ],
    '3_chefs_small_kitchen_two_resources': [
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

def get_all_teammates_for_evaluation(args, primary_agent, num_players, layout_names, deterministic, max_num_teams_per_layout_per_x):
    '''
    x = 0 means all N-1 teammates are primary_agent
    x = 1 means 1 teammate out of N-1 is unseen agent
    x = 2 means 2 teammates out of N-1- are unseen agents
    '''

    N = num_players
    X = list(range(N))
    
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
            for num_teams in range(max_num_teams_per_layout_per_x):
                teammates = [primary_agent] * (N-1-unseen_count)
                for i in range(unseen_count):
                    try: 
                        teammates.append(agents[i + (num_teams)])
                    except:
                        continue
                if len(teammates) == N-1:
                    teammates_list.append(teammates)
            all_teammates[layout_name][unseen_count] = teammates_list
    return all_teammates


def plot_evaluation_results(all_mean_rewards, all_std_rewards, layout_names, num_players, plot_name):
    num_layouts = len(layout_names) 
    fig, axes = plt.subplots(1, num_layouts, figsize=(5 * num_layouts, 6), sharey=True)

    if num_layouts == 1:
        axes = [axes]

    x_values = np.arange(num_players)

    for i, layout_name in enumerate(layout_names):
        ax = axes[i]
        for agent_name in all_mean_rewards:
            mean_values = []
            std_values = []
            
            for unseen_count in range(num_players):
                mean_rewards = all_mean_rewards[agent_name][layout_name][unseen_count]
                std_rewards = all_std_rewards[agent_name][layout_name][unseen_count]
                
                mean_values.append(np.mean(mean_rewards))
                std_values.append(np.mean(std_rewards))
            ax.errorbar(x_values, mean_values, yerr=std_values, fmt='-o', 
                         label=f'Agent: {agent_name}', capsize=5)

        ax.set_title(f'{layout_name}')
        ax.set_xlabel('Number of Unseen Teammates')
        ax.set_xticks(x_values)
        ax.legend(loc='upper right', fontsize='small', fancybox=True, framealpha=0.5)

        if i == 0:
            ax.set_ylabel('Mean Reward')

    plt.tight_layout()
    plt.savefig(f'data/eval/{plot_name}.png')
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

def generate_plot_name(num_players, deterministic, p_idxes, num_eps, max_num_teams):
    plot_name = f'{num_players}-players'
    plot_name += '-det' if deterministic else '-stoch'
    p_idexes_str = ''.join([str(p_idx) for p_idx in p_idxes])
    plot_name += f'-pidx{p_idexes_str}'
    plot_name += f'-eps{num_eps}'
    plot_name += f'-maxteams{str(max_num_teams)}'
    return plot_name+'.png'


def get_3_player_input(args):
    args.num_players = 3
    layout_names = ['3_chefs_small_kitchen_two_resources',
                    '3_chefs_forced_coordination_3OP2S1D',
                    '3_chefs_asymmetric_advantages',
                    '3_chefs_counter_circuit'
                    ]
    p_idxes = [0,1,2]

    # the name on the left is shown on the plot
    all_agents_paths = {
        'N-1-SP_cur': 'agent_models/N-1-SP-vs-AP-supporter-howlong-4/N-1-SP_s1010_h256_tr(SPH_SPM_SPL)_cur/best',
        'N-1-SP_ran': 'agent_models/N-1-SP-vs-AP-supporter-howlong-4/N-1-SP_s1010_h256_tr(SPH_SPM_SPL)_ran/best',
        'Saboteur Play': 'agent_models/four-layouts/supporter-fishplay/024_14/pwsab_s68_h512_tr(SP_SPH)_ran/best',
        'FCP': 'agent_models/all_layouts_supporters/fcp_s2020_h256_tr(AMX)_ran/best',
        'SP': 'agent_models/N-1-SP-vs-AP-supporter-howlong-4/SP_hd64_seed14/ck_40_rew_316.0'
    }

    return layout_names, p_idxes, all_agents_paths, args

def get_5_player_input(args):
    args.num_players = 5
    layout_names = ['5_chefs_storage_room_lots_resources',
                    '5_chefs_clustered_kitchen',
                    '5_chefs_coordination_ring'
                    ]
    p_idxes = [0]
    all_agents_paths = {
        'FCP': 'agent_models/5-p-layouts/FCP_s2020_h256_tr(AMX)_ran/best',
        'SP': 'agent_models/5-p-layouts/SP_hd256_seed13/ck_20_rew_441.3333333333333',
        'N-1-SP ran': 'agent_models/N-X-SP/N-1-SP_s1010_h256_tr(SPH_SPM_SPL)_ran/best',
        'N-1-SP cur': 'agent_models/N-4-SP/N-1-SP_s1010_h256_tr(SPH_SPM_SPL)_cur/best',
        'N-3-SP ran': 'agent_models/N-3-SP/N-3-SP_s1010_h256_tr(SPH_SPM_SPL)_ran/best',
        'N-3-SP cur': 'agent_models/N-3-SP/N-3-SP_s1010_h256_tr(SPH_SPM_SPL)_cur/best',
        'N-4-SP ran': 'agent_models/N-4-SP/N-4-SP_s1010_h256_tr(SPH_SPM_SPL)_ran/best',
        'N-4-SP cur': 'agent_models/N-4-SP/N-4-SP_s1010_h256_tr(SPH_SPM_SPL)_cur/best'
    }

    return layout_names, p_idxes, all_agents_paths, args


if __name__ == "__main__":
    args = get_arguments()
    # layout_names, p_idxes, all_agents_paths, args = get_3_player_input(args)
    layout_names, p_idxes, all_agents_paths, args = get_5_player_input(args)    

    deterministic = True
    max_num_teams_per_layout_per_x = 10
    number_of_eps = 10

    all_mean_rewards, all_std_rewards = {}, {}
    for name, path in all_agents_paths.items():
        agent = load_agent(Path(path), args)
        agent.deterministic = deterministic

        all_teammates = get_all_teammates_for_evaluation(args=args,
                                                        primary_agent=agent,
                                                        num_players=args.num_players,
                                                        layout_names=layout_names,
                                                        deterministic=deterministic,
                                                        max_num_teams_per_layout_per_x=max_num_teams_per_layout_per_x)
        mean_rewards, std_rewards = evaluate_agent(args = args,
                                                    primary_agent = agent,
                                                    p_idxes = p_idxes,
                                                    layout_names = layout_names,
                                                    all_teammates = all_teammates,
                                                    deterministic = deterministic,
                                                    number_of_eps = number_of_eps)

        all_mean_rewards[name] = mean_rewards
        all_std_rewards[name] = std_rewards
        
    plot_name = generate_plot_name(num_players=args.num_players,
                                    deterministic=deterministic,
                                    p_idxes=p_idxes,
                                    num_eps=number_of_eps,
                                    max_num_teams=max_num_teams_per_layout_per_x)
    plot_evaluation_results(all_mean_rewards=all_mean_rewards,
                            all_std_rewards=all_std_rewards,
                            layout_names=layout_names,
                            num_players=args.num_players,
                            plot_name=plot_name)
