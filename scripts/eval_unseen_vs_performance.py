import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from oai_agents.common.arguments import get_arguments
from evaluate_agents import run_parallel_evaluation, get_2_player_input_classic, get_3_player_input_complex, get_5_player_input_complex

eval_key_lut = {
    'l': "LOW",
    'm': "MEDIUM",
    'h': "HIGH"
}

# Set font for all plots
plt.rcParams['font.family'] = 'Times New Roman'
FONT_SIZE = 30
LEGEND_FONT_SIZE = 22
LINEWIDTH = 4

def get_normalized_result(  all_mean_rewards,
                            all_std_rewards,
                            layout_names,
                            teammate_lvl_sets, 
                            unseen_counts):

    team_lvl_set_keys = [str(t) for t in teammate_lvl_sets]
    team_lvl_set_names = [str([eval_key_lut[l] for l in t]) for t in teammate_lvl_sets]

    def process_reward(reward, max_reward=None):
        if max_reward:
            reward = reward / max_reward
        return reward


    cross_exp_mean = {
        agent_name: {
            unseen_count: [] for unseen_count in unseen_counts
        } for agent_name in all_mean_rewards
    }

    cross_exp_std = {
        agent_name: {  
            unseen_count: [] for unseen_count in unseen_counts
        } for agent_name in all_mean_rewards
    }
    

    for i, layout_name in enumerate(layout_names):
        rewards_for_all_agents = []
        for agent in all_mean_rewards:
            for team in team_lvl_set_keys:
                for unseen_count in unseen_counts:
                    for reward in all_mean_rewards[agent][team][layout_name][unseen_count]:
                        rewards_for_all_agents.append(reward)
        max_mean_reward = max(rewards_for_all_agents)

        for j, (team, team_name) in enumerate(zip(team_lvl_set_keys, team_lvl_set_names)):
            for agent_name in all_mean_rewards:
                for unseen_count in unseen_counts:
                    mean_rewards = [process_reward(r, max_reward=max_mean_reward) for r in all_mean_rewards[agent_name][team][layout_name][unseen_count]]
                    std_rewards = [process_reward(r, max_reward=max_mean_reward) for r in all_std_rewards[agent_name][team][layout_name][unseen_count]]
                    cross_exp_mean[agent_name][unseen_count].extend(mean_rewards)
                    cross_exp_std[agent_name][unseen_count].extend(std_rewards)


    cross_exp_mean_final = {
        agent_name: {
            unseen_count: np.mean(cross_exp_mean[agent_name][unseen_count])
            for unseen_count in unseen_counts
        } for agent_name in all_mean_rewards
    }

    cross_exp_std_final = {
        agent_name: {
            unseen_count: np.std(cross_exp_mean[agent_name][unseen_count])
            for unseen_count in unseen_counts
        } for agent_name in all_mean_rewards
    }

    return cross_exp_mean_final, cross_exp_std_final


def plot_unseen_over_teamsize_vs_performance(five_all_mean_rewards, five_all_std_rewards, three_all_mean_rewards, three_all_std_rewards, two_all_mean_rewards, two_all_std_rewards):
    cross_exp_five, cross_exp_std = get_normalized_result(five_all_mean_rewards, five_all_std_rewards, five_layout_names, five_teammate_lvl_sets, unseen_counts=[0, 1, 2, 3, 4])
    cross_exp_three, cross_exp_std = get_normalized_result(three_all_mean_rewards, three_all_std_rewards, three_layout_names, three_teammate_lvl_sets, unseen_counts=[0, 1, 2])
    cross_exp_two, cross_exp_std = get_normalized_result(two_all_mean_rewards, two_all_std_rewards, two_layout_names, two_teammate_lvl_sets, unseen_counts=[0, 1])

# result_dictionary_2 = {'0/2': {'SP': np.float64(0.9857485875706218), 'N-XPlay': np.float64(0.7144774011299436)}, '1/2': {'SP': np.float64(0.4777995527306968), 'N-XPlay': np.float64(0.5038059086629002)}}
# result_dictionary_3 = {'0/3': {'SP': np.float64(0.9346016184251478), 'N-XPlay': np.float64(0.8753631082062453)}, '1/3': {'SP': np.float64(0.5904885102189025), 'N-XPlay': np.float64(0.7413729380641146)}, '2/3': {'SP': np.float64(0.3720627139744787), 'N-XPlay': np.float64(0.5653270567486254)}}
# result_dictionary_5 = {'0/5': {'SP': np.float64(0.9444596914220454), 'N-XPlay': np.float64(0.5489397002785127)}, '1/5': {'SP': np.float64(0.5591021518689864), 'N-XPlay': np.float64(0.4976990719160655)}, '2/5': {'SP': np.float64(0.3969753135694281), 'N-XPlay': np.float64(0.3556219094620473)}, '3/5': {'SP': np.float64(0.29125677927586735), 'N-XPlay': np.float64(0.30076252626835864)}, '4/5': {'SP': np.float64(0.209715555918101), 'N-XPlay': np.float64(0.22736886602263062)}}

    result_dictionary_2 = {
        '0/2': {
            'SP':      cross_exp_two['SP'][0],
            'N-XPlay': cross_exp_two['N-1Play'][0],
        },
        '1/2': {
            'SP':      cross_exp_two['SP'][1],
            'N-XPlay': cross_exp_two['N-1Play'][1],
        },
    }

    result_dictionary_3 = {
        '0/3': {
            'SP':       cross_exp_three['SP'][0],
            'N-XPlay': max(cross_exp_three['N-1Play'][0], cross_exp_three['N-2Play'][0]),
        },
        '1/3': {
            'SP':       cross_exp_three['SP'][1],
            'N-XPlay': max(cross_exp_three['N-1Play'][1], cross_exp_three['N-2Play'][1]),
        },
        '2/3': {
            'SP':       cross_exp_three['SP'][2],
            'N-XPlay': max(cross_exp_three['N-1Play'][2], cross_exp_three['N-2Play'][2]),
        },
    }

    result_dictionary_5 = {
        '0/5': {
           'SP':       cross_exp_five['SP'][0],
           'N-XPlay': max(cross_exp_five['N-1Play'][0], cross_exp_five['N-3Play'][0]),
        },

        '1/5': {
            'SP':       cross_exp_five['SP'][1],
            'N-XPlay': max(cross_exp_five['N-1Play'][1], cross_exp_five['N-3Play'][1]),
        },

        '2/5': {
            'SP':       cross_exp_five['SP'][2],
            'N-XPlay': max(cross_exp_five['N-1Play'][2], cross_exp_five['N-3Play'][2]),
        },

        '3/5': {
            'SP':       cross_exp_five['SP'][3],
            'N-XPlay': max(cross_exp_five['N-1Play'][3], cross_exp_five['N-3Play'][3]),
        },
        '4/5': {
            'SP':       cross_exp_five['SP'][4],
            'N-XPlay': max(cross_exp_five['N-1Play'][4], cross_exp_five['N-3Play'][4]),
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    # plt.rcParams.update({'font.size': FONT_SIZE})
    def plot_dictionary_data(ax, result_dict, title):
        x = []
        y_sp = []
        y_nxplay = []
        
        for key in sorted(result_dict.keys(), key=lambda s: eval(s)):
            x.append(key)
            y_sp.append(result_dict[key]['SP'])
            y_nxplay.append(result_dict[key]['N-XPlay'])
        
        ax.plot(x, y_sp, marker='o', label='SP', color='orange', linewidth=LINEWIDTH)
        ax.plot(x, y_nxplay, marker='s', label='N-XPlay', color='seagreen', linewidth=LINEWIDTH)
        
        if title == 'Team Size = 2':
            ax.set_ylabel('Normalized Reward', fontsize=FONT_SIZE)
            ax.legend(loc='best', fontsize=LEGEND_FONT_SIZE, fancybox=True, framealpha=0.5)

        if title == 'Team Size = 3': 
            ax.set_xlabel('Unseen Count/Team Size', fontsize=FONT_SIZE)

        ax.set_title(title, fontsize=FONT_SIZE)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', labelsize=FONT_SIZE-10)


    plot_dictionary_data(axes[0], result_dictionary_2, 'Team Size = 2')
    plot_dictionary_data(axes[1], result_dictionary_3, 'Team Size = 3')
    plot_dictionary_data(axes[2], result_dictionary_5, 'Team Size = 5')

    plt.tight_layout()
    plt.savefig('data/plots/rss_mrs/unseen_vs_performance_comparison.png')
    plt.show()    
    

if __name__ == "__main__":
    args = get_arguments()
    args.max_workers=1

    five_layout_names, five_p_idxes, five_all_agents_paths, five_teammate_lvl_sets, five_args, five_prefix = get_5_player_input_complex(args)
    five_all_mean_rewards, five_all_std_rewards = run_parallel_evaluation(
        args=five_args,
        all_agents_paths=five_all_agents_paths,
        layout_names=five_layout_names,
        p_idxes=five_p_idxes,
        deterministic=False,
        max_num_teams_per_layout_per_x=4,
        number_of_eps=5,
        teammate_lvl_sets=five_teammate_lvl_sets
    )

    three_layout_names, three_p_idxes, three_all_agents_paths, three_teammate_lvl_sets, three_args, three_prefix = get_3_player_input_complex(args)
    three_all_mean_rewards, three_all_std_rewards = run_parallel_evaluation(
        args=three_args,
        all_agents_paths=three_all_agents_paths,
        layout_names=three_layout_names,
        p_idxes=three_p_idxes,
        deterministic=False,
        max_num_teams_per_layout_per_x=4,
        number_of_eps=5,
        teammate_lvl_sets=three_teammate_lvl_sets
    )

    two_layout_names, two_p_idxes, two_all_agents_paths, two_teammate_lvl_sets, two_args, two_prefix = get_2_player_input_classic(args)
    two_all_mean_rewards, two_all_std_rewards = run_parallel_evaluation(
        args=two_args,
        all_agents_paths=two_all_agents_paths,
        layout_names=two_layout_names,
        p_idxes=two_p_idxes,
        deterministic=False,
        max_num_teams_per_layout_per_x=4,
        number_of_eps=5,
        teammate_lvl_sets=two_teammate_lvl_sets
    )

    # Plotting the results
    plot_unseen_over_teamsize_vs_performance(five_all_mean_rewards, five_all_std_rewards, three_all_mean_rewards, three_all_std_rewards, two_all_mean_rewards, two_all_std_rewards)


