import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

from oai_agents.common.arguments import get_arguments
from evaluate_agents import run_parallel_evaluation, eval_key_lut, Eval

AXIS_FONT_SIZE = 18
FONT_SIZE = 24
LEGEND_FONT_SIZE = 16
plt.rcParams['font.family'] = 'Times New Roman'

def get_c1_input(args):
    args.num_players = 2
    args.layout_names = ['c1']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c1/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'MB': 'agent_models/bbe/c1_best_EGO/best_c1/best',
        }
    return args.layout_names, all_agents_paths, args, 'c1'


def get_c2_input(args):
    args.num_players = 2
    args.layout_names = ['c2']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c2/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'MB': 'agent_models/bbe/c2_best_EGO/best_c2/best',
    }
    return args.layout_names, all_agents_paths, args, 'c2'


def get_c3_input(args):
    args.num_players = 2
    args.layout_names = ['c3']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c3/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'MB': 'agent_models/bbe/c3_best_EGO/best_c3/best',
    }
    return args.layout_names, all_agents_paths, args, 'c3'


def get_c4_input(args):
    args.num_players = 2
    args.layout_names = ['c4']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c4/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'MB': 'agent_models/bbe/c4_best_EGO/best_c4/best',
    }
    return args.layout_names, all_agents_paths, args, 'c4'


def get_normalized_reward(args, all_mean_rewards, all_std_rewards, layout_names):
    teammate_lvl_sets = [[Eval.LOW],[Eval.MEDIUM],[Eval.HIGH]]
    team_lvl_set_keys = [str(t) for t in teammate_lvl_sets]
    team_lvl_set_names = [str([eval_key_lut[l] for l in t]) for t in teammate_lvl_sets]
    unseen_counts = range(args.num_players)

    def process_reward(reward, max_reward=None):
        if max_reward:
            reward = reward / max_reward
        return reward

    cross_exp_mean_per_layout = {
        layout_name: { 
            agent_name: {
                unseen_count: [] for unseen_count in unseen_counts
            } for agent_name in all_mean_rewards
        } for layout_name in layout_names
    }

    cross_exp_var_per_layout = {
        layout_name: { 
            agent_name: {
                unseen_count: [] for unseen_count in unseen_counts
            } for agent_name in all_mean_rewards
        } for layout_name in layout_names
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
                    cross_exp_mean_per_layout[layout_name][agent_name][unseen_count].extend(mean_rewards)
                    cross_exp_var_per_layout[layout_name][agent_name][unseen_count].extend([s**2 for s in std_rewards])  # store variance!

    
    cross_exp_mean_final = {
        layout_name: {
            agent_name: {
                unseen_count: np.mean(cross_exp_mean_per_layout[layout_name][agent_name][unseen_count])
                for unseen_count in unseen_counts
            } for agent_name in all_mean_rewards
        } for layout_name in layout_names
    }

    cross_exp_std_final = {
        layout_name: {
            agent_name: {
                unseen_count: np.sqrt(np.mean(cross_exp_var_per_layout[layout_name][agent_name][unseen_count]))  # average the variances, then sqrt
                for unseen_count in unseen_counts
            } for agent_name in all_mean_rewards
        } for layout_name in layout_names
    }

    return cross_exp_mean_final, cross_exp_std_final

def plot_all_mean_reward(args, all_mean_rewards, all_std_rewards, layout_names):
    cross_exp_mean_final, cross_exp_std_final = get_normalized_reward(args, all_mean_rewards, all_std_rewards, layout_names)
    unseen_count = 1

    custom_colors = {
        'MB': 'seagreen',
        'MEP': 'orange',
    }

    num_layouts = len(layout_names)
    bar_width = 0.35
    spacing = 0.2
    agent_names = list(custom_colors.keys())

    fig, ax = plt.subplots(figsize=(5 * num_layouts, 3))

    for i, layout in enumerate(layout_names):
        for j, agent in enumerate(agent_names):
            if agent not in cross_exp_mean_final[layout]:
                continue

            mean = cross_exp_mean_final[layout][agent][unseen_count]
            std = cross_exp_std_final[layout][agent][unseen_count]
            x_pos = i * (len(agent_names) * (bar_width + spacing)) + j * (bar_width + spacing)
            ax.bar(x_pos, mean, yerr=std, width=bar_width, color=custom_colors[agent], label=agent if i == 0 else "", capsize=5)

    # Formatting
    total_bars = num_layouts * len(agent_names)
    xtick_positions = [
        i * (len(agent_names) * (bar_width + spacing)) + (len(agent_names) - 1) * (bar_width + spacing) / 2
        for i in range(num_layouts)
    ]

    ax.set_xticks(xtick_positions)
    ax.tick_params(axis='both', labelsize=AXIS_FONT_SIZE)
    ax.set_xticklabels(['Unseen Count = 0'], rotation=0, ha='center', fontsize=FONT_SIZE)

    # ax.set_ylabel('Normalized Mean Reward', ha='center', fontsize=19)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    # ax.set_title(f'Layout: {layout_names[0]}', fontsize=FONT_SIZE, ha='center', va='center')
    ax.legend(loc='best', fontsize=LEGEND_FONT_SIZE, fancybox=True, framealpha=0.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0)

    plt.tight_layout()
    plt.savefig(f'data/plots/{layout_names[0]}_bar.png')
    plt.show()


if __name__ == "__main__":
    args = get_arguments()
    args.max_workers = 4

    # layout_names, all_agents_paths, args, prefix = get_c1_input(args)
    # all_mean_rewards, all_std_rewards = run_parallel_evaluation(
    #     args=args,
    #     all_agents_paths=all_agents_paths,
    #     layout_names=layout_names,
    #     p_idxes=[0, 1],
    #     deterministic=False,
    #     max_num_teams_per_layout_per_x=4,
    #     number_of_eps=5,
    #     teammate_lvl_sets= [[Eval.LOW],[Eval.MEDIUM],[Eval.HIGH]]
    # )

    # plot_all_mean_reward(args, all_mean_rewards, all_std_rewards, layout_names)


    # layout_names, all_agents_paths, args, prefix = get_c2_input(args)
    # all_mean_rewards, all_std_rewards = run_parallel_evaluation(
    #     args=args,
    #     all_agents_paths=all_agents_paths,
    #     layout_names=layout_names,
    #     p_idxes=[0, 1],
    #     deterministic=False,
    #     max_num_teams_per_layout_per_x=4,
    #     number_of_eps=5,
    #     teammate_lvl_sets= [[Eval.LOW],[Eval.MEDIUM],[Eval.HIGH]]
    # )

    # plot_all_mean_reward(args, all_mean_rewards, all_std_rewards, layout_names)

    # layout_names, all_agents_paths, args, prefix = get_c3_input(args)
    # all_mean_rewards, all_std_rewards = run_parallel_evaluation(
    #     args=args,
    #     all_agents_paths=all_agents_paths,
    #     layout_names=layout_names,
    #     p_idxes=[0, 1],
    #     deterministic=False,
    #     max_num_teams_per_layout_per_x=4,
    #     number_of_eps=5,
    #     teammate_lvl_sets= [[Eval.LOW],[Eval.MEDIUM],[Eval.HIGH]]
    # )
    # plot_all_mean_reward(args, all_mean_rewards, all_std_rewards, layout_names)

    # layout_names, all_agents_paths, args, prefix = get_c1_input(args)
    # layout_names, all_agents_paths, args, prefix = get_c2_input(args)
    # layout_names, all_agents_paths, args, prefix = get_c3_input(args)
    layout_names, all_agents_paths, args, prefix = get_c4_input(args)
    all_mean_rewards, all_std_rewards = run_parallel_evaluation(
        args=args,
        all_agents_paths=all_agents_paths,
        layout_names=layout_names,
        p_idxes=[0, 1],
        deterministic=False,
        max_num_teams_per_layout_per_x=4,
        number_of_eps=5,
        teammate_lvl_sets= [[Eval.LOW],[Eval.MEDIUM],[Eval.HIGH]]
    )
    plot_all_mean_reward(args, all_mean_rewards, all_std_rewards, layout_names)

