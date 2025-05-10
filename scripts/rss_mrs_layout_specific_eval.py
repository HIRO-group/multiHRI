import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

from oai_agents.common.arguments import get_arguments
from evaluate_agents import run_parallel_evaluation, get_2_player_input_classic, get_3_player_input_complex, get_5_player_input_complex, eval_key_lut
from evaluate_agents_v3_rss_mrs import DISPLAY_NAME_MAP


# Set font for all plots
plt.rcParams['font.family'] = 'Times New Roman'
LINEWIDTH = 5

def get_normalized_reward(args, all_mean_rewards, all_std_rewards, layout_names):
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


def plot_normalized_reward_per_layout(args, all_mean_rewards, all_std_rewards, layout_names):
    AXIS_FONT_SIZE = 22 # Kept same across N agents
    FONT_SIZE = 40
    LEGEND_FONT_SIZE = 24
    # Two players = 40
    # Three players = 26

    cross_exp_mean_final, cross_exp_std_final = get_normalized_reward(args, all_mean_rewards, all_std_rewards, layout_names)
    unseen_counts = range(args.num_players)
    custom_colors = {
        'SP':      'orange',
        'N-1Play': 'seagreen',
        'N-2Play': 'navy',
        'N-3Play': 'orchid',
        'N-4Play': 'saddlebrown',
    }

    num_layouts = len(layout_names)
    layout_images = {
        layout_name: mpimg.imread(f"data/screenshots/{layout_name}/-1.png") for layout_name in layout_names
    }
    
    fig = plt.figure(figsize=(5 * num_layouts, 6))
    gs = gridspec.GridSpec(2, num_layouts, height_ratios=[2, 3],
                            wspace=0.01,
                            hspace=0.01,
                            left=0.045,   # Left margin (as fraction of figure width)
                            right=0.999,
                            top=0.93,    
                            # bottom=0.1   
                            )

    for idx, layout in enumerate(layout_names):
        ax_title = fig.add_subplot(gs[0, idx])
        ax_title.text(0.5, 1.1, DISPLAY_NAME_MAP[layout] ,fontsize=FONT_SIZE, ha='center', va='center')
        ax_title.axis('off')
        ax_img = fig.add_subplot(gs[0, idx])
        ax_img.imshow(layout_images[layout])
        ax_img.axis('off') 

        ax = fig.add_subplot(gs[1, idx])
        for agent in all_mean_rewards:
            means = [cross_exp_mean_final[layout][agent][uc] for uc in unseen_counts]
            stds = [cross_exp_std_final[layout][agent][uc] for uc in unseen_counts]
            ax.errorbar(unseen_counts, means, yerr=stds, label=agent, marker='o', capsize=5, color=custom_colors[agent],linewidth=LINEWIDTH)

        ax.set_xlabel('Unseen Count', fontsize=FONT_SIZE)
        ax.set_xticks(unseen_counts)
        ax.tick_params(axis='both', labelsize=AXIS_FONT_SIZE)

        if idx == 0:
            ax.set_ylabel('Normalized Reward', fontsize=FONT_SIZE)
        else:
            ax.set_yticklabels([])  
            ax.set_ylabel('')       

        if idx == 2:
            ax.legend(loc='best', fontsize=LEGEND_FONT_SIZE, fancybox=True, framealpha=0.5)
    
    # plt.tight_layout()
    plt.savefig(f'data/plots/rss_mrs/layout_specific_eval_{args.num_players}.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    args = get_arguments()
    layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, prefix = get_2_player_input_classic(args)
    # layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, prefix = get_3_player_input_complex(args)
    # layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, prefix = get_5_player_input_complex(args)
    
    args.max_workers = 2
    all_mean_rewards, all_std_rewards = run_parallel_evaluation(
        args=args,
        all_agents_paths=all_agents_paths,
        layout_names=layout_names,
        p_idxes=p_idxes,
        deterministic=False,
        max_num_teams_per_layout_per_x=4,
        number_of_eps=5,
        teammate_lvl_sets=teammate_lvl_sets
    )

    plot_normalized_reward_per_layout(args, all_mean_rewards, all_std_rewards, layout_names)
