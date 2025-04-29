"""
Updated evaluate_agents.py script to support paper for RSS MRS workshop

Summary of differences:
- Added option to show normalize reward on y-axis
- Added option to only show the average (last row) of all results
- Added use of DISPLAY_NAME_MAP to clean up layout names on figures
- Added option to select which unseen counts to show on the line plot (similar to bar plot)

"""
import multiprocessing as mp
from pathlib import Path
mp.set_start_method('spawn', force=True)

import hashlib
from typing import Sequence
import itertools
import concurrent.futures
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

from utils import (
    Complex, Classic
)

class Eval:
    LOW = 'l'
    MEDIUM = 'm'
    HIGH = 'h'

eval_key_lut = {
    'l': "LOW",
    'm': "MEDIUM",
    'h': "HIGH"
}

AGENT_COLOR_MAP = {
    'SP' : 'orange',
    'N-1SP' : 'blue',
    'N-2SP' : 'green'
}

DISPLAY_NAME_MAP = {
    'secret_heaven': "Secret Resources",
    'storage_room': "Resource Corridor",

    'coordination_ring': "Coordination Ring",
    'counter_circuit': "Counter Circuit",
    'cramped_room': "Cramped Room",
    'asymmetric_advantages': "Asym. Adv.",
    'forced_coordination': "Forced Coord.",

    'dec_5_chefs_counter_circuit': "Counter Circuit",
    'dec_5_chefs_storage_room': "Resource Corridor",
    'dec_5_chefs_secret_heaven': "Secret Resources",
    'selected_5_chefs_spacious_room_no_counter_space': "No Counter Space",

    'dec_3_chefs_storage_room': "Resource Corridor",
    'dec_3_chefs_secret_heaven': "Secret Resources",
    'dec_3_chefs_counter_circuit': "Counter Circuit",

}


LAYOUT_NAMES_PATHs = {
    'secret_heaven': {
        Eval.LOW: Complex.L_2,
        Eval.MEDIUM: Complex.M_2,
        Eval.HIGH: Complex.H_2,
    },
    'storage_room': {
        Eval.LOW: Complex.L_2,
        Eval.MEDIUM: Complex.M_2,
        Eval.HIGH: Complex.H_2,
    },

    'coordination_ring': {
        Eval.LOW: Classic.L_2,
        Eval.MEDIUM: Classic.M_2,
        Eval.HIGH: Classic.H_2,
    },
    'counter_circuit': {
        Eval.LOW: Classic.L_2,
        Eval.MEDIUM: Classic.M_2,
        Eval.HIGH: Classic.H_2,
    },
    'cramped_room': {
        Eval.LOW: Classic.L_2,
        Eval.MEDIUM: Classic.M_2,
        Eval.HIGH: Classic.H_2,
    },
    'asymmetric_advantages': {
        Eval.LOW: Classic.L_2,
        Eval.MEDIUM: Classic.M_2,
        Eval.HIGH: Classic.H_2,
    },
    'forced_coordination': {
        Eval.LOW: Classic.L_2,
        Eval.MEDIUM: Classic.M_2,
        Eval.HIGH: Classic.H_2,
    },

    'dec_5_chefs_counter_circuit': {
        Eval.LOW: Complex.L_5,
        Eval.MEDIUM: Complex.M_5,
        Eval.HIGH: Complex.H_5,
    },
    'dec_5_chefs_storage_room': {
        Eval.LOW: Complex.L_5,
        Eval.MEDIUM: Complex.M_5,
        Eval.HIGH: Complex.H_5,
    },
    'dec_5_chefs_secret_heaven': {
        Eval.LOW: Complex.L_5,
        Eval.MEDIUM: Complex.M_5,
        Eval.HIGH: Complex.H_5,
    },
    'selected_5_chefs_spacious_room_no_counter_space': {
        Eval.LOW: Complex.L_5,
        Eval.MEDIUM: Complex.M_5,
        Eval.HIGH: Complex.H_5,
    },

    'dec_3_chefs_storage_room': {
        Eval.LOW: Complex.L_3,
        Eval.MEDIUM: Complex.M_3,
        Eval.HIGH: Complex.H_3,
    },
    'dec_3_chefs_secret_heaven': {
        Eval.LOW: Complex.L_3,
        Eval.MEDIUM: Complex.M_3,
        Eval.HIGH: Complex.H_3,
    },
    'dec_3_chefs_counter_circuit': {
        Eval.LOW: Complex.L_3,
        Eval.MEDIUM: Complex.M_3,
        Eval.HIGH: Complex.H_3,
    },
}


def print_all_teammates(all_teammates):
    for layout_name in all_teammates:
        print('Layout:', layout_name)
        for teammates in all_teammates[layout_name]:
            print([agent.name for agent in teammates])
        print()

def get_all_teammates_for_evaluation(args, primary_agent, num_players, layout_names, deterministic, max_num_teams_per_layout_per_x, teammate_lvl_set: Sequence[Eval]=[Eval.LOW, Eval.MEDIUM, Eval.HIGH]):
    '''
    x = 0 means all N-1 teammates are primary_agent
    x = 1 means 1 teammate out of N-1 is unseen agent
    x = 2 means 2 teammates out of N-1- are unseen agents
    '''

    N = num_players
    X = list(range(N))
    # X = [1]

    # Contains all the agents which are later used to create all_teammates
    all_agents = {layout_name: [] for layout_name in layout_names}
    # Containts teams for each layout and each x up to MAX_NUM_TEAMS_PER_LAYOUT_PER_X
    all_teammates = {
        layout_name: {
            unseen_count: [] for unseen_count in X}
        for layout_name in layout_names}

    for layout_name in layout_names:
        for lvl in teammate_lvl_set:
            for path in LAYOUT_NAMES_PATHs[layout_name][lvl]:
                agent = load_agent(Path(path), args)
                agent.deterministic = deterministic
                all_agents[layout_name].append(agent)

    for layout_name in layout_names:
        agents = all_agents[layout_name]

        for unseen_count in X:
            teammates_list = []

            if unseen_count == 0:
                num_teams_max = 1
            else:
                num_teams_max = min(max_num_teams_per_layout_per_x, len(agents)//unseen_count)

            for num_teams in range(num_teams_max):
                teammates = [primary_agent] * (N-1-unseen_count)
                for i in range(unseen_count):
                    try:
                        teammates.append(agents[i + (num_teams)])
                    except RuntimeError:
                        continue
                if len(teammates) == N-1:
                    teammates_list.append(teammates)
            all_teammates[layout_name][unseen_count] = teammates_list
    return all_teammates


def generate_plot_name(prefix, num_players, deterministic, p_idxes, num_eps, max_num_teams, teammate_lvl_sets):
    plot_name = f'{prefix}-{num_players}-players'
    plot_name += '-det' if deterministic else '-stoch'
    p_idexes_str = ''.join([str(p_idx) for p_idx in p_idxes])
    plot_name += f'-pidx{p_idexes_str}'
    plot_name += f'-eps{num_eps}'
    plot_name += f'-maxteams{str(max_num_teams)}'
    teams = ''.join([str(t[0]) for t in teammate_lvl_sets])
    plot_name += f"-teams({str(teams)})"
    return plot_name


def plot_evaluation_results_bar(all_mean_rewards, 
                                all_std_rewards, 
                                layout_names, 
                                teammate_lvl_sets, 
                                plot_name, 
                                unseen_counts=None, 
                                display_delivery=False, 
                                normalize_rewards=False,
                                only_show_avg_plots=False):

    unseen_counts = unseen_counts or [0]
    if display_delivery:
        plot_name += "_delivery"
    elif normalize_rewards:
        plot_name += "_normalized"
    uc = ''.join([str(u) for u in unseen_counts])
    plot_name += f"_uc{uc}"

    num_layouts = len(layout_names)
    team_lvl_set_keys = [str(t) for t in teammate_lvl_sets]
    team_lvl_set_names = [str([eval_key_lut[l] for l in t]) for t in teammate_lvl_sets]
    num_teamsets = len(team_lvl_set_names)
    if only_show_avg_plots:
        fig, axes = plt.subplots(1, num_layouts, figsize=(5 * num_layouts, 5), sharey=True)
    else:
        fig, axes = plt.subplots(num_teamsets + 1, num_layouts, figsize=(5 * num_layouts, 5 * (num_teamsets + 1)), sharey=True)

    if num_layouts == 1:
        axes = [[axes]]

    x_values = np.arange(len(unseen_counts))
    num_agents = len(all_mean_rewards)
    width = 0.8 / num_agents  # Adjust bar width based on number of agents

    def process_reward(reward, max_reward=None):
        if display_delivery:
            # Each delivery provides 20 points so divide the total reward by 20 to get number of deliveries
            reward = reward / 20
        elif max_reward:
            # Normalize reward and turn it into a percentage using the provided maximum
            reward = reward / max_reward
        return reward

    for i, layout_name in enumerate(layout_names):
        cross_exp_mean = {}
        cross_exp_std = {}
        for j, (team, team_name) in enumerate(zip(team_lvl_set_keys, team_lvl_set_names)):

            if only_show_avg_plots:
                ax = axes[i]
            else:
                ax = axes[j][i]

            # Determine the max reward received by any agent in this layout across all unseen counts
            rewards_for_all_agents = []
            for agent in all_mean_rewards:
                rewards_for_all_agents.extend(np.concatenate(list(all_mean_rewards[agent][team][layout_name].values())))
            max_mean_reward = max(rewards_for_all_agents)

            for idx, agent_name in enumerate(all_mean_rewards):
                mean_values = []
                std_values = []

                for unseen_count in unseen_counts:
                    if normalize_rewards:
                        # Use same max for both the mean reward and the std 
                        mean_rewards = [process_reward(r, max_reward=max_mean_reward) for r in all_mean_rewards[agent_name][team][layout_name][unseen_count]]
                        std_rewards = [process_reward(r, max_reward=max_mean_reward) for r in all_std_rewards[agent_name][team][layout_name][unseen_count]]
                    else:
                        mean_rewards = [process_reward(r) for r in all_mean_rewards[agent_name][team][layout_name][unseen_count]]
                        std_rewards = [process_reward(r) for r in all_std_rewards[agent_name][team][layout_name][unseen_count]]

                    mean_values.append(np.mean(mean_rewards))
                    std_values.append(np.mean(std_rewards))
                    if agent_name not in cross_exp_mean:
                        cross_exp_mean[agent_name] = [0] * len(unseen_counts)
                    if agent_name not in cross_exp_std:
                        cross_exp_std[agent_name] = [0] * len(unseen_counts)
                    cross_exp_mean[agent_name][unseen_counts.index(unseen_count)] += mean_values[-1]
                    cross_exp_std[agent_name][unseen_counts.index(unseen_count)] += std_values[-1]

                # Plot bars for each agent
                if not only_show_avg_plots:
                    x = x_values + idx * width - width * (num_agents - 1) / 2
                    ax.bar(x, mean_values, width, yerr=std_values, label=f'{agent_name}', color=AGENT_COLOR_MAP[agent_name], capsize=5)

            if not only_show_avg_plots:
                team_name_print = team_name.strip("[]'\"")
                # ax.set_title(f'{DISPLAY_NAME_MAP[layout_name]}\n{team_name_print}')
                ax.set_xlabel('Unseen Teammates', fontsize=30)
                ax.set_xticks(x_values)
                ax.set_xticklabels(unseen_counts)
                if display_delivery:
                    ax.set_yticks(np.arange(0, 20, 1))
                    axes[0,i].set_ylabel('Number Deliveries', fontsize=30)
                elif normalize_rewards:
                    ax.set_yticks(np.arange(0, 1 + max(std_values), 0.2))
                    axes[0,i].set_ylabel('Normalized Reward', fontsize=30)
                else:
                    ax.set_yticks(np.arange(0, max(mean_rewards), 1))
                    axes[0][i].set_ylabel('Reward', fontsize=30)
                ax.legend(loc='best', fontsize=28, fancybox=True, framealpha=0.5)


        # Average plot across all teamsets
        if only_show_avg_plots:
            ax = axes[i]
        else:
            ax = axes[-1][i]
        for idx, agent_name in enumerate(all_mean_rewards):
            mean_values = [v / num_teamsets for v in cross_exp_mean[agent_name]]
            std_values = [v / num_teamsets for v in cross_exp_std[agent_name]]

            x = x_values + idx * width - width * (num_agents - 1) / 2
            ax.bar(x, mean_values, width, yerr=std_values, label=f"{agent_name}", color=AGENT_COLOR_MAP[agent_name], capsize=5)


        # ax.set_title(f"Avg. {DISPLAY_NAME_MAP[layout_name]}")
        ax.set_xlabel('Unseen Teammates', fontsize=30)
        ax.set_xticks(x_values)
        ax.set_xticklabels(unseen_counts)
        if display_delivery:
            ax.set_yticks(np.arange(0, 20, 1))
        elif normalize_rewards:
            ax.set_yticks(np.arange(0, 1 + max(std_values), 0.1))
        else:
            ax.set_yticks(np.arange(0, max(mean_rewards), 1))
        ax.legend(loc='best', fontsize=28, fancybox=True, framealpha=0.5)

    # Set y-axis label based on display_delivery
    if display_delivery:
        y_label = 'Number of Deliveries'
    elif normalize_rewards:
        y_label = 'Normalized Reward'
    else:
        y_label = 'Reward'
    fig.text(0.0, 0.5, y_label, va='center', fontsize=30, rotation='vertical')

    plt.tight_layout()
    plt.savefig(f'data/plots/{plot_name}_{"deliveries" if display_delivery else "rewards"}_bar.png')


def plot_evaluation_results_line(all_mean_rewards, 
                                 all_std_rewards, 
                                 layout_names, 
                                 teammate_lvl_sets, 
                                 plot_name,
                                 unseen_counts=None, 
                                 display_delivery=False,
                                 normalize_rewards=False,
                                 only_show_avg_plots=False):

    unseen_counts = unseen_counts or [0]
    if display_delivery:
        plot_name += "_delivery"
    elif normalize_rewards:
        plot_name += "_normalized"
    uc = ''.join([str(u) for u in unseen_counts])
    plot_name += f"_uc{uc}"

    num_layouts = len(layout_names)
    team_lvl_set_keys = [str(t) for t in teammate_lvl_sets]
    team_lvl_set_names = [str([eval_key_lut[l] for l in t]) for t in teammate_lvl_sets]
    num_teamsets = len(team_lvl_set_names)
    if only_show_avg_plots:
        fig, axes = plt.subplots(1, num_layouts, figsize=(5 * num_layouts, 5), sharey=True)
    else:
        fig, axes = plt.subplots(num_teamsets + 1, num_layouts, figsize=(5 * num_layouts, 5 * (num_teamsets + 1)), sharey=True)

    if num_layouts == 1:
        axes = [[axes]]

    x_values = unseen_counts

    def process_reward(reward, max_reward=None):
        if display_delivery:
            # Each delivery provides 20 points so divide the total reward by 20 to get number of deliveries
            reward = reward / 20
        elif max_reward:
            # Normalize reward and turn it into a percentage using the provided maximum
            reward = reward / max_reward
        return reward

    for i, layout_name in enumerate(layout_names):
        cross_exp_mean = {}
        cross_exp_std = {}
        for j, (team, team_name) in enumerate(zip(team_lvl_set_keys, team_lvl_set_names)):

            if only_show_avg_plots:
                ax = axes[i]
            else:
                ax = axes[j][i]

            # Determine the max reward received by any agent in this layout across all unseen counts
            rewards_for_all_agents = []
            for agent in all_mean_rewards:
                rewards_for_all_agents.extend(np.concatenate(list(all_mean_rewards[agent][team][layout_name].values())))

            max_mean_reward = max(rewards_for_all_agents)

            for agent_name in all_mean_rewards:
                mean_values = []
                std_values = []

                # for unseen_count in range(num_players):
                for unseen_count in unseen_counts:
                    if normalize_rewards:
                        # Use same max for both the mean reward and the std
                        mean_rewards = [process_reward(r, max_reward=max_mean_reward) for r in all_mean_rewards[agent_name][team][layout_name][unseen_count]]
                        std_rewards = [process_reward(r, max_reward=max_mean_reward) for r in all_std_rewards[agent_name][team][layout_name][unseen_count]]
                    else:
                        mean_rewards = [process_reward(r) for r in all_mean_rewards[agent_name][team][layout_name][unseen_count]]
                        std_rewards = [process_reward(r) for r in all_std_rewards[agent_name][team][layout_name][unseen_count]]

                    mean_values.append(np.mean(mean_rewards))
                    std_values.append(np.mean(std_rewards))
                    if agent_name not in cross_exp_mean:
                        cross_exp_mean[agent_name] = [0] * len(unseen_counts)
                    if agent_name not in cross_exp_std:
                        cross_exp_std[agent_name] = [0] * len(unseen_counts)
                    cross_exp_mean[agent_name][unseen_count] += mean_values[-1]
                    cross_exp_std[agent_name][unseen_count] += std_values[-1]

                if not only_show_avg_plots:
                    ax.errorbar(x_values, mean_values, yerr=std_values, fmt='-o',
                                label=f'{agent_name}', color=AGENT_COLOR_MAP[agent_name], capsize=5)

            if not only_show_avg_plots:
                team_name_print = team_name.strip("[]'\"")
                # ax.set_title(f'{DISPLAY_NAME_MAP[layout_name]}\n{team_name_print}')
                ax.set_xlabel('Unseen Teammates', fontsize=30)
                ax.set_xticks(x_values)
                if display_delivery:
                    ax.set_yticks(np.arange(0, 20, 1))
                    axes[0][i].set_ylabel('Number Deliveries')
                elif normalize_rewards:
                    ax.set_yticks(np.arange(0, 1.1, 0.2))
                    axes[0][i].set_ylabel('Normalized Reward')
                else:
                    ax.set_yticks(np.arange(0, max(mean_rewards), 1))
                    axes[0][i].set_ylabel('Reward')

                ax.legend(loc='best', fontsize=28, fancybox=True, framealpha=0.5)

        if only_show_avg_plots:
            ax = axes[i]
            if display_delivery:
                axes[0].set_ylabel('Number Deliveries', fontsize=30)
            elif normalize_rewards:
                axes[0].set_ylabel('Normalized Reward', fontsize=30)
            else:
                axes[0].set_ylabel('Reward', fontsize=30)
        else:
            ax = axes[-1][i]
            if display_delivery:
                axes[0][i].set_ylabel('Number Deliveries', fontsize=30)
            elif normalize_rewards:
                axes[0][i].set_ylabel('Normalized Reward', fontsize=30)
            else:
                axes[0][i].set_ylabel('Reward', fontsize=30)

        for agent_name in all_mean_rewards:
            mean_values = [v / num_teamsets for v in cross_exp_mean[agent_name]]
            std_values = [v / num_teamsets for v in cross_exp_std[agent_name]]
            ax.errorbar(x_values, mean_values, yerr=std_values, fmt="-o", label=f"{agent_name}", color=AGENT_COLOR_MAP[agent_name], capsize=5)

        # ax.set_title(f"Avg. {DISPLAY_NAME_MAP[layout_name]}")
        ax.set_xlabel('Unseen Teammates', fontsize=30)
        ax.set_xticks(x_values)
        if display_delivery:
                ax.set_yticks(np.arange(0, 20, 1))
        elif normalize_rewards:
            ax.set_yticks(np.arange(0, 1.1, 0.2))
        else:
            ax.set_yticks(np.arange(0, max(mean_rewards), 1))
        ax.tick_params(axis='both', labelsize=22)
        ax.legend(loc='best', fontsize=28, fancybox=True, framealpha=0.5)



    plt.tight_layout()
    plt.savefig(f'data/plots/{plot_name}_line.png')
    # plt.show()



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
        # for unseen_count in [1]:
        for unseen_count in range(args.num_players):
            for teammates in all_teammates[layout_name][unseen_count]:
                env = OvercookedGymEnv(args=args,
                                       layout_name=layout_name,
                                       ret_completed_subtasks=False,
                                       is_eval_env=True,
                                       horizon=400,
                                       deterministic=deterministic,
                                       learner_type='originaler'
                                       )
                env.set_teammates(teammates)
                for p_idx in p_idxes:
                    env.set_reset_p_idx(p_idx)
                    mean_reward, std_reward = evaluate_policy(primary_agent, env,
                                                              n_eval_episodes=number_of_eps,
                                                              deterministic=deterministic,
                                                              warn=False,
                                                              render=False)
                    all_mean_rewards[layout_name][unseen_count].append(mean_reward)
                    all_std_rewards[layout_name][unseen_count].append(std_reward)

    return all_mean_rewards, all_std_rewards


def evaluate_agent_for_layout(agent_name, path, layout_names, p_idxes, args, deterministic, max_num_teams_per_layout_per_x, number_of_eps, teammate_lvl_set: Sequence[Eval]):
    fn_args = (args.num_players, path, tuple(layout_names), tuple(p_idxes), deterministic, max_num_teams_per_layout_per_x, number_of_eps, tuple(teammate_lvl_set))
    m = hashlib.md5()
    for s in fn_args:
        m.update(str(s).encode())
    arg_hash = m.hexdigest()
    cached_eval = Path(f"eval_cache/eval_{arg_hash}.pkl")

    if cached_eval.is_file():
        print(f"Loading cached evaluation for agent {agent_name}")
        with open(cached_eval, "rb") as f:
            teammate_lvl_set, mean_rewards, std_rewards = pkl.load(f)

    else:
        print(f"Evaluating agent: {agent_name}")
        agent = load_agent(Path(path), args)
        agent.deterministic = deterministic

        all_teammates = get_all_teammates_for_evaluation(args=args,
                                                        primary_agent=agent,
                                                        num_players=args.num_players,
                                                        layout_names=layout_names,
                                                        deterministic=deterministic,
                                                        max_num_teams_per_layout_per_x=max_num_teams_per_layout_per_x,
                                                        teammate_lvl_set=teammate_lvl_set)

        mean_rewards, std_rewards = evaluate_agent(args=args,
                                                primary_agent=agent,
                                                p_idxes=p_idxes,
                                                layout_names=layout_names,
                                                all_teammates=all_teammates,
                                                deterministic=deterministic,
                                                number_of_eps=number_of_eps)

        Path('eval_cache').mkdir(parents=True, exist_ok=True)
        with open(cached_eval, "wb") as f:
            pkl.dump((teammate_lvl_set, mean_rewards, std_rewards), f)

    return agent_name, str(teammate_lvl_set), mean_rewards, std_rewards


def run_parallel_evaluation(args, all_agents_paths, layout_names, p_idxes, deterministic, max_num_teams_per_layout_per_x, number_of_eps, teammate_lvl_sets: Sequence[Sequence[Eval]]):
    # for path in all_agents_paths.values():
    #     assert Path(path+'/trainer_file').is_file(), f"File {path+'/trainer_file'} does not exist"

    all_mean_rewards, all_std_rewards = {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(evaluate_agent_for_layout, name, path, layout_names, p_idxes, args, deterministic, max_num_teams_per_layout_per_x, number_of_eps, teammate_lvl_set)
            for (name, path), teammate_lvl_set in itertools.product(all_agents_paths.items(), teammate_lvl_sets)
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating Agents"):
            name, teammate_lvl_set_str, mean_rewards, std_rewards = future.result()
            if name not in all_mean_rewards:
                all_mean_rewards[name] = {}
            if name not in all_std_rewards:
                all_std_rewards[name] = {}
            all_mean_rewards[name][teammate_lvl_set_str] = mean_rewards
            all_std_rewards[name][teammate_lvl_set_str] = std_rewards

    return all_mean_rewards, all_std_rewards



def get_2_player_input_classic(args):
    args.num_players = 2
    args.layout_names = [
        'coordination_ring',
        'counter_circuit',
        'cramped_room',
        'asymmetric_advantages',
        'forced_coordination'
        ]
    p_idxes = [0, 1]
    all_agents_paths = {
        'SP':    'agent_models/RSS_MRS/Training/Classic/2/SP_hd256_seed1010/best',
        'N-1SP': 'agent_models/RSS_MRS/Training/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
    }

    teammate_lvl_sets = [
        [Eval.LOW],
        [Eval.MEDIUM],
        [Eval.HIGH]
    ]
    return args.layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, 'classic'



def get_2_player_input_complex(args):
    args.num_players = 2
    args.layout_names = [
        'secret_heaven',
        'storage_room'
        ]
    p_idxes = [0, 1]
    all_agents_paths = {
        'SP':    'agent_models/RSS_MRS/Training/Complex/2/SP_hd256_seed1010/best',
        'N-1SP': 'agent_models/RSS_MRS/Training/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
    }

    teammate_lvl_sets = [
        [Eval.LOW],
        [Eval.MEDIUM],
        [Eval.HIGH]
    ]
    return args.layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, 'complex'


def get_3_player_input_complex(args):
    args.num_players = 3
    args.layout_names = [
        'dec_3_chefs_storage_room',
        'dec_3_chefs_secret_heaven',
        'dec_3_chefs_counter_circuit',
    ]

    p_idxes = [0, 1, 2]
    all_agents_paths = {
        'SP':    'agent_models/RSS_MRS/Training/Complex/3/SP_hd256_seed1010/best',
        'N-1SP': 'agent_models/RSS_MRS/Training/Complex/3/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
    }
    teammate_lvl_sets = [
        [Eval.LOW],
        [Eval.MEDIUM],
        [Eval.HIGH]
    ]
    return args.layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, 'complex'



def get_5_player_input_complex(args):
    args.num_players = 5
    args.layout_names = [
        'dec_5_chefs_counter_circuit',
        'dec_5_chefs_storage_room',
        'dec_5_chefs_secret_heaven',
        'selected_5_chefs_spacious_room_no_counter_space',
        ]

    p_idxes = [0, 1, 2, 3, 4]
    all_agents_paths = {
        'SP':    'agent_models/RSS_MRS/Training/Complex/5/SP_hd256_seed1010/best',
        'N-1SP': 'agent_models/RSS_MRS/Training/Complex/5/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
    }
    teammate_lvl_sets = [
        [Eval.LOW],
        [Eval.MEDIUM],
        [Eval.HIGH]
    ]
    return args.layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, 'complex'


if __name__ == "__main__":
    args = get_arguments()
    # layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, prefix = get_2_player_input_classic(args)
    # layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, prefix = get_2_player_input_complex(args)
    layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, prefix = get_3_player_input_complex(args)
    # layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, prefix = get_5_player_input_complex(args)

    deterministic = False # deterministic = True does not actually work :sweat_smile:
    max_num_teams_per_layout_per_x = 4
    number_of_eps = 5

    # Number of parallel workers for evaluation
    args.max_workers = 1

    # For display_purposes
    unseen_counts = [0, 1, 2]
    show_delivery_num = False
    normalize_rewards = True    # Normalized rewards is ignored if when show_delivery_num is True
    only_show_avg_plots = True

    plot_name = generate_plot_name( prefix=prefix,
                                    num_players=args.num_players,
                                    deterministic=deterministic,
                                    p_idxes=p_idxes,
                                    num_eps=number_of_eps,
                                    max_num_teams=max_num_teams_per_layout_per_x,
                                    teammate_lvl_sets=teammate_lvl_sets)

    all_mean_rewards, all_std_rewards = run_parallel_evaluation(
            args=args,
            all_agents_paths=all_agents_paths,
            layout_names=layout_names,
            p_idxes=p_idxes,
            deterministic=deterministic,
            max_num_teams_per_layout_per_x=max_num_teams_per_layout_per_x,
            number_of_eps=number_of_eps,
            teammate_lvl_sets=teammate_lvl_sets
    )

    plot_evaluation_results_bar(all_mean_rewards=all_mean_rewards,
                           all_std_rewards=all_std_rewards,
                           layout_names=layout_names,
                           teammate_lvl_sets=teammate_lvl_sets,
                           unseen_counts=unseen_counts,
                           display_delivery=show_delivery_num,
                           plot_name=plot_name,
                           normalize_rewards=normalize_rewards,
                           only_show_avg_plots=only_show_avg_plots)


    plot_evaluation_results_line(all_mean_rewards=all_mean_rewards,
                                     all_std_rewards=all_std_rewards,
                                     layout_names=layout_names,
                                     teammate_lvl_sets=teammate_lvl_sets,
                                     unseen_counts=unseen_counts,
                                     plot_name=plot_name,
                                     display_delivery=show_delivery_num,
                                     normalize_rewards=normalize_rewards,
                                     only_show_avg_plots=only_show_avg_plots
                                     )
