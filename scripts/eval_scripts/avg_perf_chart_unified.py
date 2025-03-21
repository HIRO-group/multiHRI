import multiprocessing as mp
import os
from pathlib import Path

import matplotlib
from torch import index_select
from wandb import agent
mp.set_start_method('spawn', force=True)

import hashlib
import sys
from typing import Sequence
import itertools
import concurrent.futures
from tqdm import tqdm
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle as pkl
import warnings

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

from utils import (
    Complex, Classic,
    TWO_PLAYERS_LOW_EVAL,
    TWO_PLAYERS_MEDIUM_EVAL,
    TWO_PLAYERS_HIGH_EVAL,
    THREE_PLAYERS_LOW_EVAL,
    THREE_PLAYERS_MEDIUM_EVAL,
    THREE_PLAYERS_HIGH_EVAL,
    FIVE_PLAYERS_LOW_EVAL,
    FIVE_PLAYERS_MEDIUM_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL,
    FIVE_PLAYERS_HIGH_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL,
    FIVE_PLAYERS_MEDIUM_STORAGE_EVAL,
    FIVE_PLAYERS_HIGH_STORAGE_EVAL
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

DISPLAY_NAME_MAP = {
    'secret_heaven': "Secret Resources",
    'storage_room': "Resource Corridor",
    'coordination_ring': "Coord. Ring",
    'counter_circuit': "Counter Circuit",
    'cramped_room': "Cramped Room",
    'asymmetric_advantages': "Asym. Adv.",
    'forced_coordination': "Forced Coord.",
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


def generate_plot_name(prefix, num_players, deterministic, p_idxes, num_eps, max_num_teams, teammate_lvl_sets):
    plot_name = f'final-chart-{prefix}-{num_players}-players'
    plot_name += '-det' if deterministic else '-stoch'
    p_idexes_str = ''.join([str(p_idx) for p_idx in p_idxes])
    plot_name += f'-pidx{p_idexes_str}'
    plot_name += f'-eps{num_eps}'
    plot_name += f'-maxteams{str(max_num_teams)}'
    teams = ''.join([str(t[0]) for t in teammate_lvl_sets])
    plot_name += f"-teams({str(teams)})"
    return plot_name


def plot_evaluation_results_bar(fig, axes, all_mean_rewards, all_std_rewards, layout_names, teammate_lvl_sets, plot_name, unseen_counts=[0], display_delivery=False, start=0):
    #cmap = matplotlib.colormaps.get_cmap("Set3")
    plot_name = plot_name + "_delivery" if display_delivery else plot_name
    uc = ''.join([str(u) for u in unseen_counts])
    plot_name += f"_uc{uc}"

    num_layouts = len(layout_names)
    team_lvl_set_keys = [str(t) for t in teammate_lvl_sets]
    team_lvl_set_names = [str([eval_key_lut[l] for l in t]) for t in teammate_lvl_sets]
    num_teamsets = len(team_lvl_set_names)

    if num_layouts == 1:
        axes = [[axes]]

    axes = [[axes]]

    x_values = np.arange(len(unseen_counts))
    num_agents = len(all_mean_rewards)
    width = 0.8 / num_agents  # Adjust bar width based on number of agents

    # Function to process rewards (divide by 20 if display_delivery is True)
    def process_reward(reward):
        return reward / 20 if display_delivery else reward

    chart_data = {}
    for i, layout_name in enumerate(layout_names):
        cross_exp_mean = {}
        cross_exp_std = {}
        for j, (team, team_name) in enumerate(zip(team_lvl_set_keys, team_lvl_set_names)):
            #ax = axes[j][i]
            for idx, agent_name in enumerate(all_mean_rewards):
                mean_values = []
                std_values = []

                for unseen_count in unseen_counts:
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
                x = x_values + idx * width - width * (num_agents - 1) / 2
        #         ax.bar(x, mean_values, width, yerr=std_values, label=f'{agent_name}', capsize=5, color=cmap.colors)

        #     team_name_print = team_name.strip("[]'\"")
        #     ax.set_title(f'{layout_name}\n{team_name_print}')
        #     ax.set_xlabel('Number of Unseen Teammates')
        #     ax.set_xticks(x_values)
        #     ax.set_xticklabels(unseen_counts)
        #     ax.set_yticks(np.arange(0, 20, 1))
        #     ax.legend(loc='upper right', fontsize='small', fancybox=True, framealpha=0.5)

        # Average plot across all teamsets
        ax = axes[-1][0]
        for idx, agent_name in enumerate(all_mean_rewards):
            mean_values = [v / num_teamsets for v in cross_exp_mean[agent_name]]
            std_values = [v / num_teamsets for v in cross_exp_std[agent_name]]

            x = x_values + idx * width - width * (num_agents - 1) / 2

            if not (agent_name in chart_data):
                chart_data[agent_name] = {
                    "mean": [],
                    "std": [],
                    "layout": []
                }


            chart_data[agent_name]["mean"].append(mean_values[0])
            chart_data[agent_name]["std"].append(std_values[0])
            chart_data[agent_name]["layout"].append(layout_name)

            #ax.bar(x, mean_values, width, yerr=std_values, label=f"Agent: {agent_name}", capsize=5)

    ax = axes[-1][0]
    agents = list(chart_data.values())
    num_agents = len(agents)
    width = .95 / num_agents
    layouts = agents[0]["layout"]
    idxs = np.arange(len(layouts))
    cmap = matplotlib.colormaps["tab20b"]

    for i, (agent_name, d) in enumerate(chart_data.items()):
        ax.bar(start + idxs + (i * width), d["mean"], width, yerr=d["std"], label=agent_name if start == 0 else None, color=cmap(i*5), capsize=4)


def plot_evaluation_results_line(all_mean_rewards, all_std_rewards, layout_names, teammate_lvl_sets, num_players, plot_name):
    num_layouts = len(layout_names)
    team_lvl_set_keys = [str(t) for t in teammate_lvl_sets]
    team_lvl_set_names = [str([eval_key_lut[l] for l in t]) for t in teammate_lvl_sets]
    num_teamsets = len(team_lvl_set_names)
    fig, axes = plt.subplots(num_teamsets + 1, num_layouts, figsize=(5 * num_layouts, 5 * (num_teamsets + 1)), sharey=True)

    if num_layouts == 1:
        axes = [[axes]]

    x_values = np.arange(num_players)

    for i, layout_name in enumerate(layout_names):
        cross_exp_mean = {}
        cross_exp_std = {}
        for j, (team, team_name) in enumerate(zip(team_lvl_set_keys, team_lvl_set_names)):
            ax = axes[j][i]
            for agent_name in all_mean_rewards:
                mean_values = []
                std_values = []

                for unseen_count in range(num_players):
                    mean_rewards = all_mean_rewards[agent_name][team][layout_name][unseen_count]
                    std_rewards = all_std_rewards[agent_name][team][layout_name][unseen_count]

                    mean_values.append(np.mean(mean_rewards))
                    std_values.append(np.mean(std_rewards))
                    if agent_name not in cross_exp_mean:
                        cross_exp_mean[agent_name] = [0] * num_players
                    if agent_name not in cross_exp_std:
                        cross_exp_std[agent_name] = [0] * num_players
                    cross_exp_mean[agent_name][unseen_count] += mean_values[-1]
                    cross_exp_mean[agent_name][unseen_count] += std_values[-1]


                ax.errorbar(x_values, mean_values, yerr=std_values, fmt='-o',
                            label=f'Agent: {agent_name}', capsize=5)
            team_name_print = team_name.strip("[]'\"")
            ax.set_title(f'{layout_name}\n{team_name_print}')
            ax.set_xlabel('Number of Unseen Teammates')
            ax.set_xticks(x_values)
            ax.legend(loc='upper right', fontsize='small', fancybox=True, framealpha=0.5)

        ax = axes[-1][i]
        for agent_name in all_mean_rewards:
            mean_values = [v / num_teamsets for v in cross_exp_mean[agent_name]]
            std_values = [v / num_teamsets for v in cross_exp_std[agent_name]]
            ax.errorbar(x_values, mean_values, yerr=std_values, fmt="-o", label=f"Agent: {agent_name}", capsize=5)


        my_cmap = plt.get_cmap("viridis")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

        ax.set_title(f"Avg. {layout_name}")
        ax.set_xlabel('Number of Unseen Teammates')
        ax.set_xticks(x_values)
        ax.legend(loc='upper right', fontsize='small', fancybox=True, framealpha=0.5)


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
    for path in all_agents_paths.values():
        if not Path(path+'/trainer_file').is_file():
            warnings.warn(f"File {path+'/trainer_file'} does not exist")

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
        # 'SP_s13_h256': 'agent_models/Classic/2/SP_hd256_seed13/best',
        'SP': 'agent_models/Classic/2/SP_hd256_seed1010/best',
        'FCP': 'agent_models/Classic/2/FCP_s1010_h256_tr[AMX]_ran/best',

        # 'dsALMH 1d[2t] 1s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0/best',
        # 'dsALMH 2d[2t] 2s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1/best',
        'CAP 3d 3s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2/best',

        # 'sALMH 1s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack0/best',
        # 'sALMH 2s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack1/best',
        # 'sALMH 3s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack2/best',

        # 'dALMH 1s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA]_ran_originaler_attack0/best',
        # 'dALMH 2s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA]_ran_originaler_attack1/best',
        # 'dALMH 3s': 'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA]_ran_originaler_attack2/best',

        # 'dsALMH 1d[5t] 1s': 'agent_models/Classic/2/5_steps_in_dynamic_advs/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0/best',
        # 'dsALMH 2d[5t] 2s': 'agent_models/Classic/2/5_steps_in_dynamic_advs/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1/best',
        # 'dsALMH 3d[5t] 3s': 'agent_models/Classic/2/5_steps_in_dynamic_advs/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2/best',

    }
    teammate_lvl_sets = [
        [Eval.LOW],
        # [Eval.MEDIUM],
        # [Eval.HIGH]
    ]
    return args.layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, 'classic'



def get_2_player_input_complex(args):
    args.num_players = 2
    args.layout_names = [
        'secret_heaven',
        'storage_room',
    ]
    p_idxes = [0, 1]
    all_agents_paths = {
        # 'SP_s13_h256': 'agent_models/Complex/2/SP_hd256_seed13/best',
        'SP': 'agent_models/Complex/2/SP_hd256_seed1010/best',
        'FCP': 'agent_models/Complex/2/FCP_s1010_h256_tr[AMX]_ran/best',

        # 'dsALMH 1d[2t] 1s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0/best',
        # 'dsALMH 2d[2t] 2s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1/best',
        'CAP 3d 3s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2/best',

        # 'sALMH 1s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack0/best',
        # 'sALMH 2s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack1/best',
        # 'sALMH 3s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPSA]_ran_originaler_attack2/best',

        # 'dALMH 1s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA]_ran_originaler_attack0/best',
        # 'dALMH 2s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA]_ran_originaler_attack1/best',
        # 'dALMH 3s': 'agent_models/Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA]_ran_originaler_attack2/best',

        # 'dsALMH 1d[5t] 1s': 'agent_models/Complex/2/5_steps_in_dynamic_advs/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0/best',
        # 'dsALMH 2d[5t] 2s': 'agent_models/Complex/2/5_steps_in_dynamic_advs/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1/best',
        # 'dsALMH 3d[5t] 3s': 'agent_models/Complex/2/5_steps_in_dynamic_advs/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2/best',

    }
    teammate_lvl_sets = [
        [Eval.LOW],
        # [Eval.MEDIUM],
        # [Eval.HIGH]
    ]
    return args.layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args, 'complex'



if __name__ == "__main__":
    args = get_arguments()
    fig, axes = plt.subplots(1, 1, figsize=(30, 5), sharey=True)
    classic_layout_names, classic_p_idxes, classic_agents_paths, classic_teammate_lvl_sets, classic_args, classic_prefix = get_2_player_input_classic(args)

    axes = [axes]

    deterministic = False # deterministic = True does not actually work :sweat_smile:
    max_num_teams_per_layout_per_x = 4
    number_of_eps = 5

    # Number of parallel workers for evaluation
    args.max_workers = 4

    # For display_purposes
    unseen_counts = [1]
    show_delivery_num = True

    plot_name = generate_plot_name(prefix=classic_prefix,
                                    num_players=args.num_players,
                                    deterministic=deterministic,
                                    p_idxes=classic_p_idxes,
                                    num_eps=number_of_eps,
                                    max_num_teams=max_num_teams_per_layout_per_x,
                                    teammate_lvl_sets=classic_teammate_lvl_sets)

    all_mean_rewards, all_std_rewards = run_parallel_evaluation(
            args=args,
            all_agents_paths=classic_agents_paths,
            layout_names=classic_layout_names,
            p_idxes=classic_p_idxes,
            deterministic=deterministic,
            max_num_teams_per_layout_per_x=max_num_teams_per_layout_per_x,
            number_of_eps=number_of_eps,
            teammate_lvl_sets=classic_teammate_lvl_sets
    )

    plot_evaluation_results_bar(fig, axes[0], all_mean_rewards=all_mean_rewards,
                           all_std_rewards=all_std_rewards,
                           layout_names=classic_layout_names,
                           teammate_lvl_sets=classic_teammate_lvl_sets,
                           unseen_counts=unseen_counts,
                           display_delivery=show_delivery_num,
                           plot_name=plot_name)

    complex_layout_names, complex_p_idxes, complex_agents_paths, complex_teammate_lvl_sets, complex_args, complex_prefix = get_2_player_input_complex(args)

    deterministic = False # deterministic = True does not actually work :sweat_smile:
    max_num_teams_per_layout_per_x = 4
    number_of_eps = 5

    # Number of parallel workers for evaluation
    args.max_workers = 4

    # For display_purposes
    unseen_counts = [1]
    show_delivery_num = True

    plot_name = generate_plot_name(prefix=complex_prefix,
                                    num_players=args.num_players,
                                    deterministic=deterministic,
                                    p_idxes=complex_p_idxes,
                                    num_eps=number_of_eps,
                                    max_num_teams=max_num_teams_per_layout_per_x,
                                    teammate_lvl_sets=complex_teammate_lvl_sets)

    all_mean_rewards, all_std_rewards = run_parallel_evaluation(
            args=args,
            all_agents_paths=complex_agents_paths,
            layout_names=complex_layout_names,
            p_idxes=complex_p_idxes,
            deterministic=deterministic,
            max_num_teams_per_layout_per_x=max_num_teams_per_layout_per_x,
            number_of_eps=number_of_eps,
            teammate_lvl_sets=complex_teammate_lvl_sets
    )

    plot_evaluation_results_bar(fig, axes[0], all_mean_rewards=all_mean_rewards,
                           all_std_rewards=all_std_rewards,
                           layout_names=complex_layout_names,
                           teammate_lvl_sets=complex_teammate_lvl_sets,
                           unseen_counts=unseen_counts,
                           display_delivery=show_delivery_num,
                           plot_name=plot_name,
                           start=5)

    layouts = (classic_layout_names + complex_layout_names)
    idxs = np.arange(len(layouts))
    num_agents = 3
    width = .95 / num_agents
    axes[0].set_xticks(idxs + (num_agents/3 * width), labels=[DISPLAY_NAME_MAP[l] for l in layouts], fontsize='28')
    axes[0].set_yticks(np.arange(0, 20, 5), )
    axes[0].set_ylabel("Number of Soup Deliveries", fontsize='24')
    axes[0].autoscale_view()
    axes[0].legend(loc='best', fontsize='28', fancybox=True, framealpha=0.5)

    plt.tight_layout()
    plt.savefig(f'data/plots/{plot_name}_deliveries_bar.png')

    # plot_evaluation_results_line(all_mean_rewards=all_mean_rewards,
    #                                  all_std_rewards=all_std_rewards,
    #                                  layout_names=layout_names,
    #                                  teammate_lvl_sets=teammate_lvl_sets,
    #                                  num_players=args.num_players,
    #                                  plot_name=plot_name)
