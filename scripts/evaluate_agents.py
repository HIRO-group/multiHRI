import multiprocessing as mp
import os
from pathlib import Path
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

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv

from utils import (
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

LAYOUT_NAMES_PATHs = {
    'selected_2_chefs_coordination_ring': {
        Eval.LOW: TWO_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: TWO_PLAYERS_MEDIUM_EVAL,
        Eval.HIGH:TWO_PLAYERS_HIGH_EVAL
    },
    'selected_2_chefs_counter_circuit': {
        Eval.LOW: TWO_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: TWO_PLAYERS_MEDIUM_EVAL,
        Eval.HIGH:TWO_PLAYERS_HIGH_EVAL
    },
    'selected_2_chefs_cramped_room': {
        Eval.LOW: TWO_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: TWO_PLAYERS_MEDIUM_EVAL,
        Eval.HIGH:TWO_PLAYERS_HIGH_EVAL
    },

    'selected_3_chefs_coordination_ring': {
        Eval.LOW: THREE_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: THREE_PLAYERS_MEDIUM_EVAL,
        Eval.HIGH: THREE_PLAYERS_HIGH_EVAL,
    },
    'selected_3_chefs_counter_circuit': {
        Eval.LOW: THREE_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: THREE_PLAYERS_MEDIUM_EVAL,
        Eval.HIGH: THREE_PLAYERS_HIGH_EVAL,
    },
    'selected_3_chefs_cramped_room': {
        Eval.LOW: THREE_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: THREE_PLAYERS_MEDIUM_EVAL,
        Eval.HIGH: THREE_PLAYERS_HIGH_EVAL,
    },

    'selected_5_chefs_counter_circuit': {
        Eval.LOW: FIVE_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: FIVE_PLAYERS_MEDIUM_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL,
        Eval.HIGH: FIVE_PLAYERS_HIGH_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL,
    },
    'selected_5_chefs_secret_coordination_ring': {
        Eval.LOW: FIVE_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: FIVE_PLAYERS_MEDIUM_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL,
        Eval.HIGH: FIVE_PLAYERS_HIGH_FOR_ALL_BESIDES_STORAGE_ROOM_EVAL,
    },
    'selected_5_chefs_storage_room': {
        Eval.LOW: FIVE_PLAYERS_LOW_EVAL,
        Eval.MEDIUM: FIVE_PLAYERS_MEDIUM_STORAGE_EVAL,
        Eval.HIGH: FIVE_PLAYERS_HIGH_STORAGE_EVAL,
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


def generate_plot_name(num_players, deterministic, p_idxes, num_eps, max_num_teams, teammate_lvl_sets):
    plot_name = f'{num_players}-players'
    plot_name += '-det' if deterministic else '-stoch'
    p_idexes_str = ''.join([str(p_idx) for p_idx in p_idxes])
    plot_name += f'-pidx{p_idexes_str}'
    plot_name += f'-eps{num_eps}'
    plot_name += f'-maxteams{str(max_num_teams)}'
    teams = ''.join([str(t[0]) for t in teammate_lvl_sets])
    plot_name += f"-teams({str(teams)})"
    return plot_name


def plot_evaluation_results(all_mean_rewards, all_std_rewards, layout_names, teammate_lvl_sets, num_players, plot_name):
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

        ax.set_title(f"Avg. {layout_name}")
        ax.set_xlabel('Number of Unseen Teammates')
        ax.set_xticks(x_values)
        ax.legend(loc='upper right', fontsize='small', fancybox=True, framealpha=0.5)


    plt.tight_layout()
    plt.savefig(f'data/plots/{plot_name}.png')
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


def evaluate_agent_for_layout(agent_name, path, layout_names, p_idxes, args, deterministic, max_num_teams_per_layout_per_x, number_of_eps, teammate_lvl_set: Sequence[Eval]):
    fn_args = (agent_name, path, tuple(layout_names), tuple(p_idxes), tuple([(k, tuple(v) if isinstance(v, list) else v) for k,v in vars(args).items()]), deterministic, max_num_teams_per_layout_per_x, number_of_eps, tuple(teammate_lvl_set))
    m = hashlib.md5()
    for s in fn_args:
        m.update(str(s).encode())
    arg_hash = m.hexdigest()

    print(f"Eval Hash: {arg_hash}")
    cache_file_path = f"eval_cache/eval_{arg_hash}.pkl"
    cached_eval = Path(cache_file_path)
    if cached_eval.is_file():
        print(f"Loading cached evaluation from {cached_eval}")
        with open(cached_eval, "rb") as f:
            agent_name, teammate_lvl_set, mean_rewards, std_rewards = pkl.load(f)

    else:
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
            pkl.dump((agent_name, teammate_lvl_set, mean_rewards, std_rewards), f)

    return agent_name, str(teammate_lvl_set), mean_rewards, std_rewards


def run_parallel_evaluation(args, all_agents_paths, layout_names, p_idxes, deterministic, max_num_teams_per_layout_per_x, number_of_eps, teammate_lvl_sets: Sequence[Sequence[Eval]]):
    all_mean_rewards, all_std_rewards = {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
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



def get_2_player_input(args):
    args.num_players = 2
    layout_names = ['selected_2_chefs_coordination_ring',
                    'selected_2_chefs_counter_circuit',
                    'selected_2_chefs_cramped_room']
    p_idxes = [0, 1]

    all_agents_paths = {
        'N-1-SP FCP CUR':  'agent_models/Result/2/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_cur/best',
        'N-1-SP FCP RAN':  'agent_models/Result/2/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_ran/best',
        'SP':              'agent_models/Result/2/SP_hd64_seed14/best',
        'FCP':             'agent_models/Result/2/FCP_s2020_h256_tr(AMX)_ran/best',
        'N-1-SP ADV':      'agent_models/Result/2/MAP_SP_hd64_seed14/originaler-selfisherplay/2/pwadv_s14_h64_tr(SP_SPADV)_ran/best',
    }
    teammate_lvl_sets = [
        [Eval.LOW],
        [Eval.MEDIUM],
        [Eval.HIGH]
    ]
    return layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args


def get_3_player_input(args):
    args.num_players = 3
    layout_names = ['selected_3_chefs_coordination_ring',
                    'selected_3_chefs_counter_circuit',
                    'selected_3_chefs_cramped_room']
    p_idxes = [0, 1, 2]
    all_agents_paths = {
        'N-1-SP FCP CUR':  'agent_models/Result/3/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_cur/best',
        'N-1-SP FCP RAN':  'agent_models/Result/3/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_ran/best',
        'SP':              'agent_models/Result/3/SP_hd64_seed14/best',
        'FCP':             'agent_models/Result/3/FCP_s2020_h256_tr(AMX)_ran/best'}
    teammate_lvl_sets = [
        [Eval.LOW],
        [Eval.MEDIUM],
        [Eval.HIGH]
    ]
    return layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args


def get_five_player_input(args):
    args.num_players = 5
    layout_names = ['selected_5_chefs_counter_circuit',
                    'selected_5_chefs_secret_coordination_ring',
                    'selected_5_chefs_storage_room']
    p_idxes = [0, 1, 2, 3, 4]
    all_agents_paths = {
        'N-1-SP FCP CUR':  'agent_models/Result/5/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_cur/best',
        'N-1-SP FCP RAN':  'agent_models/Result/5/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_ran/best',
        'SP':              'agent_models/Result/5/SP_hd64_seed14/best',
        'FCP':             'agent_models/Result/5/FCP_s2020_h256_tr(AMX)_ran/best'}
    teammate_lvl_sets = [
        [Eval.LOW],
        [Eval.MEDIUM],
        [Eval.HIGH]
    ]
    return layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args


if __name__ == "__main__":
    args = get_arguments()
    # layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args = get_2_player_input(args)
    layout_names, p_idxes, all_agents_paths, teammate_lvl_sets, args = get_3_player_input(args)
    
    deterministic = False
    max_num_teams_per_layout_per_x = 4
    number_of_eps = 5

    plot_name = generate_plot_name(num_players=args.num_players,
                                    deterministic=deterministic,
                                    p_idxes=p_idxes,
                                    num_eps=number_of_eps,
                                    max_num_teams=max_num_teams_per_layout_per_x,
                                    teammate_lvl_sets=teammate_lvl_sets)

    pre_evaluated_results_file = Path(f"data/plots/{plot_name}.pkl")

    if pre_evaluated_results_file.is_file():
        with open(pre_evaluated_results_file, "rb") as f:
            all_mean_rewards, all_std_rewards = pkl.load(f)
    else:
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

        with open(pre_evaluated_results_file, "wb") as f:
            pkl.dump((all_mean_rewards, all_std_rewards), f)


    plot_evaluation_results(all_mean_rewards=all_mean_rewards,
                            all_std_rewards=all_std_rewards,
                            layout_names=layout_names,
                            teammate_lvl_sets=teammate_lvl_sets,
                            num_players=args.num_players,
                            plot_name=plot_name)
