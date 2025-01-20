from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, TeammatesCollection, KeyCheckpoints
from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.agents.agent_utils import load_agent
from oai_agents.gym_environments.base_overcooked_env import OvercookedGymEnv
from oai_agents.agents.base_agent import OAIAgent
from scripts.utils.agents_finder import AgentsFinder, AgentsFinderBySuffix, AMMAS23AgentsFinderBySuffix

from stable_baselines3.common.evaluation import evaluate_policy

from scripts.utils import Eval, POPULATION_EVAL_AGENTS, print_selected_agents_for_evaluation
from scripts.utils.layout_config import (
    two_chefs_dec_layouts,
    three_chefs_dec_layouts,
    five_chefs_dec_layouts,
    two_chefs_aamas24_layouts,
    three_chefs_aamas24_layouts,
    four_chefs_aamas24_layouts,
    five_chefs_aamas24_layouts,
    classic_layouts
)

def get_2_player_input(args):
    args.teammates_len = 1
    args.num_players = args.teammates_len + 1
    args.layout_names = classic_layouts
    # args.layout_names = ['selected_2_chefs_coordination_ring',
    #                     'selected_2_chefs_counter_circuit',
    #                     'selected_2_chefs_cramped_room']

    args.p_idxes = [0, 1]
    all_agents_paths = {
        'SP_1010':      'agent_models/Classic/2/SP_hd256_seed1010/best',
        'SP_2020':      'agent_models/Classic/2/SP_hd256_seed2020/best',
        'SP_2602':      'agent_models/Classic/2/SP_hd256_seed2602/best',
        'SP_13':        'agent_models/Classic/2/SP_hd256_seed13/best',
        'FCP_1010':     'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
        'FCP_2020':     'agent_models/Classic/2/N-1-SP_s2020_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
        'FCP_2602':     'agent_models/Classic/2/N-1-SP_s2602_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
        'FCP_13':       'agent_models/Classic/2/N-1-SP_s13_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
        'F2P_1010':    'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPFO]_ran_originaler/best',
        'F2P_2020':    'agent_models/Classic/2/N-1-SP_s2020_h256_tr[SPH_SPM_SPL_SPFO]_ran_originaler/best',
        'F2P_2602':    'agent_models/Classic/2/N-1-SP_s2602_h256_tr[SPH_SPM_SPL_SPFO]_ran_originaler/best',
        'F2P_13':      'agent_models/Classic/2/N-1-SP_s13_h256_tr[SPH_SPM_SPL_SPFO]_ran_originaler/best',
        'A2P_1010':    'agent_models/Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPADV]_ran_originaler/best',
        'A2P_2020':    'agent_models/Classic/2/N-1-SP_s2020_h256_tr[SPH_SPM_SPL_SPADV]_ran_originaler/best',
        'A2P_2602':    'agent_models/Classic/2/N-1-SP_s2602_h256_tr[SPH_SPM_SPL_SPADV]_ran_originaler/best',
        'A2P_13':      'agent_models/Classic/2/N-1-SP_s13_h256_tr[SPH_SPM_SPL_SPADV]_ran_originaler/best',
        # 'SP':          'agent_models/Result/2/SP_hd64_seed14/best',
        # 'FCP':         'agent_models/FCP_correct/2/FCP_s2020_h256_tr(AMX)_ran/best',
        # 'ALMH CUR 1A': 'agent_models/ALMH_CUR/2/PWADV-N-1-SP_s1010_h256_tr[SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL_SPADV]_cur_originaler_attack0/best',
    }

    teammate_lvl_sets = [
        # Eval.LOW,
        # Eval.MEDIUM,
        # Eval.HIGH,
        Eval.HUMAN_PROXY
    ]

    return args, all_agents_paths, teammate_lvl_sets


def get_all_players_and_teammates(args, teammate_lvl_sets, all_players):
    all_teammates = {layout_name: {teammate_lvl: [] for teammate_lvl in Eval.ALL} for layout_name in args.layout_names}

    for layout_name in args.layout_names:
        layout_population, _, _ = RLAgentTrainer.load_agents(
            args,
            path=Path(POPULATION_EVAL_AGENTS[layout_name]),
            tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL
        )

        sorted_agents = sorted(layout_population, key=lambda x: x.layout_scores[layout_name], reverse=False)
        unique_agents = {}
        for agent in layout_population:
            agent_pair = (agent.name, agent.layout_scores[layout_name])
            if agent_pair not in unique_agents:
                unique_agents[agent_pair] = agent

        sorted_agents = sorted(unique_agents.values(), key=lambda x: x.layout_scores[layout_name], reverse=False)
        part_size = len(sorted_agents) // 3
        # all_teammates[layout_name][Eval.LOW] = sorted_agents[:part_size]
        # all_teammates[layout_name][Eval.MEDIUM] = sorted_agents[part_size:2 * part_size]
        # all_teammates[layout_name][Eval.HIGH] = sorted_agents[2 * part_size:]

        human_proxy_suffix = f"bc_{layout_name}"
        args.exp_dir = "Classic/2"
        args.base_dir = Path.cwd()
        print(f"args.base_dir: {args.base_dir}")
        agent_finder = AMMAS23AgentsFinderBySuffix(args=args)
        human_proxies = agent_finder.get_agents(key=human_proxy_suffix, tag=KeyCheckpoints.BEST_EVAL_REWARD)
        all_teammates[layout_name][Eval.HUMAN_PROXY] = human_proxies


    selected_agents_for_evaluation = {
        primary_agent: {
            layout_name: {
                unseen_count: {
                    teammate_lvl: [] for teammate_lvl in Eval.ALL}
                         for unseen_count in range(args.num_players)}
                            for layout_name in args.layout_names}
                                for primary_agent in all_players}

    for primary_agent in all_players:
        for teammate_lvl in teammate_lvl_sets:
            for layout_name in args.layout_names:
                for unseen_count in range(args.num_players):
                    seen_members = [primary_agent for _ in range(args.teammates_len - unseen_count)]
                    if unseen_count == 0:
                        selected_agents_for_evaluation[primary_agent][layout_name][unseen_count][teammate_lvl] = [seen_members]
                    elif unseen_count == 1:
                        teams = [seen_members + [tm] for tm in all_teammates[layout_name][teammate_lvl]]
                        selected_agents_for_evaluation[primary_agent][layout_name][unseen_count][teammate_lvl] = teams
                    else:
                        raise NotImplementedError
    return selected_agents_for_evaluation


def get_all_players(args, all_agents_paths):
    all_players = {agent_name: load_agent(Path(all_agents_paths[agent_name]), args) for agent_name in all_agents_paths}
    for agent_name in all_agents_paths:
        all_players[agent_name].name = agent_name
    all_players = [all_players[agent_name] for agent_name in all_agents_paths]
    return all_players


def evaluate_on_layout_with_teammates(args, primary_agent:OAIAgent, layout_name, unseen_count, teammate_lvl, all_teammates):
    print(f"Evaluating {primary_agent.name} on layout {layout_name} with {unseen_count} unseen teammates at level {teammate_lvl}")

    mean_rewards = []
    std_rewards = []
    for teammates in all_teammates[:args.max_teammates_for_eval]:
        env = OvercookedGymEnv(
            args=args,
            layout_name=layout_name,
            p_encoding_fn=primary_agent.encoding_fn,
            ret_completed_subtasks=False,
            is_eval_env=True,
            horizon=400,
            deterministic=False,
            learner_type='originaler'
        )
        env.set_teammates(teammates)
        for p_idx in args.p_idxes:
            env.reset(p_idx=p_idx)
            mean_reward, std_reward = evaluate_policy(
                primary_agent,
                env,
                n_eval_episodes=args.max_num_eval_episodes,
                deterministic=False,
                warn=False,
                render=False
            )
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)

    return {
        "primary_agent_name": primary_agent.name,
        "layout_name": layout_name,
        "unseen_count": unseen_count,
        "teammate_lvl": teammate_lvl,
        "mean_rewards": sum(mean_rewards) / len(mean_rewards) if mean_rewards else 0,
        "std_rewards": sum(std_rewards) / len(std_rewards) if std_rewards else 0
    }


def evaluate_all_agents(args, all_players_and_their_teammates):
    # Nested dictionary to store results
    mean_rewards = {
        primary_agent.name: {
            layout_name: {
                unseen_count: {teammate_lvl: -1 for teammate_lvl in Eval.ALL}
                    for unseen_count in range(args.num_players)}
            for layout_name in args.layout_names}
        for primary_agent in all_players_and_their_teammates.keys()}

    std_rewards = {
        primary_agent.name: {
            layout_name: {
                unseen_count: {teammate_lvl: -1 for teammate_lvl in Eval.ALL}
                for unseen_count in range(args.num_players)}
            for layout_name in args.layout_names}
        for primary_agent in all_players_and_their_teammates.keys()}

    for primary_agent, player_data in all_players_and_their_teammates.items():
        for layout_name, teammates_data in player_data.items():
            for unseen_count, teammates_by_lvl in teammates_data.items():
                if unseen_count == 0:
                    continue
                for teammate_lvl, all_teammates in teammates_by_lvl.items():
                    result = evaluate_on_layout_with_teammates(args, primary_agent, layout_name, unseen_count, teammate_lvl, all_teammates)
                    mean_rewards[primary_agent.name][layout_name][unseen_count][teammate_lvl] = result["mean_rewards"]
                    std_rewards[primary_agent.name][layout_name][unseen_count][teammate_lvl] = result["std_rewards"]
    return mean_rewards, std_rewards

import matplotlib.pyplot as plt
import numpy as np

def plot_performance_grid(mean_rewards, std_rewards, layout_names, skill_levels):
    # Number of layouts and skill levels
    n_layouts = len(layout_names)
    n_skills = len(skill_levels) + 1  # +1 for the average row

    # Create figure and axes grid
    fig, axes = plt.subplots(n_skills, n_layouts, figsize=(5*n_layouts, 4*n_skills))
    fig.suptitle('Agent Performance Comparison', fontsize=16, y=1.02)

    # If only one layout, wrap axes in a list for consistent indexing
    if n_layouts == 1:
        axes = axes[:, np.newaxis]

    # Colors for different agents
    colors = plt.cm.Set3(np.linspace(0, 1, len(mean_rewards)))

    # Plot for each layout and skill level
    for layout_idx, layout in enumerate(layout_names):
        # Plot for each skill level
        for skill_idx, skill in enumerate(skill_levels):
            ax = axes[skill_idx, layout_idx]

            # Plot data for each primary agent
            x = np.arange(len(mean_rewards))
            width = 0.35

            for agent_idx, (agent_name, agent_data) in enumerate(mean_rewards.items()):
                reward = agent_data[layout][1][skill]  # unseen_count = 1
                std = std_rewards[agent_name][layout][1][skill]

                ax.bar(agent_idx, reward, width,
                      yerr=std,
                      label=agent_name,
                      color=colors[agent_idx],
                      capsize=5)

            # Customize subplot
            ax.set_title(f'{layout} - {skill}')
            ax.set_xticks(x)
            ax.set_xticklabels([name[:10] for name in mean_rewards.keys()],
                              rotation=45, ha='right')
            ax.set_ylabel('Reward')
            if skill_idx == 0:
                ax.legend(bbox_to_anchor=(0.5, 1.15),
                         loc='center',
                         ncol=len(mean_rewards))

        # Plot averages in the last row
        ax = axes[-1, layout_idx]
        for agent_idx, (agent_name, agent_data) in enumerate(mean_rewards.items()):
            # Calculate average across all skill levels
            avg_reward = np.mean([agent_data[layout][1][skill]
                                for skill in skill_levels])
            avg_std = np.mean([std_rewards[agent_name][layout][1][skill]
                             for skill in skill_levels])

            ax.bar(agent_idx, avg_reward, width,
                  yerr=avg_std,
                  color=colors[agent_idx],
                  capsize=5)

        # Customize average subplot
        ax.set_title(f'{layout} - Average')
        ax.set_xticks(x)
        ax.set_xticklabels([name[:10] for name in mean_rewards.keys()],
                          rotation=45, ha='right')
        ax.set_ylabel('Average Reward')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # plt.show()
    plt.savefig('data/plots/performance_grid.png')
    return fig

if __name__ == "__main__":
    args = get_arguments()
    args, all_agents_paths, teammate_lvl_sets = get_2_player_input(args)
    args.max_teammates_for_eval = 1
    args.max_num_eval_episodes = 4
    args.max_parallel_workers = 5

    all_players = get_all_players(args=args, all_agents_paths=all_agents_paths)
    all_players_and_their_teammates = get_all_players_and_teammates(args=args, teammate_lvl_sets=teammate_lvl_sets, all_players=all_players)

    # print_selected_agents_for_evaluation(all_players_and_their_teammates)

    mean_rewards, std_rewards = evaluate_all_agents(args, all_players_and_their_teammates)

    print(mean_rewards)
    print(std_rewards)

    plot_performance_grid(mean_rewards, std_rewards, args.layout_names, teammate_lvl_sets)
