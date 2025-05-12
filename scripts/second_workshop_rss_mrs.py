import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

from oai_agents.common.arguments import get_arguments
from evaluate_agents import run_parallel_evaluation, eval_key_lut, Eval


def get_c1_input(args):
    args.num_players = 2
    args.layout_names = ['c1']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c1/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'UB': 'agent_models/bbe/c1_best_EGO/best_c1/best',
        }
    return args.layout_names, all_agents_paths, args, 'c1'


def get_c2_input(args):
    args.num_players = 2
    args.layout_names = ['c2']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c2/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'UB': 'agent_models/bbe/c2_best_EGO/best_c2/best',
    }
    return args.layout_names, all_agents_paths, args, 'c2'


def get_c3_input(args):
    args.num_players = 2
    args.layout_names = ['c3']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c3/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'UB': 'agent_models/bbe/c3_best_EGO/best_c3/best',
    }
    return args.layout_names, all_agents_paths, args, 'c3'


def get_c4_input(args):
    args.num_players = 2
    args.layout_names = ['c4']
    all_agents_paths = {
        'MEP': 'agent_models/bbe/MEP_on_c_layouts/MEP_c4/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ps_originaler/best',
        'UB': 'agent_models/bbe/c4_best_EGO/best_c4/best',
    }
    return args.layout_names, all_agents_paths, args, 'c4'


if __name__ == "__main__":
    args = get_arguments()
    args.max_workers = 4
    
    layout_names, all_agents_paths, args, prefix = get_c1_input(args)
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


    layout_names, all_agents_paths, args, prefix = get_c2_input(args)
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


    layout_names, all_agents_paths, args, prefix = get_c3_input(args)
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

