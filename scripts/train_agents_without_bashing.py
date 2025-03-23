import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, AdversaryPlayConfig, KeyCheckpoints
from oai_agents.common.learner import LearnerType
from oai_agents.common.curriculum import Curriculum
from oai_agents.common.agents_finder import HMLProfileCollection, SelfPlayAgentsFinder
from oai_agents.agents.mep_population_manager import MEPPopulationManager

from pathlib import Path

from scripts.utils import (
    get_SP_agents,
    get_FCP_agent_w_pop,
    get_N_X_FCP_agents,
    get_N_X_SP_agents,
)

from scripts.utils.layout_config import (
    storage_room_1_chef_layouts,
    storage_room_2_chefs_layouts,
    cramped_room_ot_1_chef_layuouts,
    cramped_room_ot_2_chef_layuouts,
    classic_2_chefs_layouts,
    complex_2_chefs_layouts,
    complex_3_chefs_layouts,
    complex_5_chefs_layouts
)

from scripts.train_agents import (
    SP, SPN_1ADV, SPN_1ADV_XSPCKP, SPN_XSPCKP, 
    FCP_mhri, FCP_traditional, N_1_FCP, MEP_POPULATION
)


def set_input(args):
    args.num_players = args.teammates_len + 1

    one_chef_layouts = storage_room_1_chef_layouts
    two_chefs_layouts = storage_room_2_chefs_layouts
    three_chefs_layouts = complex_3_chefs_layouts
    five_chefs_layouts = complex_5_chefs_layouts

    if args.num_players == 2:
        args.layout_names = two_chefs_layouts
    elif args.num_players == 3:
        args.layout_names = three_chefs_layouts
    elif args.num_players == 5:
        args.layout_names = five_chefs_layouts
    elif args.num_players == 1:
        args.layout_names = one_chef_layouts

    args.custom_agent_ck_rate_generation = args.num_players + 1
    args.num_steps_in_traj_for_dyn_adv = 2
    args.num_static_advs_per_heatmap = 1
    args.num_dynamic_advs_per_heatmap = 1
    args.use_val_func_for_heatmap_gen = True
    args.prioritized_sampling = False

    if not args.quick_test:
        args.gen_pop_for_eval = False
        args.n_envs = 210
        args.epoch_timesteps = 1e5

        args.pop_total_training_timesteps = int(5e6 * args.how_long)
        args.n_x_sp_total_training_timesteps = int(5e6 * args.how_long)
        args.fcp_total_training_timesteps = int(5e6 * args.how_long)

        args.adversary_total_training_timesteps = int(5e6 * args.how_long)
        args.n_x_fcp_total_training_timesteps = int(2 * args.fcp_total_training_timesteps * args.how_long)

        args.total_ego_agents = 8
        print(f"args.layout_names: {args.layout_names}")
        if args.layout_names == complex_2_chefs_layouts:
            prefix = 'Complex'
        elif args.layout_names == complex_5_chefs_layouts:
            prefix = 'Complex'
        elif args.layout_names == classic_2_chefs_layouts:
            prefix = 'Classic'
        elif args.layout_names == storage_room_2_chefs_layouts:
            prefix = 'storage_room_2_chef_layouts'
        elif args.layout_names == cramped_room_ot_2_chef_layuouts:
            prefix = 'cramped_room_ot_2_chef_layouts'
        elif args.layout_names == storage_room_1_chef_layouts:
            prefix = 'storage_room_1_chef_layouts'
        elif args.layout_names == cramped_room_ot_1_chef_layuouts:
            prefix = 'cramped_room_ot_1_chef_layouts'

        args.exp_dir = f'{prefix}/{args.num_players}'

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.num_of_ckpoints = 5
        args.epoch_timesteps = 800
        args.pop_total_training_timesteps = 4000
        args.n_x_sp_total_training_timesteps = 4000
        args.adversary_total_training_timesteps = 1500
        args.fcp_total_training_timesteps = 1500
        args.n_x_fcp_total_training_timesteps = 1500 * 2
        args.total_ego_agents = 2
        args.exp_dir = f'Test/{args.num_players}'



if __name__ == '__main__':
    args = get_arguments()
    args.quick_test = True
    args.pop_force_training = False
    args.adversary_force_training = False
    args.primary_force_training = False
    args.teammates_len = 1

    if args.teammates_len <= 1:
        args.how_long = 20
        args.num_of_ckpoints = 35
    elif args.teammates_len == 2:
        args.how_long = 25
        args.num_of_ckpoints = 40
    elif args.teammates_len == 4:
        args.how_long = 35
        args.num_of_ckpoints = 50

    set_input(args=args)

    # MEP_POPULATION(args=args)

    # SPN_XSPCKP(args=args)

    # FCP_traditional(args=args)

    # SP(args)

    # FCP_mhri(args=args)

    # SPN_1ADV(args=args)

    # N_1_FCP(args=args)
