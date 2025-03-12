import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, AdversaryPlayConfig, KeyCheckpoints
from oai_agents.common.learner import LearnerType
from oai_agents.common.curriculum import Curriculum

from scripts.utils import (
    get_SP_agents,
    get_FCP_agent_w_pop,
    get_N_X_FCP_agents,
    get_N_X_SP_agents,
)

def SP(args):
    primary_train_types = [TeamType.SELF_PLAY]
    primary_eval_types = {
        'generate': [TeamType.SELF_PLAY],
        'load': []
    }
    curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_SP_agents(
        args=args,
        train_types=curriculum.train_types,
        eval_types=primary_eval_types,
        curriculum=curriculum,
        tag_for_returning_agent=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL
    )


def SPN_1ADV(args) -> None:
    '''
    In N-agents games, a randomly initialized agent will be trained with either one of two conditions:
    (a)N-1 copies of itself and 1 unseen adversary teammate.
    (b)N copies of itself

    e.g.
    when N is 4, the team can be composed by [SP, SP, SP, SP] or [SP, SP, SP, ADV] in a 4-chef layout.
    '''
    attack_rounds = 3
    unseen_teammates_len = 1
    adversary_play_config = AdversaryPlayConfig.MAP
    primary_train_types = [TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY]

    primary_eval_types = {
        'generate': [
            TeamType.SELF_PLAY_HIGH,
            TeamType.SELF_PLAY_LOW,
            TeamType.SELF_PLAY_ADVERSARY
        ],
        'load': []
    }

    curriculum = Curriculum(
        train_types = primary_train_types, is_random = True)
    get_N_X_SP_agents(
        args,
        n_x_sp_train_types=curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
        adversary_play_config=adversary_play_config,
        attack_rounds=attack_rounds
    )


def SPN_1ADV_XSPCKP(args) -> None:
    '''
    In N-agents games, a randomly initialized agent will be trained with N-X copies of itself and X unseen teammates.
    X unseen teammates can be composed by either one of the two conditions:
    (a) 1 adversary and X-1 self-play checkpoints.
    (b) X self-play checkpoints.
    e.g.
    when N is 4 and X is 1, the team can be composed by [SP, SP, SP, ADV] or [SP, SP, SP, H] or [SP, SP, SP, M] or [SP, SP, SP, L] in a 4-chef layout.
    when N is 4 and X is 2, the team can be composed
    [SP, SP, ADV, H] or [SP, SP, ADV, M] or [SP, SP, ADV, L] or
    [SP, SP, H, H] or [SP, SP, M, M] or [SP, SP, L, L] in a 4-chef layout.

    - X is the number of unseen teammate.
    - X is assigned by the variable, unseen_teammates_len, in the funciton.
    '''
    attack_rounds = 3
    unseen_teammates_len = 1
    adversary_play_config = AdversaryPlayConfig.MAP
    primary_train_types = [
        TeamType.SELF_PLAY_HIGH,
        TeamType.SELF_PLAY_MEDIUM,
        TeamType.SELF_PLAY_ADVERSARY
    ]

    primary_eval_types = {
        'generate': [
            TeamType.SELF_PLAY_HIGH,
            TeamType.SELF_PLAY_MEDIUM,
            TeamType.SELF_PLAY_ADVERSARY
        ],
        'load': []
    }

    if args.prioritized_sampling:
        curriculum = Curriculum(
            train_types = primary_train_types,
            eval_types=primary_eval_types,
            prioritized_sampling=True,
        )

    else:
        curriculum = Curriculum(
            train_types = primary_train_types,
            is_random=False,
            prioritized_sampling=True,
            total_steps = args.n_x_sp_total_training_timesteps//args.epoch_timesteps,
            training_phases_durations_in_order={
                (TeamType.SELF_PLAY_ADVERSARY): 0.5,
            },
            rest_of_the_training_probabilities={
                TeamType.SELF_PLAY_MEDIUM: 0.3,
                TeamType.SELF_PLAY_HIGH: 0.3,
                TeamType.SELF_PLAY_ADVERSARY: 0.4,
            },
            probabilities_decay_over_time=0
        )
    get_N_X_SP_agents(
        args,
        n_x_sp_train_types=curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
        adversary_play_config=adversary_play_config,
        attack_rounds=attack_rounds
    )

def FCP_mhri(args):
    '''
    There are two types of FCP, one is the traditional FCP that uses random teammates (i.e. ALL_MIX),
    one is our own version that uses certain types HIGH_FIRST, MEDIUM_FIRST, etc.
    The reason we have our version is that when we used the traditional FCP it got ~0 reward so we
    decided to add different types for teammates_collection.
    '''
    primary_train_types = [TeamType.LOW_FIRST, TeamType.MEDIUM_FIRST, TeamType.HIGH_FIRST]
    primary_eval_types = {'generate' : [TeamType.HIGH_FIRST],
                          'load': []}

    fcp_curriculum = Curriculum(
        train_types = primary_train_types,
        is_random=False,
        total_steps = args.fcp_total_training_timesteps//args.epoch_timesteps,
        training_phases_durations_in_order={
            (TeamType.LOW_FIRST): 0.5,
            (TeamType.MEDIUM_FIRST): 0.125,
            (TeamType.HIGH_FIRST): 0.125,
        },
        rest_of_the_training_probabilities={
            TeamType.LOW_FIRST: 0.4,
            TeamType.MEDIUM_FIRST: 0.3,
            TeamType.HIGH_FIRST: 0.3,
        },
        probabilities_decay_over_time=0
    )

    _, _ = get_FCP_agent_w_pop(
        args,
        fcp_train_types = fcp_curriculum.train_types,
        fcp_eval_types=primary_eval_types,
        fcp_curriculum=fcp_curriculum
    )


def N_1_FCP(args):
    unseen_teammates_len = 1 # This is the X in FCP_X_SP

    fcp_train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    fcp_eval_types = {'generate' : [], 'load': []}
    fcp_curriculum = Curriculum(train_types=fcp_train_types, is_random=True)

    primary_train_types = [
        TeamType.SELF_PLAY_LOW,
        TeamType.SELF_PLAY_MEDIUM,
        TeamType.SELF_PLAY_HIGH
    ]
    primary_eval_types = {
        'generate': [
            TeamType.SELF_PLAY_LOW,
            TeamType.SELF_PLAY_MEDIUM,
            TeamType.SELF_PLAY_HIGH
        ],
        'load': []
    }
    n_1_fcp_curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_N_X_FCP_agents(
        args=args,
        fcp_train_types=fcp_curriculum.train_types,
        fcp_eval_types=fcp_eval_types,
        n_1_fcp_train_types=n_1_fcp_curriculum.train_types,
        n_1_fcp_eval_types=primary_eval_types,
        fcp_curriculum=fcp_curriculum,
        n_1_fcp_curriculum=n_1_fcp_curriculum,
        unseen_teammates_len=unseen_teammates_len
    )



def FCP_traditional(args):
    '''
    The ALL_MIX TeamType enables truly random teammates when training (like in the original FCP
    implementation)
    '''
    primary_train_types = [TeamType.ALL_MIX]
    primary_eval_types = {
        'generate' : [TeamType.HIGH_FIRST, TeamType.LOW_FIRST],
        'load': []
    }
    fcp_curriculum = Curriculum(train_types=primary_train_types, is_random=True)
    _, _ = get_FCP_agent_w_pop(
        args,
        fcp_train_types=fcp_curriculum.train_types,
        fcp_eval_types=primary_eval_types,
        fcp_curriculum=fcp_curriculum,
    )


def SPN_XSPCKP(args) -> None:
    '''
    In N-agents games, a randomly initialized agent will be trained with N-X copies of itself
    and X homogeneous unseen teammates, which are checkpoints saved during a previous self-play process.
    when N is 4 and X is 1, the team can be composed by [SP, SP, SP, H], [SP, SP, SP, M], [SP, SP, SP, L] in a 4-chef layout.
    when N is 4 and X is 2, the team can be composed [SP, SP, H, H], [SP, SP, M, M], [SP, SP, L, L] in a 4-chef layout.
    - X is the number of unseen teammate.
    - X is assigned by the variable, unseen_teammates_len, in the funciton.
    '''
    unseen_teammates_len = 1
    primary_train_types = [
        TeamType.SELF_PLAY_HIGH,
        TeamType.SELF_PLAY_MEDIUM,
        TeamType.SELF_PLAY_LOW,
        # TeamType.SELF_PLAY_DYNAMIC_ADV,
        TeamType.SELF_PLAY_STATIC_ADV,
    ]
    primary_eval_types = {
        'generate': [TeamType.SELF_PLAY_HIGH,
                     TeamType.SELF_PLAY_LOW,
                    #  TeamType.SELF_PLAY_DYNAMIC_ADV,
                     TeamType.SELF_PLAY_STATIC_ADV,
                    ],
        'load': []
    }
    if args.prioritized_sampling:
        curriculum = Curriculum(train_types=primary_train_types,
                                eval_types=primary_eval_types,
                                is_random=False,
                                prioritized_sampling=True,
                                priority_scaling=2.0)
    else:
        curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_N_X_SP_agents(
        args,
        n_x_sp_train_types = curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
    )


def set_input(args):
    args.num_players = args.teammates_len + 1

    storage_room_1_chef_layouts = [
        'storage_room_single_left_sym',
        'storage_room_single_right_sym',
        # 'storage_room_single_left_sym_rightpotblocked',
        # 'storage_room_single_right_sym_rightpotblocked',
        # 'storage_room_single_left_sym_leftpotblocked',
        # 'storage_room_single_right_sym_leftpotblocked',
    ]

    cramped_room_ot_1_chef_layuouts = [
        'cramped_room_single_ot_oblocked',
        'cramped_room_single_ot_tblocked',
        'cramped_room_single_ot',
    ]

    cramped_room_ot_2_chef_layuouts = [
        'cramped_room_pair_ot_oblocked',
        'cramped_room_pair_ot_tblocked',
        'cramped_room_pair_ot',
    ]

    classic_2_chefs_layouts = [
        'coordination_ring',
        'counter_circuit',
        'cramped_room',
        'asymmetric_advantages',
        'forced_coordination',
    ]

    complex_2_chefs_layouts = [
        'secret_heaven',
        'storage_room'
    ]

    complex_5_chefs_layouts = [
        'dec_5_chefs_counter_circuit',
        'dec_5_chefs_storage_room',
        'dec_5_chefs_secret_heaven',
        'selected_5_chefs_spacious_room_no_counter_space',
    ]

    one_chef_layouts = cramped_room_ot_1_chef_layuouts
    two_chefs_layouts = cramped_room_ot_2_chef_layuouts
    five_chefs_layouts = complex_5_chefs_layouts

    if args.num_players == 2:
        args.layout_names = two_chefs_layouts
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
        args.n_envs = 210
        args.epoch_timesteps = 2
        args.pop_total_training_timesteps = 4000
        args.n_x_sp_total_training_timesteps = 4000
        args.adversary_total_training_timesteps = 1500
        args.fcp_total_training_timesteps = 1500
        args.n_x_fcp_total_training_timesteps = 1500 * 2
        args.total_ego_agents = 2
        args.exp_dir = f'Test/{args.num_players}'


if __name__ == '__main__':
    args = get_arguments()
    args.quick_test = False
    args.pop_force_training = False
    args.adversary_force_training = False
    args.primary_force_training = False
    args.teammates_len = 1

    if args.teammates_len == 1 or args.teammates_len == 0:
        args.how_long = 20
        args.num_of_ckpoints = 35
    elif args.teammates_len == 4:
        args.how_long = 35
        args.num_of_ckpoints = 50

    set_input(args=args)

    # SPN_XSPCKP(args=args)

    # FCP_traditional(args=args)

    SP(args)

    # FCP_mhri(args=args)

    # SPN_1ADV(args=args)

    # N_1_FCP(args=args)
