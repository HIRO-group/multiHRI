import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, AdversaryPlayConfig
from oai_agents.common.learner import LearnerType
from oai_agents.common.curriculum import Curriculum

from utils import (get_SP_agent, 
                    get_FCP_agent_w_pop, 
                    get_eval_types_to_load, 
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

    get_SP_agent(args=args,
                train_types=curriculum.train_types,
                eval_types=primary_eval_types,
                curriculum=curriculum
                )


def N_X_SP(args) -> None:
    unseen_teammates_len = 1 # This is the X in N_X_SP
    primary_train_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW]
    primary_eval_types = {
                            'generate': [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_LOW],
                            'load': []
                            }

    curriculum = Curriculum(train_types = primary_train_types,
                            is_random=False,
                            total_steps = args.n_x_sp_total_training_timesteps//args.epoch_timesteps,
                            training_phases_durations_in_order={
                                TeamType.SELF_PLAY_LOW: 0.5,
                                TeamType.SELF_PLAY_MEDIUM: 0.125,
                                TeamType.SELF_PLAY_HIGH: 0.125,
                            },
                            rest_of_the_training_probabilities={
                                TeamType.SELF_PLAY_LOW: 0.4,
                                TeamType.SELF_PLAY_MEDIUM: 0.3, 
                                TeamType.SELF_PLAY_HIGH: 0.3,
                            },
                            probabilities_decay_over_time=0
                            )

    get_N_X_SP_agents(
        args,
        n_x_sp_train_types=curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
        )


def N_X_SP_w_adversaries(args) -> None:
    attack_rounds = 3
    unseen_teammates_len = 1
    adversary_play_config = AdversaryPlayConfig.MAP
    primary_train_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_HIGH,
                           TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MEDIUM,
                           TeamType.SELF_PLAY_ADVERSARY]

    primary_eval_types = {'generate': [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_ADVERSARY],
                          'load': []}

    curriculum = Curriculum(train_types = primary_train_types,
                            is_random=False,
                            total_steps = args.n_x_sp_total_training_timesteps//args.epoch_timesteps,
                            training_phases_durations_in_order={
                                TeamType.SELF_PLAY_ADVERSARY: 0.5,
                            },
                            rest_of_the_training_probabilities={
                                TeamType.SELF_PLAY_MEDIUM: 0.3, 
                                TeamType.SELF_PLAY_HIGH: 0.3,
                                TeamType.SELF_PLAY_ADVERSARY: 0.4,
                            },
                            probabilities_decay_over_time=0)
    get_N_X_SP_agents(
        args,
        n_x_sp_train_types=curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
        adversary_play_config=adversary_play_config,
        attack_rounds=attack_rounds
    )



def N_1_SP(args) -> None:
    '''
    The randomly initialized agent will train with itself and one other unseen teammate (e.g. [SP, SP, SP, SP_H] in a 4-chef layout)
    
    :param pop_force_training: Boolean that, if true, indicates population should be generated, otherwise load it from file
    :param primary_force_training: Boolean that, if true, indicates the SP agent teammates_collection should be trained  instead of loaded from file
    :param  : Boolean indicating if   envs should be used for training or not
    '''
    unseen_teammates_len = 1
    primary_train_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_HIGH, 
                                TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_MEDIUM,
                                TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_LOW,
                                ]
    primary_eval_types = {
                            'generate': [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_LOW],
                            'load': []
                            }
    
    curriculum = Curriculum(train_types = primary_train_types,
                            is_random=False,
                            total_steps = args.n_x_sp_total_training_timesteps//args.epoch_timesteps,
                            training_phases_durations_in_order={
                                TeamType.SELF_PLAY_LOW: 0.5,
                                TeamType.SELF_PLAY_MEDIUM: 0.125,
                                TeamType.SELF_PLAY_HIGH: 0.125,
                            },
                            rest_of_the_training_probabilities={
                                TeamType.SELF_PLAY_LOW: 0.4,
                                TeamType.SELF_PLAY_MEDIUM: 0.3, 
                                TeamType.SELF_PLAY_HIGH: 0.3,
                            },
                            probabilities_decay_over_time=0
                            )

    get_N_X_SP_agents(
        args,
        n_x_sp_train_types = curriculum.train_types,
        n_x_sp_eval_types=primary_eval_types,
        curriculum=curriculum,
        unseen_teammates_len=unseen_teammates_len,
    )


def FCP_mhri(args):
    '''
    There are two types of FCP, one is the traditional FCP that uses random teammates (i.e. ALL_MIX), 
    one is our own version that uses certain types HIGH_FIRST, MEDIUM_FIRST, etc. 
    The reason we have our version is that when we used the traditional FCP it got ~0 reward so we 
    decided to add different types for teammates_collection.
    '''
    primary_train_types = [TeamType.LOW_FIRST, TeamType.MEDIUM_FIRST, TeamType.HIGH_FIRST]
    primary_eval_types = {'generate' : [],
                          'load': []}

    fcp_curriculum = Curriculum(train_types = primary_train_types,
                                is_random=False,
                                total_steps = args.fcp_total_training_timesteps//args.epoch_timesteps,
                                training_phases_durations_in_order={
                                    TeamType.LOW_FIRST: 0.5,
                                    TeamType.MEDIUM_FIRST: 0.125,
                                    TeamType.HIGH_FIRST: 0.125,
                                },
                                rest_of_the_training_probabilities={
                                    TeamType.LOW_FIRST: 0.4,
                                    TeamType.MEDIUM_FIRST: 0.3, 
                                    TeamType.HIGH_FIRST: 0.3,
                                },
                                probabilities_decay_over_time=0
                            )

    _, _ = get_FCP_agent_w_pop(args,
                                fcp_train_types = fcp_curriculum.train_types,
                                fcp_eval_types=primary_eval_types,
                                fcp_curriculum=fcp_curriculum)



def FCP_traditional(args):
    '''
    The ALL_MIX TeamType enables truly random teammates when training (like in the original FCP 
    implementation)
    '''

    primary_train_types = [TeamType.ALL_MIX]
    primary_eval_types = {'generate' : [TeamType.HIGH_FIRST, TeamType.LOW_FIRST],
                            'load': []}
    fcp_curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    _, _ = get_FCP_agent_w_pop(args,
                                fcp_train_types=fcp_curriculum.train_types,
                                fcp_eval_types=primary_eval_types,
                                fcp_curriculum=fcp_curriculum,
                                )


def N_1_FCP(args):
    unseen_teammates_len = 1 # This is the X in FCP_X_SP

    fcp_train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    fcp_eval_types = {'generate' : [], 'load': []}
    fcp_curriculum = Curriculum(train_types=fcp_train_types, is_random=True)
    
    primary_train_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    primary_eval_types = {'generate': [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH],
                                'load': []}
    n_1_fcp_curriculum = Curriculum(train_types=primary_train_types, is_random=True)

    get_N_X_FCP_agents(args=args,
                        fcp_train_types=fcp_curriculum.train_types,
                        fcp_eval_types=fcp_eval_types,
                        n_1_fcp_train_types=n_1_fcp_curriculum.train_types,
                        n_1_fcp_eval_types=primary_eval_types,
                        fcp_curriculum=fcp_curriculum,
                        n_1_fcp_curriculum=n_1_fcp_curriculum,
                        unseen_teammates_len=unseen_teammates_len)


def set_input(args):
    args.num_players = args.teammates_len + 1

    two_chefs_layouts = [
        'selected_2_chefs_coordination_ring',
        'selected_2_chefs_counter_circuit',
        'selected_2_chefs_cramped_room'
    ]

    three_chefs_layouts = [
        'selected_3_chefs_coordination_ring',
        'selected_3_chefs_counter_circuit',
        # 'selected_3_chefs_cramped_room'
    ]

    five_chefs_layouts = [
        'selected_5_chefs_counter_circuit',
        'selected_5_chefs_secret_coordination_ring',
        'selected_5_chefs_storage_room'
    ]

    if args.num_players == 2:
        args.layout_names = two_chefs_layouts
    elif args.num_players == 3:
        args.layout_names = three_chefs_layouts
    elif args.num_players == 5:
        args.layout_names = five_chefs_layouts

    args.dynamic_reward = True
    args.final_sparse_r_ratio = 0.5

    if not args.quick_test:
        args.n_envs = 200
        args.epoch_timesteps = 1e5

        args.primary_learner_type = LearnerType.ORIGINALER
        args.adversary_learner_type = LearnerType.SELFISHER
        args.pop_learner_type = LearnerType.ORIGINALER

        args.pop_total_training_timesteps = int(5e6 * args.how_long)
        args.n_x_sp_total_training_timesteps = int(5e6 * args.how_long)
        args.adversary_total_training_timesteps = int(5e6 * args.how_long)
        args.fcp_total_training_timesteps = int(5e6 * args.how_long)
        args.n_x_fcp_total_training_timesteps = int(2 * args.fcp_total_training_timesteps * args.how_long)

        args.SP_seed, args.SP_h_dim = 68, 256
        args.N_X_SP_seed, args.N_X_SP_h_dim = 1010, 256
        args.FCP_seed, args.FCP_h_dim = 2020, 256
        args.N_X_FCP_seed, args.N_X_FCP_h_dim = 2602, 256
        args.ADV_seed, args.ADV_h_dim = 68, 512

        args.num_SPs_to_train = 4
        args.exp_dir = f'Experiment/{args.num_players}/'

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        
        args.pop_total_training_timesteps = 3500
        args.n_x_sp_total_training_timesteps = 1500
        args.adversary_total_training_timesteps = 1500

        args.fcp_total_training_timesteps = 1500
        args.n_x_fcp_total_training_timesteps = 1500 * 2

        args.num_SPs_to_train = 2
        args.exp_dir = f'Test_f/{args.num_players}'


if __name__ == '__main__':
    args = get_arguments()
    args.quick_test = True
    args.parallel = True
    
    args.pop_force_training = True
    args.adversary_force_training = True
    args.primary_force_training = True

    args.teammates_len = 2
    args.how_long = 4 # not effective when quick_test is True

    set_input(args=args)

    N_X_SP_w_adversaries(args=args)
    
    # SP(args)
    
    # N_X_SP(args=args)

    # FCP_traditional(args=args)

    # FCP_mhri(args=args)
    
    # N_1_SP(args=args)

    # N_1_FCP(args=args)
