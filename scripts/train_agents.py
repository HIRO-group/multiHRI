import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType
from oai_agents.common.learner import LearnerType, Learner
from utils import (get_selfplay_agent_w_tms_collection,
                get_fcp_agent_w_tms_clction,
                get_eval_types_to_load,
                get_fcp_trained_w_selfplay_types,
                get_selfplay_agent_trained_w_selfplay_types,
                Curriculum
                )


def SP(args, pop_force_training):
    args.sp_train_types = [TeamType.SELF_PLAY]
    args.sp_eval_types = {
        'generate': [TeamType.SELF_PLAY],
        'load': []
    }
    curriculum = Curriculum(train_types=args.sp_train_types, is_random=True)

    get_selfplay_agent_w_tms_collection(args=args,
                       eval_types=args.sp_eval_types,
                       total_training_timesteps=args.pop_total_training_timesteps,
                       force_training=pop_force_training,
                       curriculum=curriculum)



def SP_w_SP_Types(args,
                  pop_force_training:bool,
                  sp_w_sp_force_training:bool,
                  parallel:bool) -> None:
    '''
    Set up and run the training for self-play with self-play types
    Similar to FCP_w_SP_TYPES, this function will first train a population of SP agents, organize them into a teammates_collection
    based on TeamType, and then select agents from th the teams to SP with a randomly initialized SP agent
    So the randomly initialized agent will train with itself and one other unseen teammate (e.g. [SP, SP, SP, SP_H] in a 4-chef layout)

    :param pop_force_training: Boolean that, if true, indicates population should be generated, otherwise load it from file
    :param sp_w_sp_force_training: Boolean that, if true, indicates the SP agent teammates_collection should be trained  instead of loaded from file
    :param parallel: Boolean indicating if parallel envs should be used for training or not
    '''

    # If you use train/eval types TeamType.SELF_PLAY_X then X_FIRST should be in pop_train_types
    # pop_train_types can be passed to get_selfplay_agent_trained_w_selfplay_types and
    # it's default values are [HIGH_FIRST, MEDIUM_FIRST, LOW_FIRST]
    args.sp_w_sp_train_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW]
    args.sp_w_sp_eval_types = {
                            'generate': [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                            'load': get_eval_types_to_load()
                            }

    curriculum = Curriculum(train_types = args.sp_w_sp_train_types,
                            is_random=False,
                            total_steps = args.sp_w_sp_total_training_timesteps//args.epoch_timesteps,
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

    get_selfplay_agent_trained_w_selfplay_types(
        args,
        pop_total_training_timesteps=args.pop_total_training_timesteps,
        sp_w_sp_total_training_timesteps=args.sp_w_sp_total_training_timesteps,
        sp_w_sp_eval_types=args.sp_w_sp_eval_types,
        pop_force_training=pop_force_training,
        sp_w_sp_force_training=sp_w_sp_force_training,
        parallel=parallel,
        curriculum=curriculum,
        num_self_play_agents_to_train=args.num_sp_agents_to_train
        )


def FCP(args, pop_force_training, fcp_force_training, parallel):
    args.fcp_train_types = [TeamType.LOW_FIRST, TeamType.MEDIUM_FIRST, TeamType.HIGH_FIRST]
    args.fcp_eval_types = {'generate' : [],
                            'load': get_eval_types_to_load()}

    fcp_curriculum = Curriculum(train_types = args.fcp_train_types,
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

    _, _ = get_fcp_agent_w_tms_clction(args,
                                        pop_total_training_timesteps=args.pop_total_training_timesteps,
                                        fcp_total_training_timesteps=args.fcp_total_training_timesteps,
                                        fcp_eval_types=args.fcp_eval_types,
                                        pop_force_training=pop_force_training,
                                        fcp_force_training=fcp_force_training,
                                        fcp_curriculum=fcp_curriculum,
                                        num_self_play_agents_to_train=args.num_sp_agents_to_train,
                                        parallel=parallel
                                        )


def FCP_w_SP_TYPES(args, pop_force_training, fcp_force_training, fcp_w_sp_force_training, parallel):
    args.fcp_train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    args.fcp_eval_types = {'generate' : [],
                           'load': get_eval_types_to_load()}
    args.fcp_w_sp_train_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    args.fcp_w_sp_eval_types = {'generate': [],
                                'load': get_eval_types_to_load()}

    fcp_curriculum = Curriculum(train_types = args.fcp_train_types,is_random=True)
    fcp_w_sp_curriculum = Curriculum(train_types=args.fcp_w_sp_train_types, is_random=True)

    get_fcp_trained_w_selfplay_types(args=args,
                                    pop_total_training_timesteps=args.pop_total_training_timesteps,
                                    fcp_total_training_timesteps=args.fcp_total_training_timesteps,
                                    fcp_w_sp_total_training_timesteps=args.fcp_w_sp_total_training_timesteps,
                                    fcp_eval_types=args.fcp_eval_types,
                                    fcp_w_sp_eval_types=args.fcp_w_sp_eval_types,
                                    pop_force_training=pop_force_training,
                                    fcp_force_training=fcp_force_training,
                                    fcp_w_sp_force_training=fcp_w_sp_force_training,
                                    num_self_play_agents_to_train=args.num_sp_agents_to_train,
                                    parallel=parallel,
                                    fcp_curriculum=fcp_curriculum,
                                    fcp_w_sp_curriculum=fcp_w_sp_curriculum
                                    )


def set_input(args, quick_test=False, supporter_run=False):
    '''
    Suggested 3-Chefs Layouts are '3_chefs_small_kitchen_two_resources',
    '3_chefs_counter_circuit', '3_chefs_asymmetric_advantages',
    '3_chefs_forced_coordination_3OP2S1D'.
    '''
    args.layout_names = ['3_chefs_long_counter_circuit', '3_chefs_long_onion', '3_chefs_unequal_kitchens']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates

    if not quick_test:
        args.learner_type = LearnerType.Originaler
        args.n_envs = 200
        how_long = 1.0
        args.epoch_timesteps = 1e5
        args.pop_total_training_timesteps = 5e6 * how_long
        args.fcp_total_training_timesteps = 2 * 5e6 * how_long
        args.sp_w_sp_total_training_timesteps = 5e6 * how_long
        args.fcp_w_sp_total_training_timesteps = 4 * 5e6 * how_long
        args.SP_seed, args.SP_h_dim = 68, 256
        args.SPWSP_seed, args.SPWSP_h_dim = 1010, 256
        args.FCP_seed, args.FCP_h_dim = 2020, 256
        args.FCPWSP_seed, args.FCPWSP_h_dim = 2602, 256
        args.num_sp_agents_to_train = 3
        args.exp_dir = 'experiment-new_layouts'
    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        args.pop_total_training_timesteps = 3500
        args.fcp_total_training_timesteps = 3500
        args.sp_w_sp_total_training_timesteps = 3500
        args.fcp_w_sp_total_training_timesteps = 3500 * 2
        args.num_sp_agents_to_train = 3
        args.exp_dir = 'test-new_layouts-1'


if __name__ == '__main__':
    args = get_arguments()
    quick_test = False
    parallel = True

    pop_force_training = True
    fcp_force_training = False
    fcp_w_sp_force_training = False
    sp_w_sp_force_training = True

    set_input(args=args, quick_test=quick_test)

    SP(args=args,
       pop_force_training=pop_force_training)

    # FCP(args=args,
    #     pop_force_training=pop_force_training,
    #     fcp_force_training=fcp_force_training,
    #     parallel=parallel)

    SP_w_SP_Types(args=args,
                   pop_force_training=pop_force_training,
                   sp_w_sp_force_training=sp_w_sp_force_training,
                   parallel=parallel)


    # FCP_w_SP_TYPES(args=args,
    #                pop_force_training=pop_force_training,
    #                fcp_force_training=fcp_force_training,
    #                fcp_w_sp_force_training=fcp_w_sp_force_training,
    #                parallel=parallel)
