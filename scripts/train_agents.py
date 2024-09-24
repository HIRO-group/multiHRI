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
                get_N_X_selfplay_agents_trained_w_selfplay_types,
                Curriculum
                )


def SP(args, pop_force_training):
    args.primary_train_types = [TeamType.SELF_PLAY]
    args.primary_eval_types = {
        'generate': [TeamType.SELF_PLAY],
        'load': []
    }
    curriculum = Curriculum(train_types=args.primary_train_types, is_random=True)

    get_selfplay_agent_w_tms_collection(args=args,
                                        train_types=curriculum.train_types,
                                        eval_types=args.primary_eval_types,
                                        total_training_timesteps=args.pop_total_training_timesteps,
                                        force_training=pop_force_training,
                                        curriculum=curriculum)



def N_X_SP(args, 
           pop_force_training:bool,
           primary_force_training:bool,
           parallel:bool) -> None:

    args.unseen_teammates_len = 1 # This is the X in N_X_SP
    args.primary_train_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW]
    args.primary_eval_types = {
                            'generate': [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                            'load': []
                            }

    curriculum = Curriculum(train_types = args.primary_train_types,
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

    get_N_X_selfplay_agents_trained_w_selfplay_types(
        args,
        pop_total_training_timesteps=args.pop_total_training_timesteps,
        pop_force_training=pop_force_training,
        n_x_sp_train_types = curriculum.train_types,
        n_x_sp_eval_types=args.primary_eval_types,
        n_x_sp_total_training_timesteps=args.n_x_sp_total_training_timesteps,
        n_x_sp_force_training=primary_force_training,
        curriculum=curriculum,
        parallel=parallel,
        num_SPs_to_train=args.num_SPs_to_train
        )


def N_1_SP(args, 
                  pop_force_training:bool,
                  primary_force_training:bool,
                  parallel:bool) -> None:
    '''
    The randomly initialized agent will train with itself and one other unseen teammate (e.g. [SP, SP, SP, SP_H] in a 4-chef layout)
    
    :param pop_force_training: Boolean that, if true, indicates population should be generated, otherwise load it from file
    :param primary_force_training: Boolean that, if true, indicates the SP agent teammates_collection should be trained  instead of loaded from file
    :param parallel: Boolean indicating if parallel envs should be used for training or not
    '''
    args.unseen_teammates_len = 1
    args.primary_train_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW]
    args.primary_eval_types = {
                            'generate': [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                            'load': get_eval_types_to_load()
                            }
    
    curriculum = Curriculum(train_types = args.primary_train_types,
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

    get_N_X_selfplay_agents_trained_w_selfplay_types(
        args,
        pop_total_training_timesteps=args.pop_total_training_timesteps,
        pop_force_training=pop_force_training,
        n_x_sp_train_types = curriculum.train_types,
        n_x_sp_eval_types=args.primary_eval_types,
        n_x_sp_total_training_timesteps=args.n_x_sp_total_training_timesteps,
        n_x_sp_force_training=primary_force_training,
        curriculum=curriculum,
        parallel=parallel,
        num_SPs_to_train=args.num_SPs_to_train
    )


def FCP_mhri(args, pop_force_training, primary_force_training, parallel):
    '''
    There are two types of FCP, one is the traditional FCP that uses random teammates (i.e. ALL_MIX), 
    one is our own version that uses certain types HIGH_FIRST, MEDIUM_FIRST, etc. 
    The reason we have our version is that when we used the traditional FCP it got ~0 reward so we 
    decided to add different types for teammates_collection.
    '''
    args.primary_train_types = [TeamType.LOW_FIRST, TeamType.MEDIUM_FIRST, TeamType.HIGH_FIRST]
    args.primary_eval_types = {'generate' : [],
                            'load': get_eval_types_to_load()}

    fcp_curriculum = Curriculum(train_types = args.primary_train_types,
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
                                        fcp_train_types = fcp_curriculum.train_types,
                                        fcp_eval_types=args.primary_eval_types,
                                        pop_force_training=pop_force_training,
                                        primary_force_training=primary_force_training,
                                        fcp_curriculum=fcp_curriculum,
                                        num_SPs_to_train=args.num_SPs_to_train,
                                        parallel=parallel,
                                        )



def FCP_traditional(args, pop_force_training, primary_force_training, parallel):
    '''
    The ALL_MIX TeamType enables truly random teammates when training (like in the original FCP 
    implementation)
    '''

    args.primary_train_types = [TeamType.ALL_MIX]
    args.primary_eval_types = {'generate' : [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST],
                            'load': []}

    fcp_curriculum = Curriculum(train_types=args.primary_train_types, is_random=True)

    _, _ = get_fcp_agent_w_tms_clction(args,
                                        pop_total_training_timesteps=args.pop_total_training_timesteps,
                                        fcp_total_training_timesteps=args.fcp_total_training_timesteps,
                                        
                                        fcp_train_types=fcp_curriculum.train_types,
                                        fcp_eval_types=args.primary_eval_types,

                                        pop_force_training=pop_force_training,
                                        primary_force_training=primary_force_training,

                                        fcp_curriculum=fcp_curriculum,
                                        num_SPs_to_train=args.num_SPs_to_train,
                                        parallel=parallel
                                        )


def N_1_FCP(args, pop_force_training, primary_force_training, n_1_fcp_force_training, parallel):
    args.fcp_train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    args.fcp_eval_types = {'generate' : [],
                           'load': get_eval_types_to_load()}
    
    args.primary_train_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    args.primary_eval_types = {'generate': [],
                                'load': get_eval_types_to_load()}

    fcp_curriculum = Curriculum(train_types = args.fcp_train_types,is_random=True)
    n_1_fcp_curriculum = Curriculum(train_types=args.primary_train_types, is_random=True)

    get_fcp_trained_w_selfplay_types(args=args,
                                    pop_total_training_timesteps=args.pop_total_training_timesteps,
                                    fcp_total_training_timesteps=args.fcp_total_training_timesteps,
                                    n_1_fcp_total_training_timesteps=args.n_1_fcp_total_training_timesteps,

                                    fcp_eval_types=args.fcp_eval_types,
                                    n_1_fcp_eval_types=args.primary_eval_types,


                                    pop_force_training=pop_force_training,
                                    fcp_force_training=primary_force_training,
                                    primary_force_training=n_1_fcp_force_training,

                                    num_SPs_to_train=args.num_SPs_to_train,
                                    parallel=parallel,
                                    fcp_curriculum=fcp_curriculum,
                                    n_1_fcp_curriculum=n_1_fcp_curriculum,
                                    )


def set_input(args, quick_test=False, supporter_run=False):
    '''
    Suggested 3-Chefs Layouts are '3_chefs_small_kitchen_two_resources', 
    '3_chefs_counter_circuit', '3_chefs_asymmetric_advantages', 
    '3_chefs_forced_coordination_3OP2S1D'.
    '''
    args.layout_names = ['3_chefs_small_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
        
    if not quick_test:
        args.learner_type = LearnerType.Originaler
        args.n_envs = 200
        args.epoch_timesteps = 1e5

        how_long = 1.0
        args.pop_total_training_timesteps = int(5e6 * how_long)
        args.n_x_sp_total_training_timesteps = int(5e6 * how_long)

        args.fcp_total_training_timesteps = int(5e6 * how_long)
        args.n_1_fcp_total_training_timesteps = int(2 * args.fcp_total_training_timesteps * how_long)

        args.SP_seed, args.SP_h_dim = 68, 256
        args.N_X_SP_seed, args.N_X_SP_h_dim = 1010, 256

        args.FCP_seed, args.FCP_h_dim = 2020, 256
        args.FCPWSP_seed, args.FCPWSP_h_dim = 2602, 256

        args.num_SPs_to_train = 2
        args.exp_dir = 'experiment/1' # This is the directory where the experiment will be saved. Change it to your desired directory
    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        
        args.pop_total_training_timesteps = 3500
        args.fcp_total_training_timesteps = 3500
        args.n_x_sp_total_training_timesteps = 3500
        args.n_1_fcp_total_training_timesteps = 3500 * 2

        args.num_SPs_to_train = 2
        args.exp_dir = 'test/1'


if __name__ == '__main__':
    args = get_arguments()
    quick_test = True
    parallel = True

    pop_force_training = False
    primary_force_training = True
    
    set_input(args=args, quick_test=quick_test)

    # SP(args=args,
    #    pop_force_training=pop_force_training)
    
    
    # N_X_SP(args=args,
    #        pop_force_training=pop_force_training,
    #        primary_force_training=primary_force_training,
    #        parallel=parallel)

    
    # FCP_traditional(args=args,
    #                 pop_force_training=pop_force_training,
    #                 primary_force_training=primary_force_training,
    #                 parallel=parallel)

    # FCP_mhri(args=args,
    #         pop_force_training=pop_force_training,
    #         primary_force_training=primary_force_training,
    #         parallel=parallel)

    # N_1_SP(args=args,
    #         pop_force_training=pop_force_training,
    #         primary_force_training=primary_force_training,
    #         parallel=parallel)


    N_1_FCP(args=args,
            pop_force_training=pop_force_training,
            primary_force_training=primary_force_training,
            parallel=parallel)