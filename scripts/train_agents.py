import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType
from utils import (get_selfplay_agent_w_tms_collection, 
                get_fcp_agent_w_tms_clction, 
                get_eval_types_to_load, 
                get_fcp_trained_w_selfplay_types, 
                get_selfplay_agent_trained_w_selfplay_types)


def SP(args, pop_force_training):
    args.sp_train_types = [TeamType.SELF_PLAY]
    args.sp_eval_types = {
        'generate': [TeamType.SELF_PLAY],
        'load': get_eval_types_to_load()
    }
    _, _ = get_selfplay_agent_w_tms_collection(args=args,
                                            train_types=args.sp_train_types,
                                            eval_types=args.sp_eval_types,
                                            total_training_timesteps=args.pop_total_training_timesteps,
                                            force_training=pop_force_training)


def SP_w_SP_Types(args, pop_force_training:bool, parallel:bool) -> None:
    '''
    Set up and run the training for self-play with self-play types
    Similar to FCP_w_SP_TYPES, this function will first train a population of SP agents, organize them into a teammates_collection
    based on TeamType, and then select agents from th the teams to SP with a randomly initialized SP agent
    So the randomly initialized agent will train with itself and one other unseen teammate (e.g. [SP, SP, SP, SP_H] in a 4-chef layout)
    
    :param pop_force_training: Boolean that, if true, indicates population should be generated, otherwise load it from file
    :param parallel: Boolean indicating if parallel envs should be used for training or not
    '''

    # Set the train and eval types that will be used to organize the first "FCP-like" population
    # NOTE: When in order to use SELF_PLAY_X later, we need X_FIRST here
    args.sp_train_types = [TeamType.HIGH_FIRST]
    args.sp_eval_types = {
        'generate': [TeamType.HIGH_FIRST],
        'load': []
        }

    # Set the train and eval types for the teammates that will be used to play with the SP agent
    args.sp_w_sp_train_types = [TeamType.SELF_PLAY_HIGH]
    args.sp_w_sp_eval_types = {
                            'generate': [TeamType.SELF_PLAY_HIGH],
                            'load': get_eval_types_to_load()
                            }

    get_selfplay_agent_trained_w_selfplay_types(
        args,
        pop_total_training_timesteps=args.pop_total_training_timesteps,
        sp_train_types=args.sp_train_types,
        sp_eval_types=args.sp_eval_types,
        num_self_play_agents_to_train=args.num_sp_agents_to_train,
        sp_w_sp_total_training_timesteps=args.sp_w_sp_total_training_timesteps,
        sp_w_sp_train_types=args.sp_w_sp_train_types,
        sp_w_sp_eval_types=args.sp_w_sp_eval_types,
        force_training=pop_force_training,
        parallel=parallel)


def FCP(args, pop_force_training, fcp_force_training, parallel):
    args.fcp_train_types = [TeamType.HIGH_FIRST]
    args.fcp_eval_types = {'generate' : [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST],
                            'load': get_eval_types_to_load()}
    _, _ = get_fcp_agent_w_tms_clction(args,
                                        pop_total_training_timesteps=args.pop_total_training_timesteps,
                                        fcp_total_training_timesteps=args.fcp_total_training_timesteps,
                                        fcp_train_types=args.fcp_train_types,
                                        fcp_eval_types=args.fcp_eval_types,
                                        pop_force_training=pop_force_training,
                                        fcp_force_training=fcp_force_training,
                                        num_self_play_agents_to_train=args.num_sp_agents_to_train,
                                        parallel=parallel)


def FCP_w_SP_TYPES(args, pop_force_training, fcp_force_training, fcp_w_sp_force_training, parallel):
    args.fcp_train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    args.fcp_eval_types = {'generate' : [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST],
                           'load': get_eval_types_to_load()}
    args.fcp_w_sp_train_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    args.fcp_w_sp_eval_types = {'generate': [],
                                'load': get_eval_types_to_load()}
    get_fcp_trained_w_selfplay_types(args=args,
                                    pop_total_training_timesteps=args.pop_total_training_timesteps,
                                    fcp_total_training_timesteps=args.fcp_total_training_timesteps,
                                    fcp_w_sp_total_training_timesteps=args.fcp_w_sp_total_training_timesteps,
                                    fcp_train_types=args.fcp_train_types,
                                    fcp_eval_types=args.fcp_eval_types,
                                    fcp_w_sp_train_types=args.fcp_w_sp_train_types,
                                    fcp_w_sp_eval_types=args.fcp_w_sp_eval_types,
                                    pop_force_training=pop_force_training,
                                    fcp_force_training=fcp_force_training,
                                    fcp_w_sp_force_training=fcp_w_sp_force_training,
                                    parallel=parallel)


def set_input(args, quick_test=False):
    args.layout_names = ['3_chefs_small_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
    
    if not quick_test: 
        args.n_envs = 50
        args.epoch_timesteps = 1e5
        args.pop_total_training_timesteps = 5e6
        args.fcp_total_training_timesteps = 5e6
        args.sp_w_sp_total_training_timesteps = 5e6
        args.fcp_w_sp_total_training_timesteps = 2 * 5e6
        args.num_sp_agents_to_train = 2

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        args.pop_total_training_timesteps = 3500
        args.fcp_total_training_timesteps = 3500
        args.sp_w_sp_total_training_timesteps = 3500
        args.fcp_w_sp_total_training_timesteps = 3500 * 2
        args.num_sp_agents_to_train = 2
    

if __name__ == '__main__':
    args = get_arguments()
    quick_test = False
    parallel = True
    
    pop_force_training = True
    fcp_force_training = True
    fcp_w_sp_force_training = True
    
    set_input(args=args, quick_test=quick_test)

    # SP(args=args,
    #    pop_force_training=pop_force_training)

    SP_w_SP_Types(args=args,
                  pop_force_training=pop_force_training,
                  parallel=parallel)


    # FCP(args=args,
    #     pop_force_training=pop_force_training,
    #     fcp_force_training=fcp_force_training,
    #     parallel=parallel)


    # FCP_w_SP_TYPES(args=args,
    #                pop_force_training=pop_force_training,
    #                fcp_force_training=fcp_force_training,
    #                fcp_w_sp_force_training=fcp_w_sp_force_training,
    #                parallel=parallel)
