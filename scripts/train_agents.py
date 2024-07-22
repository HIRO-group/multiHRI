import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType
from utils import get_selfplay_agent, get_fcp_agent_w_tms_clction, get_eval_types_to_load, get_fcp_trained_w_selfplay_types


def SP(args, pop_force_training):
    args.sp_train_types = [TeamType.SELF_PLAY]
    args.sp_eval_types = {
        'generate': [TeamType.SELF_PLAY],
        'load': get_eval_types_to_load()
    }
    get_selfplay_agent(args=args,
                       train_types=args.sp_train_types,
                       eval_types=args.sp_eval_types,
                       total_training_timesteps=args.pop_total_training_timesteps,
                       force_training=pop_force_training,
                        )


def FCP(args, pop_force_training, fcp_force_training, parallel):
    args.fcp_train_types = [TeamType.HIGH_FIRST]
    args.fcp_eval_types = {'generate' : [],
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
    args.fcp_eval_types = {'generate' : [],
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
        args.fcp_w_sp_total_training_timesteps = 2 * 5e6
        args.num_sp_agents_to_train = 2

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        args.pop_total_training_timesteps = 3500
        args.fcp_total_training_timesteps = 3500
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


    FCP(args=args,
        pop_force_training=pop_force_training,
        fcp_force_training=fcp_force_training,
        parallel=parallel)


    # FCP_w_SP_TYPES(args=args,
    #                pop_force_training=pop_force_training,
    #                fcp_force_training=fcp_force_training,
    #                fcp_w_sp_force_training=fcp_w_sp_force_training,
    #                parallel=parallel)
