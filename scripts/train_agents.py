import multiprocessing as mp
import os
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, CheckedPoints
from oai_agents.common.learner import LearnerType, Learner
from utils import (get_selfplay_agent_w_tms_collection, 
                get_fcp_agent_w_tms_clction, 
                get_eval_types_to_load, 
                get_fcp_trained_w_selfplay_types, 
                get_selfplay_agent_trained_w_selfplay_types,
                get_adversary,
                get_agent_play_w_adversarys,
                Curriculum
                )
from scripts.utils.common import generate_name

def SingleAdversaryPlay(args, 
                        exp_tag = 'S2FP', 
                        # main_agent_path = 'S2FP/sp_s68_h512_tr(SP)_ran',
                        main_agent_path = None,
                        main_agent_seed = 68,
                        main_agent_h_dim = 512,
                        main_agent_type = LearnerType.SUPPORTER, 
                        adversary_seed = 68,
                        adversary_h_dim = 512,
                        adversary_type = LearnerType.SELFISHER, 
                        checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
                        how_long_init = 4.0,
                        how_long_for_agent = 1.0,
                        how_long_for_adv = 1.0,
                        rounds_of_advplay = 101,
                        reward_magnifier = 3.0):
    
    args.dynamic_reward = False
    args.final_sparse_r_ratio = 0.5
    if main_agent_path is None:
        how_long = how_long_init
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.SP_seed, args.SP_h_dim = main_agent_seed, main_agent_h_dim
        args.learner_type = main_agent_type
        args.reward_magnifier = reward_magnifier
        args.exp_dir = exp_tag 
        sp, _ = SP(args=args, pop_force_training=True)
        main_agent_path = f"{exp_tag}/{sp.name}"
    root = main_agent_path
    root_adv = f"{root}/{exp_tag}"

    how_long = how_long_for_adv
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.exp_dir = f"{root_adv}/{adversary_type}/0"
    _, _, adv_tag = ADV(
        args = args, 
        agent_folder_path = root, 
        agent_file_tag = CheckedPoints.BEST_EVAL_REWARD, 
        adversary_type = adversary_type,
        adversary_seed = adversary_seed,
        adversary_h_dim = adversary_h_dim,
        reward_magnifier = reward_magnifier)

    how_long = how_long_init + how_long_for_agent
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.exp_dir = f"{root_adv}/{main_agent_type}-{adversary_type}play/0"
    _, _, pwadv_tag = PwADVs( 
            args=args, 
            agent_folder_path = root, 
            agent_file_tag = CheckedPoints.BEST_EVAL_REWARD,
            adv_folder_paths = [f"{root_adv}/{adversary_type}/0"], 
            adv_file_tag = f"{adv_tag}/{checked_adversary}",
            main_agent_type = main_agent_type,
            main_agent_seed = main_agent_seed,
            main_agent_h_dim = main_agent_h_dim,
            reward_magnifier = reward_magnifier)
    ###################################################################
    for round in range(1,rounds_of_advplay):
        how_long = how_long_for_adv
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.exp_dir = f"{root_adv}/{adversary_type}/{str(round)}"
        _, _, adv_tag = ADV(
            args=args,
            agent_folder_path = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round-1)}", 
            agent_file_tag = f"{pwadv_tag}/{CheckedPoints.FINAL_TRAINED_MODEL}", 
            adversary_type = adversary_type,
            adversary_seed = adversary_seed,
            adversary_h_dim = adversary_h_dim,
            reward_magnifier = reward_magnifier)
        
        how_long = how_long_init + how_long_for_agent*(round+1)
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.exp_dir = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round)}"
        _, _, pwadv_tag = PwADVs( 
                args=args, 
                agent_folder_path = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round-1)}", 
                agent_file_tag = f"{pwadv_tag}/{CheckedPoints.FINAL_TRAINED_MODEL}",
                adv_folder_paths = [f"{root_adv}/{adversary_type}/{str(round)}"], 
                adv_file_tag = f"{adv_tag}/{checked_adversary}",
                main_agent_type = main_agent_type,
                main_agent_seed = main_agent_seed,
                main_agent_h_dim = main_agent_h_dim,
                reward_magnifier = reward_magnifier)
        
def MultiAdversaryPlay( args, 
                        exp_tag = 'M2FP', 
                        # main_agent_path = 'M2FP/sp_s68_h512_tr(SP)_ran',
                        main_agent_path = None,
                        main_agent_seed = 68,
                        main_agent_h_dim = 512,
                        main_agent_type = LearnerType.SUPPORTER, 
                        adversary_seed = 68,
                        adversary_h_dim = 512,
                        adversary_type = LearnerType.SELFISHER, 
                        checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
                        how_long_init = 4.0,
                        how_long_for_agent = 2.0,
                        how_long_for_adv = 1.0,
                        rounds_of_advplay = 101,
                        reward_magnifier = 3.0):
    
    args.dynamic_reward = False
    args.final_sparse_r_ratio = 0.5
    if main_agent_path is None:
        how_long = how_long_init
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.SP_seed, args.SP_h_dim = main_agent_seed, main_agent_h_dim
        args.learner_type = main_agent_type
        args.reward_magnifier = reward_magnifier
        args.exp_dir = exp_tag 
        sp, _ = SP(args=args, pop_force_training=True)
        main_agent_path = f"{exp_tag}/{sp.name}"
    root = main_agent_path
    root_adv = f"{root}/{exp_tag}"

    how_long = how_long_for_adv
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.exp_dir = f"{root_adv}/{adversary_type}/0"
    _, _, adv_tag = ADV(
        args = args, 
        agent_folder_path = root, 
        agent_file_tag = CheckedPoints.BEST_EVAL_REWARD, 
        adversary_type = adversary_type,
        adversary_seed = adversary_seed,
        adversary_h_dim = adversary_h_dim,
        reward_magnifier = reward_magnifier)

    how_long = how_long_init + how_long_for_agent
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.exp_dir = f"{root_adv}/{main_agent_type}-{adversary_type}play/0"
    _, _, pwadv_tag = PwADVs( 
            args=args, 
            agent_folder_path = root, 
            agent_file_tag = CheckedPoints.BEST_EVAL_REWARD,
            adv_folder_paths = [f"{root_adv}/{adversary_type}/0"], 
            adv_file_tag = f"{adv_tag}/{checked_adversary}",
            main_agent_type = main_agent_type,
            main_agent_seed = main_agent_seed,
            main_agent_h_dim = main_agent_h_dim,
            reward_magnifier = reward_magnifier)
    ###################################################################
    for round in range(1,rounds_of_advplay):
        how_long = how_long_for_adv
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.exp_dir = f"{root_adv}/{adversary_type}/{str(round)}"
        _, _, adv_tag = ADV(
            args=args,
            agent_folder_path = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round-1)}", 
            agent_file_tag = f"{pwadv_tag}/{CheckedPoints.FINAL_TRAINED_MODEL}", 
            adversary_type = adversary_type,
            adversary_seed = adversary_seed,
            adversary_h_dim = adversary_h_dim,
            reward_magnifier = reward_magnifier)
        
        how_long = how_long_init + how_long_for_agent*(round+1)
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.exp_dir = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round)}"
        _, _, pwadv_tag = PwADVs( 
                args=args, 
                agent_folder_path = root, 
                agent_file_tag = CheckedPoints.BEST_EVAL_REWARD,
                adv_folder_paths = [f"{root_adv}/{adversary_type}/{str(round_num)}" for round_num in range(round)], 
                adv_file_tag = f"{adv_tag}/{checked_adversary}",
                main_agent_type = main_agent_type,
                main_agent_seed = main_agent_seed,
                main_agent_h_dim = main_agent_h_dim,
                reward_magnifier = reward_magnifier)
        
def MultiAdversaryScheduledPlay(args, 
                                exp_tag = 'M2FSP', 
                                # main_agent_path = 'M2FSP/sp_s68_h512_tr(SP)_ran',
                                main_agent_path = None,
                                main_agent_seed = 68,
                                main_agent_h_dim = 512,
                                main_agent_type = LearnerType.SUPPORTER, 
                                adversary_seed = 68,
                                adversary_h_dim = 512,
                                adversary_type = LearnerType.SELFISHER, 
                                checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
                                how_long_init = 4.0,
                                how_long_for_agent = 2.0,
                                how_long_for_adv = 1.0,
                                rounds_of_advplay = 101,
                                reward_magnifier = 3.0):
    
    args.dynamic_reward = False
    args.final_sparse_r_ratio = 0.5
    if main_agent_path is None:
        how_long = how_long_init
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.SP_seed, args.SP_h_dim = main_agent_seed, main_agent_h_dim
        args.learner_type = main_agent_type
        args.reward_magnifier = reward_magnifier
        args.exp_dir = exp_tag 
        sp, _ = SP(args=args, pop_force_training=True)
        main_agent_path = f"{exp_tag}/{sp.name}"
    root = main_agent_path
    root_adv = f"{root}/{exp_tag}"

    how_long = how_long_for_adv
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.exp_dir = f"{root_adv}/{adversary_type}/0"
    _, _, adv_tag = ADV(
        args = args, 
        agent_folder_path = root, 
        agent_file_tag = CheckedPoints.BEST_EVAL_REWARD, 
        adversary_type = adversary_type,
        adversary_seed = adversary_seed,
        adversary_h_dim = adversary_h_dim,
        reward_magnifier = reward_magnifier)

    how_long = how_long_init + how_long_for_agent
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.exp_dir = f"{root_adv}/{main_agent_type}-{adversary_type}play/0"
    _, _, pwadv_tag = PwADVs( 
            args=args, 
            agent_folder_path = root, 
            agent_file_tag = CheckedPoints.BEST_EVAL_REWARD,
            adv_folder_paths = [f"{root_adv}/{adversary_type}/0"], 
            adv_file_tag = f"{adv_tag}/{checked_adversary}",
            main_agent_type = main_agent_type,
            main_agent_seed = main_agent_seed,
            main_agent_h_dim = main_agent_h_dim,
            reward_magnifier = reward_magnifier)
    ###################################################################
    for round in range(1,rounds_of_advplay):
        how_long = how_long_for_adv
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.exp_dir = f"{root_adv}/{adversary_type}/{str(round)}"
        _, _, adv_tag = ADV(
            args=args,
            agent_folder_path = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round-1)}", 
            agent_file_tag = f"{pwadv_tag}/{CheckedPoints.FINAL_TRAINED_MODEL}", 
            adversary_type = adversary_type,
            adversary_seed = adversary_seed,
            adversary_h_dim = adversary_h_dim,
            reward_magnifier = reward_magnifier)
        
        how_long = how_long_init + how_long_for_agent*round + how_long_for_agent*0.5
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.exp_dir = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round)}"
        _, _, pwadv_tag = PwADVs( 
                args=args, 
                agent_folder_path = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round-1)}", 
                agent_file_tag = f"{pwadv_tag}/{CheckedPoints.FINAL_TRAINED_MODEL}",
                adv_folder_paths = [f"{root_adv}/{adversary_type}/{str(round)}"], 
                adv_file_tag = f"{adv_tag}/{checked_adversary}",
                main_agent_type = main_agent_type,
                main_agent_seed = main_agent_seed,
                main_agent_h_dim = main_agent_h_dim,
                reward_magnifier = reward_magnifier)
        
        how_long = how_long_init + how_long_for_agent*round + how_long_for_agent*1
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.exp_dir = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round)}"
        _, _, pwadv_tag = PwADVs( 
                args=args, 
                agent_folder_path = f"{root_adv}/{main_agent_type}-{adversary_type}play/{str(round)}", 
                agent_file_tag = f"{pwadv_tag}/{CheckedPoints.FINAL_TRAINED_MODEL}",
                adv_folder_paths = [f"{root_adv}/{adversary_type}/{str(round_num)}" for round_num in range(round)], 
                adv_file_tag = f"{adv_tag}/{checked_adversary}",
                main_agent_type = main_agent_type,
                main_agent_seed = main_agent_seed,
                main_agent_h_dim = main_agent_h_dim,
                reward_magnifier = reward_magnifier)

def PwADVs(args, 
          agent_folder_path, 
          agent_file_tag,
          adv_folder_paths, 
          adv_file_tag,
          main_agent_type = LearnerType.SUPPORTER,
          main_agent_seed = 68,
          main_agent_h_dim = 512,
          reward_magnifier = 3.0
          ):
    train_types = [TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY]
    eval_types = {
        'generate': [TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY],
        'load': []
    }
    curriculum = Curriculum(train_types=train_types, is_random=True)
    agent_path = f"agent_models/{agent_folder_path}/{agent_file_tag}"
    adv_paths = [f"agent_models/{adv_folder_path}/{adv_file_tag}" for adv_folder_path in adv_folder_paths]
    args.learner_type = main_agent_type
    args.PwADV_seed = main_agent_seed
    args.PwADV_h_dim = main_agent_h_dim
    args.reward_magnifier = reward_magnifier
    agent_model, teammates, agent_tag = get_agent_play_w_adversarys(
        args=args, 
        train_types=train_types,
        eval_types=eval_types,
        total_training_timesteps=args.pop_total_training_timesteps,
        curriculum=curriculum, 
        agent_path=agent_path,
        adv_paths=adv_paths)
    return agent_model, teammates, agent_tag
    
    

def ADV(args, 
        agent_folder_path, 
        agent_file_tag,
        adversary_type = LearnerType.SELFISHER,
        adversary_seed = 68,
        adversary_h_dim = 512,
        reward_magnifier = 3.0):
    train_types = [TeamType.HIGH_FIRST]
    eval_types = {
        'generate': [TeamType.HIGH_FIRST],
        'load': []
    }
    curriculum = Curriculum(train_types=train_types, is_random=True)
    agent_path = f"agent_models/{agent_folder_path}/{agent_file_tag}"
    args.learner_type = adversary_type
    args.ADV_seed = adversary_seed
    args.ADV_h_dim = adversary_h_dim
    args.reward_magnifier = reward_magnifier
    agent_model, teammates, agent_tag = get_adversary(  
                    args=args, 
                    train_types=train_types,
                    eval_types=eval_types,
                    total_training_timesteps=args.pop_total_training_timesteps,
                    curriculum=curriculum, 
                    agent_path=agent_path)
    return agent_model, teammates, agent_tag

def SP(args, pop_force_training):
    args.sp_train_types = [TeamType.SELF_PLAY]
    args.sp_eval_types = {
        'generate': [TeamType.SELF_PLAY],
        'load': []
    }
    curriculum = Curriculum(train_types=args.sp_train_types, is_random=True)
    agent_model, teammates = get_selfplay_agent_w_tms_collection(
        args=args,
        eval_types=args.sp_eval_types,
        total_training_timesteps=args.pop_total_training_timesteps,
        force_training=pop_force_training,
        curriculum=curriculum)
    return agent_model, teammates



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


def FCP_mhri(args, pop_force_training, fcp_force_training, parallel):
    '''
    There are two types of FCP, one is the traditional FCP that uses random teammates (i.e. ALL_MIX), 
    one is our own version that uses certain types HIGH_FIRST, MEDIUM_FIRST, etc. 
    The reason we have our version is that when we used the traditional FCP it got ~0 reward so we 
    decided to add different types for teammates_collection.
    '''
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



def FCP_traditional(args, pop_force_training, fcp_force_training, parallel):
    '''
    The ALL_MIX TeamType enables truly random teammates when training (like in the original FCP 
    implementation)
    '''

    args.fcp_train_types = [TeamType.ALL_MIX]
    args.fcp_eval_types = {'generate' : [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST],
                            'load': []}

    fcp_curriculum = Curriculum(train_types=args.fcp_train_types, is_random=True)

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


def set_input(args, quick_test=False, how_long=6):
    '''
    Suggested 3-Chefs Layouts are '3_chefs_small_kitchen_two_resources', 
    '3_chefs_counter_circuit', '3_chefs_asymmetric_advantages', 
    '3_chefs_forced_coordination_3OP2S1D'.
    '''
    args.layout_names = ['3_chefs_small_kitchen_two_resources', '3_chefs_counter_circuit', '3_chefs_asymmetric_advantages']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
    args.dynamic_reward = True
    args.final_sparse_r_ratio = 1.0
        
    if not quick_test: 
        args.learner_type = LearnerType.ORIGINALER
        args.n_envs = 200
        args.epoch_timesteps = 1e5
        args.pop_total_training_timesteps = 5e6 * how_long
        args.fcp_total_training_timesteps = 2 * 5e6 * how_long
        args.sp_w_sp_total_training_timesteps = 5e6 * how_long
        args.fcp_w_sp_total_training_timesteps = 4 * 5e6 * how_long        
        args.SP_seed, args.SP_h_dim = 68, 512
        args.SPWSP_seed, args.SPWSP_h_dim = 1010, 512
        args.FCP_seed, args.FCP_h_dim = 2020, 512
        args.FCPWSP_seed, args.FCPWSP_h_dim = 2602, 512
        args.ADV_seed, args.ADV_h_dim = 68, 512
        args.num_sp_agents_to_train = 3
        args.exp_dir = 'experiment/1' # This is the directory where the experiment will be saved. Change it to your desired directory
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
        args.exp_dir = 'test/1'


if __name__ == '__main__':
    args = get_arguments()
    quick_test = False
    parallel = True
    how_long = 5
    
    pop_force_training = True
    fcp_force_training = True
    fcp_w_sp_force_training = True
    sp_w_sp_force_training = True
    
    set_input(args=args, quick_test=quick_test)
    
    # SingleAdversaryPlay(args, 
    #                     exp_tag = 'S2FP', 
    #                     main_agent_path = None,
    #                     main_agent_seed = 68,
    #                     main_agent_h_dim = 512,
    #                     main_agent_type = LearnerType.SUPPORTER, 
    #                     adversary_seed = 68,
    #                     adversary_h_dim = 512,
    #                     adversary_type = LearnerType.SELFISHER, 
    #                     checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
    #                     how_long_init = 4.0,
    #                     how_long_for_agent = 1.0,
    #                     how_long_for_adv = 1.0,
    #                     rounds_of_advplay = 101,
    #                     reward_magnifier = 3.0)

    # MultiAdversaryPlay( args, 
    #                     exp_tag = 'M2FP', 
    #                     main_agent_path = None,
    #                     main_agent_seed = 68,
    #                     main_agent_h_dim = 512,
    #                     main_agent_type = LearnerType.SUPPORTER, 
    #                     adversary_seed = 68,
    #                     adversary_h_dim = 512,
    #                     adversary_type = LearnerType.SELFISHER, 
    #                     checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
    #                     how_long_init = 4.0,
    #                     how_long_for_agent = 2.0,
    #                     how_long_for_adv = 1.0,
    #                     rounds_of_advplay = 101,
    #                     reward_magnifier = 3.0)
    
    MultiAdversaryScheduledPlay(args, 
                                exp_tag = 'M2FSP', 
                                main_agent_path = None,
                                main_agent_seed = 68,
                                main_agent_h_dim = 512,
                                main_agent_type = LearnerType.SUPPORTER, 
                                adversary_seed = 68,
                                adversary_h_dim = 512,
                                adversary_type = LearnerType.SELFISHER, 
                                checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
                                how_long_init = 4.0,
                                how_long_for_agent = 2.0,
                                how_long_for_adv = 1.0,
                                rounds_of_advplay = 101,
                                reward_magnifier = 3.0)

    # SingleAdversaryPlay(args=args, 
    #                     exp_tag = 'S2FP', 
    #                     main_agent_path = None,
    #                     main_agent_type = LearnerType.SUPPORTER, 
    #                     adversary_type = LearnerType.SELFISHER, 
    #                     checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
    #                     how_long_init = 0.02,
    #                     how_long_for_agent = 0.02,
    #                     how_long_for_adv = 0.02,
    #                     rounds_of_advplay = 2,
    #                     reward_magnifier = 3.0)
    
    # MultiAdversaryPlay(args=args, 
    #                     exp_tag = 'M2FP', 
    #                     main_agent_path = None,
    #                     main_agent_type = LearnerType.SUPPORTER, 
    #                     adversary_type = LearnerType.SELFISHER, 
    #                     checked_adversary = CheckedPoints.FINAL_TRAINED_MODEL, 
    #                     how_long_init = 0.02,
    #                     how_long_for_agent = 0.02,
    #                     how_long_for_adv = 0.02,
    #                     rounds_of_advplay = 2,
    #                     reward_magnifier = 3.0)

    # SP(args=args,
    #    pop_force_training=pop_force_training)

    # FCP_traditional(args=args,
    #               pop_force_training=pop_force_training,
    #               fcp_force_training=fcp_force_training,
    #               parallel=parallel)

    # FCP_mhri(args=args,
    #       pop_force_training=pop_force_training,
    #       fcp_force_training=fcp_force_training,
    #       parallel=parallel)

    # SP_w_SP_Types(args=args,
    #               pop_force_training=pop_force_training,
    #               sp_w_sp_force_training=sp_w_sp_force_training,
    #               parallel=parallel)


    # FCP_w_SP_TYPES(args=args,
    #                pop_force_training=pop_force_training,
    #                fcp_force_training=fcp_force_training,
    #                fcp_w_sp_force_training=fcp_w_sp_force_training,
    #                parallel=parallel)