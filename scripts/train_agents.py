import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, CheckedPoints
from oai_agents.common.learner import LearnerType, Learner
from utils import (get_selfplay_agent_w_tms_collection, 
                get_fcp_agent_w_tms_clction, 
                get_eval_types_to_load, 
                get_fcp_trained_w_selfplay_types, 
                get_selfplay_agent_trained_w_selfplay_types,
                get_saboteur,
                get_agent_play_w_saboteur,
                get_agent_play_w_saboteurs,
                Curriculum
                )

def SingleSelfisherPlay(args, pop_force_training, where_to_save='S2FP-1010-worst-asy-sk-cc', checked_selfisher=CheckedPoints.FINAL_TRAINED_MODEL):
    # TODO: 
    # 3. allow user to choose the type of primary agents
    # 4. allow user to choose the type of saboteurs 
    root = where_to_save
    how_long_init = 4
    how_long_for_agent = 1.0
    how_long_for_sab = 1.0

    # how_long = 4
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = root + '/supporter/0'
    # SP(args=args, pop_force_training=pop_force_training)

    how_long = how_long_for_sab
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SELFISHER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/selfisher/0'
    SAB(args=args, 
        agent_folder_path='agent_models/' + root + '/supporter/0', 
        agent_file_tag='sp_s68_h512_tr(SP)_ran/' + CheckedPoints.BEST_EVAL_REWARD)

    how_long = how_long_init + how_long_for_agent
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SUPPORTER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/supporter-fishplay/0'
    PwSABs(args=args, 
          agent_folder_path='agent_models/' + root + '/supporter/0', 
          agent_file_tag='sp_s68_h512_tr(SP)_ran/' + CheckedPoints.BEST_EVAL_REWARD,
          sab_folder_paths=['agent_models/' + root + '/selfisher/0'], 
          sab_file_tag='sab_s68_h512_tr(H)_ran/' + checked_selfisher
          )
    ###################################################################
    for round in range(1,401):
        how_long = how_long_for_sab
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.learner_type = LearnerType.SELFISHER
        args.reward_magnifier = 3.0
        args.exp_dir = root + '/selfisher/'+ str(round)
        SAB(args=args,
            agent_folder_path='agent_models/' + root + '/supporter-fishplay/'+ str(round-1), 
            agent_file_tag='pwsab_s68_h512_tr(SP_SPSAB)_ran/' + CheckedPoints.FINAL_TRAINED_MODEL)
        
        how_long = how_long_init + how_long_for_agent*(round+1)
        set_input(args=args, quick_test=quick_test, how_long=how_long)
        args.learner_type = LearnerType.SUPPORTER
        args.reward_magnifier = 3.0
        args.exp_dir = root + '/supporter-fishplay/'+ str(round)
        PwSABs( args=args, 
                agent_folder_path='agent_models/' + root + '/supporter-fishplay/'+ str(round-1), 
                agent_file_tag='pwsab_s68_h512_tr(SP_SPSAB)_ran/' + CheckedPoints.FINAL_TRAINED_MODEL,
                sab_folder_paths=['agent_models/' + root + '/selfisher/'+ str(round)], 
                sab_file_tag='sab_s68_h512_tr(H)_ran/' + checked_selfisher)

def DiverseSelfisherPlay(args, pop_force_training, file_folder='asy-sk-cc-18-best'):
    # TODO: 
    # 1. allow user to choose the last or the best agent for training saboteur
    # 2. allow user to choose the last or the best or the most adversarial saboteurs for training supporter
    # 3. allow user to choose the type of primary agents
    # 4. allow user to choose the type of saboteurs 
    root = file_folder

    # how_long = 4
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = root + '/supporter/0'
    # SP(args=args, pop_force_training=pop_force_training)

    how_long = 1
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SELFISHER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/selfisher/0'
    SAB(args=args, 
        agent_folder_path='agent_models/' + root + '/supporter/0', 
        agent_file_tag='sp_s68_h512_tr(SP)_ran/best')

    how_long = 8
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SUPPORTER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/supporter-fishplay/0'
    PwSABs(args=args, 
          agent_folder_path='agent_models/' + root + '/supporter/0', 
          agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
          sab_folder_paths=['agent_models/' + root + '/selfisher/0'], 
          sab_file_tag='sab_s68_h512_tr(H)_ran/best'
          )
    ###################################################################
    how_long = 1
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SELFISHER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/selfisher/1'
    SAB(args=args,
        agent_folder_path='agent_models/' + root + '/supporter-fishplay/0', 
        agent_file_tag='pwsab_s68_h512_tr(SP_SPSAB)_ran/best')
    
    how_long = 8
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SUPPORTER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/supporter-fishplay/1'
    PwSABs(args=args, 
          agent_folder_path='agent_models/' + root + '/supporter/0', 
          agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
          sab_folder_paths=['agent_models/' + root + '/selfisher/1', 'agent_models/' + root + '/selfisher/0'], 
          sab_file_tag='sab_s68_h512_tr(H)_ran/best'
          )
    
    ####################################################################      
    how_long = 1
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SELFISHER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/selfisher/2'
    SAB(args=args,
        agent_folder_path='agent_models/' + root + '/supporter-fishplay/1', 
        agent_file_tag='pwsab_s68_h512_tr(SP_SPSAB)_ran/best')
    
    how_long = 8
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SUPPORTER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/supporter-fishplay/2'
    PwSABs(args=args, 
          agent_folder_path='agent_models/' + root + '/supporter/0', 
          agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
          sab_folder_paths=['agent_models/' + root + '/selfisher/2', 'agent_models/' + root + '/selfisher/1', 'agent_models/' + root + '/selfisher/0'], 
          sab_file_tag='sab_s68_h512_tr(H)_ran/best'
          )
    
    ####################################################################
    how_long = 12
    set_input(args=args, quick_test=quick_test, how_long=how_long)
    args.learner_type = LearnerType.SUPPORTER
    args.reward_magnifier = 3.0
    args.exp_dir = root + '/supporter-fishplay/2_extend'
    PwSABs(args=args, 
          agent_folder_path='agent_models/' + root + '/supporter-fishplay/2', 
          agent_file_tag='pwsab_s68_h512_tr(SP_SPSAB)_ran/best',
          sab_folder_paths=['agent_models/' + root + '/selfisher/2', 'agent_models/' + root + '/selfisher/1', 'agent_models/' + root + '/selfisher/0'], 
          sab_file_tag='sab_s68_h512_tr(H)_ran/best'
          )

def PwSABs(args, 
          agent_folder_path='agent_models/four-layouts/supporter/0', 
          agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
          sab_folder_paths=['agent_models/four-layouts/saboteur/0', 'agent_models/four-layouts/saboteur/1'], 
          sab_file_tag='sab_s68_h512_tr(H)_ran/best'
          ):
    args.sp_train_types = [TeamType.SELF_PLAY, TeamType.SELF_PLAY_SABOTEUR]
    args.sp_eval_types = {
        'generate': [TeamType.SELF_PLAY, TeamType.SELF_PLAY_SABOTEUR],
        'load': []
    }
    curriculum = Curriculum(train_types=args.sp_train_types, is_random=True)
    agent_path = agent_folder_path + '/' + agent_file_tag
    sab_paths = [sab_folder_path + '/' + sab_file_tag for sab_folder_path in sab_folder_paths]
    get_agent_play_w_saboteurs(
        args=args, 
        train_types=args.sp_train_types,
        eval_types=args.sp_eval_types,
        total_training_timesteps=args.pop_total_training_timesteps,
        curriculum=curriculum, 
        agent_path=agent_path,
        sab_paths=sab_paths)


def PwSAB(args, 
          agent_folder_path='agent_models/four-layouts/supporter/0', 
          agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
          sab_folder_path='agent_models/four-layouts/saboteur/0', 
          sab_file_tag='sab_s68_h512_tr(H)_ran/best'
          ):
    args.sp_train_types = [TeamType.SELF_PLAY, TeamType.SELF_PLAY_HIGH]
    args.sp_eval_types = {
        'generate': [TeamType.SELF_PLAY, TeamType.SELF_PLAY_HIGH],
        'load': []
    }
    curriculum = Curriculum(train_types=args.sp_train_types, is_random=True)
    agent_path = agent_folder_path + '/' + agent_file_tag
    sab_path = sab_folder_path + '/' + sab_file_tag
    get_agent_play_w_saboteur(
        args=args, 
        train_types=args.sp_train_types,
        eval_types=args.sp_eval_types,
        total_training_timesteps=args.pop_total_training_timesteps,
        curriculum=curriculum, 
        agent_path=agent_path,
        sab_path=sab_path)

def SAB(args, agent_folder_path='agent_models/four-layouts/supporter/0', agent_file_tag='sp_s68_h512_tr(SP)_ran/best'):
    args.sp_train_types = [TeamType.HIGH_FIRST]
    args.sp_eval_types = {
        'generate': [TeamType.HIGH_FIRST],
        'load': []
    }
    curriculum = Curriculum(train_types=args.sp_train_types, is_random=True)
    agent_path = agent_folder_path + '/' + agent_file_tag
    get_saboteur(   args=args, 
                    train_types=args.sp_train_types,
                    eval_types=args.sp_eval_types,
                    total_training_timesteps=args.pop_total_training_timesteps,
                    curriculum=curriculum, 
                    agent_path=agent_path)

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


def set_input(args, quick_test=False, how_long=6):
    '''
    Suggested 3-Chefs Layouts are '3_chefs_small_kitchen_two_resources', 
    '3_chefs_counter_circuit', '3_chefs_asymmetric_advantages', 
    '3_chefs_forced_coordination_3OP2S1D'.
    '''
    args.layout_names = ['3_chefs_small_kitchen_two_resources', '3_chefs_counter_circuit', '3_chefs_asymmetric_advantages']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
        
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
        args.num_sp_agents_to_train = 3
        args.exp_dir = 'experiment-1'
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
        args.exp_dir = 'test-1'


if __name__ == '__main__':
    args = get_arguments()
    quick_test = False
    parallel = True
    how_long = 5
    
    pop_force_training = True
    fcp_force_training = True
    fcp_w_sp_force_training = True
    sp_w_sp_force_training = True
    
    args.SP_seed, args.SP_h_dim = 68, 512
    # DiverseSelfisherPlay(args=args, pop_force_training=pop_force_training)
    SingleSelfisherPlay(args=args, pop_force_training=pop_force_training)

    # how_long = 5
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-fishplay/0'
    # PwSAB(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter/0', 
    #       agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
    #       sab_folder_path='agent_models/four-layouts/selfisher/0', 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )
    
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-sabplay/0'
    # PwSAB(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter/0', 
    #       agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
    #       sab_folder_path='agent_models/four-layouts/saboteur/0', 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )
    
    # how_long = 7
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-fishplay/1'
    # PwSAB(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter-fishplay/0', 
    #       agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best',
    #       sab_folder_path='agent_models/four-layouts/selfisher/0', 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )

    # how_long = 4
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SELFISHER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/selfisher/2'
    # SAB(args=args, 
    #     agent_folder_path='agent_models/four-layouts/supporter-fishplay/1', 
    #     agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best')
    
    # how_long = 4
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SABOTEUR
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/saboteur/1'
    # SAB(args=args, 
    #     agent_folder_path='agent_models/four-layouts/supporter-sabplay/0', 
    #     agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best')

    # how_long = 8
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-fishplay/2'
    # PwSABs(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter/0', 
    #       agent_file_tag='pwsab_s68_h512_tr(SP)_ran/best',
    #       sab_folder_paths=['agent_models/four-layouts/selfisher/0', 'agent_models/four-layouts/selfisher/2'], 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )
    
    # how_long = 10
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-fishplay/3'
    # PwSABs(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter-fishplay/2', 
    #       agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best',
    #       sab_folder_paths=['agent_models/four-layouts/selfisher/0', 'agent_models/four-layouts/selfisher/2'], 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )

    # how_long = 4
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SELFISHER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/selfisher/4'
    # SAB(args=args, 
    #     agent_folder_path='agent_models/four-layouts/supporter-fishplay/3', 
    #     agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best')
    
    # how_long = 14
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-fishplay/4'
    # PwSABs(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter-fishplay/3', 
    #       agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best',
    #       sab_folder_paths=['agent_models/four-layouts/selfisher/4', 'agent_models/four-layouts/selfisher/2', 'agent_models/four-layouts/selfisher/0'], 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )

    # how_long = 12
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-fishplay/024'
    # PwSABs(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter/0', 
    #       agent_file_tag='sp_s68_h512_tr(SP)_ran/best',
    #       sab_folder_paths=['agent_models/four-layouts/selfisher/4', 'agent_models/four-layouts/selfisher/2', 'agent_models/four-layouts/selfisher/0'], 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )
    
    # how_long = 14
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-fishplay/024_14'
    # PwSABs(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter-fishplay/024', 
    #       agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best',
    #       sab_folder_paths=['agent_models/four-layouts/selfisher/4', 'agent_models/four-layouts/selfisher/2', 'agent_models/four-layouts/selfisher/0'], 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )

    # how_long = 12
    # set_input(args=args, quick_test=quick_test, how_long=how_long)
    # args.learner_type = LearnerType.SUPPORTER
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'four-layouts/supporter-sabplay/1'
    # PwSABs(args, 
    #       agent_folder_path='agent_models/four-layouts/supporter-sabplay/0', 
    #       agent_file_tag='pwsab_s68_h512_tr(SP_SPH)_ran/best',
    #       sab_folder_paths=['agent_models/four-layouts/saboteur/0', 'agent_models/four-layouts/saboteur/1'], 
    #       sab_file_tag='sab_s68_h512_tr(H)_ran/best'
    #       )

    # args.learner_type = LearnerType.SABOTEUR
    # args.reward_magnifier = 3.0
    # args.exp_dir = 'small_kitchen_2/saboteur/0'
    # SAB(args=args, 
    #     folder_path='agent_models/four-layouts/supporter/0', 
    #     tag='sp_s68_h512_tr(SP)_ran/best')

    # SP(args=args,
    #    pop_force_training=pop_force_training)

    # FCP(args=args,
    #     pop_force_training=pop_force_training,
    #     fcp_force_training=fcp_force_training,
    #     parallel=parallel)

    # SP_w_SP_Types(args=args,
    #               pop_force_training=pop_force_training,
    #               sp_w_sp_force_training=sp_w_sp_force_training,
    #               parallel=parallel)


    # FCP_w_SP_TYPES(args=args,
    #                pop_force_training=pop_force_training,
    #                fcp_force_training=fcp_force_training,
    #                fcp_w_sp_force_training=fcp_w_sp_force_training,
    #                parallel=parallel)