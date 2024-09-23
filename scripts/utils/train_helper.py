from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.agents.base_agent import OAIAgent
from oai_agents.common.tags import TeamType

from .common import load_agents, generate_name
from .pop_helper import get_fcp_population, get_population
from .tc_helper import generate_TC_for_FCP_w_SP_types, generate_TC_for_SP, generate_TC_for_N_X_SP
from .curriculum import Curriculum


def get_selfplay_agent_w_tms_collection(args, total_training_timesteps, eval_types, curriculum, tag=None, force_training=False):
    name = generate_name(args, 
                         prefix='sp',
                         seed=args.SP_seed,
                         h_dim=args.SP_h_dim, 
                         train_types=curriculum.train_types,
                         has_curriculum= not curriculum.is_random)
    
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
    tc = generate_TC_for_SP(args=args,
                            train_types=curriculum.train_types,
                            eval_types_to_generate=eval_types['generate'],
                            eval_types_to_read_from_file=eval_types['load'])

    if agents:
        return agents[0], tc

    selfplay_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection=tc,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        curriculum=curriculum,
        seed=args.SP_seed,
        hidden_dim=args.SP_h_dim,
    )

    selfplay_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return selfplay_trainer.get_agents()[0], tc


def get_N_X_selfplay_agents_trained_w_selfplay_types(args,
                                                     pop_total_training_timesteps:int,
                                                     pop_force_training:bool,

                                                     n_x_sp_train_types:list,
                                                     n_x_sp_eval_types:list,
                                                     n_x_sp_force_training:list,
                                                     n_x_sp_total_training_timesteps:int,

                                                     curriculum:Curriculum,
                                                     
                                                     tag:str=None,
                                                     parallel:bool=True,
                                                     num_SPs_to_train=2) -> tuple:
    

    
    curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                                         unallowed_types = TeamType.ALL_TYPES_BESIDES_SP)

    name = generate_name(args,
                         prefix = 'n-x-sp',
                         seed = args.N_X_SP_seed,
                         h_dim = args.N_X_SP_h_dim, 
                         train_types = n_x_sp_train_types,
                         has_curriculum = not curriculum.is_random)
    
    agents = load_agents(args, name=name, tag=tag, force_training=n_x_sp_force_training)
    if agents:
        return agents[0]
    
    population = get_population(
        args=args,
        ck_rate=pop_total_training_timesteps // 10,
        total_training_timesteps=pop_total_training_timesteps,

        train_types=n_x_sp_train_types,
        eval_types=n_x_sp_eval_types['generate'],

        unseen_teammates_len = args.unseen_teammates_len,
        num_SPs_to_train=num_SPs_to_train,
        
        parallel=parallel,
        force_training=pop_force_training,
        tag = tag
    )

    randomly_init_agent = RLAgentTrainer.generate_randomly_initialized_agent(args=args, seed=args.N_X_SP_seed)
    teammates_collection = generate_TC_for_N_X_SP(args=args,
                                                  population=population,
                                                  agent=randomly_init_agent,
                                                  train_types=n_x_sp_train_types,
                                                  eval_types_to_generate=n_x_sp_eval_types['generate'],
                                                  eval_types_to_read_from_file=n_x_sp_eval_types['load'],
                                                  unseen_teammates_len=args.unseen_teammates_len
                                                  )

    n_x_sp_types_trainer = RLAgentTrainer(name=name,
                                           args=args,
                                           agent=randomly_init_agent,
                                           teammates_collection=teammates_collection,
                                           epoch_timesteps=args.epoch_timesteps,
                                           n_envs=args.n_envs,
                                           curriculum=curriculum,
                                           seed=args.N_X_SP_seed,
                                           hidden_dim=args.N_X_SP_h_dim)

    n_x_sp_types_trainer.train_agents(total_train_timesteps=n_x_sp_total_training_timesteps)
    return n_x_sp_types_trainer.get_agents()[0], teammates_collection


def get_selfplay_agent_trained_w_selfplay_types(args,
                                                pop_total_training_timesteps:int,
                                                n_x_sp_total_training_timesteps:int,
                                                sp_w_sp_eval_types:list,
                                                curriculum:Curriculum,
                                                pop_train_types:list=[TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST],
                                                pop_eval_types:list=[TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST],
                                                tag:str=None,
                                                pop_force_training:bool=True,
                                                sp_w_sp_force_training:bool=True,
                                                num_SPs_to_train=2,
                                                parallel:bool=True) -> tuple:
    '''
    Train a SP agent using SP train types. This function will first train a SP agent and then let that
    agent train with itself and one other unseen teammate (e.g. [SP, SP, SP, SP_H] in a 4-chef layout)
    
    :param args: Parsed arguments list
    :param pop_total_training_timesteps: Total number of timesteps to train the initial population of agents
    :param sp_w_sp_eval_types: List of TeamTypes to be used for evaluating SP agents against
    :param tag: File name to use when loading agent files
    :param pop_force_training: Boolean that (when true) indicates the SP agent population should be trained instead of loaded from file
    :param sp_w_sp_force_training: Boolean that (when true) indicates the SP agent teammates_collection should be trained instead of loaded from file
    :returns: Trained self-play agent and the teammates collection used to generate it
    '''

    # To use SP-types, the curriculum needs to contain only SP types
    curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                                         unallowed_types= TeamType.ALL_TYPES_BESIDES_SP)

    # Generate a teammates collection (the same kind used for FCP training) by training some SP agents,
    # saving them periodically (to represent various skill levels), and then oragnizing them into teams of
    # different TeamTypes for training and evaluation, use all TeamTypes so we can generate whatever teammates_collection 
    # needed for the final SP training
    population_of_all_train_types = get_fcp_population(args=args,
                                            ck_rate = pop_total_training_timesteps // 5,
                                            total_training_timesteps=pop_total_training_timesteps,
                                            train_types=pop_train_types,
                                            eval_types_to_generate=pop_eval_types,
                                            eval_types_to_load_from_file=[],
                                            num_SPs_to_train=num_SPs_to_train,
                                            force_training=pop_force_training,
                                            parallel=parallel)

    
    name = generate_name(args, 
                         prefix='spWsp',
                         seed=args.N_X_SP,
                         h_dim=args.N_X_SP, 
                         train_types=curriculum.train_types,
                         has_curriculum = not curriculum.is_random)

    
    agents = load_agents(args, name=name, tag=tag, force_training=sp_w_sp_force_training)
    if agents:
        return agents[0], population_of_all_train_types

    # Generate a randomly initialized SP agent
    randomly_init_sp_agent = RLAgentTrainer.generate_randomly_initialized_agent(args=args)

    teammates_collection_for_sp_w_sp_types_training = generate_TC_for_FCP_w_SP_types(args=args,
                                                                                     teammates_collection=population_of_all_train_types,
                                                                                     agent=randomly_init_sp_agent,
                                                                                     train_types=curriculum.train_types,
                                                                                     eval_types=sp_w_sp_eval_types['generate'])

    sp_w_sp_types_trainer = RLAgentTrainer(name=name,
                                           args=args,
                                           agent=randomly_init_sp_agent,
                                           teammates_collection=teammates_collection_for_sp_w_sp_types_training,
                                           epoch_timesteps=args.epoch_timesteps,
                                           n_envs=args.n_envs,
                                           curriculum=curriculum,
                                           seed=args.N_X_SP,
                                           hidden_dim=args.N_X_SP)

    sp_w_sp_types_trainer.train_agents(total_train_timesteps=n_x_sp_total_training_timesteps)

    return sp_w_sp_types_trainer.get_agents()[0], teammates_collection_for_sp_w_sp_types_training


def get_fcp_agent_w_tms_clction(args, 
                                pop_total_training_timesteps,
                                fcp_total_training_timesteps,
                                fcp_eval_types,
                                fcp_curriculum,
                                pop_force_training, fcp_force_training,
                                num_SPs_to_train=2, tag=None, parallel=True):
    teammates_collection = get_fcp_population(args,
                                              ck_rate = pop_total_training_timesteps // 5,
                                              train_types = fcp_curriculum.train_types,
                                              eval_types_to_generate = fcp_eval_types['generate'],
                                              eval_types_to_load_from_file = fcp_eval_types['load'],
                                              num_SPs_to_train= num_SPs_to_train,
                                              total_training_timesteps = pop_total_training_timesteps,
                                              force_training=pop_force_training,
                                              parallel=parallel)
    name = generate_name(args, 
                         prefix='fcp',
                         seed=args.FCP_seed,
                         h_dim=args.FCP_h_dim, 
                         train_types=fcp_curriculum.train_types,
                         has_curriculum = not fcp_curriculum.is_random)
    
    agents = load_agents(args, name=name, tag=tag, force_training=fcp_force_training)
    if agents:
        return agents[0], teammates_collection

    fcp_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection=teammates_collection,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=args.FCP_seed,
        hidden_dim=args.FCP_h_dim,
        curriculum=fcp_curriculum,
    )

    fcp_trainer.train_agents(total_train_timesteps=fcp_total_training_timesteps)
    return fcp_trainer.get_agents()[0], teammates_collection



def get_fcp_trained_w_selfplay_types(args,
                                    pop_total_training_timesteps,
                                    fcp_total_training_timesteps,
                                    fcp_w_sp_total_training_timesteps,
                                    pop_force_training,
                                    fcp_force_training,
                                    fcp_w_sp_force_training,
                                    fcp_eval_types,
                                    fcp_w_sp_eval_types,
                                    fcp_curriculum,
                                    fcp_w_sp_curriculum,
                                    num_SPs_to_train=2,
                                    parallel=True,
                                    tag=None):

    # To use SP-types, the curriculum needs to contain only SP types
    fcp_w_sp_curriculum.validate_curriculum_types(expected_types = [TeamType.SELF_PLAY_HIGH, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_LOW],
                                                  unallowed_types= TeamType.ALL_TYPES_BESIDES_SP)

    fcp_agent, fcp_teammates_collection = get_fcp_agent_w_tms_clction(args, 
                                                                  pop_total_training_timesteps=pop_total_training_timesteps,
                                                                  fcp_total_training_timesteps=fcp_total_training_timesteps,
                                                                  fcp_train_types=fcp_curriculum.train_types,
                                                                  fcp_eval_types=fcp_eval_types,
                                                                  pop_force_training=pop_force_training,
                                                                  fcp_force_training=fcp_force_training,
                                                                  num_SPs_to_train=num_SPs_to_train,
                                                                  fcp_curriculum=fcp_curriculum,
                                                                  parallel=parallel)

    teammates_collection = generate_TC_for_FCP_w_SP_types(args=args,
                                                        teammates_collection=fcp_teammates_collection,
                                                        agent=fcp_agent,
                                                        train_types=fcp_w_sp_curriculum.train_types,
                                                        eval_types=fcp_w_sp_eval_types['generate'],
                                                        )
    
    name = generate_name(args, 
                         prefix='fcpWsp',
                         seed=args.FCPWSP_seed,
                         h_dim=args.FCPWSP_h_dim, 
                         train_types=fcp_w_sp_curriculum.train_types,
                         has_curriculum = not fcp_curriculum.is_random)

    agents = load_agents(args, name=name, tag=tag, force_training=fcp_w_sp_force_training)
    if agents:

        # If agents were loaded, we already trained them and don't need to continue to the training step
        return agents[0], teammates_collection

    fcp_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=fcp_agent,
        teammates_collection=teammates_collection,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=args.FCPWSP_seed,
        hidden_dim=args.FCPWSP_h_dim,
        curriculum=fcp_w_sp_curriculum,
    )

    fcp_trainer.train_agents(total_train_timesteps=fcp_w_sp_total_training_timesteps)
    return fcp_trainer.get_agents()[0], teammates_collection


