from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.agents.base_agent import OAIAgent
from oai_agents.common.tags import TeamType

from .common import load_agents
from .fcp_pop_helper import get_fcp_population
from .tc_helper import generate_TC_for_FCP_w_SP_types, generate_TC_for_SP
from .curriculum import Curriculum


def get_selfplay_agent_w_tms_collection(args, total_training_timesteps, train_types, eval_types, curriculum, tag=None, force_training=False):
    name = 'sp'
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
    tc = generate_TC_for_SP(args=args,
                            train_types=train_types,
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
        seed=678,
    )

    selfplay_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return selfplay_trainer.get_agents()[0], tc


def get_selfplay_agent_trained_w_selfplay_types(args,
                                                pop_total_training_timesteps:int,
                                                sp_w_sp_total_training_timesteps:int,
                                                sp_w_sp_train_types:list,
                                                sp_w_sp_eval_types:list,
                                                curriculum:Curriculum,
                                                tag:str=None,
                                                pop_force_training:bool=True,
                                                sp_w_sp_force_training:bool=True,
                                                parallel:bool=True) -> tuple:
    '''
    Train a SP agent using SP train types. This function will first train a SP agent and then let that
    agent train with itself and one other unseen teammate (e.g. [SP, SP, SP, SP_H] in a 4-chef layout)
    
    :param args: Parsed arguments list
    :param pop_total_training_timesteps: Total number of timesteps to train the initial population of agents
    :param sp_w_sp_train_types: List of TeamTypes to be used for training SP agents against
    :param sp_w_sp_eval_types: List of TeamTypes to be used for evaluating SP agents against
    :param tag: File name to use when loading agent files
    :param pop_force_training: Boolean that (when true) indicates the SP agent population should be trained instead of loaded from file
    :param sp_w_sp_force_training: Boolean that (when true) indicates the SP agent teammates_collection should be trained instead of loaded from file
    :returns: Trained self-play agent and the teammates collection used to generate it
    '''


    # Generate a teammates collection (the same kind used for FCP training) by training some SP agents,
    # saving them periodically (to represent various skill levels), and then oragnizing them into teams of
    # different TeamTypes for training and evaluation, use all TeamTypes so we can generate whatever teammates_collection 
    # needed for the final SP training
    population_of_all_train_types = get_fcp_population(args=args,
                                            ck_rate = pop_total_training_timesteps // 5,
                                            total_training_timesteps=pop_total_training_timesteps,
                                            train_types=TeamType.ALL_TYPES_BESIDES_SP,
                                            eval_types_to_generate=[TeamType.HIGH_FIRST, TeamType.LOW_FIRST],
                                            eval_types_to_load_from_file=[],
                                            num_self_play_agents_to_train=args.teammates_len * len(TeamType.ALL_TYPES_BESIDES_SP),
                                            force_training=pop_force_training,
                                            parallel=parallel)

    name = 'sp_w_selfplay_types'
    
    agents = load_agents(args, name=name, tag=tag, force_training=sp_w_sp_force_training)
    if agents:

        # If agents were loaded, we already trained them and don't need to continue to the training step
        return agents[0], population_of_all_train_types

    # Generate a randomly initialized SP agent
    randomly_init_sp_agent = RLAgentTrainer.generate_randomly_initialized_SP_agent(args=args)

    teammates_collection_for_sp_w_sp_types_training = generate_TC_for_FCP_w_SP_types(args=args,
                                                                                     teammates_collection=population_of_all_train_types,
                                                                                     agent=randomly_init_sp_agent,
                                                                                     train_types=sp_w_sp_train_types,
                                                                                     eval_types=sp_w_sp_eval_types)

    sp_w_sp_types_trainer = RLAgentTrainer(name=name,
                                           args=args,
                                           agent=randomly_init_sp_agent,
                                           teammates_collection=teammates_collection_for_sp_w_sp_types_training,
                                           epoch_timesteps=args.epoch_timesteps,
                                           n_envs=args.n_envs,
                                           curriculum=curriculum,
                                           seed=1010)

    sp_w_sp_types_trainer.train_agents(total_train_timesteps=sp_w_sp_total_training_timesteps)

    return sp_w_sp_types_trainer.get_agents()[0], teammates_collection_for_sp_w_sp_types_training


def get_fcp_agent_w_tms_clction(args, 
                                pop_total_training_timesteps,
                                fcp_total_training_timesteps,
                                fcp_train_types,
                                fcp_eval_types,
                                fcp_curriculum,
                                pop_force_training, fcp_force_training,
                                num_self_play_agents_to_train=2, tag=None, parallel=True):
    teammates_collection = get_fcp_population(args,
                                              ck_rate = pop_total_training_timesteps // 5,
                                              train_types = fcp_train_types,
                                              eval_types_to_generate = fcp_eval_types['generate'],
                                              eval_types_to_load_from_file = fcp_eval_types['load'],
                                              num_self_play_agents_to_train= num_self_play_agents_to_train,
                                              total_training_timesteps = pop_total_training_timesteps,
                                              force_training=pop_force_training,
                                              parallel=parallel)
    name = 'fcp'
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
        seed=2602,
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
                                    fcp_train_types, 
                                    fcp_eval_types,
                                    fcp_w_sp_train_types,
                                    fcp_w_sp_eval_types,
                                    fcp_curriculum,
                                    fcp_w_sp_curriculum,
                                    num_self_play_agents_to_train=2,
                                    parallel=True,
                                    tag=None):

    fcp_agent, fcp_teammates_collection = get_fcp_agent_w_tms_clction(args, 
                                                                  pop_total_training_timesteps=pop_total_training_timesteps,
                                                                  fcp_total_training_timesteps=fcp_total_training_timesteps,
                                                                  fcp_train_types=fcp_train_types,
                                                                  fcp_eval_types=fcp_eval_types,
                                                                  pop_force_training=pop_force_training,
                                                                  fcp_force_training=fcp_force_training,
                                                                  num_self_play_agents_to_train=num_self_play_agents_to_train,
                                                                  fcp_curriculum=fcp_curriculum,
                                                                  parallel=parallel)

    teammates_collection = generate_TC_for_FCP_w_SP_types(args=args,
                                                        teammates_collection=fcp_teammates_collection,
                                                        agent=fcp_agent,
                                                        train_types=fcp_w_sp_train_types,
                                                        eval_types=fcp_w_sp_eval_types,
                                                        )

    name = 'fcp_w_selfplay_types'

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
        seed=2602,
        curriculum=fcp_w_sp_curriculum,
    )

    fcp_trainer.train_agents(total_train_timesteps=fcp_w_sp_total_training_timesteps)
    return fcp_trainer.get_agents()[0], teammates_collection


