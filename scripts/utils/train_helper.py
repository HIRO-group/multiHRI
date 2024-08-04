from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.agents.base_agent import OAIAgent

from .common import load_agents
from .fcp_pop_helper import get_fcp_population
from .tc_helper import generate_TC_for_FCP_w_SP_types, generate_TC_for_SP


def get_selfplay_agent_w_tms_collection(args, total_training_timesteps, train_types, eval_types, tag=None, force_training=False):
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
        seed=678,
    )

    selfplay_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return selfplay_trainer.get_agents()[0], tc


def generate_randomly_initialized_SP_agent(args,
                                           teammates_collection:dict) -> OAIAgent:
    '''
    Generate a randomly initiralized learning agent using the RLAgentTrainer class
    This function does not perform any learning

    :param args: Parsed args object
    :param teammates_collection: 
    :returns: An untrained, randomly inititalized RL agent
    '''

    name = 'randomized_agent'

    # TODO: Is teammates_collection necessary here? or can it be initialized with None since no training
    # is performed
    sp_trainer = RLAgentTrainer(name=name,
                                args=args,
                                agent=None,
                                teammates_collection=teammates_collection,
                                epoch_timesteps=args.epoch_timesteps,
                                n_envs=args.n_envs,
                                seed=8080)

    # Don't need to do any training, just return the agent
    return sp_trainer.get_agents()[0]



def get_selfplay_agent_trained_w_selfplay_types(args,
                                                pop_total_training_timesteps:int,
                                                sp_train_types:list,
                                                sp_eval_types:list,
                                                num_self_play_agents_to_train:int,
                                                sp_w_sp_total_training_timesteps:int,
                                                sp_w_sp_train_types:list,
                                                sp_w_sp_eval_types:list,
                                                tag:str=None,
                                                force_training:bool=None,
                                                parallel:bool=True) -> tuple:
    '''
    Train a SP agent using SP train types. This function will first train a SP agent and then let that
    agent train with itself and one other unseen teammate (e.g. [SP, SP, SP, SP_H] in a 4-chef layout)
    
    TODO: add parameter descriptions
    '''


    # Generate a teammates collection (the same kind used for FCP training) by training some SP agents,
    # saving them periodically (to represent various skill levels), and then oragnizing them into teams of
    # different TeamTypes for training and evaluation
    teammates_collection = get_fcp_population(args=args,
                                            ck_rate = pop_total_training_timesteps // 5,
                                            total_training_timesteps=pop_total_training_timesteps,
                                            train_types=sp_train_types,
                                            eval_types_to_generate=sp_eval_types['generate'],
                                            eval_types_to_load_from_file=sp_eval_types['load'],
                                            num_self_play_agents_to_train=num_self_play_agents_to_train,
                                            force_training=force_training,
                                            parallel=parallel)

    name = 'sp_w_selfplay_types'
    
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
    if agents:

        # If agents were loaded, we already trained them and don't need to continue to the training step
        return agents[0], teammates_collection

    # Generate a randomly initialized SP agent
    sp_agent = generate_randomly_initialized_SP_agent(args=args, teammates_collection=teammates_collection)

    # Generate a new teammates collection using a randomly inititalized SP agent and a teammates collection
    teammates_collection_for_sp_w_sp_types_training = generate_TC_for_FCP_w_SP_types(args=args,
                                                                                     teammates_collection=teammates_collection,
                                                                                     agent=sp_agent,
                                                                                     train_types=sp_w_sp_train_types,
                                                                                     eval_types=sp_w_sp_eval_types)

    # Create a new SP trainer from the previously trained SP and the newly generated teammates collection
    sp_w_sp_types_trainer = RLAgentTrainer(name=name,
                                           args=args,
                                           agent=sp_agent,
                                           teammates_collection=teammates_collection_for_sp_w_sp_types_training,
                                           epoch_timesteps=args.epoch_timesteps,
                                           n_envs=args.n_envs,
                                           seed=1010)

    # Train the new SP agent
    sp_w_sp_types_trainer.train_agents(total_train_timesteps=sp_w_sp_total_training_timesteps)

    return sp_w_sp_types_trainer.get_agents()[0], teammates_collection_for_sp_w_sp_types_training


def get_fcp_agent_w_tms_clction(args, 
                                pop_total_training_timesteps,
                                fcp_total_training_timesteps,
                                fcp_train_types,
                                fcp_eval_types,
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
    )

    fcp_trainer.train_agents(total_train_timesteps=fcp_w_sp_total_training_timesteps)
    return fcp_trainer.get_agents()[0], teammates_collection


