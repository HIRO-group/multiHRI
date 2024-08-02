from oai_agents.agents.rl import RLAgentTrainer

from .common import load_agents
from .fcp_pop_helper import get_fcp_population
from .tc_helper import generate_TC_for_FCP_w_SP_types, generate_TC_for_SP


def get_selfplay_agent(args, total_training_timesteps, train_types, eval_types, tag=None, force_training=False):
    name = 'sp'
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
    if agents:
        return agents[0]

    tc = generate_TC_for_SP(args=args,
                            train_types=train_types,
                            eval_types_to_generate=eval_types['generate'],
                            eval_types_to_read_from_file=eval_types['load'])
    
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
    return selfplay_trainer.get_agents()


def get_fcp_agent_w_tms_clction(args, 
                                pop_total_training_timesteps,
                                fcp_total_training_timesteps,
                                fcp_train_types,
                                fcp_eval_types,
                                curriculum,
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
        curriculum=curriculum,
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
    name = 'fcp_w_selfplay_types'
    agents = load_agents(args, name=name, tag=tag, force_training=fcp_w_sp_force_training)
    if agents:
        return agents[0]

    teammates_collection = generate_TC_for_FCP_w_SP_types(args=args,
                                                                  teammates_collection=fcp_teammates_collection,
                                                                  agent=fcp_agent,
                                                                  train_types=fcp_w_sp_train_types,
                                                                  eval_types=fcp_w_sp_eval_types,
                                                                  )
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
    return fcp_trainer.get_agents()[0]    

