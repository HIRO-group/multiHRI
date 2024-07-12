import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

import torch as th

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.population_tags import TeamType
from scripts.utils import get_fcp_population, update_tms_clction_with_selfplay_types, load_agents, print_teammates_collection


def get_selfplay_agent(args, total_training_timesteps, tag=None, force_training=False):
    name = 'sp'
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
    if agents:
        return agents[0]


    selfplay_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection={},
        train_types=[TeamType.SELF_PLAY],
        eval_types=[TeamType.SELF_PLAY],
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
                                pop_force_training,
                                fcp_force_training,
                                tag=None,
                                parallel=True):

    teammates_collection = get_fcp_population(args,
                                              ck_rate=pop_total_training_timesteps // 5,
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
        train_types=fcp_train_types,
        eval_types=fcp_eval_types,
        seed=2602,
    )

    fcp_trainer.train_agents(total_train_timesteps=fcp_total_training_timesteps)
    return fcp_trainer.get_agents()[0], teammates_collection


def get_fcp_agent_trained_with_selfplay_types(args,
                                              fcp_agent,
                                              tms_clction,
                                              total_training_timesteps, 
                                              train_types,
                                              eval_types,
                                              force_training=False,
                                              tag=None):
    name = 'fcp_w_selfplay_types'
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
    if agents:
        return agents[0]

    teammates_collection = update_tms_clction_with_selfplay_types(teammates_collection=tms_clction,
                                                                  agent=fcp_agent,
                                                                  args=args)
    fcp_trainer = RLAgentTrainer(
        name=name,
        args=args,
        agent=fcp_agent,
        teammates_collection=teammates_collection,
        train_types=train_types,
        eval_types=eval_types,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=2602,
    )

    fcp_trainer.train_agents(total_train_timesteps=total_training_timesteps)
    return fcp_trainer.get_agents()[0]


def train_fcp_and_fcp_w_selfplay(args,
                                 pop_total_training_timesteps,
                                 fcp_total_training_timesteps,
                                 fcp_w_sp_total_training_timesteps,
                                 pop_force_training,
                                 fcp_force_training,
                                 fcp_w_sp_force_training,
                                 parallel=True):

    fcp_train_types = [TeamType.HIGH_FIRST]
    fcp_eval_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.MIDDLE_FIRST, 
                       TeamType.LOW_FIRST, TeamType.RANDOM, TeamType.HIGH_MEDIUM,
                       TeamType.HIGH_LOW, TeamType.MEDIUM_LOW, TeamType.HIGH_LOW_RANDOM]
    fcp_agent, teammates_collection = get_fcp_agent_w_tms_clction(args, 
                                                                  pop_total_training_timesteps=pop_total_training_timesteps,
                                                                  fcp_total_training_timesteps=fcp_total_training_timesteps,
                                                                  fcp_train_types=fcp_train_types,
                                                                  fcp_eval_types=fcp_eval_types,
                                                                  pop_force_training=pop_force_training,
                                                                  fcp_force_training=fcp_force_training,
                                                                  parallel=parallel)


    fcp_w_sp_train_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    fcp_w_sp_eval_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    get_fcp_agent_trained_with_selfplay_types(args,
                                              fcp_agent=fcp_agent, 
                                              tms_clction=teammates_collection,
                                              total_training_timesteps=fcp_w_sp_total_training_timesteps,
                                              train_types=fcp_w_sp_train_types,
                                              eval_types=fcp_w_sp_eval_types,
                                              force_training=fcp_w_sp_force_training,
                                              )


def get_input(args, use_gpu=True):
    args.layout_names = ['3_chefs_small_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
    
    if use_gpu: 
        args.n_envs = 50
        args.epoch_timesteps = 1e5
        pop_total_training_timesteps = 5e6
        fcp_total_training_timesteps = 5e6
        fcp_w_sp_total_training_timesteps = 2 * 5e6

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        pop_total_training_timesteps = 3500
        fcp_total_training_timesteps = 3500
        fcp_w_sp_total_training_timesteps = 3500 * 2
    
    return pop_total_training_timesteps, fcp_total_training_timesteps, fcp_w_sp_total_training_timesteps


if __name__ == '__main__':
    args = get_arguments()
    use_gpu = False
    parallel = False
    pop_force_training = False
    fcp_force_training = False
    fcp_w_sp_force_training = False
    
    pop_total_training_timesteps, fcp_total_training_timesteps, fcp_w_sp_total_training_timesteps = get_input(args=args,
                                                                                                              use_gpu=use_gpu)

    train_fcp_and_fcp_w_selfplay(args=args,
                                 pop_total_training_timesteps=pop_total_training_timesteps,
                                 fcp_total_training_timesteps=fcp_total_training_timesteps,
                                 fcp_w_sp_total_training_timesteps=fcp_w_sp_total_training_timesteps,
                                 pop_force_training=pop_force_training,
                                 fcp_force_training=fcp_force_training,
                                 fcp_w_sp_force_training=fcp_w_sp_force_training,
                                 parallel=parallel)


    # get_selfplay_agent(args, force_training=True, total_training_timesteps=3500)
    # args.train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    # get_fcp_agent(args, force_training=False, parallel=True)

    # args.train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST,
    #                     TeamType.HIGH_MEDIUM, TeamType.HIGH_LOW, TeamType.MEDIUM_LOW]
    # get_fcp_agent(args, force_training=False, parallel=True)
