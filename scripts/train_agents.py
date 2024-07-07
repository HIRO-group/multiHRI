import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

import torch as th

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.population_tags import TeamType
from scripts.utils import get_fcp_population, update_tms_clction_with_selfplay_types, load_agents, print_teammates_collection


def get_selfplay_agent(args, tag=None, force_training=False):
    name = 'sp'
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
    if agents:
        return agents[0]


    selfplay_trainer = RLAgentTrainer(
        name=name,
        args=args,
        teammates_collection={},
        agent=None,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=678,
    )

    selfplay_trainer.train_agents(total_train_timesteps=args.total_training_timesteps)
    return selfplay_trainer.get_agents()


def get_fcp_agent_w_tms_clction(args, tag=None, force_training=False, parallel=True):
    teammates_collection = get_fcp_population(args,
                                              ck_rate = args.total_training_timesteps // 5,
                                              force_training=force_training,
                                              parallel=parallel)
    name = 'fcp'    
    agents = load_agents(args, name=name, tag=tag, force_training=force_training)
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

    fcp_trainer.train_agents(total_train_timesteps=args.total_training_timesteps)
    return fcp_trainer.get_agents()[0], teammates_collection


def get_fcp_agent_trained_with_selfplay_types(args, fcp_agent, tms_clction, force_training=False, tag=None):
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
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=2602,
    )

    fcp_trainer.train_agents(total_train_timesteps=args.total_training_timesteps)
    return fcp_trainer.get_agents()[0]


def set_input(args, use_gpu=True):
    args.layout_names = ['3_chefs_small_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
    
    if use_gpu: 
        args.n_envs = 50
        args.epoch_timesteps = 1e5
        args.total_training_timesteps = 5e6

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 2
        args.epoch_timesteps = 2
        args.total_training_timesteps = 3500


if __name__ == '__main__':
    args = get_arguments()
    use_gpu = True
    parallel = True
    force_training = True
    set_input(args, use_gpu=use_gpu)
    
    args.eval_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.MIDDLE_FIRST, 
                       TeamType.LOW_FIRST, TeamType.RANDOM, TeamType.HIGH_MEDIUM,
                       TeamType.HIGH_LOW, TeamType.MEDIUM_LOW, TeamType.HIGH_LOW_RANDOM]
    args.train_types = [TeamType.HIGH_FIRST]
    fcp_agent, teammates_collection = get_fcp_agent_w_tms_clction(args, force_training=force_training,
                                                                  parallel=parallel)

    args.train_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    args.eval_types = [TeamType.SELF_PLAY_LOW, TeamType.SELF_PLAY_MEDIUM, TeamType.SELF_PLAY_HIGH]
    get_fcp_agent_trained_with_selfplay_types(args, fcp_agent=fcp_agent, 
                                              force_training=force_training,
                                              tms_clction=teammates_collection)
    
    # get_selfplay_agent(args, force_training=force_training)

    # args.train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    # get_fcp_agent(args, force_training=False, parallel=True)

    # args.train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST,
    #                     TeamType.HIGH_MEDIUM, TeamType.HIGH_LOW, TeamType.MEDIUM_LOW]
    # get_fcp_agent(args, force_training=False, parallel=True)
