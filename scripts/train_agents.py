import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

import torch as th

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.population_tags import TeamType
from scripts.utils import get_fcp_population


def get_selfplay_agent(args, tag=None, force_training=False):
    name = 'sp'
    if not force_training:
        try:
            agents = RLAgentTrainer.load_agents(args, name=name, tag=tag or 'best')
            return agents
        except FileNotFoundError as e:
            print(f'Could not find saved selfplay agent, creating them from scratch...\nFull Error: {e}')

    selfplay_trainer = RLAgentTrainer(
        name=name,
        args=args,
        selfplay=True,
        teammates_collection={},
        agent=None,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=678,
    )

    selfplay_trainer.train_agents(total_train_timesteps=args.total_training_timesteps)
    return selfplay_trainer.get_agents()


def get_fcp_agent(args, force_training=False, parallel=True):
    teammates_collection = get_fcp_population(args,
                                              ck_rate = args.total_training_timesteps // 5,
                                              force_training=force_training,
                                              parallel=parallel)
    fcp_trainer = RLAgentTrainer(
        name='fcp',
        args=args,
        selfplay=False,
        teammates_collection=teammates_collection,
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        seed=2602,
    )

    agents = fcp_trainer.train_agents(total_train_timesteps=args.total_training_timesteps)
    # agent = 

    # selfplay_trainer = RLAgentTrainer(
    #     name='fcp_selfplay',
    #     args=args,
    #     agent = agents[0],
    #     teammates_collection=[],
    #     epoch_timesteps=args.epoch_timesteps,
    #     n_envs=args.n_envs,
    #     seed=678,
    # )

    return fcp_trainer.get_agents()[0]

def set_input(args, use_gpu=True):
    args.layout_names = ['3_chefs_small_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates
    
    if use_gpu: 
        args.n_envs = 100
        args.epoch_timesteps = 1e5
        args.total_training_timesteps = 5e6

    else: # Used for doing quick tests
        args.sb_verbose = 1
        args.wandb_mode = 'disabled'
        args.n_envs = 1
        args.epoch_timesteps = 2
        args.total_training_timesteps = 2500


if __name__ == '__main__':
    args = get_arguments()


    set_input(args, use_gpu=False)

    args.eval_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.MIDDLE_FIRST, 
                       TeamType.LOW_FIRST, TeamType.RANDOM, TeamType.HIGH_MEDIUM,
                       TeamType.HIGH_LOW, TeamType.MEDIUM_LOW, TeamType.HIGH_LOW_RANDOM]


    args.train_types = [TeamType.HIGH_FIRST]

    get_selfplay_agent(args, force_training=True)
    # get_fcp_agent(args, force_training=False, parallel=True)

    # args.train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST]
    # get_fcp_agent(args, force_training=False, parallel=True)

    # args.train_types = [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.LOW_FIRST,
    #                     TeamType.HIGH_MEDIUM, TeamType.HIGH_LOW, TeamType.MEDIUM_LOW]
    # get_fcp_agent(args, force_training=False, parallel=True)
