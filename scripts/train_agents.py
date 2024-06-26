import multiprocessing as mp
mp.set_start_method('spawn', force=True) # should be called before any other module imports

import torch as th

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
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
        teammates_collection=[],
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

    fcp_trainer.train_agents(total_train_timesteps=args.total_training_timesteps)
    return fcp_trainer.get_agents()[0]


if __name__ == '__main__':
    args = get_arguments()
    # args.sb_verbose = 0
    # args.wandb_mode = 'disabled'

    args.layout_names = ['3_players_small_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1  # 3 players = 1 agent + 2 teammates

    args.n_envs = 50
    args.epoch_timesteps = 1e5
    args.total_training_timesteps = 5e6

    # get_selfplay_agent(args, force_training=True)

    get_fcp_agent(args, force_training=False, parallel=True)
