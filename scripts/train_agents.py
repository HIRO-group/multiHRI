from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.rl import RLAgentTrainer, SB3Wrapper, VEC_ENV_CLS
from oai_agents.agents.hrl import RLManagerTrainer, HierarchicalRL, DummyAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.agents.agent_utils import load_agent
from oai_agents.gym_environments.worker_env import OvercookedSubtaskGymEnv

from overcooked_ai_py.mdp.overcooked_mdp import Action
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from gym import Env, spaces

from copy import deepcopy
import numpy as np
from pathlib import Path
import random


def get_selfplay_agent(args, tag=None, force_training=False):
    name = 'sp_det'
    try:
        if force_training:
            raise FileNotFoundError
        tag = tag or 'best'
        agents = RLAgentTrainer.load_agents(args, name=name, tag=tag)
    except FileNotFoundError as e:
        print(f'Could not find saved selfplay agent, creating them from scratch...\nFull Error: {e}')
        selfplay_trainer = RLAgentTrainer(
                                        args=args,
                                        selfplay=True,
                                        teammates_collection=[], 
                                        epoch_timesteps=args.epoch_timesteps,
                                        n_envs = args.n_envs,
                                        seed=678, num_layers=2, name=name,
                                        use_frame_stack=False, deterministic=False,
                                        use_lstm=False, use_cnn=False,
                                        )
        
        selfplay_trainer.train_agents(total_train_timesteps=args.total_training_timesteps)
        agents = selfplay_trainer.get_agents()
    return agents



def get_fcp_population(args, training_steps, force_training=False):
    try:
        if force_training:
            raise FileNotFoundError
        fcp_pop = {}
        for layout_name in args.layout_names:
            fcp_pop[layout_name] = RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}', tag='aamas24')
            print(f'Loaded fcp_pop with {len(fcp_pop[layout_name])} agents.')
    except FileNotFoundError as e:
        print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
        fcp_pop = {layout_name: [] for layout_name in args.layout_names}
        agents = []
        num_layers = 2
        for use_fs in [True]:#[False, True]:
            for seed, h_dim in [(2907, 64), (2907, 256)]:  #(105, 64), (105, 256),# [8,16], [32, 64], [128, 256], [512, 1024]
                ck_rate = training_steps // 10
                name = 'fcp_sp'
                name += f'fs_' if use_fs else ''
                name += f'hd{h_dim}_'
                name += f'seed{seed}'
                print(f'Starting training for: {name}')
                rlat = RLAgentTrainer([], args, selfplay=True, name=name, hidden_dim=h_dim, use_frame_stack=use_fs,
                                      fcp_ck_rate=ck_rate, seed=seed, num_layers=num_layers, epoch_timesteps=args.epoch_timesteps)
                rlat.train_agents(total_training_timesteps=args.total_training_timesteps)

                for layout_name in args.layout_names:
                    agents = rlat.get_fcp_agents(layout_name)
                    fcp_pop[layout_name] += agents

        for layout_name in args.layout_names:
            pop = RLAgentTrainer([], args, selfplay=True, name=f'fcp_pop_{layout_name}')
            pop.agents = fcp_pop[layout_name]
            pop.save_agents(tag='aamas24')
    
    teammates_collection = generate_teammates_collection(fcp_pop, args)
    return teammates_collection


def generate_teammates_collection(fcp_pop, args):
    len_teammates = args.teammates_len
    teammates_collection = {layout_name: [] for layout_name in args.layout_names}

    for layout_name in args.layout_names:
        for _ in range(args.groups_num_in_population):
            if len(fcp_pop[layout_name]) >= len_teammates:
                teammates = random.sample(fcp_pop[layout_name], len_teammates)
                teammates_collection[layout_name].append(teammates)
            else:
                raise ValueError(f"Not enough agents in fcp_pop to form a team of {len_teammates} members for layout {layout_name}")
    return teammates_collection


def get_fcp_agent(args, seed=100, training_steps=1e7, force_training=False):
    name = f'fcp_{seed}'
    teammates_collection = get_fcp_population(args, training_steps, force_training)

    for layout_name in args.layout_names:
        print(f'Loaded fcp_pop with {len(teammates_collection[layout_name])} agents.')

    fcp_trainer = RLAgentTrainer(teammates_collection, args, name=name, use_subtask_counts=False, use_policy_clone=False,
                                 seed=2602, deterministic=False, epoch_timesteps=args.epoch_timesteps)
    fcp_trainer.train_agents(train_timesteps=training_steps)
    return fcp_trainer.get_agents()[0]



if __name__ == '__main__':
    args = get_arguments()
    args.sb_verbose = 1
    args.wandb_mode = 'disabled'

    args.layout_names = ['3_players_clustered_kitchen']
    args.teammates_len = 2
    args.num_players = args.teammates_len + 1 # 3 players = 1 agent + 2 teammates
    
    args.n_envs = 1
    args.epoch_timesteps = 2
    args.total_training_timesteps = 1500
    get_selfplay_agent(args, force_training=True)


    # args.groups_num_in_population = 3
    # get_fcp_agent(args, force_training=True)