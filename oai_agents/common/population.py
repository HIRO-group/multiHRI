
import concurrent
import dill
import os

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, KeyCheckpoints, TeamType


from .curriculum import Curriculum

import random


def train_agent_with_checkpoints(args, total_training_timesteps, ck_rate, seed, h_dim, serialize, force_training):
    '''
        Returns ckeckpoints_list
        either serialized or not based on serialize flag
    '''
    name = f'SP_hd{h_dim}_seed{seed}'

    agent_ckpt = None
    start_step = 0
    start_timestep = 0
    ck_rewards = None
    if args.resume:
        last_ckpt = RLAgentTrainer.get_most_recent_checkpoint(args, name=name)
        agent_ckpt_info, env_info, training_info = RLAgentTrainer.load_agents(args, name=name, tag=last_ckpt)
        agent_ckpt = agent_ckpt_info[0]
        start_step = env_info["step_count"]
        start_timestep = env_info["timestep_count"]
        ck_rewards = training_info["ck_list"]
        print(f"Restarting training from step: {start_step} (timestep: {start_timestep})")


    rlat = RLAgentTrainer(
        name=name,
        args=args,
        agent=agent_ckpt,
        teammates_collection={}, # automatically creates SP type
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        hidden_dim=h_dim,
        seed=seed,
        checkpoint_rate=ck_rate,
        learner_type=args.pop_learner_type,
        curriculum=Curriculum(train_types=[TeamType.SELF_PLAY], is_random=True),
        start_step=start_step,
        start_timestep=start_timestep
    )
    '''
    For curriculum, whenever we don't care about the order of the training types, we can set is_random=True.
    For SP agents, they only are trained with themselves so the order doesn't matter.
    '''

    rlat.train_agents(total_train_timesteps=total_training_timesteps, tag_for_returning_agent=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL, resume_ck_list=ck_rewards)
    checkpoints_list = rlat.ck_list

    if serialize:
        return dill.dumps(checkpoints_list)
    return checkpoints_list


def ensure_we_will_have_enough_agents_in_population(teammates_len,
                                                    train_types,
                                                    eval_types,
                                                    num_SPs_to_train,
                                                    unseen_teammates_len=0, # only used for SPX teamtypes
                                                ):

    total_population_len = len(AgentPerformance.ALL) * num_SPs_to_train

    train_agents_len, eval_agents_len = 0, 0

    for train_type in train_types:
        if train_type in TeamType.ALL_TYPES_BESIDES_SP:
            train_agents_len += teammates_len
        elif train_type == TeamType.SELF_PLAY or train_type == TeamType.SELF_PLAY_ADVERSARY:
            train_agents_len += 0
        else:
            train_agents_len += unseen_teammates_len

    for eval_type in eval_types:
        if eval_type in TeamType.ALL_TYPES_BESIDES_SP:
            eval_agents_len += teammates_len
        elif train_type == TeamType.SELF_PLAY or train_type == TeamType.SELF_PLAY_ADVERSARY:
            train_agents_len += 0
        else:
            eval_agents_len += unseen_teammates_len

    assert total_population_len >= train_agents_len + eval_agents_len, "Not enough agents to train and evaluate." \
                                                                        " Should increase num_SPs_to_train." \
                                                                        f" Total population len: {total_population_len}," \
                                                                        f" train_agents len: {train_agents_len}," \
                                                                        f" eval_agents len: {eval_agents_len}, "\
                                                                        f" num_SPs_to_train: {num_SPs_to_train}."


def generate_hdim_and_seed(num_of_required_agents):
    '''
    Generates lists of seeds and hidden dimensions for a given number of agents.

    Each setting is a pair (hidden_dim, seed). If the number of required agents
    is less than or equal to the number of predefined settings, it selects from
    the predefined seeds and hidden dimensions. Otherwise, it generates random
    seeds and hidden dimensions to fill the remaining number of agents.

    Arguments:
    num_of_required_agents -- the number of (hidden_dim, seed) pairs to generate.

    Returns:
    selected_seeds -- list of selected seeds
    selected_hdims -- list of selected hidden dimensions
    '''

    # Predefined seeds and hidden dimensions
    seeds = [1010, 2020, 2602, 13, 68, 2907, 105, 128]
    hdims = [256] * len(seeds)

    # Initialize selected lists
    selected_seeds = []
    selected_hdims = []

    # Check if we have enough predefined pairs
    if num_of_required_agents <= len(seeds):
        # Select predefined seeds and hdims
        selected_seeds = seeds[:num_of_required_agents]
        selected_hdims = hdims[:num_of_required_agents]
    else:
        # Use all predefined settings
        selected_seeds = seeds[:]
        selected_hdims = hdims[:]

        # Generate additional random settings if more agents are needed
        remaining = num_of_required_agents - len(seeds)
        available_seeds = set(range(0, 5000)) - set(selected_seeds)
        random_seeds = random.sample(available_seeds, remaining)  # Generate random seeds
        random_hdims = random.choices([256, 512], k=remaining)  # Generate random hidden dimensions

        # Append randomly generated settings to selected lists
        selected_seeds += random_seeds
        selected_hdims += random_hdims

    return selected_seeds, selected_hdims

def save_categorized_population(args, population):
    name_prefix = 'pop'
    for layout_name in args.layout_names:
        rt = RLAgentTrainer(
            name=f'{name_prefix}_{layout_name}',
            args=args,
            agent=None,
            teammates_collection={},
            train_types=[TeamType.SELF_PLAY],
            eval_types=[TeamType.SELF_PLAY],
            epoch_timesteps=args.epoch_timesteps,
            n_envs=args.n_envs,
            learner_type=args.pop_learner_type,
            seed=None,
        )
        rt.agents = population[layout_name]
        rt.save_agents(tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)


def get_categorized_population(args,
                   ck_rate,
                   total_training_timesteps,
                   train_types,
                   eval_types,
                   num_SPs_to_train,
                   unseen_teammates_len=0,
                   force_training=False,
                   tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL,
                   ):

    population = {layout_name: [] for layout_name in args.layout_names}

    try:
        if force_training:
            raise FileNotFoundError
        for layout_name in args.layout_names:
            name = f'pop_{layout_name}'
            population[layout_name], _, _ = RLAgentTrainer.load_agents(args, name=name, tag=tag)
            print(f'Loaded pop with {len(population[layout_name])} agents.')
    except FileNotFoundError as e:
        print(f'Could not find saved population, creating them from scratch...\nFull Error: {e}')

        ensure_we_will_have_enough_agents_in_population(teammates_len=args.teammates_len,
                                                        unseen_teammates_len=unseen_teammates_len,
                                                        train_types=train_types,
                                                        eval_types=eval_types,
                                                        num_SPs_to_train=num_SPs_to_train)

        seed, h_dim = generate_hdim_and_seed(num_SPs_to_train)
        inputs = [
            (args, total_training_timesteps, ck_rate, seed[i], h_dim[i], True) for i in range(num_SPs_to_train)
        ]


        if args.parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_concurrent_jobs) as executor:
                arg_lists = list(zip(*inputs))
                dilled_results = list(executor.map(train_agent_with_checkpoints, *arg_lists))
            for dilled_res in dilled_results:
                checkpoints_list = dill.loads(dilled_res)
                for layout_name in args.layout_names:
                    layout_pop = RLAgentTrainer.get_checkedpoints_agents(args, checkpoints_list, layout_name)
                    population[layout_name].extend(layout_pop)
        else:
            for inp in inputs:
                checkpoints_list = train_agent_with_checkpoints(args=inp[0],
                                                   total_training_timesteps = inp[1],
                                                   ck_rate=inp[2],
                                                   seed=inp[3],
                                                   h_dim=inp[4],
                                                   serialize=False)
                for layout_name in args.layout_names:
                    layout_pop = RLAgentTrainer.get_checkedpoints_agents(args, checkpoints_list, layout_name)
                    population[layout_name].extend(layout_pop)

        save_categorized_population(args=args, population=population)

    return population