
import concurrent
import dill
import os

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, KeyCheckpoints, TeamType

from .curriculum import Curriculum


def _get_most_recent_checkpoint(args, name: str) -> str:
    if args.exp_dir:
        path = args.base_dir / 'agent_models' / args.exp_dir / name
    else:
        path = args.base_dir / 'agent_models' / name


    ckpts = [name for name in os.listdir(path) if name.startswith("ck")]
    ckpts_nums = [int(c.split('_')[1]) for c in ckpts]
    last_ckpt_num = max(ckpts_nums)
    return [c for c in ckpts if c.startswith(f"ck_{last_ckpt_num}")][0]

def train_agent_with_checkpoints(args, total_training_timesteps, ck_rate, seed, h_dim, serialize):
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
        last_ckpt = _get_most_recent_checkpoint(args, name)
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
        elif train_type in [TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY, TeamType.SELF_PLAY_STATIC_ADV, TeamType.SELF_PLAY_DYNAMIC_ADV]:
            train_agents_len += 0
        else:
            eval_agents_len += unseen_teammates_len

    assert total_population_len >= train_agents_len + eval_agents_len, "Not enough agents to train and evaluate." \
                                                                        " Should increase num_SPs_to_train." \
                                                                        f" Total population len: {total_population_len}," \
                                                                        f" train_agents len: {train_agents_len}," \
                                                                        f" eval_agents len: {eval_agents_len}, "\
                                                                        f" num_SPs_to_train: {num_SPs_to_train}."


def generate_hdim_and_seed(num_SPs_to_train):
    '''
    (hidden_dim, seed) = reward of selfplay
    (256, 68)=362, (64, 14)=318
    (256, 13)=248, (64, 0)=230
    (256, 48)=20, (64, 30)=0
    '''
    # Tested in 3-chefs-small-kitchen:
    good_seeds = [68, 14, 13, 0]
    good_hdims = [256, 64, 256, 64]

    # Not tested:
    other_seeds_copied_from_HAHA = [2907, 2907, 105, 105, 8, 32, 128, 512]
    other_hdims_copied_from_HAHA = [64, 256, 64, 256, 16, 64, 256, 1024]

    all_seeds = good_seeds + other_seeds_copied_from_HAHA
    all_hdims = good_hdims + other_hdims_copied_from_HAHA

    selected_seeds = all_seeds[:num_SPs_to_train]
    selected_hdims = all_hdims[:num_SPs_to_train]
    return selected_seeds, selected_hdims


def save_population(args, population):
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


def get_population(args,
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

        save_population(args=args, population=population)

    return population
