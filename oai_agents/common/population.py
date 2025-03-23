
import concurrent
import dill

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, KeyCheckpoints, TeamType


from .curriculum import Curriculum


def train_SP_with_checkpoints(args, total_training_timesteps, ck_rate, seed, h_dim, serialize):
    '''
        Returns ckeckpoints_list
        either serialized or not based on serialize flag
    '''
    name = f'SP_hd{h_dim}_seed{seed}'

    agent_ckpt = None
    start_step = 0
    start_timestep = 0
    ck_rewards = None
    n_envs=args.n_envs
    if args.resume:
        last_ckpt = RLAgentTrainer.get_most_recent_checkpoint(args, name=name)
        if last_ckpt:
            agent_ckpt_info, env_info, training_info = RLAgentTrainer.load_agents(args, name=name, tag=last_ckpt)
            agent_ckpt = agent_ckpt_info[0]
            start_step = env_info["step_count"]
            start_timestep = env_info["timestep_count"]
            ck_rewards = training_info["ck_list"]
            n_envs = training_info["n_envs"]
            print(f"Restarting training from step: {start_step} (timestep: {start_timestep})")


    rlat = RLAgentTrainer(
        name=name,
        args=args,
        agent=agent_ckpt,
        teammates_collection={}, # automatically creates SP type
        epoch_timesteps=args.epoch_timesteps,
        n_envs=n_envs,
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

    rlat.train_agents(
        total_train_timesteps=total_training_timesteps,
        tag_for_returning_agent=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL,
        resume_ck_list=ck_rewards
    )
    checkpoints_list = rlat.ck_list

    if serialize:
        return dill.dumps(checkpoints_list)
    return checkpoints_list


def ensure_enough_SP_agents(teammates_len,
                            train_types,
                            eval_types,
                            total_ego_agents,
                            unseen_teammates_len=0, # only used for SPX teamtypes
                        ):

    total_population_len = len(AgentPerformance.ALL) * total_ego_agents

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
        elif eval_type in [TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY, TeamType.SELF_PLAY_STATIC_ADV, TeamType.SELF_PLAY_DYNAMIC_ADV]:
            eval_agents_len += 0
        else:
            eval_agents_len += unseen_teammates_len

    assert total_population_len >= train_agents_len + eval_agents_len, "Not enough agents to train and evaluate." \
                                                                        " Should increase total_ego_agents." \
                                                                        f" Total population len: {total_population_len}," \
                                                                        f" train_agents len: {train_agents_len}," \
                                                                        f" eval_agents len: {eval_agents_len}, "\
                                                                        f" total_ego_agents: {total_ego_agents}."


def generate_hdim_and_seed(for_evaluation: bool, total_ego_agents: int):
    '''
    Generates lists of seeds and hidden dimensions for a given number of agents for training or evaluation.

    Each setting is a pair (hidden_dim, seed). If the number of required agents
    is less than or equal to the number of predefined settings, it selects from
    the predefined seeds and hidden dimensions. Otherwise, it generates random
    seeds and hidden dimensions to fill the remaining number of agents.

    Arguments:
    for_evaluation -- a boolean indicating whether to generate settings for evluation (True) or training (False).
    total_ego_agents -- the number of (hidden_dim, seed) pairs to generate.

    Returns:
    selected_seeds -- list of selected seeds
    selected_hdims -- list of selected hidden dimensions
    '''
    # Predefined seeds and hidden dimensions for evaluation
    evaluation_seeds = [3031, 4041, 5051, 3708, 3809, 3910, 4607, 5506]
    evaluation_hdims = [256] * len(evaluation_seeds)

    # Predefined seeds and hidden dimensions for training
    training_seeds = [1010, 2020, 2602, 13, 68, 2907, 105, 128]
    training_hdims = [256] * len(training_seeds)


    # Select appropriate predefined settings based on the input setting
    if for_evaluation:
        assert total_ego_agents <= len(evaluation_seeds), (
            f"Total ego agents ({total_ego_agents}) cannot exceed the number of evaluation seeds ({len(evaluation_seeds)}). "
            "Please either increase the number of evaluation seeds in the `generate_hdim_and_seed` function or decrease "
            f"`self.total_ego_agents` (currently set to {total_ego_agents}, based on `args.total_ego_agents`)."
        )
        seeds = evaluation_seeds
        hdims = evaluation_hdims
    else:
        assert total_ego_agents <= len(training_seeds), (
            f"Total ego agents ({total_ego_agents}) cannot exceed the number of training seeds ({len(training_seeds)}). "
            "Please either increase the number of training seeds in the `generate_hdim_and_seed` function or decrease "
            f"`self.total_ego_agents` (currently set to {total_ego_agents}, based on `args.total_ego_agents`)."
        )
        seeds = training_seeds
        hdims = training_hdims

    # Initialize selected lists
    selected_seeds = seeds[:total_ego_agents]
    selected_hdims = hdims[:total_ego_agents]

    return selected_seeds, selected_hdims

def save_performance_based_population_by_layouts(args, population):
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


def get_performance_based_population_by_layouts(
        args,
        ck_rate,
        total_training_timesteps,
        train_types,
        eval_types,
        total_ego_agents,
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

        ensure_enough_SP_agents(
            teammates_len=args.teammates_len,
            unseen_teammates_len=unseen_teammates_len,
            train_types=train_types,
            eval_types=eval_types,
            total_ego_agents=total_ego_agents
        )

        seed, h_dim = generate_hdim_and_seed(
            for_evaluation=args.gen_pop_for_eval, total_ego_agents=total_ego_agents)
        inputs = [
            (args, total_training_timesteps, ck_rate, seed[i], h_dim[i], True)
            for i in range(total_ego_agents)
        ]


        if args.parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_concurrent_jobs) as executor:
                arg_lists = list(zip(*inputs))
                dilled_results = list(executor.map(train_SP_with_checkpoints, *arg_lists))
            for dilled_res in dilled_results:
                checkpoints_list = dill.loads(dilled_res)
                for layout_name in args.layout_names:
                    layout_pop = RLAgentTrainer.get_checkedpoints_agents(
                        args, checkpoints_list, layout_name)
                    population[layout_name].extend(layout_pop)
        else:
            for inp in inputs:
                checkpoints_list = train_SP_with_checkpoints(
                    args=inp[0],
                    total_training_timesteps = inp[1],
                    ck_rate=inp[2],
                    seed=inp[3],
                    h_dim=inp[4],
                    serialize=False
                )
                for layout_name in args.layout_names:
                    layout_pop = RLAgentTrainer.get_checkedpoints_agents(
                        args, checkpoints_list, layout_name)
                    population[layout_name].extend(layout_pop)

        save_performance_based_population_by_layouts(args=args, population=population)

    return population
