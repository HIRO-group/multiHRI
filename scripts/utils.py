from oai_agents.agents.rl import RLAgentTrainer
import random
import multiprocessing
import dill


def train_a_agent_with_checkpoints(args, ck_rate, seed, h_dim, serialize):
    name = f'fcp_hd{h_dim}_seed{seed}'
    fcp_pop = {layout_name: [] for layout_name in args.layout_names}

    rlat = RLAgentTrainer(
        name=name,
        args=args,
        selfplay=True,
        teammates_collection=[],
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        hidden_dim=h_dim,
        seed=seed,
        fcp_ck_rate=ck_rate,
    )
    rlat.train_agents(total_train_timesteps=args.total_training_timesteps)
    for layout_name in args.layout_names:
        fcp_pop[layout_name] = rlat.get_fcp_agents(layout_name)

    if serialize:
        return dill.dumps(fcp_pop)
    return fcp_pop


def get_fcp_population(args, parallel=True, force_training=False):
    fcp_pop = {layout_name: [] for layout_name in args.layout_names}
    print("Force Training: ", force_training)
    try:
        if force_training:
            raise FileNotFoundError
        for layout_name in args.layout_names:
            fcp_pop[layout_name] = RLAgentTrainer.load_agents(
                args, name=f'fcp_pop_{layout_name}', tag='aamas25')
            print(f'Loaded fcp_pop with {len(fcp_pop[layout_name])} agents.')

    except FileNotFoundError as e:
        print(
            f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
        ck_rate = args.total_training_timesteps // 10
        seed, h_dim = [2907, 2907], [64, 256]
        inputs = [(args, ck_rate, seed[0], h_dim[0], True), # serialize=True
                  (args, ck_rate, seed[1], h_dim[1], True)]

        if parallel:
            with multiprocessing.Pool() as pool:
                dilled_results = pool.starmap(
                    train_a_agent_with_checkpoints, inputs)
            for dilled_res in dilled_results:
                res = dill.loads(dilled_res)
                for layout_name in args.layout_names:
                    fcp_pop[layout_name].extend(res[layout_name])
        else:
            for inp in inputs:
                res = train_a_agent_with_checkpoints(args=inp[0],
                                                     ck_rate=inp[1],
                                                     seed=inp[2],
                                                     h_dim=inp[3],
                                                     serialize=False)
                for layout_name in args.layout_names:
                    fcp_pop[layout_name].extend(res[layout_name])

        save_fcp_pop(args, fcp_pop)
    return generate_teammates_collection(fcp_pop, args)


def save_fcp_pop(args, fcp_pop):
    for layout_name in args.layout_names:
        rt = RLAgentTrainer(
            name=f'fcp_pop_{layout_name}',
            args=args,
            teammates_collection=[],
            selfplay=True,
            epoch_timesteps=args.epoch_timesteps,
            n_envs=args.n_envs,
            seed=None,
        )
        rt.agents = fcp_pop[layout_name]
        rt.save_agents(tag='aamas25')


def generate_teammates_collection(fcp_pop, args):
    len_teammates = args.teammates_len
    teammates_collection = {layout_name: []
                            for layout_name in args.layout_names}
    for layout_name in args.layout_names:
        for _ in range(args.groups_num_in_population):
            if len(fcp_pop[layout_name]) >= len_teammates:
                teammates = random.sample(fcp_pop[layout_name], len_teammates)
                teammates_collection[layout_name].append(teammates)
            else:
                raise ValueError(
                    f"Not enough agents in fcp_pop to form a team of {len_teammates} members for layout {layout_name}")
    return teammates_collection
