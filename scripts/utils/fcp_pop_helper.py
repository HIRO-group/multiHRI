from .categorization import generate_teammates_collection_w_NO_SP_types, get_teammates_per_type_and_layout, print_teammates_collection

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, TeamType

import multiprocessing
import dill

def get_fcp_population(args,
                       ck_rate,
                       total_training_timesteps,
                       train_types, 
                       eval_types_to_generate,
                       eval_types_to_load_from_file=[],
                       num_self_play_agents_to_train=2,
                       parallel=True,
                       force_training=False,
                       save_path_prefix=None,
                       ):

    population = {layout_name: [] for layout_name in args.layout_names}

    try:
        if force_training:
            raise FileNotFoundError
        for layout_name in args.layout_names:

            if save_path_prefix:
                name = f'{save_path_prefix}/fcp_pop_{layout_name}'
            else:
                name = f'fcp_pop_{layout_name}'

            population[layout_name] = RLAgentTrainer.load_agents(args, name=name, tag='aamas25')
            print(f'Loaded fcp_pop with {len(population[layout_name])} agents.')
    except FileNotFoundError as e:
        print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')

        ensure_we_have_enough_train_and_eval_agents(teammates_len=args.teammates_len,
                                                    train_types=train_types,
                                                    eval_types=eval_types_to_generate,
                                                    num_self_play_agents_to_train=num_self_play_agents_to_train,
                                                    )

        seed, h_dim = generate_hdim_and_seed(num_self_play_agents_to_train)
        inputs = [
            (args, total_training_timesteps, ck_rate, seed[i], h_dim[i], True, save_path_prefix) for i in range(num_self_play_agents_to_train)
        ]

        if parallel:
            with multiprocessing.Pool() as pool:
                dilled_results = pool.starmap(train_agent_with_checkpoints, inputs)
            for dilled_res in dilled_results:
                res = dill.loads(dilled_res)
                for layout_name in args.layout_names:
                    population[layout_name].extend(res[layout_name])
        else:
            for inp in inputs:
                res = train_agent_with_checkpoints(args=inp[0],
                                                   total_training_timesteps = inp[1],
                                                   ck_rate=inp[2],
                                                   seed=inp[3],
                                                   h_dim=inp[4],
                                                   serialize=False,
                                                   save_path_prefix=save_path_prefix)
                for layout_name in args.layout_names:
                    population[layout_name].extend(res[layout_name])

        save_fcp_pop(args=args, population=population, save_path_prefix=save_path_prefix)

    return generate_teammates_collection_w_NO_SP_types(args=args,
                                                       population=population,
                                                       train_types=train_types,
                                                       eval_types_to_generate=eval_types_to_generate,
                                                       eval_types_to_read_from_file=eval_types_to_load_from_file)


def train_agent_with_checkpoints(args, total_training_timesteps, ck_rate, seed, h_dim, serialize, save_path_prefix=None):
    '''
        Returns population = {layout: [agent1,  agent2, ...] }
        either serialized or not based on serialize flag
    '''

    name = f'fcp_hd{h_dim}_seed{seed}'
    if save_path_prefix:
        name = f'{save_path_prefix}/{name}'

    population = {layout_name: [] for layout_name in args.layout_names}

    rlat = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection={},
        epoch_timesteps=args.epoch_timesteps,
        n_envs=args.n_envs,
        hidden_dim=h_dim,
        seed=seed,
        fcp_ck_rate=ck_rate,
    )
    rlat.train_agents(total_train_timesteps=total_training_timesteps)
    for layout_name in args.layout_names:
        population[layout_name] = rlat.get_fcp_agents(layout_name)
    if serialize:
        return dill.dumps(population)
    return population


def ensure_we_have_enough_train_and_eval_agents(teammates_len,
                                                train_types,
                                                eval_types,
                                                num_self_play_agents_to_train,
                                                ):

    total_population_len = len(AgentPerformance.ALL) * num_self_play_agents_to_train
    train_agents_len = len(train_types) * teammates_len
    eval_agents_len = len(eval_types) * teammates_len
    assert total_population_len > train_agents_len + eval_agents_len, "Not enough agents to train and evaluate. Should increase num_sp_agents_to_train"


def generate_hdim_and_seed(num_self_play_agents_to_train):
    '''
    Returns seed = [13, 2 * 13, 3 * 13, ...] and h_dim = [64, 64, 256, 256, ...]
    '''
    seed = [i*13 for i in range(num_self_play_agents_to_train)]
    h_dim = [64 for _ in range(num_self_play_agents_to_train//2)] 
    h_dim += [256 for _ in range(num_self_play_agents_to_train-len(h_dim))]
    return seed, h_dim



def save_fcp_pop(args, population, save_path_prefix=None):
    if save_path_prefix:
        name_prefix = save_path_prefix+'/fcp_pop'
    else:
        name_prefix = 'fcp_pop'
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
            seed=None,
        )
        rt.agents = population[layout_name]
        rt.save_agents(tag='aamas25')