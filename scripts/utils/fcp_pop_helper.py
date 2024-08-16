from .tc_helper import generate_TC_for_FCP_w_NO_SP_types, get_teammates_per_type_and_layout, print_teammates_collection

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, TeamType
from .curriculum import Curriculum

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
                       ):

    population = {layout_name: [] for layout_name in args.layout_names}

    try:
        if force_training:
            raise FileNotFoundError
        for layout_name in args.layout_names:
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
            (args, total_training_timesteps, ck_rate, seed[i], h_dim[i], True) for i in range(num_self_play_agents_to_train)
        ]

        if parallel:
            with multiprocessing.Pool() as pool:
                dilled_results = pool.starmap(train_agent_with_checkpoints, inputs)
            for dilled_res in dilled_results:
                checkpoints_list = dill.loads(dilled_res)
                for layout_name in args.layout_names:
                    layout_pop = RLAgentTrainer.get_fcp_agents(args, checkpoints_list, layout_name)
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
                    layout_pop = RLAgentTrainer.get_fcp_agents(args, checkpoints_list, layout_name)
                    population[layout_name].extend(layout_pop)

        save_fcp_pop(args=args, population=population)

    return generate_TC_for_FCP_w_NO_SP_types(args=args,
                                                       population=population,
                                                       train_types=train_types,
                                                       eval_types_to_generate=eval_types_to_generate,
                                                       eval_types_to_read_from_file=eval_types_to_load_from_file)


def train_agent_with_checkpoints(args, total_training_timesteps, ck_rate, seed, h_dim, serialize):
    '''
        Returns ckeckpoints_list
        either serialized or not based on serialize flag
    '''

    name = f'fcp_hd{h_dim}_seed{seed}'

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
        curriculum=Curriculum(train_types=args.fcp_train_types, is_random=True)
    )
    '''
    For curriculum, whenever we don't care about the order of the training types, we can set is_random=True.
    For SP agents, they only are trained with themselves so the order doesn't matter.
    '''
    
    rlat.train_agents(total_train_timesteps=total_training_timesteps)
    checkpoints_list = rlat.ck_list

    if serialize:
        return dill.dumps(checkpoints_list)
    return checkpoints_list


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
    
    selected_seeds = all_seeds[:num_self_play_agents_to_train]
    selected_hdims = all_hdims[:num_self_play_agents_to_train]
    return selected_seeds, selected_hdims


def save_fcp_pop(args, population):
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