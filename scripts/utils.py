from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.population_tags import AgentPerformance, TeamType

import random
import multiprocessing
import dill


def train_a_agent_with_checkpoints(args, ck_rate, seed, h_dim, serialize):
    '''
    Returns agnts_prftg_scr_per_lyt = [(agent, performance_tag, score), ...] 
    either serialized or not based on serialize flag
    '''
    name = f'fcp_hd{h_dim}_seed{seed}'
    agnts_prftg_scr_per_lyt = {layout_name: [] for layout_name in args.layout_names}

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
        agnts_prftg_scr_per_lyt[layout_name] = rlat.get_fcp_agents(layout_name)
    if serialize:
        return dill.dumps(agnts_prftg_scr_per_lyt)
    return agnts_prftg_scr_per_lyt


def get_fcp_population(args, ck_rate, parallel=True, force_training=False):
    agnts_prftg_scr_per_lyt = {layout_name: [] for layout_name in args.layout_names} # = [(agent, score, tag), ...]
    try:
        if force_training:
            raise FileNotFoundError
        for layout_name in args.layout_names:
            # TODO
            agnts_prftg_scr_per_lyt[layout_name] = RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}', tag='aamas25')
            print(f'Loaded fcp_pop with {len(agnts_prftg_scr_per_lyt[layout_name])} agents.')

    except FileNotFoundError as e:
        print(
            f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
        seed, h_dim = [2907, 2907], [64, 256]
        inputs = [(args, ck_rate, seed[0], h_dim[0], True), # serialize = True
                  (args, ck_rate, seed[1], h_dim[1], True)]

        if parallel:
            with multiprocessing.Pool() as pool:
                dilled_results = pool.starmap(train_a_agent_with_checkpoints, inputs)
            for dilled_res in dilled_results:
                res = dill.loads(dilled_res)
                for layout_name in args.layout_names:
                    agnts_prftg_scr_per_lyt[layout_name].extend(res[layout_name])
        else:
            for inp in inputs:
                res = train_a_agent_with_checkpoints(args=inp[0],
                                                     ck_rate=inp[1],
                                                     seed=inp[2],
                                                     h_dim=inp[3],
                                                     serialize=False)
                for layout_name in args.layout_names:
                    agnts_prftg_scr_per_lyt[layout_name].extend(res[layout_name])

        save_fcp_pop(args, agnts_prftg_scr_per_lyt)
    return generate_teammates_collection(agnts_prftg_scr_per_lyt, args)


def save_fcp_pop(args, agnts_prftg_scr_per_lyt):
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
        agnts_prftg_scr = agnts_prftg_scr_per_lyt[layout_name]
        rt.agents = [agent_perf_score[0] for agent_perf_score in agnts_prftg_scr]
        rt.save_agents(tag='aamas25')


def generate_teammates_collection(agnts_prftg_scr_per_lyt, args):
    '''
    AgentPerformance 
    agnts_prftg_scr_per_lyt = {layout_name: [(agent, perftag, score), ...]}
    
    TeamType
    returns teammates_collection = {
                'layout_name': {
                    'high': [agent1, agent2],
                    'medium': [agent3, agent4],
                    'low': [agent5, agent6],
                    'random': [agent7, agent8],
                },
            }
    '''
    teammates_collection = \
        {layout_name: {
            TeamType.HIGH_FIRST: [],
            TeamType.MEDIUM_FIRST: [],
            TeamType.LOW_FIRST: [],
            TeamType.RANDOM: [],
                                } for layout_name in args.layout_names}

    for layout_name in args.layout_names:
        for tag in TeamType.ALL:
            if tag == TeamType.HIGH_FIRST:
                tms_prftg_scr = sorted(agnts_prftg_scr_per_lyt[layout_name], key=lambda x: x[2], reverse=True)[:args.teammates_len]
                teammates_collection[layout_name][TeamType.HIGH_FIRST] = [tm[0] for tm in tms_prftg_scr]

            elif tag == TeamType.MEDIUM_FIRST:
                tms_prftg_scr = sorted(agnts_prftg_scr_per_lyt[layout_name], key=lambda x: x[2], reverse=True)[args.teammates_len: 2*args.teammates_len]
                teammates_collection[layout_name][TeamType.MEDIUM_FIRST] = [tm[0] for tm in tms_prftg_scr]

            elif tag == TeamType.LOW_FIRST:
                tms_prftg_scr = sorted(agnts_prftg_scr_per_lyt[layout_name], key=lambda x: x[2])[:args.teammates_len]
                teammates_collection[layout_name][TeamType.LOW_FIRST] = [tm[0] for tm in tms_prftg_scr]

            elif tag == TeamType.RANDOM:
                tms_prftg_scr = random.sample(agnts_prftg_scr_per_lyt[layout_name], args.teammates_len)
                teammates_collection[layout_name][TeamType.RANDOM] = [tm[0] for tm in tms_prftg_scr]
    return teammates_collection


def print_teammates_collection(teammates_collection):
    for layout_name in teammates_collection:
        print(f'Layout: {layout_name}')
        for tag in teammates_collection[layout_name]:
            print(f'\t{tag}:')
            teammates = teammates_collection[layout_name][tag]
            for agent in teammates:
                print(f'\t{agent.name}')
            print('\n')
        print('\n')
