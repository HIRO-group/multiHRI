from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.population_tags import AgentPerformance, TeamType

import random
import multiprocessing
import dill


def train_a_agent_with_checkpoints(args, ck_rate, seed, h_dim, serialize):
    name = f'fcp_hd{h_dim}_seed{seed}'
    fcp_pop_perftag = {layout_name: [] for layout_name in args.layout_names}

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
        fcp_pop_perftag[layout_name] = rlat.get_fcp_agents(layout_name)
    if serialize:
        return dill.dumps(fcp_pop_perftag)
    return fcp_pop_perftag


def get_fcp_population(args, ck_rate, parallel=True, force_training=False):
    fcp_pop_perftag = {layout_name: [] for layout_name in args.layout_names} # = [(agent, score, tag), ...]
    try:
        if force_training:
            raise FileNotFoundError
        for layout_name in args.layout_names:
            fcp_pop_perftag[layout_name] = RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}', tag='aamas25')
            print(f'Loaded fcp_pop with {len(fcp_pop_perftag[layout_name])} agents.')

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
                    fcp_pop_perftag[layout_name].extend(res[layout_name])
        else:
            for inp in inputs:
                res = train_a_agent_with_checkpoints(args=inp[0],
                                                     ck_rate=inp[1],
                                                     seed=inp[2],
                                                     h_dim=inp[3],
                                                     serialize=False)
                for layout_name in args.layout_names:
                    fcp_pop_perftag[layout_name].extend(res[layout_name])

        save_fcp_pop(args, fcp_pop_perftag)
    return generate_teammates_collection(fcp_pop_perftag, args)


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


def generate_teammates_collection(fcp_pop_perftag, args):
    '''
    AgentPerformance 
    fcp_pop_perftag = {layout_name: [(agent, agent_perftag, score), ...]}
    
    TeamType
    returns teammates_collection = {
                'layout_name': [
                    'high': [agent1, agent2],
                    'medium': [agent3, agent4],
                    'low': [agent5, agent6],
                    'random': [agent7, agent8],
                ],
            }
    '''
    # print("3_players_clustered_kitchen: ", len(fcp_pop_perftag[args.layout_names[0]][0]))
    teammates_collection = {layout_name: [] for layout_name in args.layout_names}
    for layout_name in args.layout_names:
        for tag in TeamType.ALL:
            if tag == TeamType.HIGH_FIRST:
                teammates = sorted(fcp_pop_perftag[layout_name], key=lambda x: x[1], reverse=True)[:args.teammates_len]
                teammates_collection[layout_name].append({TeamType.HIGH_FIRST : teammates})
            elif tag == TeamType.MEDIUM_FIRST:
                teammates = sorted(fcp_pop_perftag[layout_name], key=lambda x: x[1], reverse=True)[args.teammates_len: 2*args.teammates_len]
                teammates_collection[layout_name].append({TeamType.MEDIUM_FIRST : teammates})
            elif tag == TeamType.LOW_FIRST:
                teammates = sorted(fcp_pop_perftag[layout_name], key=lambda x: x[1])[:args.teammates_len]
                teammates_collection[layout_name].append({TeamType.LOW_FIRST : teammates})
            elif tag == TeamType.RANDOM:
                teammates = random.sample(fcp_pop_perftag[layout_name], args.teammates_len)
                teammates_collection[layout_name].append({TeamType.RANDOM : teammates})
    return teammates_collection
