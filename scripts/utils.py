from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.population_tags import AgentPerformance, TeamType

import random
import multiprocessing
import dill
import math


def train_a_agent_with_checkpoints(args, total_training_timesteps, ck_rate, seed, h_dim, serialize):
    '''
        Returns population = {layout: [agent1,  agent2, ...] }
        either serialized or not based on serialize flag
    '''
    name = f'fcp_hd{h_dim}_seed{seed}'
    population = {layout_name: [] for layout_name in args.layout_names}

    rlat = RLAgentTrainer(
        name=name,
        args=args,
        agent=None,
        teammates_collection={},
        train_types=[TeamType.SELF_PLAY],
        eval_types=[TeamType.SELF_PLAY],
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


def get_fcp_population(args, ck_rate, total_training_timesteps, parallel=True, force_training=False):
    population = {layout_name: [] for layout_name in args.layout_names} # = [(agent, score, tag), ...]
    try:
        if force_training:
            raise FileNotFoundError
        for layout_name in args.layout_names:
            population[layout_name] = RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}', tag='aamas25')
            print(f'Loaded fcp_pop with {len(population[layout_name])} agents.')

    except FileNotFoundError as e:
        print(f'Could not find saved FCP population, creating them from scratch...\nFull Error: {e}')
        
        seed, h_dim = [2907, 2907], [64, 256]
        inputs = [(args, total_training_timesteps, ck_rate, seed[0], h_dim[0], True), # serialize = True
                  (args, total_training_timesteps, ck_rate, seed[1], h_dim[1], True)]

        if parallel:
            with multiprocessing.Pool() as pool:
                dilled_results = pool.starmap(train_a_agent_with_checkpoints, inputs)
            for dilled_res in dilled_results:
                res = dill.loads(dilled_res)
                for layout_name in args.layout_names:
                    population[layout_name].extend(res[layout_name])
        else:
            for inp in inputs:
                res = train_a_agent_with_checkpoints(args=inp[0],
                                                     total_training_timesteps = inp[1],
                                                     ck_rate=inp[2],
                                                     seed=inp[3],
                                                     h_dim=inp[4],
                                                     serialize=False)
                for layout_name in args.layout_names:
                    population[layout_name].extend(res[layout_name])

        save_fcp_pop(args, population)
    return generate_teammates_collection(population, args)


def save_fcp_pop(args, population):
    for layout_name in args.layout_names:
        rt = RLAgentTrainer(
            name=f'fcp_pop_{layout_name}',
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


def generate_teammates_collection(population, args):
    '''
    dict 
    teammates_collection = {
        'layout_name': {
            'training': {
                'TeamType.HIGH_FIRST': [[agent1, agent2], ...],
                'TeamType.MEDIUM_FIRST': [agent3, agent4],
                'TeamType.LOW_FIRST': [agent5, agent6],
                'TeamType.RANDOM': [agent7, agent8],
            }
            'eval': {
            
            }
        },
    }
    '''
    # teammates_collection = \
    #     {layout_name: {
    #         TeamType.HIGH_FIRST: [],
    #         TeamType.MEDIUM_FIRST: [],
    #         TeamType.MIDDLE_FIRST: [],
    #         TeamType.LOW_FIRST: [],
    #         TeamType.RANDOM: [],
    #         TeamType.RANDOM_HIGH_MEDIUM: [],
    #         TeamType.RANDOM_HIGH_LOW: [],
    #         TeamType.RANDOM_MEDIUM_LOW: [],
    #         TeamType.HIGH_LOW_RANDOM: [],
    #                             } for layout_name in args.layout_names}

    non_sp_tags = [
            TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, TeamType.MIDDLE_FIRST,
            TeamType.LOW_FIRST, TeamType.RANDOM, TeamType.HIGH_MEDIUM,
            TeamType.HIGH_LOW, TeamType.MEDIUM_LOW, TeamType.HIGH_LOW_RANDOM
            ]
    
    teammates_collection = {
        layout_name: {tag: [] for tag in non_sp_tags}
        for layout_name in args.layout_names
    }
    

    for layout_name in args.layout_names:
        layout_population = population[layout_name]
        agents_perftag_score = [(agent,
                                agent.layout_performance_tags[layout_name], 
                                agent.layout_scores[layout_name])
                                for agent in layout_population]

        sorted_agents_perftag_score = sorted(agents_perftag_score, key=lambda x: x[2], reverse=True)
        t_len = args.teammates_len
        half_floor = math.floor(t_len/2) 
        half_ceil =math.ceil(t_len/2) 
        for tag in non_sp_tags:
            if tag == TeamType.HIGH_FIRST:
                tms_prftg_scr = sorted_agents_perftag_score[:t_len]
                teammates_collection[layout_name][tag] = [tm[0] for tm in tms_prftg_scr]
            
            elif tag == TeamType.MEDIUM_FIRST:
                mean_score = (sorted_agents_perftag_score[0][2]+sorted_agents_perftag_score[-1][2])/2
                # Sort scores by their distance to the mean_score
                sorted_by_closeness = sorted(agents_perftag_score, key=lambda x: abs(x[2] - mean_score))[:t_len]
                # Select the top num_teammates scores closest to the mean
                teammates_collection[layout_name][tag] = [tm[0] for tm in sorted_by_closeness]

            elif tag == TeamType.MIDDLE_FIRST:
                l = len(sorted_agents_perftag_score)
                l_div_2 = math.floor(l/2)
                lower_mid_id = l_div_2-half_floor
                higher_mid_id = l_div_2+half_ceil-1
                assert lower_mid_id >= 0
                assert higher_mid_id < l
                tms_prftg_scr = sorted_agents_perftag_score[lower_mid_id:higher_mid_id+1]
                teammates_collection[layout_name][tag] = [tm[0] for tm in tms_prftg_scr]

            elif tag == TeamType.LOW_FIRST:
                tms_prftg_scr = sorted_agents_perftag_score[-t_len:]
                teammates_collection[layout_name][tag] = [tm[0] for tm in tms_prftg_scr]

            elif tag == TeamType.RANDOM:
                tms_prftg_scr = random.sample(agents_perftag_score, t_len)
                teammates_collection[layout_name][tag] = [tm[0] for tm in tms_prftg_scr]
            
            elif tag == TeamType.HIGH_MEDIUM:
                if t_len >= 2:
                    first_half = random.sample(teammates_collection[layout_name][TeamType.MEDIUM_FIRST], half_floor)
                    second_half = random.sample(teammates_collection[layout_name][TeamType.HIGH_FIRST], half_ceil)
                    teammates_collection[layout_name][tag] = first_half+second_half
                    random.shuffle(teammates_collection[layout_name][tag])

            elif tag == TeamType.HIGH_LOW:
                if t_len >= 2:
                    first_half = random.sample(teammates_collection[layout_name][TeamType.LOW_FIRST], half_floor)
                    second_half = random.sample(teammates_collection[layout_name][TeamType.HIGH_FIRST], half_ceil)
                    teammates_collection[layout_name][tag] = first_half+second_half
                    random.shuffle(teammates_collection[layout_name][tag])

            elif tag == TeamType.MEDIUM_LOW:
                if t_len >= 2:
                    first_half = random.sample(teammates_collection[layout_name][TeamType.LOW_FIRST], half_floor)
                    second_half = random.sample(teammates_collection[layout_name][TeamType.MEDIUM_FIRST], half_ceil)
                    teammates_collection[layout_name][tag] = first_half+second_half
                    random.shuffle(teammates_collection[layout_name][tag])
            
            elif tag == TeamType.HIGH_LOW_RANDOM:
                if t_len >= 2:
                    high, low = sorted_agents_perftag_score[0], sorted_agents_perftag_score[-1]
                    rand = random.sample(agents_perftag_score, t_len - 2)
                    tms_prftg_scr = [high, low] + rand
                    teammates_collection[layout_name][TeamType.HIGH_LOW_RANDOM] = [tm[0] for tm in tms_prftg_scr]
    
    return teammates_collection



def update_tms_clction_with_selfplay_types(teammates_collection, agent, args):
    for layout in args.layout_names:
        high_p_agent = random.choice([a for a in teammates_collection[layout][TeamType.HIGH_FIRST]])
        medium_p_agent = random.choice([a for a in teammates_collection[layout][TeamType.MEDIUM_FIRST]])
        low_p_agent = random.choice([a for a in teammates_collection[layout][TeamType.LOW_FIRST]])

        self_teammates = [agent for _ in range(args.teammates_len-1)]
        teammates_collection[layout][TeamType.SELF_PLAY_HIGH] = self_teammates + [high_p_agent]
        teammates_collection[layout][TeamType.SELF_PLAY_MEDIUM] = self_teammates + [medium_p_agent]
        teammates_collection[layout][TeamType.SELF_PLAY_LOW] = self_teammates + [low_p_agent]
    
    print_teammates_collection(teammates_collection=teammates_collection)
    return teammates_collection



def print_teammates_collection(teammates_collection):
    for layout_name in teammates_collection:
        for tag in teammates_collection[layout_name]:
            print(f'\t{tag}:')
            teammates = teammates_collection[layout_name][tag]
            for agent in teammates:
                print(f'\t{agent.name}, score for layout {layout_name} is: {agent.layout_scores[layout_name]}, len: {len(teammates)}')
            print('\n')


def load_agents(args, name, tag, force_training=False):
    if force_training:
        return []
    
    try:
        agents = RLAgentTrainer.load_agents(args, name=name, tag=tag or 'best')
        return agents
    except FileNotFoundError as e:
        print(f'Could not find saved {name} agent \nFull Error: {e}')
        return []