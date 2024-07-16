from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, TeamType, TeammatesCollection

import random
import multiprocessing
import dill
import math



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
            population[layout_name] = RLAgentTrainer.load_agents(args, name=f'fcp_pop_{layout_name}', tag='aamas25')
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

    
def get_teammates_per_type_and_layout(agents_perftag_score, team_types, t_len):
    teammates_per_type = {
        ttype: [] for ttype in team_types
    }
    sorted_agents_perftag_score = sorted(agents_perftag_score, key=lambda x: x[2], reverse=True)

    for ttype in team_types:
        if ttype == TeamType.HIGH_FIRST:
            tms_prftg_scr = sorted_agents_perftag_score[:t_len]
            teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])
        
        elif ttype == TeamType.MEDIUM_FIRST:
            mean_score = (sorted_agents_perftag_score[0][2] + sorted_agents_perftag_score[-1][2])/2
            sorted_by_closeness = sorted(agents_perftag_score, key=lambda x: abs(x[2] - mean_score))[:t_len]
            teammates_per_type[ttype].append([tm[0] for tm in sorted_by_closeness])

        elif ttype == TeamType.MIDDLE_FIRST:
            middle_index = len(sorted_agents_perftag_score)//2
            start_index_for_mid = middle_index - t_len//2
            end_index_for_mid = start_index_for_mid + t_len
            tms_prftg_scr = sorted_agents_perftag_score[start_index_for_mid:end_index_for_mid]
            teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])

        elif ttype == TeamType.LOW_FIRST:
            tms_prftg_scr = sorted_agents_perftag_score[-t_len:]
            teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])

        elif ttype == TeamType.RANDOM:
            tms_prftg_scr = random.sample(agents_perftag_score, t_len)
            teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])

        elif ttype == TeamType.HIGH_MEDIUM:
            if t_len >= 2:
                first_half = random.sample(teammates_per_type[TeamType.MEDIUM_FIRST][0], t_len//2)
                second_half = random.sample(teammates_per_type[TeamType.HIGH_FIRST][0],  t_len - t_len//2)
                teammates_per_type[ttype].append(first_half + second_half)
                random.shuffle(teammates_per_type[ttype][0])


        elif ttype == TeamType.HIGH_LOW:
            if t_len >= 2:
                first_half = random.sample(teammates_per_type[TeamType.LOW_FIRST][0], t_len//2)
                second_half = random.sample(teammates_per_type[TeamType.HIGH_FIRST][0], t_len - t_len//2)
                teammates_per_type[ttype].append(first_half+second_half)
                random.shuffle(teammates_per_type[ttype][0])

        elif ttype == TeamType.MEDIUM_LOW:
            if t_len >= 2:
                first_half = random.sample(teammates_per_type[TeamType.LOW_FIRST][0], t_len//2)
                second_half = random.sample(teammates_per_type[TeamType.MEDIUM_FIRST][0], t_len - t_len//2)
                teammates_per_type[ttype].append(first_half+second_half)
                random.shuffle(teammates_per_type[ttype][0])
        
        elif ttype == TeamType.HIGH_LOW_RANDOM:
            if t_len >= 2:
                high, low = sorted_agents_perftag_score[0], sorted_agents_perftag_score[-1]
                rand = random.sample(agents_perftag_score, t_len - 2)
                tms_prftg_scr = [high, low] + rand
                teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])
    
    selected_agents = []
    for ttype in team_types:
        selected_agents.extend(teammates_per_type[ttype])

    return teammates_per_type, selected_agents


def update_eval_collection_with_eval_types_from_file(args,
                                                     eval_types,
                                                     eval_collection):

    for teammates in eval_types:
        if teammates.team_type not in eval_collection[teammates.layout_name]:
            eval_collection[teammates.layout_name][teammates.team_type] = []
    
        if teammates.load_from_pop_structure:
            print(f'Loading {teammates.names[0]} agent for {teammates.layout_name} layout')
            layout_population = RLAgentTrainer.load_agents(args, name=teammates.names[0], tag=teammates.tags[0])

            agents_perftag_score_all = [(agent,
                                         agent.layout_performance_tags[teammates.layout_name], 
                                         agent.layout_scores[teammates.layout_name]) for agent in layout_population]
            ec_ln, _ = get_teammates_per_type_and_layout(agents_perftag_score=agents_perftag_score_all,
                                                         team_types=[teammates.team_type],
                                                         t_len=args.teammates_len)
            eval_collection[teammates.layout_name][teammates.team_type].append(ec_ln[teammates.team_type][0])
        else:
            group = []
            for (name, tag) in zip(teammates.names, teammates.tags):
                agents = load_agents(args, name, tag)
                if agents:
                    group.append(agents[0])
            if len(group) == args.teammates_len:
                eval_collection[teammates.layout_name][teammates.team_type].append(group)
    
    print_teammates_collection(eval_collection)



def generate_teammates_collection_w_NO_SP_types(args,
                                                population,
                                                train_types,
                                                eval_types_to_generate,
                                                eval_types_to_read_from_file=[]):
    '''
    Input:
        population = [(Agent, Score, Tag), ...]
        train_types: [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, ...]
        eval_types_to_generate: [TeamType.HIGH_FIRST, TeamType.MEDIUM_FIRST, ...]
        eval_types_to_read_from_file: [EvalMembersToBeLoaded, ...]

    Returns dict
        teammates_collection = {
            TeammatesCollection.Train: {
                'layout_name': {
                        'TeamType.HIGH_FIRST': [[agent1, agent2], ...],
                        'TeamType.RANDOM': [[agent7, agent8],...]
                        ....
                    }
                },
            TeammatesCollection.Eval: {
                    ...
            }
        }
    '''
    eval_collection = {
            layout_name: {ttype: [] for ttype in set(eval_types_to_generate + [t.team_type for t in eval_types_to_read_from_file])}
            for layout_name in args.layout_names
    }

    train_collection = {
        layout_name: {ttype: [] for ttype in train_types}
        for layout_name in args.layout_names
    }

    teammates_collection = {
        TeammatesCollection.TRAIN: train_collection,
        TeammatesCollection.EVAL: eval_collection
    }

    for layout_name in args.layout_names:
        layout_population = population[layout_name]
        agents_perftag_score_all = [(agent,
                                    agent.layout_performance_tags[layout_name], 
                                    agent.layout_scores[layout_name])
                                    for agent in layout_population]

        train_collection[layout_name], train_agents = get_teammates_per_type_and_layout(agents_perftag_score=agents_perftag_score_all,
                                                                                        team_types=train_types,
                                                                                        t_len=args.teammates_len
                                                                                        )

        agents_perftag_score_eval = [agent for agent in agents_perftag_score_all if agent[0] not in train_agents]
        eval_collection[layout_name], _eval_agents = get_teammates_per_type_and_layout(agents_perftag_score=agents_perftag_score_eval, 
                                                                                      team_types=eval_types_to_generate,
                                                                                      t_len=args.teammates_len)
    
    update_eval_collection_with_eval_types_from_file(args=args,
                                                     eval_types=eval_types_to_read_from_file,
                                                     eval_collection=eval_collection)

    teammates_collection[TeammatesCollection.TRAIN] = train_collection
    teammates_collection[TeammatesCollection.EVAL] = eval_collection

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
    return teammates_collection


def print_tc(teammates_collection):
    for layout_name in teammates_collection:
        for tag in teammates_collection[layout_name]:
            print(f'\t{tag}:')
            teammates_c = teammates_collection[layout_name][tag]
            for teammates in teammates_c:
                for agent in teammates:
                    print(f'\t{agent.name}, score for layout {layout_name} is: {agent.layout_scores[layout_name]}, len: {len(teammates)}')
            print('\n')

def print_teammates_collection(teammates_collection):
    if TeammatesCollection.TRAIN in teammates_collection:
        print('Train: ')
        print_tc(teammates_collection[TeammatesCollection.TRAIN])
    if TeammatesCollection.EVAL in teammates_collection:
        print('Eval: ')
        print_tc(teammates_collection[TeammatesCollection.EVAL])
    else:
        print_tc(teammates_collection)


def load_agents(args, name, tag, force_training=False):
    if force_training:
        return []
    try:
        agents = RLAgentTrainer.load_agents(args, name=name, tag=tag or 'best')
        return agents
    except FileNotFoundError as e:
        print(f'Could not find saved {name} agent \nFull Error: {e}')
        return []