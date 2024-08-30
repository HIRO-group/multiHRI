from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import TeamType, TeammatesCollection
from .common import load_agents
from oai_agents.agents.agent_utils import load_agent

import random
from pathlib import Path
    

def generate_TC_for_SP(args,
                       train_types,
                       eval_types_to_generate,
                       eval_types_to_read_from_file):
    eval_collection = {
            layout_name: {ttype: [] for ttype in set(eval_types_to_generate + [t.team_type for t in eval_types_to_read_from_file])}
            for layout_name in args.layout_names
    }

    train_collection = {
        layout_name: {ttype: [] for ttype in train_types}
        for layout_name in args.layout_names
    }

    assert len(train_types) == 1, 'Only one train type is allowed for SP'
    assert train_types[0] == TeamType.SELF_PLAY, 'Only SELF_PLAY is allowed for SP train_type'
    assert len(eval_types_to_generate) == 1, 'Only one eval type can be generated for SP'
    assert eval_types_to_generate[0] == TeamType.SELF_PLAY, 'Only SELF_PLAY is allowed for SP eval_type_to_generate'

    for layout_name in args.layout_names:
        train_collection[layout_name][TeamType.SELF_PLAY] = [[]]
        eval_collection[layout_name][TeamType.SELF_PLAY] = [[]]

    update_eval_collection_with_eval_types_from_file(args=args,
                                                     eval_types=eval_types_to_read_from_file,
                                                     eval_collection=eval_collection)
    teammates_collection = {
        TeammatesCollection.TRAIN: train_collection,
        TeammatesCollection.EVAL: eval_collection
    }
    return teammates_collection



def generate_TC_for_FCP_w_NO_SP_types(args,
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

    teammates_collection = {
        TeammatesCollection.TRAIN: train_collection,
        TeammatesCollection.EVAL: eval_collection
    }

    return teammates_collection


def get_teammates_per_type_and_layout(agents_perftag_score:list, team_types:list, t_len:int):
    '''
    Sort through a list of agents, scores, tags to form different "teams" of different types.
    Each team will be of length t_len and type team_type

    :param agents_perftag_score: List of tuples, [(agent, performance tag, score), (...), ...]
    :param team_types: List of TeamTypes to use for defining each team
    :param t_len: Number of agents in each team
    :returns: Dictionary mapping Each team_type to a list of teammates (agents) and a list of all agents in all teams
    '''
    all_teammates_per_type = {
        ttype: [] for ttype in TeamType.ALL_TYPES_BESIDES_SP
    }
    sorted_agents_perftag_score = sorted(agents_perftag_score, key=lambda x: x[2], reverse=True)

    for ttype in TeamType.ALL_TYPES_BESIDES_SP:
        if ttype == TeamType.HIGH_FIRST:
            tms_prftg_scr = sorted_agents_perftag_score[:t_len]
            all_teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])
        
        elif ttype == TeamType.MEDIUM_FIRST:
            mean_score = (sorted_agents_perftag_score[0][2] + sorted_agents_perftag_score[-1][2])/2
            sorted_by_closeness = sorted(agents_perftag_score, key=lambda x: abs(x[2] - mean_score))[:t_len]
            all_teammates_per_type[ttype].append([tm[0] for tm in sorted_by_closeness])

        elif ttype == TeamType.MIDDLE_FIRST:
            middle_index = len(sorted_agents_perftag_score)//2
            start_index_for_mid = middle_index - t_len//2
            end_index_for_mid = start_index_for_mid + t_len
            tms_prftg_scr = sorted_agents_perftag_score[start_index_for_mid:end_index_for_mid]
            all_teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])

        elif ttype == TeamType.LOW_FIRST:
            tms_prftg_scr = sorted_agents_perftag_score[-t_len:]
            all_teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])

        elif ttype == TeamType.RANDOM:
            tms_prftg_scr = random.sample(agents_perftag_score, t_len)
            all_teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])

        elif ttype == TeamType.HIGH_MEDIUM:
            if t_len >= 2:
                first_half = random.sample(all_teammates_per_type[TeamType.MEDIUM_FIRST][0], t_len//2)
                second_half = random.sample(all_teammates_per_type[TeamType.HIGH_FIRST][0],  t_len - t_len//2)
                all_teammates_per_type[ttype].append(first_half + second_half)
                random.shuffle(all_teammates_per_type[ttype][0])


        elif ttype == TeamType.HIGH_LOW:
            if t_len >= 2:
                first_half = random.sample(all_teammates_per_type[TeamType.LOW_FIRST][0], t_len//2)
                second_half = random.sample(all_teammates_per_type[TeamType.HIGH_FIRST][0], t_len - t_len//2)
                all_teammates_per_type[ttype].append(first_half+second_half)
                random.shuffle(all_teammates_per_type[ttype][0])

        elif ttype == TeamType.MEDIUM_LOW:
            if t_len >= 2:
                first_half = random.sample(all_teammates_per_type[TeamType.LOW_FIRST][0], t_len//2)
                second_half = random.sample(all_teammates_per_type[TeamType.MEDIUM_FIRST][0], t_len - t_len//2)
                all_teammates_per_type[ttype].append(first_half+second_half)
                random.shuffle(all_teammates_per_type[ttype][0])
        
        elif ttype == TeamType.HIGH_LOW_RANDOM:
            if t_len >= 2:
                high, low = sorted_agents_perftag_score[0], sorted_agents_perftag_score[-1]
                rand = random.sample(agents_perftag_score, t_len - 2)
                tms_prftg_scr = [high, low] + rand
                all_teammates_per_type[ttype].append([tm[0] for tm in tms_prftg_scr])


    # Only select the agents that are in the selected team_types
    selected_teammates_per_type = {
        ttype: all_teammates_per_type[ttype] for ttype in team_types
    }

    selected_agents = []
    for ttype in team_types:
        selected_agents.extend(selected_teammates_per_type[ttype])

    return selected_teammates_per_type, selected_agents


# TODO: clean this function
def generate_TC_for_FCP_w_SP_types(args, teammates_collection, agent, train_types, eval_types):
    for layout in args.layout_names:
        train_collection = teammates_collection[TeammatesCollection.TRAIN][layout]
        eval_collection = teammates_collection[TeammatesCollection.EVAL][layout]

        # The TC used for FCP_w_SP_types consists of the original TC and copies of the learning agent itself
        self_teammates = [agent for _ in range(args.teammates_len-1)]
        
        tr_teammates = []
        for train_type in train_types:
            if train_type == TeamType.SELF_PLAY_HIGH:
                high_p_agent_tr = random.choice([a for a in train_collection[TeamType.HIGH_FIRST][0]])
                tr_teammates = [high_p_agent_tr] + self_teammates
            elif train_type == TeamType.SELF_PLAY_MEDIUM:
                medium_p_agent_tr = random.choice([a for a in train_collection[TeamType.MEDIUM_FIRST][0]])
                tr_teammates = [medium_p_agent_tr] + self_teammates
            elif train_type == TeamType.SELF_PLAY_LOW:
                low_p_agent_tr = random.choice([a for a in train_collection[TeamType.LOW_FIRST][0]])
                tr_teammates = [low_p_agent_tr] + self_teammates
            elif train_type == TeamType.SELF_PLAY:
                tr_teammates = [agent for _ in range(args.teammates_len)]
            
            if tr_teammates:
                teammates_collection[TeammatesCollection.TRAIN][layout][train_type] = [tr_teammates]

        e_teammates = []
        for eval_type in eval_types:
            if eval_type == TeamType.SELF_PLAY_HIGH:
                high_p_agent_ev = random.choice([a for a in eval_collection[TeamType.HIGH_FIRST][0]])
                e_teammates = [high_p_agent_ev] + self_teammates
            elif eval_type == TeamType.SELF_PLAY_MEDIUM:
                medium_p_agent_ev = random.choice([a for a in eval_collection[TeamType.MEDIUM_FIRST][0]])
                e_teammates = [medium_p_agent_ev] + self_teammates
            elif eval_type == TeamType.SELF_PLAY_LOW:
                low_p_agent_ev = random.choice([a for a in eval_collection[TeamType.LOW_FIRST][0]])
                e_teammates = [low_p_agent_ev] + self_teammates
            elif eval_type == TeamType.SELF_PLAY:
                e_teammates = [agent for _ in range(args.teammates_len)]
            if e_teammates:
                teammates_collection[TeammatesCollection.EVAL][layout][eval_type] = [e_teammates]
    return teammates_collection



def update_eval_collection_with_eval_types_from_file(args, eval_types, eval_collection):
    for teammates in eval_types:
        if teammates.team_type not in eval_collection[teammates.layout_name]:
            eval_collection[teammates.layout_name][teammates.team_type] = []
        tms_path = Path.cwd() / 'agent_models' / teammates.names[0] 
        if teammates.load_from_pop_structure:
            layout_population = RLAgentTrainer.load_agents(args, path=tms_path, tag=teammates.tags[0])
            agents_perftag_score_all = [(agent,
                                         agent.layout_performance_tags[teammates.layout_name], 
                                         agent.layout_scores[teammates.layout_name]) for agent in layout_population]
            
            ec_ln, _ = get_teammates_per_type_and_layout(agents_perftag_score=agents_perftag_score_all,
                                                         team_types=[teammates.team_type],
                                                         t_len=args.teammates_len)
            print("Loaded agents from pop_file for eval: ", teammates.names[0], ", Teamtype: ", teammates.team_type)
            eval_collection[teammates.layout_name][teammates.team_type].append(ec_ln[teammates.team_type][0])
        else:
            group = []
            for (name, tag) in zip(teammates.names, teammates.tags):
                agents = load_agents(args, name=name, path=tms_path, tag=tag)
                if agents:
                    group.append(agents[0])
            if len(group) == args.teammates_len:
                eval_collection[teammates.layout_name][teammates.team_type].append(group)
                print("Loaded agents from files for eval: ", teammates.names, ", Teamtype: ", teammates.team_type)


def print_teammates_collection(teammates_collection):
    if TeammatesCollection.TRAIN in teammates_collection:
        print('Train: ')
        print_tc_helper(teammates_collection[TeammatesCollection.TRAIN])
    if TeammatesCollection.EVAL in teammates_collection:
        print('Eval: ')
        print_tc_helper(teammates_collection[TeammatesCollection.EVAL])
    else:
        print_tc_helper(teammates_collection)


def print_tc_helper(teammates_collection):
    for layout_name in teammates_collection:
        for tag in teammates_collection[layout_name]:
            print(f'\t{tag}:')
            teammates_c = teammates_collection[layout_name][tag]
            for teammates in teammates_c:
                for agent in teammates:
                    print(f'\t{agent.name}, score for layout {layout_name} is: {agent.layout_scores[layout_name]}, len: {len(teammates)}')


def generate_TC_for_Saboteur(args, 
                            folder_path='agent_models/small_kitchen/supporters', 
                            tag='sp_s68_h256_tr(SP)_ran/best',
                            train_types = TeamType.HIGH_FIRST,
                            eval_types_to_generate=None,
                            eval_types_to_read_from_file=None):
    path = folder_path + '/' + tag

    teammates = [load_agent(Path(path), args) for _ in range(args.teammates_len)] 

    eval_collection = {
            layout_name: {ttype: [] for ttype in set(eval_types_to_generate + [t.team_type for t in eval_types_to_read_from_file])}
            for layout_name in args.layout_names
    }

    train_collection = {
        layout_name: {ttype: [] for ttype in train_types}
        for layout_name in args.layout_names
    }


    for layout_name in args.layout_names:
        train_collection[layout_name][TeamType.HIGH_FIRST] = [teammates]
        eval_collection[layout_name][TeamType.HIGH_FIRST] = [teammates]

    teammates_collection = {
        TeammatesCollection.TRAIN: train_collection,
        TeammatesCollection.EVAL: eval_collection
    }
    
    return teammates_collection