from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import TeamType, TeammatesCollection

from itertools import permutations
import random
from pathlib import Path


def get_teammates(args, agents_perftag_score:list, teamtypes:list, teammates_len:int, unseen_teammates_len:int, agent:RLAgentTrainer=None):
    all_teammates = {
        teamtype: [] for teamtype in teamtypes
    }
    sorted_agents_perftag_score = sorted(agents_perftag_score, key=lambda x: x[2], reverse=True)
    used_agents = set()  # To keep track of used agents

    for teamtype in teamtypes:
        available_agents = [agent for agent in sorted_agents_perftag_score if agent[0] not in used_agents]

        if teamtype == TeamType.HIGH_FIRST:
            tms_prftg_scr = available_agents[:teammates_len]
            all_teammates[teamtype].append([tm[0] for tm in tms_prftg_scr])
            used_agents.update([tm[0] for tm in tms_prftg_scr])
        
        elif teamtype == TeamType.MEDIUM_FIRST:
            mean_score = (available_agents[0][2] + available_agents[-1][2]) / 2
            sorted_by_closeness = sorted(available_agents, key=lambda x: abs(x[2] - mean_score))[:teammates_len]
            all_teammates[teamtype].append([tm[0] for tm in sorted_by_closeness])
            used_agents.update([tm[0] for tm in sorted_by_closeness])

        elif teamtype == TeamType.MIDDLE_FIRST:
            middle_index = len(available_agents) // 2
            start_index_for_mid = middle_index - teammates_len // 2
            end_index_for_mid = start_index_for_mid + teammates_len
            tms_prftg_scr = available_agents[start_index_for_mid:end_index_for_mid]
            all_teammates[teamtype].append([tm[0] for tm in tms_prftg_scr])
            used_agents.update([tm[0] for tm in tms_prftg_scr])

        elif teamtype == TeamType.LOW_FIRST:
            tms_prftg_scr = available_agents[-teammates_len:]
            all_teammates[teamtype].append([tm[0] for tm in tms_prftg_scr])
            used_agents.update([tm[0] for tm in tms_prftg_scr])

        elif teamtype == TeamType.RANDOM:
            tms_prftg_scr = random.sample(available_agents, teammates_len)
            all_teammates[teamtype].append([tm[0] for tm in tms_prftg_scr])
            used_agents.update([tm[0] for tm in tms_prftg_scr])

        elif teamtype == TeamType.ALL_MIX:
            teammate_permutations = list(permutations(sorted_agents_perftag_score, teammates_len))
            for tp in teammate_permutations:
                all_teammates[teamtype].append([tm[0] for tm in tp])

        elif teamtype == TeamType.SELF_PLAY:
            assert agent is not None
            all_teammates[teamtype].append([agent for _ in range(teammates_len)])

        elif teamtype == TeamType.SELF_PLAY_HIGH:
            assert agent is not None
            high_p_agents = available_agents[:unseen_teammates_len]
            agents_itself = [agent for _ in range(teammates_len - unseen_teammates_len)]
            all_teammates[teamtype].append([tm[0] for tm in high_p_agents] + agents_itself)
            used_agents.update([tm[0] for tm in high_p_agents])

        elif teamtype == TeamType.SELF_PLAY_MEDIUM:
            assert agent is not None
            mean_score = (available_agents[0][2] + available_agents[-1][2]) / 2
            mean_p_agents = sorted(available_agents, key=lambda x: abs(x[2] - mean_score))[:unseen_teammates_len]
            agents_itself = [agent for _ in range(teammates_len - unseen_teammates_len)]
            all_teammates[teamtype].append([tm[0] for tm in mean_p_agents] + agents_itself)
            used_agents.update([tm[0] for tm in mean_p_agents])

        elif teamtype == TeamType.SELF_PLAY_MIDDLE:
            assert agent is not None
            middle_index = len(available_agents) // 2
            start_index_for_mid = middle_index - unseen_teammates_len // 2
            end_index_for_mid = start_index_for_mid + unseen_teammates_len
            mid_p_agents = available_agents[start_index_for_mid:end_index_for_mid]
            agents_itself = [agent for _ in range(teammates_len - unseen_teammates_len)]
            all_teammates[teamtype].append([tm[0] for tm in mid_p_agents] + agents_itself)
            used_agents.update([tm[0] for tm in mid_p_agents])

        elif teamtype == TeamType.SELF_PLAY_LOW:
            assert agent is not None
            low_p_agents = available_agents[-unseen_teammates_len:]
            agents_itself = [agent for _ in range(teammates_len - unseen_teammates_len)]
            all_teammates[teamtype].append([tm[0] for tm in low_p_agents] + agents_itself)
            used_agents.update([tm[0] for tm in low_p_agents])


    selected_agents = []
    for teamtype in teamtypes:
        selected_agents.extend(all_teammates[teamtype])

    return all_teammates, selected_agents



def generate_TC(args,
                population,
                train_types,
                eval_types_to_generate,
                eval_types_to_read_from_file,
                agent=None,
                unseen_teammates_len=0,
                ):
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
    if unseen_teammates_len > 0 and agent is None:
        raise ValueError('Unseen teammates length is greater than 0 but agent is not provided')

    eval_collection = {
        layout_name: {ttype: [] for ttype in set(eval_types_to_generate + [t.team_type for t in eval_types_to_read_from_file])}
        for layout_name in args.layout_names}

    train_collection = {
        layout_name: {ttype: [] for ttype in train_types}
        for layout_name in args.layout_names}


    for layout_name in args.layout_names:
        layout_population = population[layout_name]
        agents_perftag_score_all = [(agent,
                                     agent.layout_performance_tags[layout_name], 
                                     agent.layout_scores[layout_name]) for agent in layout_population]
        train_collection[layout_name], train_agents = get_teammates(args=args,
                                                                    agents_perftag_score=agents_perftag_score_all,
                                                                    teamtypes=train_types,
                                                                    teammates_len=args.teammates_len,
                                                                    agent=agent,
                                                                    unseen_teammates_len=unseen_teammates_len,
                                                                    )

        agents_perftag_score_eval = [agent for agent in agents_perftag_score_all if agent[0] not in train_agents]
        eval_collection[layout_name], _eval_agents = get_teammates(args=args,
                                                                    agents_perftag_score=agents_perftag_score_eval,
                                                                    teamtypes=eval_types_to_generate,
                                                                    teammates_len=args.teammates_len,
                                                                    agent=agent,
                                                                    unseen_teammates_len=unseen_teammates_len,
                                                                    )
    
    update_eval_collection_with_eval_types_from_file(args=args,
                                                     eval_types=eval_types_to_read_from_file,
                                                     eval_collection=eval_collection,
                                                     agent=agent,
                                                     unseen_teammates_len=unseen_teammates_len,
                                                    )

    teammates_collection = {
        TeammatesCollection.TRAIN: train_collection,
        TeammatesCollection.EVAL: eval_collection
    }

    return teammates_collection            


def get_best_SP_agent(args, population):
    agents_scores_averaged_over_layouts = []

    for layout_name in args.layout_names:
        all_agents = [agent for agent in population[layout_name]]

    for agent in all_agents:
        scores = [agent.layout_scores[layout_name] for layout_name in args.layout_names]
        agents_scores_averaged_over_layouts.append((agent, sum(scores)/len(scores)))
    best_agent = max(agents_scores_averaged_over_layouts, key=lambda x: x[1])
    return best_agent[0]



def update_eval_collection_with_eval_types_from_file(args, agent, unseen_teammates_len, eval_types, eval_collection):
    for teammates in eval_types:
        if teammates.team_type not in eval_collection[teammates.layout_name]:
            eval_collection[teammates.layout_name][teammates.team_type] = []
        tms_path = Path.cwd() / 'agent_models' / teammates.names[0] 
        if teammates.load_from_pop_structure:
            layout_population = RLAgentTrainer.load_agents(args, path=tms_path, tag=teammates.tags[0])
            agents_perftag_score_all = [(agent,
                                         agent.layout_performance_tags[teammates.layout_name], 
                                         agent.layout_scores[teammates.layout_name]) for agent in layout_population]
            
            ec_ln , _ = get_teammates(agents_perftag_score=agents_perftag_score_all,
                                     teamtypes=[teammates.team_type],
                                     teammates_len=args.teammates_len,
                                     agent=agent,
                                     unseen_teammates_len=unseen_teammates_len
                                     )

            print("Loaded agents from pop_file for eval: ", teammates.names[0], ", Teamtype: ", teammates.team_type)
            eval_collection[teammates.layout_name][teammates.team_type].append(ec_ln[teammates.team_type][0])
        else:
            group = []
            for (name, tag) in zip(teammates.names, teammates.tags):
                try:
                    agents = RLAgentTrainer.load_agents(args, name=name, path=tms_path, tag=tag)
                except FileNotFoundError as e:
                    print(f'Could not find saved {name} agent \nFull Error: {e}')
                    agents = []
                if agents:
                    group.append(agents[0])
            if len(group) == args.teammates_len:
                eval_collection[teammates.layout_name][teammates.team_type].append(group)
                print("Loaded agents from files for eval: ", teammates.names, ", Teamtype: ", teammates.team_type)


def update_TC_w_adversary(args,
                          teammates_collection,
                          adversaries, 
                          primary_agent):
    for layout_name in args.layout_names:
        for adversary in adversaries:
            teammates_collection[TeammatesCollection.TRAIN][layout_name][TeamType.SELF_PLAY_ADVERSARY] = [[adversary]+[primary_agent for _ in range(args.teammates_len-1)]]
            teammates_collection[TeammatesCollection.EVAL][layout_name][TeamType.SELF_PLAY_ADVERSARY] = [[adversary]+[primary_agent for _ in range(args.teammates_len-1)]]
    return teammates_collection


def generate_TC_for_Adversary(args, 
                            agent):

    teammates = [agent for _ in range(args.teammates_len)]

    eval_collection = {
            layout_name: {ttype: [] for ttype in [TeamType.HIGH_FIRST]}
            for layout_name in args.layout_names
    }

    train_collection = {
        layout_name: {ttype: [] for ttype in [TeamType.HIGH_FIRST]}
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


def generate_TC_for_AdversarysPlay(args, 
                                agent,
                                adversarys, 
                                train_types = [TeamType.SELF_PLAY, TeamType.SELF_PLAY_ADVERSARY],
                                eval_types_to_generate=None,
                                eval_types_to_read_from_file=None):

    self_teammates = [agent for _ in range(args.teammates_len-1)] 
    eval_collection = {
            layout_name: {ttype: [] for ttype in set(eval_types_to_generate + [t.team_type for t in eval_types_to_read_from_file])}
            for layout_name in args.layout_names
    }

    train_collection = {
        layout_name: {ttype: [] for ttype in train_types}
        for layout_name in args.layout_names
    }

    for layout_name in args.layout_names:
        train_collection[layout_name][TeamType.SELF_PLAY] = [[]]
        eval_collection[layout_name][TeamType.SELF_PLAY] = [[]]
        train_collection[layout_name][TeamType.SELF_PLAY_ADVERSARY] = [[adversary]+self_teammates for adversary in adversarys]
        eval_collection[layout_name][TeamType.SELF_PLAY_ADVERSARY] = [[adversary]+self_teammates for adversary in adversarys]
        

    teammates_collection = {
        TeammatesCollection.TRAIN: train_collection,
        TeammatesCollection.EVAL: eval_collection
    }

    return teammates_collection