from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance, TeamType, TeammatesCollection, AdversaryPlayConfig

from itertools import permutations
import random
from pathlib import Path


def get_teammates(agents_perftag_score:list, teamtypes:list, teammates_len:int, unseen_teammates_len:int, agent:RLAgentTrainer=None, use_entire_population:bool=False):
    '''
    Get the teammates for an agent to populate a TeammatesCollection

    :param agents_perftag_score: The population of agents to select from represented as a list of tuples (agent, performance tag, score for this layout)
    :param teamtypes: The list of TeamTypes to use to select teammates for each team
    :param teammates_len: Total number teammates (i.e. totoal number of players - 1)
    :param unseen_teammates_len: For NXSP experiments, this is X, the number of unseen players that an SP agent is playing with
    :param agent: The agent that the teammates will play with
    :param use_entire_population: Flag inidicating if the entire population should be used, if False, this function will only generate one team for each teamtype
    :returns: Dictionary of form `{<TeamType> : [[team1], [team2], ... ] ...}` and a list of all the agents that were selected for a team
    '''

    if use_entire_population:

        # NOTE: Right now, this assumes that number of train_types is less than or equal to 3 and they are all unique,
        # If train types are duplicated, only 1 set of teammates will be provided e.g. if TrainTypes = [SPH, SPH, SPM]
        # The TC will only have one list for SPH
        # {layout_name: {'SPH' : [[team1], [team2]] }, 'SPM' : [...]}
        assert len(teamtypes) == len(set(teamtypes)), "Duplicate teamtypes detected, When using entire population to generate TC, the teamtypes should be unqiue"

        # If we want to use the entire population, we must check that the population is evenly divisible by the number of required agents
        required_population_size = 0
        for team_type in teamtypes:
            if (team_type in TeamType.ALL_TYPES_BESIDES_SP):
                required_population_size += teammates_len
            elif (team_type in [TeamType.SELF_PLAY_ADVERSARY, TeamType.SELF_PLAY_STATIC_ADV, TeamType.SELF_PLAY_DYNAMIC_ADV]):
                # Adversary teammates are not added in this function so we don't need to require the population to have an agent for this type
                continue
            elif (team_type in TeamType.SELF_PLAY_X_TYPES):
                required_population_size += unseen_teammates_len

        if required_population_size > 0:
            assert len(agents_perftag_score) % required_population_size == 0, \
                f"Requested use of entire population for teammate generation but provided population size is not evenly divisible by the minimum number of required agents\n"\
                f"Population size: {len(agents_perftag_score)}\n"\
                f"Minimum number of agents required for teammate generation: {required_population_size}\n"

    all_teammates = {
        teamtype: [] for teamtype in teamtypes
    }
    sorted_agents_perftag_score = sorted(agents_perftag_score, key=lambda x: x[2], reverse=True)

    if use_entire_population:

        # Split up the sorted list into "parts" so each part corresponds to a chunk of the population
        # E.g. Assuming 3 different categories of agents {'part_1' : [first 3rd of pop], 'part_2': [second 3rd of pop], 'part_3' : [third 3rd of pop]}
        num_perf_categories = len(AgentPerformance.ALL)

        # TODO: Update this to support any number of performance tags and train types
        assert num_perf_categories == 3, 'Current TC generation assumes only three performance types are present in the population'

        PERFORMANCE_LEVELS = ['H', 'M', 'L']
        # Number of elements in each partition
        part_size = len(agents_perftag_score) // num_perf_categories
        catagorized_agents = {
            PERFORMANCE_LEVELS[i] : sorted_agents_perftag_score[i*part_size : (i + 1) * part_size]
            for i in range(num_perf_categories)
        }

        for teamtype in teamtypes:
            if teamtype in TeamType.ALL_TYPES_BESIDES_SP:
                raise NotImplementedError(f'TeamType {teamtype} not yet supported when using full population to generate TC')

            elif teamtype == TeamType.SELF_PLAY:
                assert agent is not None
                all_teammates[teamtype] = [agent for _ in range(teammates_len)]

            elif teamtype == TeamType.SELF_PLAY_HIGH:
                assert agent is not None
                perf_level = PERFORMANCE_LEVELS[0]
                agents_itself = [agent for _ in range(teammates_len - unseen_teammates_len)]
                agent_perftag_score_of_category = catagorized_agents[perf_level]
                # Extract the agent from each tuple in the list
                agents_of_category = [a[0] for a in agent_perftag_score_of_category]
                all_teammates[teamtype] = [agents_of_category[i:i + unseen_teammates_len] + agents_itself for i in range(0, len(agents_of_category), unseen_teammates_len)]

            elif teamtype == TeamType.SELF_PLAY_MEDIUM:
                assert agent is not None
                perf_level = PERFORMANCE_LEVELS[1]
                agents_itself = [agent for _ in range(teammates_len - unseen_teammates_len)]
                agent_perftag_score_of_category = catagorized_agents[perf_level]
                # Extract the agent from each tuple in the list
                agents_of_category = [a[0] for a in agent_perftag_score_of_category]
                all_teammates[teamtype] = [agents_of_category[i:i + unseen_teammates_len] + agents_itself for i in range(0, len(agents_of_category), unseen_teammates_len)]

            elif teamtype == TeamType.SELF_PLAY_MIDDLE:
                raise NotImplementedError(f'TeamType {teamtype} not yet supported when using full population to generate TC')

            elif teamtype == TeamType.SELF_PLAY_LOW:
                assert agent is not None
                perf_level = PERFORMANCE_LEVELS[2]
                agents_itself = [agent for _ in range(teammates_len - unseen_teammates_len)]
                agent_perftag_score_of_category = catagorized_agents[perf_level]
                # Extract the agent from each tuple in the list
                agents_of_category = [a[0] for a in agent_perftag_score_of_category]
                all_teammates[teamtype] = [agents_of_category[i:i + unseen_teammates_len] + agents_itself for i in range(0, len(agents_of_category), unseen_teammates_len)]

    else:
        used_agents = set()  # To keep track of used agents

        for teamtype in teamtypes:
            available_agents = [sorted_aps for sorted_aps in sorted_agents_perftag_score if sorted_aps[0] not in used_agents]

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

    return all_teammates



def generate_TC(args,
                population,
                train_types,
                eval_types_to_generate,
                eval_types_to_read_from_file,
                agent=None,
                unseen_teammates_len=0,
                use_entire_population_for_train_types_teammates=False,
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

        agents_perftag_score_all = [(layout_agent,
                                     layout_agent.layout_performance_tags[layout_name],
                                     layout_agent.layout_scores[layout_name]) for layout_agent in layout_population]

        # Generate the train TC using the entire population of SP agents
        train_collection[layout_name] = get_teammates(agents_perftag_score=agents_perftag_score_all,
                                                      teamtypes=train_types,
                                                      teammates_len=args.teammates_len,
                                                      agent=agent,
                                                      unseen_teammates_len=unseen_teammates_len,
                                                      use_entire_population=use_entire_population_for_train_types_teammates
                                                      )

        # Generate the eval TC using the same population of agents that were used to generate the training TC
        # This is ok because these evaluation agents are used to find the best performance agent and for wandb plots
        # In this case though, we won't use the entire population, we'll just make sure that each TeamType has a team
        eval_collection[layout_name] = get_teammates(agents_perftag_score=agents_perftag_score_all,
                                                     teamtypes=eval_types_to_generate,
                                                     teammates_len=args.teammates_len,
                                                     agent=agent,
                                                     unseen_teammates_len=unseen_teammates_len,
                                                     use_entire_population=False
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
    all_agents = get_all_agents_for_layout(args.layout_names[0], population)
    agents_scores_averaged_over_layouts = []

    for agent in all_agents:
        scores = [agent.layout_scores[layout_name] for layout_name in args.layout_names]
        agents_scores_averaged_over_layouts.append((agent, sum(scores)/len(scores)))
    best_agent = max(agents_scores_averaged_over_layouts, key=lambda x: x[1])
    return best_agent[0]

def get_all_agents_for_layout(layout_name, population):
    all_agents = [agent for agent in population[layout_name]]
    return all_agents

def update_eval_collection_with_eval_types_from_file(args, agent, unseen_teammates_len, eval_types, eval_collection):
    for teammates in eval_types:
        if teammates.team_type not in eval_collection[teammates.layout_name]:
            eval_collection[teammates.layout_name][teammates.team_type] = []
        tms_path = RLAgentTrainer.get_model_path(base_dir=Path.cwd(), model_name=teammates.names[0])
        if teammates.load_from_pop_structure:
            layout_population, _, _ = RLAgentTrainer.load_agents(args, path=tms_path, tag=teammates.tags[0])
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
                    agents, _, _ = RLAgentTrainer.load_agents(args, name=name, path=tms_path, tag=tag)
                except FileNotFoundError as e:
                    print(f'Could not find saved {name} agent \nFull Error: {e}')
                    agents = []
                if agents:
                    group.append(agents[0])
            if len(group) == args.teammates_len:
                eval_collection[teammates.layout_name][teammates.team_type].append(group)
                print("Loaded agents from files for eval: ", teammates.names, ", Teamtype: ", teammates.team_type)


def generate_TC_for_ADV_agent(args, agent_to_be_attacked, teamtype):
    '''
        For when we train the adversary
    '''
    teammates = [agent_to_be_attacked for _ in range(args.teammates_len)]
    collection_template = {
        layout_name: {teamtype: [teammates]} for layout_name in args.layout_names }
    teammates_collection = {
        TeammatesCollection.TRAIN: collection_template,
        TeammatesCollection.EVAL: collection_template.copy()
    }
    return teammates_collection


def update_TC_w_ADV_teammates(args, teammates_collection, adversaries, primary_agent, adversary_play_config):
    '''
        For when we train a primary agent with adversary teammates
    '''
    self_teammates = [primary_agent for _ in range(args.teammates_len-1)]
    if adversary_play_config == AdversaryPlayConfig.SAP:
        teammates = [[adversaries[-1]] + self_teammates]
    if adversary_play_config == AdversaryPlayConfig.MAP:
        teammates = [[adversary]+self_teammates for adversary in adversaries]

    for layout_name in args.layout_names:
        teammates_collection[TeammatesCollection.TRAIN][layout_name][TeamType.SELF_PLAY_ADVERSARY] = teammates
        teammates_collection[TeammatesCollection.EVAL][layout_name][TeamType.SELF_PLAY_ADVERSARY] = teammates
    return teammates_collection


def update_TC_w_dynamic_and_static_ADV_teammates(args, train_types, eval_types, teammates_collection, primary_agent, adversaries):
    itself = [primary_agent for _ in range(args.teammates_len-1)]

    for layout_name in args.layout_names:
        if TeamType.SELF_PLAY_STATIC_ADV in train_types:
            static_advs = adversaries[TeamType.SELF_PLAY_STATIC_ADV]
            teammates_collection[TeammatesCollection.TRAIN][layout_name][TeamType.SELF_PLAY_STATIC_ADV] = [[static_advs[i]] + itself for i in range(len(static_advs))]

        if TeamType.SELF_PLAY_STATIC_ADV in eval_types:
            static_advs = adversaries[TeamType.SELF_PLAY_STATIC_ADV]
            teammates_collection[TeammatesCollection.EVAL][layout_name][TeamType.SELF_PLAY_STATIC_ADV] = [[static_advs[i]] + itself for i in range(len(static_advs))]

        if TeamType.SELF_PLAY_DYNAMIC_ADV in train_types:
            dyn_advs = adversaries[TeamType.SELF_PLAY_DYNAMIC_ADV]
            teammates_collection[TeammatesCollection.TRAIN][layout_name][TeamType.SELF_PLAY_DYNAMIC_ADV] = [[dyn_advs[i]] + itself for i in range(len(dyn_advs))]

        if TeamType.SELF_PLAY_DYNAMIC_ADV in eval_types:
            dyn_advs = adversaries[TeamType.SELF_PLAY_DYNAMIC_ADV]
            teammates_collection[TeammatesCollection.EVAL][layout_name][TeamType.SELF_PLAY_DYNAMIC_ADV] = [[dyn_advs[i]] + itself for i in range(len(dyn_advs))]

    return teammates_collection
