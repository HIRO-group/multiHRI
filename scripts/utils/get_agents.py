from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.tags import AgentPerformance

def get_layoutnames_from_cklist(ck_list):
    assert ck_list is not None
    scores, _, _ = ck_list[0]
    assert isinstance(scores, dict)
    return list(ck_list.keys())

def get_HML_agents_by_layout(args, ck_list, layout_name):
    '''
    categorizes agents using performance tags based on the checkpoint list
        AgentPerformance.HIGH
        AgentPerformance.MEDIUM
        AgentPerformance.LOW
    It categorizes by setting their score and performance tag:
        OAIAgent.layout_scores
        OAIAgent.layout_performance_tags
    returns three agents with three different performance
    '''
    if len(ck_list) < len(AgentPerformance.ALL):
        raise ValueError(
            f'Must have at least {len(AgentPerformance.ALL)} checkpoints saved. \
            Currently is: {len(ck_list)}. Increase ck_rate or training length'
        )

    all_score_path_tag_sorted = []
    for scores, path, tag in ck_list:
        all_score_path_tag_sorted.append((scores[layout_name], path, tag))
    all_score_path_tag_sorted.sort(key=lambda x: x[0], reverse=True)

    highest_score = all_score_path_tag_sorted[0][0]
    lowest_score = all_score_path_tag_sorted[-1][0]
    middle_score = (highest_score + lowest_score) // 2

    high_score_path_tag = all_score_path_tag_sorted[0]
    medium_score_path_tag = RLAgentTrainer.find_closest_score_path_tag(middle_score, all_score_path_tag_sorted)
    low_score_path_tag = all_score_path_tag_sorted[-1]

    H_agents = RLAgentTrainer.get_agents_and_set_score_and_perftag(args, layout_name, high_score_path_tag, AgentPerformance.HIGH, ck_list=ck_list)
    M_agents = RLAgentTrainer.get_agents_and_set_score_and_perftag(args, layout_name, medium_score_path_tag, AgentPerformance.MEDIUM, ck_list=ck_list)
    L_agents = RLAgentTrainer.get_agents_and_set_score_and_perftag(args, layout_name, low_score_path_tag, AgentPerformance.LOW, ck_list=ck_list)

    return H_agents, M_agents, L_agents

def get_HML_agents(ck_list):
    layout_names = get_layoutnames_from_cklist(ck_list=ck_list)
    for layout_name in layout_names:
        RLAgentTrainer.get_HML_agents_by_layout(
            ck_list=ck_list,
            layout_name=layout_name
        )