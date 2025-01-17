from pathlib import Path

from oai_agents.agents.agent_utils import DummyAgent, load_agent
from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI
from scripts.utils.agents_finder import AgentsFinder, SelfPlayAgentsFinder, AgentsFinderBySuffix, AMMAS23AgentsFinderBySuffix
from oai_agents.common.learner import LearnerType
from oai_agents.common.tags import KeyCheckpoints, TeamType


def get_teammate_from_pop_file(tm_name, tm_score, pop_path, layout_name):
    population, _, _ = RLAgentTrainer.load_agents(args, path=Path(pop_path), tag='last')
    for tm in population:
        if tm.layout_scores[layout_name] == tm_score and tm.name == tm_name:
            return tm


if __name__ == "__main__":
    args = get_arguments()
    args.encoding_fn = 'OAI_contexted_egocentric'
    print("args.encoding_fn", args.encoding_fn)
    print("args.encoding_fn", args.encoding_fn)
    print("args.encoding_fn", args.encoding_fn)
    print("args.encoding_fn", args.encoding_fn)
    print("args.encoding_fn", args.encoding_fn)
    args.num_players = 2
    # args.layout = f'selected_{args.num_players}_chefs_counter_circuit'
    args.layout = 'forced_coordination'
    args.p_idx = 0
    '''
    Define a suffix parameter to place a suffix, which are used by an agent folder or multiple of them, and
    then let the agent finder to use that suffix as  and
    then it will find all agents under this suffix
    '''
    suffix = f"tr[SPH_SPM_SPL]_ran_{LearnerType.ORIGINALER}"
    fcp_suffix = "N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler"
    advp_suffix = "N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPADV]_ran_originaler"
    f2p_suffix = "N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPFO]_ran_originaler"
    human_proxy_suffix = f"bc_{args.layout}"

    args.exp_dir = "Classic/2"

    sp_agent_finder = SelfPlayAgentsFinder(args=args)
    sp_agents = sp_agent_finder.get_agents(tag=KeyCheckpoints.BEST_EVAL_REWARD)
    # agent_finder = AgentsFinderBySuffix(args=args)
    # agents = agent_finder.get_agents(key=fcp_suffix, tag=KeyCheckpoints.BEST_EVAL_REWARD)
    # human_proxy_finder = AMMAS23AgentsFinderBySuffix(args=args)
    # human_proxies = human_proxy_finder.get_agents(key=human_proxy_suffix, tag=KeyCheckpoints.BEST_EVAL_REWARD)
    teammates = [sp_agents[0]]
    team_type = TeamType.SELF_PLAY_ADVERSARY

    # teammates_path = [
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best', # green
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best', # orange
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
    # ]

    # teammates = [load_agent(Path(tm_path), args) for tm_path in teammates_path[:args.num_players - 1]]

    # player_path = 'agent_models/ALMH_CUR/2/SP_hd64_seed14/best'
    # player = load_agent(Path(player_path), args)

    # player = sp_agents[0] # blue
    player = "human"
    dc = OvercookedGUI(args, agent=player, teammates=teammates, team_type=team_type, layout_name=args.layout, p_idx=args.p_idx, fps=10, horizon=400)
    dc.on_execute()
