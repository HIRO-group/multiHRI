from pathlib import Path

from oai_agents.agents.agent_utils import DummyAgent, load_agent
from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI


def get_teammate_from_pop_file(tm_name, tm_score, pop_path, layout_name):
    population, _, _ = RLAgentTrainer.load_agents(args, path=Path(pop_path), tag='last')
    for tm in population:
        if tm.layout_scores[layout_name] == tm_score and tm.name == tm_name:
            return tm


if __name__ == "__main__":
    args = get_arguments()
    args.num_players = 3
    args.layout = f'selected_{args.num_players}_chefs_counter_circuit'
    args.p_idx = 0

    teammates_path = [
        'agent_models/ALMH_CUR/2/SP_hd64_seed14/best', # green 
        'agent_models/ALMH_CUR/2/SP_hd64_seed14/best', # orange
        'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
        'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
        'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
    ]

    teammates = [load_agent(Path(tm_path), args) for tm_path in teammates_path[:args.num_players - 1]]

    player_path = 'agent_models/ALMH_CUR/2/SP_hd64_seed14/best'
    player = load_agent(Path(player_path), args)

    # player = 'human' # blue

    dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10, horizon=400)
    dc.on_execute()
