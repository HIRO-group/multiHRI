from pathlib import Path

from oai_agents.agents.agent_utils import DummyAgent, load_agent
from oai_agents.agents.hrl import HierarchicalRL
from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.human_agents import HumanManagerHRL, HumanPlayer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI


if __name__ == "__main__":
    """
    Sample commands
    python scripts/run_overcooked_game.py --agent human --teammate agent_models/HAHA
    """
    additional_args = [
        ('--agent', {'type': str, 'default': 'human', 'help': '"human" to used keyboard inputs or a path to a saved agent'}),
        ('--teammate', {'type': str, 'default': 'agent_models/HAHA', 'help': 'Path to saved agent to use as teammate'}),
        ('--layout', {'type': str, 'default': 'counter_circuit_o_1order', 'help': 'Layout to play on'}),
        ('--p-idx', {'type': int, 'default': 0, 'help': 'Player idx of agent (teammate will have other player idx), Can be 0 or 1.'})
    ]


    args = get_arguments(additional_args)

    # args.layout = 'counter_circuit_o_1order'
    args.layout = 'n_players'
    args.p_idx = 0

    # bc, human_proxy = BehavioralCloningTrainer.load_bc_and_human_proxy(args, name=f'bc_{args.layout}')
    # agent, tm = bc, human_proxy
    # teammate = bc
    # worker = load_agent(Path('agent_models/worker_bcp/'), args)
    # tm = load_agent(Path('agent_models/HAHA_bcp_bcp'), args)
    # agent = HumanManagerHRL(worker, args)


    tm1 = load_agent(Path('/home/ava/Research/Codes/HMT/HAHA_fcp_219'), args)
    tm2 = load_agent(Path('/home/ava/Research/Codes/HMT/HAHA_fcp_219'), args)
    teammates = [tm1, tm2]
    agent = 'human'

    dc = OvercookedGUI(args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
                       horizon=100)
    dc.on_execute()
    print(dc.trajectory)
