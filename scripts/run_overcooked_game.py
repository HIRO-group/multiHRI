from pathlib import Path

from oai_agents.agents.agent_utils import DummyAgent, load_agent
from oai_agents.agents.hrl import HierarchicalRL
from oai_agents.agents.il import BehavioralCloningTrainer
from oai_agents.agents.human_agents import HumanManagerHRL, HumanPlayer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI


if __name__ == "__main__":
    # TEAMMATE and POP(TODO): replace --teammate by --teammates after figuring out how to assign multiple teammates in an argument.
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
    args.num_players = 3
    args.layout = '3_chefs_forced_coordination_one_low_pot'
    args.p_idx = 0

    # agent_models/fcp_hd64_seed0/best
    # agent_models/Backup/fcp_pop_3_chefs_forced_coordination/agent_dir/agent_0
    # agent_models/fcp_hd64_seed0/best
    tm1 = load_agent(Path('agent_models/fcp_hd64_seed0/best'), args)
    tm2 = load_agent(Path('agent_models/fcp_hd64_seed0/best'), args)
    teammates = [tm1, tm2]

    agent = load_agent(Path('agent_models/fcp_hd64_seed0/best'), args)
    # agent = 'human'

    dc = OvercookedGUI(args, agent=agent, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
                       horizon=200)
    dc.on_execute()
    print(dc.trajectory)
