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
    args.num_players = 2
    two_chefs_layouts = [
        'selected_2_chefs_coordination_ring',
        'selected_2_chefs_counter_circuit',
        'selected_2_chefs_cramped_room'
    ]
    args.layout = two_chefs_layouts[2]
    args.p_idx = 0

    tm_path = 'agent_models/Final/2/PWADV-N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPADV)_cur_originaler_attack2/best'
    agent =  load_agent(Path(tm_path), args)

    orange = agent
    green = agent
    teammates = [orange]

    blue = agent
    blue = 'human'

    dc = OvercookedGUI(args, agent=blue, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
                       horizon=400)
    dc.on_execute()
    print(dc.trajectory)