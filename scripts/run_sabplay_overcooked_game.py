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
    fc = '3_chefs_forced_coordination_3OP2S1D'
    sk = '3_chefs_small_kitchen_two_resources'
    asy = '3_chefs_asymmetric_advantages'
    cc = '3_chefs_counter_circuit'
    args.layout = sk
    args.p_idx = 0

    sup_path = 'agent_models/four-layouts/supporter/0/sp_s68_h512_tr(SP)_ran/best'
    sab_path = 'agent_models/four-layouts/saboteur/0/sab_s68_h512_tr(H)_ran/best'
    sab1_path = 'agent_models/four-layouts/saboteur/1/sab_s68_h512_tr(H)_ran/best'
    fish_path = 'agent_models/four-layouts/selfisher/0/sab_s68_h512_tr(H)_ran/best'
    fish2_path = 'agent_models/four-layouts/selfisher/2/sab_s68_h512_tr(H)_ran/best'
    fish4_path = 'agent_models/four-layouts/selfisher/4/sab_s68_h512_tr(H)_ran/best'
    sabplay_path = 'agent_models/four-layouts/supporter-sabplay/0/pwsab_s68_h512_tr(SP_SPH)_ran/best'
    sabplay1_path = 'agent_models/four-layouts/supporter-sabplay/1/pwsab_s68_h512_tr(SP_SPH)_ran/best'
    fishplay1_path = 'agent_models/four-layouts/supporter-fishplay/1/pwsab_s68_h512_tr(SP_SPH)_ran/best'
    fishplay2_path = 'agent_models/four-layouts/supporter-fishplay/2/pwsab_s68_h512_tr(SP_SPH)_ran/best'
    fishplay3_path = 'agent_models/four-layouts/supporter-fishplay/3/pwsab_s68_h512_tr(SP_SPH)_ran/best'
    fishplay4_path = 'agent_models/four-layouts/supporter-fishplay/4/pwsab_s68_h512_tr(SP_SPH)_ran/best'
    fishplay024_14_path = 'agent_models/four-layouts/supporter-fishplay/024_14/pwsab_s68_h512_tr(SP_SPH)_ran/best'

    
    
    orange = load_agent(Path(sup_path), args)
    green = load_agent(Path(sup_path), args)
    # blue = load_agent(Path(fishplay4_path), args)

    teammates = [orange, green]

    blue = 'human'


    dc = OvercookedGUI(args, agent=blue, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
                       horizon=400)
    dc.on_execute()
    print(dc.trajectory)