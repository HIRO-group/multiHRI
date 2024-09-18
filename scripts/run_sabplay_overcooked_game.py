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
    s2fp = 'S2FP-asy-sk-cc/'
    four_layouts = 'four-layouts/'
    agent_models = 'agent_models/'
    exp = s2fp

    supporter = 'supporter/'
    saboteur = 'saboteur/'
    selfisher = 'selfisher/'
    supporter_fishplay = 'supporter-fishplay/'

    sup_name = 'sp_s68_h512_tr(SP)_ran/'
    sab_name = 'sab_s68_h512_tr(H)_ran/'
    fish_name = 'sab_s68_h512_tr(H)_ran/'
    sabplay_name = 'pwsab_s68_h512_tr(SP_SPSAB)_ran/'
    fishplay_name = 'pwsab_s68_h512_tr(SP_SPSAB)_ran/'

    best = 'best'
    worst = 'worst'
    last = 'aamas25'

    id = str(25) + '/'
    s2fp_fishplay_path = agent_models + exp + supporter_fishplay + id + fishplay_name + best

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

    asc_fish0_path = 'agent_models/asy-sk-cc/selfisher/0/sab_s68_h512_tr(H)_ran/aamas25'
    asc_fish1_path = 'agent_models/asy-sk-cc/selfisher/1/sab_s68_h512_tr(H)_ran/aamas25'
    asc_fish2_path = 'agent_models/asy-sk-cc/selfisher/1/sab_s68_h512_tr(H)_ran/aamas25'
    asc_fishplay0_path = 'agent_models/asy-sk-cc/supporter-fishplay/0/pwsab_s68_h512_tr(SP_SPSAB)_ran/best'
    asc_fishplay1_path = 'agent_models/asy-sk-cc/supporter-fishplay/1/pwsab_s68_h512_tr(SP_SPSAB)_ran/best'
    asc_fishplay2_path = 'agent_models/asy-sk-cc/supporter-fishplay/2/pwsab_s68_h512_tr(SP_SPSAB)_ran/best'
    asc_fishplay2_extend_path = 'agent_models/asy-sk-cc/supporter-fishplay/2_extend/pwsab_s68_h512_tr(SP_SPSAB)_ran/best'
    asc_18_fishplay2_extend_path = 'agent_models/asy-sk-cc-18/supporter-fishplay/2_extend/pwsab_s68_h512_tr(SP_SPSAB)_ran/best'
    asc_18_fishplay2_extend_path = 'agent_models/asy-sk-cc-18/supporter-fishplay/2_extend/pwsab_s68_h512_tr(SP_SPSAB)_ran/best'
    asc_18_best_fishplay1_path = 'agent_models/asy-sk-cc-18-best/supporter-fishplay/1/pwsab_s68_h512_tr(SP_SPSAB)_ran/best'
    
    agent = load_agent(Path(s2fp_fishplay_path), args)
    tester = load_agent(Path(s2fp_fishplay_path), args)
    tester = 'human'
    
    orange = agent
    green = agent
    blue = tester

    teammates = [orange, green]


    dc = OvercookedGUI(args, agent=blue, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
                       horizon=400)
    dc.on_execute()
    print(dc.trajectory)