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

    two_chefs_layouts = [
        'selected_2_chefs_coordination_ring',
        'selected_2_chefs_counter_circuit',
        'selected_2_chefs_cramped_room'
    ]

    args = get_arguments(additional_args)
    args.num_players = 2
    layout_id = 2
    args.layout = two_chefs_layouts[layout_id]
    args.p_idx = 0

    path_list = [
        # 'agent_models/Final/eval_2/adv_reused_sp/PWADV-N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPADV_SP)_ran_originaler_attack2/PWADV-N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPADV_SP)_ran_originaler_attack2/best',
        # 200, 220, 320
        'agent_models/Final/eval_2/FCP_s2020_h256_tr(AMX)_ran/best', # 200, 240, 480 // 220, 220, 460
        # 'agent_models/Final/eval_2/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_cur/best', # 220, 200, 360
        # 'agent_models/Final/eval_2/N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL)_ran/best', # 200, 220, 420
        # 'agent_models/Final/eval_2/PWADV-N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPADV)_cur_supporter_attack0/best', #
        # 'agent_models/Final/eval_2/PWADV-N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPADV)_cur_supporter_attack1/best', #
        # 'agent_models/Final/eval_2/PWADV-N-1-SP_s1010_h256_tr(SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPADV)_cur_supporter_attack2/best', # 220, 240, 400
        'agent_models/Final/eval_2/SP_hd64_seed14/best', # 200, 220, 440
        # 'agent_models/Final/eval_2/SP_hd256_seed13/best', # 200, 160, 420
        'agent_models/Final/eval_2/PWADV-N-1-SP_s1010_h256_tr[SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPADV]_cur_originaler_attack2/best', # 220, 220, 460 // 240, 240, 460
        'agent_models/Final/eval_2/SP_hd64_seed14_MAP/originaler-selfisherplay/2/pwadv_s14_h64_tr(SP_SPADV)_ran/best' # 240, 220, 420
    ]
    for layout in two_chefs_layouts:
        args.layout = layout
        for tm_path in path_list:
            agent =  load_agent(Path(tm_path), args)

            orange = agent
            green = agent
            teammates = [orange, green]

            blue = agent
            blue = 'human'

            dc = OvercookedGUI(args, agent=blue, teammates=teammates[:args.num_players-1], layout_name=args.layout, p_idx=args.p_idx, fps=10,
                            horizon=400)
            dc.on_execute()
            # print(dc.trajectory)