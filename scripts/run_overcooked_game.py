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
    args.layout ='selected_2_chefs_coordination_ring'
    args.p_idx = 1

    # tm_path = 'agent_models/Result/Eval/2/SP_hd64_seed11/ck_0'
    tm_path = 'agent_models/Result/Eval/2/SP_hd64_seed11/ck_2_rew_108.66666666666667'
    # tm_path = 'agent_models/Result/Eval/2/SP_hd64_seed11/best'
    green =  load_agent(Path(tm_path), args)
    teammates = [green]

    tm_path = 'agent_models/ALMH_CUR/2/PWADV-N-1-SP_s1010_h256_tr[SPH_SPH_SPH_SPH_SPM_SPM_SPM_SPM_SPL_SPL_SPL_SPL_SPADV]_cur_originaler_attack0/best'
    # tm_path = 'agent_models/Result/2/SP_hd64_seed14/best'
    blue =  load_agent(Path(tm_path), args)

    dc = OvercookedGUI(args, agent=blue, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
                       horizon=400)
    dc.on_execute()
    # print(dc.trajectory)