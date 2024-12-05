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
    args.num_players = 2

    args.layout = f'coordination_ring_dummy'
    args.p_idx = 0

    hmlda1_teammates_path = [
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack1/best', # green
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack1/best', # orange
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack1/best',
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack1/best',
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack1/best',
    ]
    hmlda1_teammates = [load_agent(Path(tm_path), args) for tm_path in hmlda1_teammates_path[:args.num_players - 1]]

    hmlda0_teammates_path = [
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack0/best', # green
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack0/best', # orange
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack0/best',
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack0/best',
        f'agent_models/DummyADV/{args.num_players}/PWADV-N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SPADV]_ran_originaler_attack0/best',
    ]
    hmlda0_teammates = [load_agent(Path(tm_path), args) for tm_path in hmlda0_teammates_path[:args.num_players - 1]]

    hmldsp_teammates_path = [
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SP]_ran_originaler/best', # green
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SP]_ran_originaler/best', # orange
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SP]_ran_originaler/best',
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SP]_ran_originaler/best',
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM_SP]_ran_originaler/best',
    ]
    hmldsp_teammates = [load_agent(Path(tm_path), args) for tm_path in hmldsp_teammates_path[:args.num_players - 1]]

    hmld_teammates_path = [
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM]_ran_originaler/best', # green
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM]_ran_originaler/best', # orange
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM]_ran_originaler/best',
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM]_ran_originaler/best',
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDUM]_ran_originaler/best',
    ]
    hmld_teammates = [load_agent(Path(tm_path), args) for tm_path in hmld_teammates_path[:args.num_players - 1]]

    hml_teammates_path = [
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best', # green
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best', # orange
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
        f'agent_models/DummyADV/{args.num_players}/N-1-SP_s1010_h256_tr[SPH_SPM_SPL]_ran_originaler/best',
    ]

    hml_teammates = [load_agent(Path(tm_path), args) for tm_path in hml_teammates_path[:args.num_players - 1]]

    sp_teammates_path = [
        f'agent_models/DummyADV/{args.num_players}/SP_hd64_seed0/best', # green
        f'agent_models/DummyADV/{args.num_players}/SP_hd64_seed0/best', # orange
        f'agent_models/DummyADV/{args.num_players}/SP_hd64_seed0/best',
        f'agent_models/DummyADV/{args.num_players}/SP_hd64_seed0/best',
        f'agent_models/DummyADV/{args.num_players}/SP_hd64_seed0/best',
    ]

    sp_teammates = [load_agent(Path(tm_path), args) for tm_path in sp_teammates_path[:args.num_players - 1]]

    # player_path = 'agent_models/ALMH_CUR/2/SP_hd64_seed14/best'
    # player = load_agent(Path(player_path), args)

    player = 'human' # blue
    # player = sp_teammates[0]

    dc = OvercookedGUI(args, agent=player, teammates=hmlda1_teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10, horizon=400)
    dc.on_execute()
