from pathlib import Path

from oai_agents.agents.agent_utils import DummyAgent, load_agent, CustomAgent
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
    args.num_players = 1

    'dec_5_chefs_counter_circuit',
    'dec_5_chefs_storage_room',
    'dec_5_chefs_secret_heaven',
    'selected_5_chefs_spacious_room_no_counter_space',


    # args.layout = f'dec_5_chefs_counter_circuit'
    args.p_idx = 0
    args.n_envs = 1

    if args.num_players == 1:
        args.layout = 'storage_room_single_right_sym_blocked'
        args.layout_names = [args.layout]
        player_path = 'agent_models/storage_room_1_chef_layouts/1/SP_s13_h256_tr[SP]_ran/best'
        player = load_agent(Path(player_path), args)
        teammates = []
    elif args.num_players == 2:
        args.layout = 'storage_room'
        args.layout_names = [args.layout]
        player_path = 'agent_models/Complex/2/SP_s1010_h256_tr[SP]_ran/best'
        player = load_agent(Path(player_path), args)
        teammates = [player]

    # teammates_path = [
    #     'agent_models/Classic/2/SP_s1010_h256_tr[SP]_ran/ck_0'
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best', # green
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best', # orange
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
    #     'agent_models/ALMH_CUR/2/SP_hd64_seed14/best',
    # ]

    # teammates = [load_agent(Path(tm_path), args) for tm_path in teammates_path[:args.num_players - 1]]

    # trajectories = tile locations. Top left of the layout is (0, 0), bottom right is (M, N)
    # teammates = [CustomAgent(args=args, name='human', trajectories={args.layout: [(8, 1), (8, 2), (7, 2), (6, 2)]})]
    # teammates = [DummyAgent(action='random') for _ in range(args.num_players - 1)]

    # player_path = 'agent_models/storage_room_1_chef_layouts/1/random_positions/SP_s2602_h256_tr[SP]_ran/best'
    # player = load_agent(Path(player_path), args)
    # teammates = []
    # player = 'human' # blue
    # player = DummyAgent(action='random')

    # dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
    #                     horizon=400, gif_name=args.layout)
    dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10,
                        horizon=400, gif_name=args.layout)
    dc.on_execute()
