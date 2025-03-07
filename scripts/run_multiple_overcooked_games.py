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

    args.p_idx = 0
    args.n_envs = 1


    seeds = [13, 68, 105, 128, 1010, 2020, 2602, 2907]

    exps = [
        'fixed_positions_final_sparse_ratio_05_sym_nonblocked',
        'fixed_positions_final_sparse_ratio_05_sym_eitherblocked',
        'fixed_positions_final_sparse_ratio_10_sym_nonblocked',
        'fixed_positions_final_sparse_ratio_10_sym_eitherblocked',
    ]
    for exp in exps:
        print(f"{exp} is simulating!")
        for seed in seeds:
            args.layout = 'storage_room_single_right_sym_leftpotblocked'
            args.layout_names = [args.layout]
            player_path = f'agent_models/storage_room_1_chef_layouts/1/{exp}/SP_s{seed}_h256_tr[SP]_ran/best'
            player = load_agent(Path(player_path), args)
            teammates = []
            print(f'seed: {seed}')
            print(f'layout:{args.layout}')
            dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10000,
                                horizon=400, gif_name=args.layout)
            dc.on_execute()

            args.layout = 'storage_room_single_left_sym_leftpotblocked'
            args.layout_names = [args.layout]
            print(f'layout:{args.layout}')
            dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10000,
                                horizon=400, gif_name=args.layout)
            dc.on_execute()

            args.layout = 'storage_room_single_right_sym_rightpotblocked'
            args.layout_names = [args.layout]
            print(f'layout:{args.layout}')
            dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10000,
                                horizon=400, gif_name=args.layout)
            dc.on_execute()

            args.layout = 'storage_room_single_left_sym_rightpotblocked'
            args.layout_names = [args.layout]
            print(f'layout:{args.layout}')
            dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10000,
                                horizon=400, gif_name=args.layout)
            dc.on_execute()

            args.layout = 'storage_room_single_right_sym'
            args.layout_names = [args.layout]
            print(f'layout:{args.layout}')
            dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10000,
                                horizon=400, gif_name=args.layout)
            dc.on_execute()

            args.layout = 'storage_room_single_left_sym'
            args.layout_names = [args.layout]
            print(f'layout:{args.layout}')
            dc = OvercookedGUI(args, agent=player, teammates=teammates, layout_name=args.layout, p_idx=args.p_idx, fps=10000,
                                horizon=400, gif_name=args.layout)
            dc.on_execute()



