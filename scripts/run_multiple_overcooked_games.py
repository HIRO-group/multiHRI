import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from oai_agents.agents.agent_utils import load_agent
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI


seeds = [13, 68, 105, 128, 1010, 2020, 2602, 2907]

class Experiment:
    def __init__(self, exps_folder: str, methods: List[str], layouts: List[str], num_players: int, exp_seeds: List[int]):
        self.exps_folder = exps_folder
        self.methods = methods
        self.layouts = layouts
        self.num_players = num_players
        self.seeds = exp_seeds


class StorageRoomSingle(Experiment):
    def __init__(self):
        super().__init__(
            exps_folder='storage_room_1_chef_layouts/1',
            methods=[
                'fixed_positions_final_sparse_ratio_05_sym_eitherblocked',
                'fixed_positions_final_sparse_ratio_10_sym_nonblocked',
                'fixed_positions_final_sparse_ratio_10_sym_eitherblocked',
            ],
            layouts=[
                'storage_room_single_right_sym_leftpotblocked',
                'storage_room_single_left_sym_leftpotblocked',
                'storage_room_single_right_sym_rightpotblocked',
                'storage_room_single_left_sym_rightpotblocked',
                'storage_room_single_right_sym',
                'storage_room_single_left_sym'
            ],
            num_players=1,
            exp_seeds=seeds,
        )


class CrampedRoomSingleOT(Experiment):
    def __init__(self):
        super().__init__(
            exps_folder='cramped_room_ot_1_chef_layouts/1',
            methods=['unblocking', 'blocking'],
            layouts=[
                'cramped_room_single_ot_oblocked',
                'cramped_room_single_ot_tblocked',
                'cramped_room_single_ot',
            ],
            num_players=1,
            exp_seeds=seeds,
        )

class StoragedRoomPair(Experiment):
    def __init__(self):
        super().__init__(
            exps_folder='storage_room_2_chef_layouts/2',
            methods=[
                'dense_reward_unblocking',
                'dense_reward_blocking',
                'sparse_reward_blocking'
            ],
            layouts=[
                'storage_room_pair_right_sym_leftpotblocked',
                'storage_room_pair_left_sym_leftpotblocked',
                'storage_room_pair_right_sym_rightpotblocked',
                'storage_room_pair_left_sym_rightpotblocked',
                'storage_room_pair_right_sym',
                'storage_room_pair_left_sym'
            ],
            num_players=2,
            exp_seeds=seeds,
        )


def run_experiment(exp: Experiment) -> pd.DataFrame:
    """Runs the experiment and collects data in a DataFrame."""
    data = []  # List to store experiment results

    for method in exp.methods:
        print(f"{method} is simulating!")
        for seed in exp.seeds:
            print(f'seed: {seed}')
            player_path = f'agent_models/{exp.exps_folder}/{method}/SP_s{seed}_h256_tr[SP]_ran/best'
            for layout in exp.layouts:
                args.layout = layout
                args.layout_names = [args.layout]

                # Load agent and run simulation
                player = load_agent(Path(player_path), args)
                teammates = [player for i in range(exp.num_players-1)]
                print(f'layout: {args.layout}')
                dc = OvercookedGUI(
                    args,
                    agent=player,
                    teammates=teammates,
                    layout_name=args.layout,
                    p_idx=args.p_idx,
                    fps=10000,
                    horizon=400,
                    gif_name=args.layout
                )
                dc.on_execute()

                # Store results in data list
                data.append({
                    "method": method,
                    "layout": args.layout,
                    "seed": seed,
                    "score": dc.score  # Assuming dc.score stores the game result
                })

    # Convert collected data to a Pandas DataFrame
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    args = get_arguments()
    args.p_idx = 0
    args.n_envs = 1

    # Choose your experiment
    exp: Experiment = StoragedRoomPair()
    args.num_players = exp.num_players

    # 1. Run experiment and store results in a DataFrame
    df = run_experiment(exp)

    # 2. Save the table to a CSV file
    csv_path = f"agent_models/{exp.exps_folder}/data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # 3. Generate a grouped bar chart showing mean ± std for each (method, layout)
    #    a) Group by (method, layout) and compute mean and std
    summary_df = df.groupby(["method", "layout"])["score"].agg(["mean", "std"]).reset_index()
    summary_df.rename(columns={"mean": "avg_score", "std": "std_dev"}, inplace=True)

    #    b) Extract unique layouts and methods
    layouts = sorted(summary_df["layout"].unique())
    methods = sorted(summary_df["method"].unique())

    #    c) Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(len(layouts))
    width = 0.8 / len(methods)  # Distribute bar widths evenly

    #    d) Plot bars with error bars for each method
    for i, method in enumerate(methods):
        # Filter summary data for this method, reindex by layout so the order matches
        method_data = summary_df[summary_df["method"] == method].set_index("layout").reindex(layouts).reset_index()

        # Offset each method so bars don't overlap
        x_offset = (i - (len(methods) - 1) / 2) * width

        ax.bar(
            x_positions + x_offset,
            method_data["avg_score"],
            yerr=method_data["std_dev"],  # ±1 std
            width=width,
            label=method,
            alpha=0.9,
            capsize=5,         # Error bar caps
            edgecolor="black"  # Optional: outline each bar
        )

    #    e) Label the x-axis with layouts
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layouts, rotation=15)
    ax.set_xlabel("Challenges (Layouts)")
    ax.set_ylabel("Average Score")
    ax.set_title("Average Scores over Seeds (±1 Std)")

    #    f) Legend and layout
    ax.legend(title="Methods", loc="best")
    plt.tight_layout()

    # 4. Save the plot to a file and/or show it
    plot_path = f"agent_models/{exp.exps_folder}/plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Optionally display the plot
    plt.show()
