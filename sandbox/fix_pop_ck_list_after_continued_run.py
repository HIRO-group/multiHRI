import os
import shutil
import re

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, KeyCheckpoints
from oai_agents.common.learner import LearnerType
from oai_agents.common.tags import KeyCheckpoints

def fix_ck_list(initial_run_root, continued_run_root, corrected_run_root, ck_starts_from):
    ck_regex = re.compile(r'ck_(\d+)(.*)')
    pop_regex = re.compile(r'pop\w+')


    for sp_folder in os.listdir(initial_run_root):
        initial_SP_path = os.path.join(initial_run_root, sp_folder)
        continued_SP_path = os.path.join(continued_run_root, sp_folder)
        corrected_SP_path = os.path.join(corrected_run_root, sp_folder)

        if os.path.isdir(initial_SP_path):
            os.makedirs(corrected_SP_path, exist_ok=True)
            for SP_tag in os.listdir(initial_SP_path):
                if pop_regex.match(sp_folder) or SP_tag == 'best' or SP_tag == KeyCheckpoints.MOST_RECENT_TRAINED_MODEL: # does not copy the content of pop as well
                    print(f"Skipping {SP_tag} in {initial_SP_path}")
                    continue
                else:
                    initial_SP_tag_folder = os.path.join(initial_SP_path, SP_tag)
                    corrected_SP_tag_folder = os.path.join(corrected_SP_path, SP_tag)
                    if os.path.isdir(initial_SP_tag_folder):
                        shutil.copytree(initial_SP_tag_folder, corrected_SP_tag_folder)
                        print(f"Copied {SP_tag} from {initial_SP_path} to {corrected_SP_path}")


            if os.path.isdir(continued_SP_path):
                for SP_tag in os.listdir(continued_SP_path):
                    if not pop_regex.match(sp_folder) and (SP_tag == 'best' or SP_tag == KeyCheckpoints.MOST_RECENT_TRAINED_MODEL):
                        continued_SP_tag_folder = os.path.join(continued_SP_path, SP_tag)
                        corrected_SP_tag_folder = os.path.join(corrected_SP_path, SP_tag)
                        if os.path.isdir(continued_SP_tag_folder):
                            shutil.copytree(continued_SP_tag_folder, corrected_SP_tag_folder)
                            print(f"Copied {SP_tag} from {continued_SP_path} to {corrected_SP_path}")

                    elif os.path.exists(os.path.join(corrected_SP_path, SP_tag)):
                        print(f"{SP_tag} already exists in {corrected_SP_path}, skipping...")
                        continue

                    match = ck_regex.match(SP_tag)
                    continued_SP_tag_folder = os.path.join(continued_SP_path, SP_tag)
                    if match and os.path.isdir(continued_SP_tag_folder):
                        ck_number = int(match.group(1))
                        suffix = match.group(2)
                        new_ck_number = ck_number + ck_starts_from
                        new_ck_folder = f"ck_{new_ck_number}{suffix}"
                        new_ck_path_with_sum = os.path.join(corrected_SP_path, new_ck_folder)

                        if not os.path.exists(new_ck_path_with_sum):
                            shutil.copytree(continued_SP_tag_folder, new_ck_path_with_sum)
                            print(f"Copied {SP_tag} from {continued_SP_path} to {new_ck_folder} in {corrected_SP_path}")


def fix_pop(args, initial_run_root, continued_run_root, corrected_run_root):
    initial_run_exp = re.search(r'agent_models/(.*)', initial_run_root).group(1)
    continued_run_exp = re.search(r'agent_models/(.*)', continued_run_root).group(1)
    corrected_run_exp = re.search(r'agent_models/(.*)', corrected_run_root).group(1)

    population_initial = {layout_name: [] for layout_name in args.layout_names}
    population_continued = {layout_name: [] for layout_name in args.layout_names}

    for layout_name in args.layout_names:
        name = f'pop_{layout_name}'
        args.exp_dir = initial_run_exp
        population_initial[layout_name] = RLAgentTrainer.load_agents(args, name=name, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)
        print(f"Loaded {name} in {initial_run_exp}, size: {len(population_initial[layout_name])}")

        args.exp_dir = continued_run_exp
        population_continued[layout_name] = RLAgentTrainer.load_agents(args, name=name, tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)
        print(f"Loaded {name} in {continued_run_exp}, size: {len(population_continued[layout_name])}")

        all_agents = population_initial[layout_name] + population_continued[layout_name]

        rt = RLAgentTrainer(
            name=f'{name}',
            args=args,
            agent=None,
            teammates_collection={},
            train_types=[TeamType.SELF_PLAY],
            eval_types=[TeamType.SELF_PLAY],
            epoch_timesteps=args.epoch_timesteps,
            n_envs=args.n_envs,
            seed=None,
        )
        rt.agents = all_agents
        args.exp_dir = corrected_run_exp
        rt.save_agents(tag=KeyCheckpoints.MOST_RECENT_TRAINED_MODEL)
        print(f"Saved {name} in {corrected_run_root}, size: {len(all_agents)}")


def set_input():
    args = get_arguments()
    args.teammates_len = 4
    args.num_players = args.teammates_len + 1
    args.layout_names = ['selected_5_chefs_counter_circuit',
                         'selected_5_chefs_secret_coordination_ring',
                         'selected_5_chefs_storage_room']
    args.n_envs = 200
    args.epoch_timesteps = 1e5
    return args


if __name__ == "__main__":
    initial_run_root = 'agent_models/Final/5_first_run'
    continued_run_root = 'agent_models/Final/5_continued_run'
    corrected_run_root = 'agent_models/Final/5'
    ck_starts_from = 20
    args = set_input()
    # ^ set all the above vars before running the script

    fix_ck_list(initial_run_root=initial_run_root,
                continued_run_root=continued_run_root,
                corrected_run_root=corrected_run_root,
                ck_starts_from=ck_starts_from
                )

    fix_pop(args, initial_run_root, continued_run_root, corrected_run_root)
