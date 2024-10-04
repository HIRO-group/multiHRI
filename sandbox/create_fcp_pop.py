import os
import re

import multiprocessing as mp
import os
mp.set_start_method('spawn', force=True) # should be called before any other module imports

from oai_agents.agents.rl import RLAgentTrainer
from oai_agents.common.arguments import get_arguments
from oai_agents.common.tags import TeamType, CheckedPoints
from oai_agents.common.learner import LearnerType
from oai_agents.common.curriculum import Curriculum

three_chefs_layouts = [
        'selected_3_chefs_coordination_ring',
        'selected_3_chefs_counter_circuit',
        'selected_3_chefs_cramped_room'
    ]

population = {layout_name: [] for layout_name in three_chefs_layouts}

# Define the root directory containing all SP_* folders
correct_root = os.path.expanduser('agent_models/corrected_wrong')

# Regular expression to match 'ck_{x}_rew_{y}'
ck_regex = re.compile(r'ck_(\d+)_rew_([\d.]+)')

# Initialize ck_list for each SP folder
ck_lists = {}

# Traverse each SP_* folder in the corrected root
for sp_folder in os.listdir(correct_root):
    sp_path = os.path.join(correct_root, sp_folder)
    
    # Only process directories
    if os.path.isdir(sp_path):
        ck_list = []  # Initialize ck_list for this SP folder
        
        # Iterate over all ck_* folders inside this SP_* directory
        for ck_folder in os.listdir(sp_path):
            ck_folder_path = os.path.join(sp_path, ck_folder)

            # Ensure we're working with a directory that matches the ck_* pattern
            match = ck_regex.match(ck_folder)
            if match and os.path.isdir(ck_folder_path):
                ck_number = int(match.group(1))  # Extract the ck number
                reward = float(match.group(2))   # Extract the reward value

                # Construct the tag using the folder name
                tag = f'ck_{ck_number}_rew_{reward}'
                ck_list.append((reward, ck_folder_path, tag))
        ck_lists[sp_folder] = ck_list


def save_population(args, population):
    name_prefix = 'pop'
    for layout_name in args.layout_names:
        rt = RLAgentTrainer(
            name=f'{name_prefix}_{layout_name}',
            args=args,
            agent=None,
            teammates_collection={},
            train_types=[TeamType.SELF_PLAY],
            eval_types=[TeamType.SELF_PLAY],
            epoch_timesteps=args.epoch_timesteps,
            n_envs=args.n_envs,
            learner_type=args.pop_learner_type,
            seed=None,
        )
        rt.agents = population[layout_name]
        rt.save_agents(tag='aamas25')

args = get_arguments()
args.teammates_len = 2
args.num_players = args.teammates_len + 1 
args.layout_names = three_chefs_layouts

for sp_folder, ck_list in ck_lists.items():
    for layout_name in args.layout_names:
        # print(ck_list)
        layout_pop = RLAgentTrainer.get_checkedpoints_agents(args, ck_list, layout_name)
        population[layout_name].extend(layout_pop)

save_population(args=args, population=population)

    
