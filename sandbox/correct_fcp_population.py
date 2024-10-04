import os
import shutil
import re

correct_root = '/home/ava/Downloads/OneDrive_1_10-3-2024'
wrong_root = '/home/ava/Research/Codes/MHRI/multiHRI/agent_models/Final/3'
new_root = '/home/ava/Desktop/corrected_wrong'

# Regular expression to match ck_* folders and extract the number
ck_regex = re.compile(r'ck_(\d+)(.*)')

# Step 1: Iterate over each SP folder in the correct folder
for sp_folder in os.listdir(correct_root):
    correct_sp_path = os.path.join(correct_root, sp_folder)
    wrong_sp_path = os.path.join(wrong_root, sp_folder)
    new_sp_path = os.path.join(new_root, sp_folder)

    # Check if it's a directory (SP folders)
    if os.path.isdir(correct_sp_path):
        # Create the corresponding SP folder in the new directory
        os.makedirs(new_sp_path, exist_ok=True)

        # Step 2: Copy all ck_* folders from the correct SP folder to the new folder
        for ck_folder in os.listdir(correct_sp_path):
            correct_ck_path = os.path.join(correct_sp_path, ck_folder)
            new_ck_path = os.path.join(new_sp_path, ck_folder)

            # If it's a folder (ck_*), copy it to the new folder
            if os.path.isdir(correct_ck_path):
                shutil.copytree(correct_ck_path, new_ck_path, dirs_exist_ok=True)
                print(f"Copied {ck_folder} from {correct_sp_path} to {new_sp_path}")

        # Step 3: Check for ck_* folders in the wrong SP folder
        if os.path.isdir(wrong_sp_path):
            for ck_folder in os.listdir(wrong_sp_path):
                wrong_ck_path = os.path.join(wrong_sp_path, ck_folder)

                # Check if the ck_folder already exists in the new folder (from correct folder)
                if os.path.exists(os.path.join(new_sp_path, ck_folder)):
                    print(f"{ck_folder} already exists in {new_sp_path}, skipping...")
                    continue  # Skip if the folder already exists

                # Match the folder name to ck_ number pattern
                match = ck_regex.match(ck_folder)
                if match and os.path.isdir(wrong_ck_path):
                    ck_number = int(match.group(1))  # Extract the number (e.g., '10' from 'ck_10')
                    suffix = match.group(2)  # Extract the rest of the name (e.g., '_rew_100')

                    new_ck_number = ck_number + 20  # Sum the number with 20
                    new_ck_folder = f"ck_{new_ck_number}{suffix}"  # Recreate the folder name with new number
                    new_ck_path_with_sum = os.path.join(new_sp_path, new_ck_folder)

                    # Only copy if the new folder doesn't already exist
                    if not os.path.exists(new_ck_path_with_sum):
                        shutil.copytree(wrong_ck_path, new_ck_path_with_sum, dirs_exist_ok=True)
                        print(f"Copied {ck_folder} from {wrong_sp_path} to {new_ck_folder} in {new_sp_path}")
