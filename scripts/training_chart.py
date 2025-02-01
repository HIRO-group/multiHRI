import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load and process CSV data
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    df['timestep'] = df['timestep'] / 1000000  # Convert timestep to millions
    return df

# Load SPSA and SPH data
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.dirname(__file__))

spl_data = load_and_process_csv(os.path.join(project_root, '../data/training_logs/wandb_export_2025-01-31T20_49_49.238-07_00.csv'))
sph_data = load_and_process_csv(os.path.join(project_root, '../data/training_logs/wandb_export_2025-01-31T20_08_58.304-07_00.csv'))

# Plot settings
plt.figure(figsize=(10, 4))

# SPL Chart
plt.subplot(1, 1, 1)
plt.plot(spl_data['timestep'], spl_data['Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0 - eval_mean_reward_asymmetric_advantages_teamtype_SPL'], label='SPL-Attack 0', color='#ff7d7d')
plt.plot(spl_data['timestep'], spl_data['Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1 - eval_mean_reward_asymmetric_advantages_teamtype_SPL'], label='SPL-Attack 1', color='#ff4a4a')
plt.plot(spl_data['timestep'], spl_data['Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2 - eval_mean_reward_asymmetric_advantages_teamtype_SPL'], label='SPL-Attack 2', color='#f70202')

# plt.title('Self-Play Low Performance Teammate Training')
# plt.xlabel('Timesteps (millions)')
# plt.ylabel('Mean Reward')
# plt.yticks(np.arange(0, 600, 50))
# plt.legend(loc='upper right')
# plt.grid(True)


# SPH Chart
#plt.subplot(2, 1, 2)
plt.plot(sph_data['timestep'], sph_data['Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0 - eval_mean_reward_asymmetric_advantages_teamtype_SPH'], label='SPH-Attack 0', color='#89a4fa')
plt.plot(sph_data['timestep'], sph_data['Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1 - eval_mean_reward_asymmetric_advantages_teamtype_SPH'], label='SPH-Attack 1', color='#668aff')
plt.plot(sph_data['timestep'], sph_data['Classic/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2 - eval_mean_reward_asymmetric_advantages_teamtype_SPH'], label='SPH-Attack 2', color='#003cff')

plt.title('CAP Performance w/ High and Low Performing Teammates Across Training')
plt.xlabel('Timesteps (millions)')
plt.ylabel('Mean Reward')
plt.yticks(np.arange(0, 600, 50))
plt.legend(loc='best', fontsize=8)
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(os.path.join(project_root, '../data/training_logs/training_chart.png'))
