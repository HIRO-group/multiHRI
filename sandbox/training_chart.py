import pandas as pd
import matplotlib.pyplot as plt

# Function to load and process CSV data
def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    df['timestep'] = df['timestep'] / 1000000  # Convert timestep to millions
    return df

# Load SPSA and SPH data
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.dirname(__file__))

spl_data = load_and_process_csv(os.path.join(project_root, '../data/training_logs/storage_room_SPL.csv'))
sph_data = load_and_process_csv(os.path.join(project_root, '../data/training_logs/storage_room_SPH.csv'))

# Plot settings
plt.figure(figsize=(10, 4))

# SPL Chart
# plt.subplot(1, 1, 1)
plt.plot(spl_data['timestep'], spl_data['Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0 - eval_mean_reward_storage_room_teamtype_SPL'], label='CAP 1 - eval L', color='#ff7d7d')
plt.plot(spl_data['timestep'], spl_data['Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1 - eval_mean_reward_storage_room_teamtype_SPL'], label='CAP 2 - eval L', color='#ff4a4a')
plt.plot(spl_data['timestep'], spl_data['Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2 - eval_mean_reward_storage_room_teamtype_SPL'], label='CAP 3 - eval L', color='#f70202')

# plt.title('Self-Play Low Performance Teammate Training')
# plt.xlabel('Timesteps (millions)')
# plt.ylabel('Mean Reward')
# plt.yticks(np.arange(0, 600, 50))
# plt.legend(loc='upper right')
# plt.grid(True)


# SPH Chart
# plt.subplot(2, 1, 2)
plt.plot(sph_data['timestep'], sph_data['Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack0 - eval_mean_reward_storage_room_teamtype_SPH'], label='CAP 1 - eval H', color='#89a4fa')
plt.plot(sph_data['timestep'], sph_data['Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack1 - eval_mean_reward_storage_room_teamtype_SPH'], label='CAP 2 - eval H', color='#668aff')
plt.plot(sph_data['timestep'], sph_data['Complex/2/N-1-SP_s1010_h256_tr[SPH_SPM_SPL_SPDA_SPSA]_ran_originaler_attack2 - eval_mean_reward_storage_room_teamtype_SPH'], label='CAP 3 - eval H', color='#003cff')

plt.title('CAP Performance w/ High and Low Performing Teammates Across Training', fontsize=16)
plt.xlabel('Timesteps (millions)', fontsize=16)
plt.ylabel('Mean Reward', fontsize=16)
# plt.yticks(np.arange(0, 600, 50))
plt.legend(loc='best', fontsize=16, ncol=3)
plt.grid(True)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig(os.path.join(project_root, '../data/training_logs/training_chart.png'), dpi=300)
plt.show()
