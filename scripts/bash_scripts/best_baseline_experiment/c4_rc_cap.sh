#!/bin/sh

#SBATCH --partition=amilian
#SBATCH --job-name=c4_rc_cap
#SBATCH --output=c4_rc_cap.%j.out
#SBATCH --time=23:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=15G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avab4907@colorado.edu

cd $(git rev-parse --show-toplevel)

if [ -d ".venv" ]; then
	# Install UV
	curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/scratch/alpine/$USER/bin" sh

	# Make sure to cache in scratch
	export UV_CACHE_DIR="/scratch/alpine/$USER/.uv"
	export PATH=$PATH:"/scratch/alpine/$USER/bin"

	uv venv -p 3.9
fi

uv pip install wandb
source .venv/bin/activate
uv pip install -e ../overcooked_ai
uv pip install pip==22 setuptools==62 torch==2.5.1
pip install -e .



WANDB_API_KEY=$(cat /home/$USER/.wandb_api_key)
WANDB_CACHE_DIR="/scratch/alpine/$USER/.wandb_cache"


ALGO="best_EGO_with_CAP"
TEAMMATES_LEN=1
HOW_LONG=20
NUM_OF_CKPOINTS=40
LAYOUT_NAMES="c4"
EXP_DIR="${LAYOUT_NAMES}_best_EGO_with_CAP"
TOTAL_SP_AGENTS=1
QUICK_TEST=false

L0="${LAYOUT_NAMES}_v1/SP_s1010_h256_tr[SP]_ran/ck_0"
L1="${LAYOUT_NAMES}_v2/SP_s1010_h256_tr[SP]_ran/ck_0"
L2="${LAYOUT_NAMES}_v3/SP_s1010_h256_tr[SP]_ran/ck_0"
L3="${LAYOUT_NAMES}_v4/SP_s1010_h256_tr[SP]_ran/ck_0"

M0="${LAYOUT_NAMES}_v1/SP_s1010_h256_tr[SP]_ran/ck_2_rew_192.0"
M1="${LAYOUT_NAMES}_v2/SP_s1010_h256_tr[SP]_ran/ck_2_rew_118.0"
M2="${LAYOUT_NAMES}_v3/SP_s1010_h256_tr[SP]_ran/ck_1_rew_54.0"
M3="${LAYOUT_NAMES}_v4/SP_s1010_h256_tr[SP]_ran/ck_1_rew_104.0"

H0="${LAYOUT_NAMES}_v1/SP_s1010_h256_tr[SP]_ran/best"
H1="${LAYOUT_NAMES}_v2/SP_s1010_h256_tr[SP]_ran/best"
H2="${LAYOUT_NAMES}_v3/SP_s1010_h256_tr[SP]_ran/best"
H3="${LAYOUT_NAMES}_v4/SP_s1010_h256_tr[SP]_ran/best"

L="${L0},${L1},${L2},${L3}"
M="${M0},${M1},${M2},${M3}"
H="${H0},${H1},${H2},${H3}"

WANDB_MODE="online"
POP_FORCE_TRAINING=false
ADVERSARY_FORCE_TRAINING=false
PRIMARY_FORCE_TRAINING=false

source scripts/bash_scripts/env_config.sh

N_ENVS=50

python scripts/train_agents.py \
    --layout-names ${LAYOUT_NAMES} \
    --algo-name ${ALGO} \
    --exp-dir ${EXP_DIR} \
    --num-of-ckpoints ${NUM_OF_CKPOINTS} \
    --teammates-len ${TEAMMATES_LEN} \
    --num-players ${NUM_PLAYERS} \
    --custom-agent-ck-rate-generation ${CUSTOM_AGENT_CK_RATE_GENERATION} \
    --num-steps-in-traj-for-dyn-adv ${NUM_STEPS_IN_TRAJ_FOR_DYN_ADV} \
    --num-static-advs-per-heatmap ${NUM_STATIC_ADVS_PER_HEATMAP} \
    --num-dynamic-advs-per-heatmap ${NUM_DYNAMIC_ADVS_PER_HEATMAP} \
    --use-val-func-for-heatmap-gen ${USE_VAL_FUNC_FOR_HEATMAP_GEN} \
    --prioritized-sampling ${PRIORITIZED_SAMPLING} \
    --n-envs ${N_ENVS} \
    --epoch-timesteps ${EPOCH_TIMESTEPS} \
    --pop-total-training-timesteps ${POP_TOTAL_TRAINING_TIMESTEPS} \
    --n-x-sp-total-training-timesteps ${N_X_SP_TOTAL_TRAINING_TIMESTEPS} \
    --fcp-total-training-timesteps ${FCP_TOTAL_TRAINING_TIMESTEPS} \
    --adversary-total-training-timesteps ${ADVERSARY_TOTAL_TRAINING_TIMESTEPS} \
    --n-x-fcp-total-training-timesteps ${N_X_FCP_TOTAL_TRAINING_TIMESTEPS} \
    --total-sp-agents ${TOTAL_SP_AGENTS} \
    --wandb-mode ${WANDB_MODE} \
    --pop-force-training ${POP_FORCE_TRAINING} \
    --adversary-force-training ${ADVERSARY_FORCE_TRAINING} \
    --primary-force-training ${PRIMARY_FORCE_TRAINING} \
    --how-long ${HOW_LONG} \
    --exp-name-prefix "${EXP_NAME_PREFIX}" \
    --low-perfs ${L} \
    --med-perfs ${M} \
    --high-perfs ${H} \