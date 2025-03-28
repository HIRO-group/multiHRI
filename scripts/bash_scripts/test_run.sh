#!/bin/sh

ALGO="FCP_traditional"
TEAMMATES_LEN=1
NUM_PLAYERS=$((TEAMMATES_LEN + 1))
NUM_OF_CKPOINTS=10
LAYOUT_NAMES="counter_circuit"
EXP_DIR="$NUM_PLAYERS" # When quick_test=True this will be overwritten to "Test/$EXP_DIR"
TOTAL_SP_AGENTS=2
QUICK_TEST=true
HOW_LONG=1
USE_CUDA=false
USE_MULTIPLEPROCESSES=false

POP_FORCE_TRAINING=false
ADVERSARY_FORCE_TRAINING=false
PRIMARY_FORCE_TRAINING=false
# EXP_NAME_PREFIX="test_"

source scripts/bash_scripts/env_config.sh
# Overwrite the default values from env_config here if needed
N_ENVS=5
WANDB_MODE="disabled"
EPOCH_TIMESTEPS=2500
N_X_SP_TOTAL_TRAINING_TIMESTEPS=10000
FCP_TOTAL_TRAINING_TIMESTEPS=10000


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
    --use-cuda ${USE_CUDA} \
    --use-multipleprocesses ${USE_MULTIPLEPROCESSES} \