#!/bin/sh

ALGO="best_EGO_with_CAP"
TEAMMATES_LEN=1
HOW_LONG=20
NUM_OF_CKPOINTS=40
LAYOUT_NAMES="c4"
EXP_DIR="${LAYOUT_NAMES}_best_EGO_with_CAP"
TOTAL_EGO_AGENTS=1
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
    --total-ego-agents ${TOTAL_EGO_AGENTS} \
    --wandb-mode ${WANDB_MODE} \
    --pop-force-training ${POP_FORCE_TRAINING} \
    --adversary-force-training ${ADVERSARY_FORCE_TRAINING} \
    --primary-force-training ${PRIMARY_FORCE_TRAINING} \
    --how-long ${HOW_LONG} \
    --exp-name-prefix "${EXP_NAME_PREFIX}" \
    --low-perfs ${L} \
    --med-perfs ${M} \
    --high-perfs ${H} \