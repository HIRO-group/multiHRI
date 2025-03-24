# Running training on the Research Cluster

## Step 0: Getting access to the Research Cluster

Follow the instructions here: https://curc.readthedocs.io/en/latest/getting_started/logging-in.html

You should now be able to log in to the Research Cluster using SSH (`ssh [identikey]@login.rc.colorado.edu`)

You will be dropped into the home directory (`/home/[identikey]`) of the login node. Only use this space for configuration files and occasional scripts. Do not store any data here.

The majority of your space will be available in the scratch directory (`/scratch/[identikey]`).

## Step 1: Setting up your environment

Clone the repository and overcooked, preferably in the scratch directory:

```bash
git clone https://github.com/hiro-group/multihri.git
git clone https://github.com/hiro-group/overcooked_ai.git
```

## Step 2: Writing a SLURM script

Slurm uses the prefix `#SBATCH` to specify job parameters inside a shell script. Here is an example script:

```bash
#!/bin/sh

#SBATCH --partition=amem
#SBATCH --job-name=SP_c3_v2
#SBATCH --output=SP_c3_v2.%j.out
#SBATCH --time=36:00:00
#SBATCH --qos=mem
#SBATCH --nodes=1
#SBATCH --ntasks=50
#SBATCH --mem=256G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[identikey]]@colorado.edu


ALGO="SP"
TEAMMATES_LEN=1
HOW_LONG=20
NUM_OF_CKPOINTS=40
LAYOUT_NAMES="c3_v2"
EXP_DIR=${LAYOUT_NAMES}
TOTAL_EGO_AGENTS=1
QUICK_TEST=false

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
    --quick-test ${QUICK_TEST} \
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
```

The partition refers to the kind of nodes you want to run on. Since we only need CPUs we can use the amilan parition. A list can be found here: https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html

The output file will contain the log of the output of the job and can be used to monitor progress using the command `tail -f SP_c3_v2_[JOB_ID].out`.

You must set an upper bound on the time needed for the job to complete. The job will be killed if this time is exceeded regardless if the job is successful or not.

QOS refers to the properties of the job. For high memory jobs like SB3 venvs use the `mem` QOS.

Nodes refers to the number of machines you want to run on for jobs that might use a system like OpenMP. Nodes typically have a large number of cores so it is unlikely that you will need more than one machine.

ntasks refers to the number of processes that you want to run. These will be mapped to cores on the machine. Here it should be equal to the number of envs you are using in stable baselines.

mem is the total memory for the job on one node. Jobs with QoS `mem` need to have at least 256GB to be accepted.

The mail commands allow you to be sent status updates about the job to your email.

**IMPORTANT** WandB uses a large amount of disk space. Make sure to explicity configure its directory in the script to use some directory in your scratch space otherwise it will fill up your home directory and cause your job to fail.

These scripts are designed to run on a compute node, not the login node you are using to launch the job. So they must be able to set up the environment correctly as part of the script. This includes installing any dependencies or potentially loading a venv.

Additional notes: https://curc.readthedocs.io/en/latest/additional-resources/CURC-cheatsheet.html

