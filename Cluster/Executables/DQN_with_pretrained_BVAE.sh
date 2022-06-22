#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/ericschr/net_scratch/BA/DQN_runs/With_Pretrained_conv_BVAE/logs/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/itet-stor/ericschr/net_scratch/BA/DQN_runs/With_Pretrained_conv_BVAE/logs/runMAR29/%j.err  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Conda 
[[ -f /itet-stor/ericschr/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/ericschr/net_scratch/conda/bin/conda shell.bash hook)"
conda activate PyTorchRL4

# Binary or script to execute

#  *************** Hyperparameters ***************
#arguments: path to model, features, likeDQN, namefolder, random seed, epsilon decay (standard: #.999985), big Agent [normal, big, superbig, superduperbig], Breakout ['Breakout', 'Pong']
/itet-stor/ericschr/net_scratch/BA/Executables/DQN_with_pretrained_BVAE_Pong.py $1 $2 $3 $4 $5 $6 $7 $8

#  *************** Example  ***************
#"//<path_to_storage>//BA/models/TCBVAE/ConvTC0.0001_Beta0.00192Lat64lr0.0001-best29_3.dat" 64 "NotDQNWOBN" "testOldLong_Seed73" 73 .999985 "big"

# sbatch --constraint='titan_xp' //<path_to_storage>//BA/Executables/DQN_with_pretrained_BVAE.sh "/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/findBest_firstBall_WithBNlikeDQNConv_B5_sumTC0_Lat10_Epochs100randomSeed5VAE/outputBeta4-5/likeDQNConvB5_TC0_Lat10_Epochs100VAErandomSeed54-5" 10 "likeDQN" "EVALB5" 73 .99985 "superbig"


# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0