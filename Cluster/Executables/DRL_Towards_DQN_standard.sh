#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/ericschr/net_scratch/BA/DQN_runs/final/logs/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/itet-stor/ericschr/net_scratch/BA/DQN_runs/final/logs/errors/%j.err  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5

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
#argument ist random_seed
/itet-stor/ericschr/net_scratch/BA/Executables/DRL_TowardsDataScience_Pong.py $1

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0