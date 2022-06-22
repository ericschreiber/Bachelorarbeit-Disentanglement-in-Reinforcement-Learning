#!/bin/bash

#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/ericschr/net_scratch/BA/Disentanglement_metrics/disentanglement_lib/.ipynb_checkpoints/logs/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/itet-stor/ericschr/net_scratch/BA/Disentanglement_metrics/disentanglement_lib/.ipynb_checkpoints/logs/errors/%j.err  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

stdbuf -o0 -e0 command

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
conda activate DisMetrics

# Binary or script to execute

pathvar=/itet-stor/ericschr/net_scratch/BA/models/TCBVAE/findBest_Breakout_runConv_FullBN_likeDQNConv_B0.0001_sumTC0.0_Lat32_Epochs100randomSeed71VAE/outputBeta20-5
cd $pathvar
mkdir results
python /itet-stor/ericschr/net_scratch/BA/Disentanglement_metrics/disentanglement_lib/compute_disentangle_metrics.py --dataset_path /itet-stor/ericschr/net_scratch/BA/train_data100kBreakout.npz --store_path $pathvar  --factor_vae


#********************************* Benjamin Code *****************************
# pathvar=/itet-stor/ericschr/net_scratch/BTCVAE_Benjamin/results/test
# python /itet-stor/ericschr/net_scratch/BA/Disentanglement_metrics/disentanglement_lib/compute_disentangle_metrics.py --dataset_path /itet-stor/ericschr/net_scratch/BA/64x64train_data100kWithLabels_benjamin.npz --store_path $pathvar  --dci --mig

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0