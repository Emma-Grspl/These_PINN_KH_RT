#!/bin/bash
#SBATCH --job-name=KH_sup_PINN_1case
#SBATCH --output=slurm/log/%x_%j.out
#SBATCH --error=slurm/log/%x_%j.err
#SBATCH -A fdb@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3

set -euo pipefail

module purge
module load pytorch-gpu/py3/2.1.1

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

mkdir -p slurm/log

export MPLCONFIGDIR="${JOBSCRATCH:-/tmp}/mplconfig_${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export DEVICE="${DEVICE:-cuda}"
export ALPHA="${ALPHA:-0.26}"
export MACH="${MACH:-1.8}"
export CR="${CR:-0.38}"
export CI="${CI:-0.024}"
export OUTPUT_DIR="${OUTPUT_DIR:-model_saved/kh_supersonic_singlecase_pressure_fixed_c}"
export EPOCHS="${EPOCHS:-3000}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export N_INTERIOR="${N_INTERIOR:-512}"
export N_BOUNDARY="${N_BOUNDARY:-64}"
export N_CENTER="${N_CENTER:-256}"
export HIDDEN_DIM="${HIDDEN_DIM:-192}"
export DEPTH="${DEPTH:-6}"
export ACTIVATION="${ACTIVATION:-tanh}"
export W_PDE="${W_PDE:-1.0}"
export W_BC="${W_BC:-20.0}"
export W_GAUGE="${W_GAUGE:-100.0}"
export W_CENTER_PDE="${W_CENTER_PDE:-1.0}"
export YMAX="${YMAX:-120.0}"
export ENVELOPE_EPS="${ENVELOPE_EPS:-1.0}"
export SEED="${SEED:-1234}"

echo "Supersonic single-case pressure-first PINN on ${SLURMD_NODENAME}"
echo "alpha=${ALPHA} mach=${MACH} cr=${CR} ci=${CI}"
echo "output_dir=${OUTPUT_DIR}"
echo "weights: pde=${W_PDE} bc=${W_BC} gauge=${W_GAUGE} center=${W_CENTER_PDE}"
echo "ymax=${YMAX} envelope_eps=${ENVELOPE_EPS}"
date
nvidia-smi || true

srun bash scripts/run_kh_supersonic_singlecase_pressure_fixed_c.sh
