#!/bin/bash
#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true
#$ -R y
#$ -cwd
#$ -S /bin/bash
#$ -j y
#$ -N vgg9_ablation
#$ -t 1-48
set -euo pipefail

hostname
date

number=$SGE_TASK_ID
paramfile="scripts/jobs_vgg9_ablation.txt"

# ---------------------------------------------------------------------
# 1.  Load toolchains and activate virtual-env
# ---------------------------------------------------------------------
if command -v source >/dev/null 2>&1; then
  source /share/apps/source_files/python/python-3.9.5.source || true
  source /share/apps/source_files/cuda/cuda-11.8.source || true
fi
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  :
else
  if [[ -f /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate ]]; then
    source /SAN/intelsys/syn_vae_datasets/MATS_anti_spur/spur_venv/bin/activate
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
  fi
fi

# ---------------------------------------------------------------------
# 2.  Keep Matplotlib out of home quota
# ---------------------------------------------------------------------
export MPLCONFIGDIR="$TMPDIR/mplcache"
mkdir -p "$MPLCONFIGDIR"

# ---------------------------------------------------------------------
# 3.  Create output directories
# ---------------------------------------------------------------------
mkdir -p checkpoints
mkdir -p results
mkdir -p logs
mkdir -p data
mkdir -p plots

# ---------------------------------------------------------------------
# 4.  Extract task-specific parameters
# ---------------------------------------------------------------------
# Format per line: <arch> <seed> <optimizer> <batchnorm> <augmentation> <dropout>
arch=$(sed -n ${number}p "$paramfile" | awk '{print $1}')
seed=$(sed -n ${number}p "$paramfile" | awk '{print $2}')
optimizer=$(sed -n ${number}p "$paramfile" | awk '{print $3}')
batchnorm=$(sed -n ${number}p "$paramfile" | awk '{print $4}')
augmentation=$(sed -n ${number}p "$paramfile" | awk '{print $5}')
dropout=$(sed -n ${number}p "$paramfile" | awk '{print $6}')

if [[ -z "$arch" || -z "$seed" || -z "$optimizer" || -z "$batchnorm" || -z "$augmentation" || -z "$dropout" ]]; then
  echo "Invalid job line at index $number in $paramfile" >&2
  exit 1
fi

date
echo "Running ablation job: arch=$arch, seed=$seed, optimizer=$optimizer, batchnorm=$batchnorm, augmentation=$augmentation, dropout=$dropout"

# Define expected checkpoint path for conditional training
ablation_name="${optimizer}_${batchnorm}_${augmentation}_${dropout}"
checkpoint_file="checkpoints/${arch}_${ablation_name}_seed${seed}_final.pt"
results_file="results/${arch}_${ablation_name}_seed${seed}_results.npz"

# ---------------------------------------------------------------------
# 5.  Train (conditional on checkpoint and results existence)
# ---------------------------------------------------------------------
if [ -f "$checkpoint_file" ] && [ -f "$results_file" ]; then
    echo "Checkpoint and results already exist; skipping training."
    echo "  Checkpoint: $checkpoint_file"
    echo "  Results: $results_file"
else
    echo "Starting training..."
    python3.9 -u train_vgg_ablation.py \
        --arch "$arch" \
        --seed $seed \
        --optimizer "$optimizer" \
        --batchnorm "$batchnorm" \
        --augmentation "$augmentation" \
        --dropout "$dropout" \
        --epochs 500 \
        --batch_size 128 \
        --lr 0.001 \
        --weight_decay 5e-4 \
        --eval_interval 10 \
        --n_masks_train 20 \
        --n_masks_final 40 \
        --max_eval_batches_train 20 \
        --max_eval_batches_final 40 \
        --checkpoint_dir checkpoints \
        --results_dir results \
        --data_dir ./data \
        --device cuda

    date
    echo "Training completed: arch=$arch seed=$seed optimizer=$optimizer batchnorm=$batchnorm augmentation=$augmentation dropout=$dropout"
fi

date
echo "Job completed: arch=$arch seed=$seed optimizer=$optimizer batchnorm=$batchnorm augmentation=$augmentation dropout=$dropout"
