#!/bin/bash
#SBATCH --job-name=meanflow_qsgd_64
#SBATCH --account=aarc
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=384G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

ml conda
ml cuda

source ~/.bashrc
conda activate meanflow

cd /scratch/gautschi/liu3688/meanflow-orderedsgd/meanflow

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"

EXP_NAME="qsgd_64"
OUTPUT_DIR="tmp/${EXP_NAME}_job${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=8 --master_port=12345 \
    train.py \
    --output_dir="$OUTPUT_DIR" \
    --dataset=cifar10 \
    --batch_size=128 \
    --lr=0.0006 \
    --eval_frequency=200 \
    --epochs=3200 \
    --compute_fid \
    --log_per_step=100 \
    --tr_sampler=v1 \
    --P_mean_t -0.6 \
    --P_std_t 1.6 \
    --P_mean_r -4.0 \
    --P_std_r 1.6 \
    --warmup_epochs 10 \
    --norm_p 0.75 \
    --ratio 0.75 \
    --dropout 0.2 \
    --use_edm_aug \
    --not_compile \
    --method 1 \
    --ssize 64
    
echo "Outputs saved in: $(realpath "$OUTPUT_DIR")"
