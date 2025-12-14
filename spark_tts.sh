#!/bin/bash
#SBATCH -p gpuqs
#SBATCH -w g18
#SBATCH --job-name=sparktts_h100
#SBATCH --output=logs/sparktts_h100_%j.log
#SBATCH --time=02-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module purge
module load python3
source /fs/atipa/app/rl9.x/python3/3.11.7/bin/activate

cd "$HOME/Spark-TTS-finetune"

# 1) keep Transformers CPU-only backends off
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1

# 2) tell everything we do NOT use DeepSpeed
export ACCELERATE_USE_DEEPSPEED=0
export TRANSFORMERS_NO_DEEPSPEED=1
export USE_DEEPSPEED=0

# 3) shadow system deepspeed with our safe stub (no CUDA checks)
export PYTHONPATH="$PWD/vendor_no_ds:$PYTHONPATH"

# (optional) avoid NFS warnings for Triton cache
export TRITON_CACHE_DIR="$PWD/.triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

echo "Python: $(which python)"
python -V

python train.py \
  --prompts_yaml ./output_prompt/Elise.jsonl \
  --model_name_or_path ./pretrained_models/Spark-TTS-0.5B/LLM \
  --output_dir ./sparktts-fullft-out-8\
  --epochs 50 \
  --batch_size 32 \
  --grad_accum 16 \
  --max_length 2048
