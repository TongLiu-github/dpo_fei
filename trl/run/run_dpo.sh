#!/bin/bash -l
#
#SBATCH --nodes=1                  # Resource requirements, job runtime, other options
#SBATCH --ntasks-per-node=1                 #All #SBATCH lines have to follow uninterrupted
#SBATCH --time=24:00:00
#SBATCH --job-name=zephyr-7b_dpo_4gpu
#SBATCH --export=NONE              # do not export environment from submitting shell
#SBATCH --output=zephyr-7b_dpo_4gpu.txt
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:a40:1

#unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun
#
#module load python              # Setup job environment (load modules, stage data, ...)


cd /home/hpc/v100dd/v100dd18/dpo/trl

source ~/.bashrc
conda activate openinstruct
#conda activate /home/hpc/v100dd/v100dd18/miniconda3/envs/simpo
#conda activate openinstruct



echo "start to run"
#python -c "import torch; print(torch.cuda.is_available())"

#export CUDA_HOME=/usr/local/cuda-11.6
#export CUDA_HOME=/home/hpc/v100dd/v100dd18/cuda-12.6
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#echo "CUDA_HOME is set to $CUDA_HOME"
#echo "PATH is set to $PATH"
#export WANDB_API_KEY=522aca12c02f85859cdc9bd70e649cec09eaa182
#export HF_HUB_DOWNLOAD_TIMEOUT=30
#wandb login --relogin $(grep wandb_api_key ./run/config.yaml | cut -d ' ' -f 2)
#export WANDB_INIT_TIMEOUT=120
python fei.py
#python -c "import wandb; print(wandb.init(project='sdfsdf'))"
## example
#python dpo_Tong.py \
#    --dataset_name trl-lib/ultrafeedback_binarized \
#    --model_name_or_path alignment-handbook/zephyr-7b-sft-full \
#    --learning_rate 5.0e-7 \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 2 \
#    --gradient_accumulation_steps 8 \
#    --gradient_checkpointing \
#    --logging_steps 25 \
#    --eval_strategy steps \
#    --eval_steps 50 \
#    --output_dir Qwen2-0.5B-DPO \
#    --no_remove_unused_columns \
#    --report_to wandb

#export NCCL_IB_GID_INDEX=3
#export NCCL_P2P_DISABLE=0
#export NCCL_P2P_LEVEL=NVL
#accelerate launch --config_file ./accelerate_configs/deepspeed_zero3.yaml ./dpo_Tong.py \
#    --dataset_name trl-lib/ultrafeedback_binarized \
#    --model_name_or_path alignment-handbook/zephyr-7b-sft-full \
#    --learning_rate 5.0e-7 \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 2 \
#    --gradient_accumulation_steps 8 \
#    --gradient_checkpointing \
#    --logging_steps 25 \
#    --eval_strategy steps \
#    --eval_steps 50 \
#    --output_dir Qwen2-0.5B-DPO \
#    --no_remove_unused_columns \
#    --report_to wandb \
#    --loss_type dpo
#export NCCL_IB_GID_INDEX=3
#export NCCL_P2P_DISABLE=0
#export NCCL_P2P_LEVEL=NVL
#accelerate launch --config_file ./src/accelerate/deepspeed_zero3.yaml \
#    ./trl/test_orpo_trainer_demo.py \
#    --learning_rate 5e-6 \
#    --per_device_train_batch_size 4 \
#    --cache_dir ./projects/hf_cache/ \
#    --output_dir ./phi-2-aligned-orpo \
#    --po_type "orpo"
    # --num_proc 4 # # of gpus