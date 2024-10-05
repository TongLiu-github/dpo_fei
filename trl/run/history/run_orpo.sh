#!/usr/bin/env bash
#
#SBATCH --job-name=phi-2
#SBATCH --output=phi-2_4gpus_worker-6.txt
#SBATCH --ntasks=1
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=worker-6
# debug info
hostname

# activate env
cd /home/wiss/liu/
#env
#python -m venv ./venv/test
#source ./venv/test/bin/activate
#conda activate test
source ~/.bashrc
conda activate simpo

cd ./dpo/orpo
echo "start to run"
python -c "import torch; print(torch.cuda.is_available())"


#python orpo.py \
#    --model_name_or_path=gpt2 \
#    --per_device_train_batch_size 4 \
#    --max_steps 1000 \
#    --learning_rate 8e-6 \
#    --gradient_accumulation_steps 1 \
#    --logging_steps 10 \
#    --eval_steps 500 \
#    --output_dir="gpt2-aligned-orpo" \
#    --warmup_steps 150 \
#    --report_to wandb \
#    --bf16 \
#    --logging_first_step \
#    --no_remove_unused_columns

#python orpo_ori.py \
#    --model_name="microsoft/phi-2" \
#    --data_name="argilla/ultrafeedback-binarized-preferences-cleaned" \
#    --output_dir="./outputs/phi-2-1gpu/" \
#    --per_device_train_batch_size 2 \

# multigpu + deepspeed
#export NUM_GPUS=4
## 指定NCCL使用Infiniband或RoCE网络中的第4个GID进行通信，适用于有多个网络接口的情况。
#export NCCL_IB_GID_INDEX=3
## 启用 P2P 通信
#export NCCL_P2P_DISABLE=0
## 指定使用 NVLink 进行GPU之间的P2P通信，如果你的系统支持 NVLink，这可以显著提升多GPU训练时的通信效率。
#export NCCL_P2P_LEVEL=NVL
#accelerate launch --config_file=accelerate_configs/deepspeed_zero11.yaml \
#    --num_processes $NUM_GPUS \
#    ./orpo_ori.py \
#    --model_name="microsoft/phi-2" \
#    --data_name="argilla/ultrafeedback-binarized-preferences-cleaned" \
#    --output_dir="./outputs/phi-2-4gpus/" \
#    --per_device_train_batch_size 2

# gpt2: multigpu + deepspeed
#export NUM_GPUS=2
#accelerate launch --config_file=accelerate_configs/deepspeed_zero11.yaml \
#    --num_processes $NUM_GPUS \
#    ./orpo.py \
#    --model_name_or_path=gpt2 \
#    --per_device_train_batch_size 4 \
#    --max_steps 1000 \
#    --learning_rate 8e-6 \
#    --gradient_accumulation_steps 1 \
#    --logging_steps 10 \
#    --eval_steps 500 \
#    --output_dir="gpt2-aligned-orpo" \
#    --warmup_steps 150 \
#    --report_to wandb \
#    --bf16 \
#    --logging_first_step \
#    --no_remove_unused_columns

export NUM_GPUS=4
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
    --num_processes $NUM_GPUS \
    orpo.py \
    --model_name_or_path="microsoft/phi-2" \
    --dataset_name="argilla/ultrafeedback-binarized-preferences-cleaned" \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="phi-2-aligned-orpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

