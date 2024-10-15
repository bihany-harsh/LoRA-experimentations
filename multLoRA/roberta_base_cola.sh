export num_gpus=2
export LOCAL_RANK=0
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./cola_rank_16"
python -m torch.distributed.launch --use_env --nproc_per_node=$num_gpus --master_port=25900 \
run_mult_lora.py \
--model_name_or_path roberta-base \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 512 \
--per_device_train_batch_size 64 \
--learning_rate 4e-4 \
--num_train_epochs 80 \
--output_dir $output_dir/model \
--logging_steps 10 \
--eval_steps 50 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--mult_lora_mode pre \
--lora_r 16 \
--lora_alpha 16 \
--seed 0 \
--weight_decay 0.1 \
--fp16 False

