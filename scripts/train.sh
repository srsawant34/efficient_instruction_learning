# !/bin/bash
set -x
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export WANDB_DISABLED="true"
port=$(shuf -i25000-30000 -n1)
export TRANSFORMERS_CACHE=/data/data/ssawan13/conda/cache

which=$1
tasklistfile=$2
experiement=$3
tasks=(`cat "$tasklistfile"`)

if [[ "$experiement" =~ ^()$ ]]; then
    echo "Exp: $experiment"
    mkdir $which
    mkdir $which/outputs
    count=1
    for task in ${tasks[@]}; do
        echo "Currently working on task "$task "($count/119)"

        deepspeed --num_gpus=6 --master_port $port scripts/run_model.py \
            --model_name_or_path "allenai/tk-instruct-3b-def-pos" \
            --cache_dir "/data/data/ssawan13/conda/cache" \
            --do_train true --do_eval true --do_predict true \
            --train_file /data/tasks/$task/$which.csv \
            --validation_file /data/tasks/$task/val.csv \
            --test_file /data/tasks/$task/test.csv \
            --output_dir /$which/outputs/output_$task \
            --per_device_train_batch_size="1" \
            --per_device_eval_batch_size="2" \
            --gradient_accumulation_steps="2"\
            --max_source_length 1024 \
            --max_target_length 128 \
            --generation_max_length 128 \
            --learning_rate 5e-05 \
            --num_train_epochs 2 \
            --warmup_steps 0 \
            --lr_scheduler_type constant \
            --predict_with_generate \
            --save_strategy=no \
            --deepspeed /stage.config \
            --bf16 \
            --run_name t5-experiment
        
        count=$((count+1))
    done
else
    mkdir crosstask
    mkdir crosstask/$which
    mkdir crosstask/$which/outputs
    echo "Experiment: $EXPERIMENT"
    deepspeed --num_gpus=6 --master_port $port scripts/run_model.py \
        --model_name_or_path "allenai/tk-instruct-3b-def-pos" \
        --cache_dir "/data/data/ssawan13/conda/cache" \
        --do_train true --do_eval true --do_predict true \
        --train_file /data/merged/$which.csv \
        --validation_file /data/merged/val.csv \
        --test_file /data/merged/test.csv \
        --output_dir /crosstask/$which/outputs/output_task \
        --per_device_train_batch_size="1" \
        --per_device_eval_batch_size="2" \
        --gradient_accumulation_steps="2"\
        --max_source_length 1024 \
        --max_target_length 128 \
        --generation_max_length 128 \
        --learning_rate 5e-05 \
        --num_train_epochs 2 \
        --warmup_steps 0 \
        --lr_scheduler_type constant \
        --predict_with_generate \
        --save_strategy=no \
        --deepspeed /stage.config \
        --bf16 \
        --run_name t5-experiment
fi