# !/bin/bash
set -x
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export WANDB_DISABLED="true"
port=$(shuf -i25000-30000 -n1)

which=$1
tasklistfile=$2
experiement=$3
tasks=(`cat "$tasklistfile"`)

if [[ "$experiement" =~ ^()$ ]]; then
    count=1
    for task in ${tasks[@]}; do
        echo "Currently evaluating task "$task "($count/119)"
        mkdir /$which/outputs/output_$task/eval
        touch /$which/outputs/output_$task/eval/results.json
        chmod 777 /$which/outputs/output_$task/eval/results.json

        python evaluation.py \
            --dataset_file /data/tasks/$task/test.csv \
            --prediction_file /$which/outputs/output_$task/test_generations.txt \
            --output_dir /$which/outputs/output_$task/eval

        count=$((count+1))
    done
    python calc_res.py $which
else
    mkdir crosstask/$which/outputs/output_task/eval
    touch crosstask/$which/outputs/output_task/eval/results.json
    chmod 777 crosstask/$which/outputs/output_task/eval/results.json

    python evaluation.py \
        --dataset_file /data/merged/test.csv \
        --prediction_file /$which/outputs/output_task/test_generations.txt \
        --output_dir crosstask/$which/outputs/output_task/eval
fi