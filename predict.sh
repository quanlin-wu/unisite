#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_data> <output_dir>"
    exit 1
fi

input_data=$1
output_dir=$2
mkdir -p $output_dir
echo "INPUT DATA: $input_data"
echo "OUTPUT DIR: $output_dir"

if [ ${input_data##*.} == "fasta" ]; then
    export model_type=seq_only
    export ckpt_path=model_weights/uniste_1d_all.pth
else
    export model_type=structure
    export ckpt_path=model_weights/uniste_3d_all.pth
fi

export test_data=$input_data
export output_dir=$output_dir

# experiment
[ -z "${run_id}" ] && run_id="null"
[ -z "${use_gpu}" ] && use_gpu=True
[ -z "${num_gpus}" ] && num_gpus=$(nvidia-smi -L | wc -l)
[ -z "${eval_batch_size}" ] && eval_batch_size=32

# data
[ -z "${test_data}" ] && test_data=""

# model
[ -z "${model_type}" ] && model_type=structure

echo "******************** experiment ********************"
echo "ckpt_path" $ckpt_path
echo "output_dir" $output_dir
echo "eval_batch_size" $eval_batch_size
echo "num_gpus" $num_gpus

echo "******************** data ********************"
echo "test_data" $test_data

echo "******************** model ********************"
echo "model_type" $model_type

#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_DISTRIBUTED_DEBUG=DETAIL

mkdir -p $output_dir/hydra_output/

# export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=$num_gpus scripts/inference.py \
      +ckpt_path=$ckpt_path \
      experiment.output_dir=$output_dir hydra.run.dir=$output_dir/hydra_output/ \
      experiment.use_gpu=$use_gpu experiment.num_gpus=$num_gpus \
      experiment.eval_batch_size=$eval_batch_size \
      data.test_data=$test_data model.model_type=$model_type 

echo "Inference completed. Results saved to $output_dir."

if [ $model_type != "seq_only" ]; then
    echo "Calculating pocket centers..."
    python scripts/calc_center.py -i $output_dir
else
    echo "Skipping pocket center calculation for seq_only model."
fi
