#!/bin/sh -v
#PBS -e /mnt/beegfs2/home/abstrac01/reinis_freibergs/logs
#PBS -o /mnt/beegfs2/home/abstrac01/reinis_freibergs/logs
#PBS -q batch
#PBS -p 1000
#PBS -l nodes=1:ppn=12:gpus=1:shared,feature=k40
#PBS -l mem=40gb
#PBS -l walltime=24:00:00

eval "$(conda shell.bash hook)"
conda activate conda_env
export LD_LIBRARY_PATH=/mnt/home/abstrac01/.conda/envs/conda_env/lib:$LD_LIBRARY_PATH

cd /mnt/beegfs2/home/abstrac01/reinis_freibergs
python ./taskgen.py \
-conclusion_number 0 \
-learning_rate 1e-4 1e-5 1e-6 1e-7 \
-batch_size 128 \
-hidden_size 64 \
-num_workers 16 \
-sequence_len 400 \
-epochs 20 \
-output_directory ./datasource \
-lstm_layers 4 \
-model model_3_PLSTM model_4_snake_LSTM \
-datasource datasource_2 \
-num_tasks_in_parallel 8 \
-is_single_cuda_device False \
-is_force_start True


