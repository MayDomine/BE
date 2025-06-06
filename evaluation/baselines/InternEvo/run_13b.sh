
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1
export SEQ_LEN=$1
export CP_SIZE=$2
DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes $WORLD_SIZE --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
cmd="torchrun $DISTRIBUTED_ARGS train.py --config configs/7B_llama2.py"
echo $cmd

$cmd
