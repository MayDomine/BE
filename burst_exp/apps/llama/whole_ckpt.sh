
#!/bin/bash

# sizes=( 1048576 2097152 )
# sizes = ( 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 )
sizes=( 32768 65536 131072 262144 )
methods=(
  "burst_whole_ckpt"
)
cp_sizes=( 2 4 8 )
# methods=(
#   # "burst_ulysses"
#   "burst"
# )
export NODES="g47"
DOCKER_DIR=/workspace/workspace/burst_exp/apps/llama
export WORLD_SIZE=1
export LOG_FILE=$DOCKER_DIR/single_node_exp_13b.log
export LOG_FILE=$DOCKER_DIR/whole_ckpt_13b.log
export MODEL="13b"
echo $LOG_FILE
for cp in ${cp_sizes[@]}; do
  for method in ${methods[@]}; do
    export CP_SIZE=$cp
    echo "Running method $method $CP_SIZE" >> summary.txt
    for size in ${sizes[@]}; do
        echo "Running size $size with method $method" >> summary.txt
        # bash submit.sh "bash build_run.sh $size $method" 
        bash multi.sh $size $method $cp
    done
  done
done