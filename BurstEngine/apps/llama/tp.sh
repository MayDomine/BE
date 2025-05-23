# ring+flash+zero+ckpt
nnodes=1
gpus=8
tp=8
sp=$3
model=13b
model=70b
model=34b
model=13b
bs=5
if [ $sp == "tp" ]; then
   sp_args = "--tp-sp" 
else
  sp_args = "--sp $sp --spzero"
fi
export TOK_PATH=/data/public/opensource_models/meta-llama/Llama-2-7b-hf/
export TOK_PATH=/home/test/testdata/models/Llama-2-7b-hf/
cmd="torchrun --nnodes=$nnodes --nproc_per_node=8 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=$1:7778 pretrain_llama_hug_tokenizer.py --model-config config/${model}/config.json --vocab config/70b/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --offload --batch-size ${bs} --max-length $2 --tp ${tp} $sp_args --flash cuda  --tokenizer-path $TOK_PATH --ckpt  2>&1 |tee  ${nnodes}nnodes_${gpus}gpus_${tp}tp_${sp}sp.log"
echo $cmd
eval $cmd
torchrun --nnodes=4 --nproc_per_node=8  --rdzv_id=2 --rdzv_backend=c10d --rdzv_endpoint=$1:12306 pretrain_llama_hug_tokenizer.py --model-config config/70b/config.json --vocab config/7b/vocab.txt --train-iters 400000 --lr 1.5e-4 --inspect-iters 100 --warmup-iters 2000 --lr-decay-style noam --weight-decay 0.1 --clip-grad 1.0 --loss-scale 1048576 --dataset datasets/_datasets_laptop.json --start-step 1 --batch-size 4 --max-length $2 --offload --tp 32 --flash cuda --tp-sp --tokenizer-path $TOK_PATH
