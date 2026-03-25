GPUS=4
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29159}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export CUDA_VISIBLE_DEVICES="4,5,6,7"
export TORCHDYNAMO_VERBOSE=1

PYTHONPATH="$(dirname $0)/..":"$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    Deliver/D_new_asymformer_train.py \
    -b 8 \
    --amp