MODEL_PATH=$1
MODEL_NAME=$(basename $MODEL_PATH)

echo $MODEL_NAME

GPU_COUNT_PARSED=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
python3 -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --tensor-parallel-size $GPU_COUNT_PARSED --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.9 --served-model-name $MODEL_NAME