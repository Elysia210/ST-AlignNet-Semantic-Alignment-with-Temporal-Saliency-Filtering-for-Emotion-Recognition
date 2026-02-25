#!/bin/bash

# =========================================================
#  DEAP Facial Expression Recognition Benchmark Script
# =========================================================

# 1. 定义要运行的模型列表
ENCODERS=(slowfast" "videoswin" "formerdfer" "logoformer" "est")

# 2. 实验通用参数设置
DATASET="deap"
MODALITY="facial"       # 指定为 facial 模态
TASK="classification"   # 任务类型
EVAL_MODE="subject_independent" # 评估模式

# 训练超参数 (根据你之前的报错信息调整，特征训练通常很快，Batch可以大)
EPOCHS=4
BATCH_SIZE=2048
LR=1e-4

# 路径设置 (根据你的环境)
DATA_PATH="/root/autodl-tmp/eeg/data/DEAP/processed"
BASE_OUTPUT_DIR="results/deap_facial_benchmark"

# 创建日志文件夹
mkdir -p logs/facial_exp

echo "========================================================="
echo "🚀 Starting DEAP Facial Benchmark Experiment"
echo "📋 Encoders: ${ENCODERS[*]}"
echo "📂 Data Path: $DATA_PATH"
echo "========================================================="

# 3. 开始循环运行
for MODEL in "${ENCODERS[@]}"
do
    # 生成当前时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # 定义该模型的输出目录 (results/deap_facial_benchmark/c3d)
    OUT_DIR="${BASE_OUTPUT_DIR}/${MODEL}"

    # 定义日志文件路径
    LOG_FILE="logs/facial_exp/${MODEL}_${TIMESTAMP}.log"

    echo -e "\n▶️  \033[1;32mRunning Model: $MODEL\033[0m"
    echo "    📂 Output Dir: $OUT_DIR"
    echo "    📝 Log File:   $LOG_FILE"

    # 运行 Python 脚本
    # 注意：因为已经在代码里做了GPU检测，这里直接运行即可
    python train.py \
        --dataset $DATASET \
        --modality $MODALITY \
        --task $TASK \
        --eval_mode $EVAL_MODE \
        --encoder $MODEL \
        --data_path $DATA_PATH \
        --output_path $OUT_DIR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --device cuda \
        > "$LOG_FILE" 2>&1

    # 检查运行状态
    if [ $? -eq 0 ]; then
        echo -e "    ✅ \033[1;32mFinished successfully: $MODEL\033[0m"
    else
        echo -e "    ❌ \033[1;31mFailed: $MODEL\033[0m (Check $LOG_FILE for details)"
    fi

    echo "---------------------------------------------------------"
done

echo "🎉 All facial experiments completed!"