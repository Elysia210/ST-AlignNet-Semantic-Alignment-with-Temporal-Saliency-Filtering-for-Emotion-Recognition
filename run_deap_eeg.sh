#!/bin/bash

# =========================================================
#  DEAP EEG Encoder Benchmark Script
# =========================================================

# 定义要运行的 EEG 模型列表
#ENCODERS=("eegnet" "dgcnn" "lggnet" "tsception" "ccnn" "bihdm" "gcbnet")
ENCODERS=("ccnn" "bihdm" "gcbnet")

# 设置通用参数
DATASET="deap"
MODALITY="eeg"
TASK="classification"
EPOCHS=4
BATCH_SIZE=2048
DATA_PATH="/root/autodl-tmp/eeg/data/DEAP/processed"

# 创建日志文件夹
mkdir -p logs

echo "🚀 Starting DEAP EEG Benchmark Experiment..."
echo "📋 List of encoders to run: ${ENCODERS[*]}"
echo "========================================================="

for MODEL in "${ENCODERS[@]}"
do
    # 生成当前时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # 定义输出目录 (防止覆盖)
    OUT_DIR="results/${DATASET}_${MODALITY}/${MODEL}"

    # 定义日志文件
    LOG_FILE="logs/${DATASET}_${MODEL}_${TIMESTAMP}.log"

    echo "▶️  Running Model: $MODEL"
    echo "    📂 Output: $OUT_DIR"
    echo "    📝 Log:    $LOG_FILE"

    # 运行 Python 脚本
    # nohup ... & 放在这里如果想并行，但为了显存安全，我们通常串行跑（不加 &）
    python train.py \
        --dataset $DATASET \
        --modality $MODALITY \
        --task $TASK \
        --encoder $MODEL \
        --data_path $DATA_PATH \
        --output_path $OUT_DIR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        > "$LOG_FILE" 2>&1

    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✅ Finished: $MODEL"
    else
        echo "❌ Failed:   $MODEL (Check $LOG_FILE for details)"
    fi

    echo "---------------------------------------------------------"
done

echo "🎉 All experiments completed!"