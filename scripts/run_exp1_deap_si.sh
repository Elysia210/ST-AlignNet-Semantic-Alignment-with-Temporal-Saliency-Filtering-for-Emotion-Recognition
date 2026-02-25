#!/bin/bash
# scripts/run_exp1_deap_si_all.sh
# 运行DEAP数据集的Subject-Independent实验
# 包含分类和回归两种任务

export CUDA_VISIBLE_DEVICES=0

DATA_PATH="data/DEAP/processed"
BATCH_SIZE=64
EPOCHS=80
LR=1e-4

echo "============================================"
echo "DEAP Subject-Independent Experiments"
echo "留一法 (32 folds)"
echo "============================================"

# ============================================
# 分类任务 (Classification)
# ============================================

echo ""
echo "========== CLASSIFICATION TASK =========="
echo ""

# EEG Encoders
echo "--- EEG Encoders (Classification) ---"

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder eegnet \
    --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR \
    --output_path results/exp1/deap_si_cls/eeg_eegnet

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder dgcnn \
    --output_path results/exp1/deap_si_cls/eeg_dgcnn

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder lggnet \
    --output_path results/exp1/deap_si_cls/eeg_lggnet

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder tsception \
    --output_path results/exp1/deap_si_cls/eeg_tsception

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder ccnn \
    --output_path results/exp1/deap_si_cls/eeg_ccnn

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder bihdm \
    --output_path results/exp1/deap_si_cls/eeg_bihdm

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder gcbnet \
    --output_path results/exp1/deap_si_cls/eeg_gcbnet

# Facial Encoders
echo ""
echo "--- Facial Encoders (Classification) ---"

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder c3d \
    --output_path results/exp1/deap_si_cls/facial_c3d

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder slowfast \
    --output_path results/exp1/deap_si_cls/facial_slowfast

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder videoswin \
    --output_path results/exp1/deap_si_cls/facial_videoswin

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder formerdfer \
    --output_path results/exp1/deap_si_cls/facial_formerdfer

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder logoformer \
    --output_path results/exp1/deap_si_cls/facial_logoformer

python train_final.py \
    --dataset deap --task classification --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder est \
    --output_path results/exp1/deap_si_cls/facial_est

# ============================================
# 回归任务 (Regression)
# ============================================

echo ""
echo "========== REGRESSION TASK =========="
echo ""

# EEG Encoders
echo "--- EEG Encoders (Regression) ---"

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder eegnet \
    --output_path results/exp1/deap_si_reg/eeg_eegnet

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder dgcnn \
    --output_path results/exp1/deap_si_reg/eeg_dgcnn

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder lggnet \
    --output_path results/exp1/deap_si_reg/eeg_lggnet

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder tsception \
    --output_path results/exp1/deap_si_reg/eeg_tsception

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder ccnn \
    --output_path results/exp1/deap_si_reg/eeg_ccnn

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder bihdm \
    --output_path results/exp1/deap_si_reg/eeg_bihdm

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality eeg --encoder gcbnet \
    --output_path results/exp1/deap_si_reg/eeg_gcbnet

# Facial Encoders
echo ""
echo "--- Facial Encoders (Regression) ---"

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder c3d \
    --output_path results/exp1/deap_si_reg/facial_c3d

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder slowfast \
    --output_path results/exp1/deap_si_reg/facial_slowfast

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder videoswin \
    --output_path results/exp1/deap_si_reg/facial_videoswin

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder formerdfer \
    --output_path results/exp1/deap_si_reg/facial_formerdfer

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder logoformer \
    --output_path results/exp1/deap_si_reg/facial_logoformer

python train_final.py \
    --dataset deap --task regression --eval_mode subject_independent \
    --data_path $DATA_PATH --modality facial --encoder est \
    --output_path results/exp1/deap_si_reg/facial_est

echo ""
echo "============================================"
echo "All experiments completed!"
echo "Classification results: results/exp1/deap_si_cls/"
echo "Regression results: results/exp1/deap_si_reg/"
echo "============================================"

# 汇总结果
echo ""
echo "Generating summaries..."
python scripts/analyze_results.py \
    --results_dir results/exp1/deap_si_cls \
    --output_file results/exp1/deap_si_cls_summary.csv

python scripts/analyze_results.py \
    --results_dir results/exp1/deap_si_reg \
    --output_file results/exp1/deap_si_reg_summary.csv

echo "Done!"