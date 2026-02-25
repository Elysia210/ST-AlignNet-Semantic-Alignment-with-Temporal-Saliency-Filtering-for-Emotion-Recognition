# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import numpy as np

# 引入 torch 相关库用于优化
import torch
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ========== 配置 ==========
VIDEO_ROOT = Path(r"/root/autodl-tmp/eeg/data/DEAP/face_video")
FRAMES_ROOT = Path(r"/root/autodl-tmp/eeg/data/DEAP/facial_frames")
FEAT_ROOT = Path(r"/root/autodl-tmp/eeg/data/DEAP/facial_ResNet50_features")

MISSING = {
    "s07": list(range(1, 22)),
    "s09": list(range(2, 41)),
    "s11": list(range(1, 19)) + [20]
}

TARGET_SIZE = (224, 224)
MIN_FRAMES = 8
BATCH_SIZE = 128  # 批处理大小，显存不够改 64
NUM_WORKERS = 4  # 读图进程数


# ========== 简单的 Dataset 类，用于批量读图 ==========
class FrameDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            # 遇到坏图返回黑图防止崩掉
            print(f"[WARN] Read error {path}: {e}")
            return torch.zeros(3, 224, 224)


# ========= 抽帧（逻辑未变，仅保持结构） =========
def extract_frames_if_needed(subj: str, t: int) -> bool:
    frames_dir = FRAMES_ROOT / subj / f"trial{t:02d}"

    # 1) 检查是否已存在
    if frames_dir.is_dir():
        # 简单统计文件数
        n = len(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")))
        if n >= MIN_FRAMES:
            print(f"[SKIP-frames] {subj} trial{t:02d} 已有 {n} 帧")
            return True

    # 2) 抽帧
    avi = VIDEO_ROOT / subj / f"{subj}_trial{t:02d}.avi"
    if not avi.exists():
        print(f"[MISS-AVI] 源视频不存在：{avi}")
        return False

    try:
        import cv2
    except ImportError:
        print("[ERR] pip install opencv-python")
        return False

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(avi))
    if not cap.isOpened():
        print(f"[ERR] 打不开视频：{avi}")
        return False

    idx, saved = 1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if TARGET_SIZE is not None:
            frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frames_dir / f"{idx:04d}.png"), frame)
        idx += 1
        saved += 1
    cap.release()

    if saved >= MIN_FRAMES:
        print(f"[OK-frames] {subj} trial{t:02d} 抽帧 {saved} 张")
        return True
    else:
        print(f"[FAIL-frames] {subj} trial{t:02d} 仅抽到 {saved} 张")
        return False


# ========= ResNet50 特征（核心优化部分） =========
def extract_feat_if_needed(subj: str, t: int, model, transform, device) -> bool:
    out_dir = FEAT_ROOT / subj
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"{subj}_trial{t:02d}.npy"

    if out_npy.exists():
        print(f"[SKIP-feat] 已有特征：{out_npy}")
        return True

    frames_dir = FRAMES_ROOT / subj / f"trial{t:02d}"
    if not frames_dir.is_dir():
        return False

    # 获取所有图片路径
    frame_files = sorted([
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])

    if len(frame_files) == 0:
        print(f"[FAIL-feat] 目录为空：{frames_dir}")
        return False

    # --- 优化：使用 DataLoader 批量处理 ---
    dataset = FrameDataset(frame_files, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    feats_list = []

    # 推理
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # ResNet 输出 [B, 2048, 1, 1] -> [B, 2048]
            output = model(batch)
            output = output.reshape(output.shape[0], -1).cpu().numpy()
            feats_list.append(output)

    if not feats_list:
        return False

    arr = np.concatenate(feats_list, axis=0)  # [Total_Frames, 2048]
    np.save(out_npy, arr)
    print(f"[OK-feat] 保存 {out_npy}  shape={arr.shape}")
    return True


def main():
    # --- 1. 全局初始化模型（只做一次，极大提升速度）---
    print("正在初始化 ResNet50...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # 去掉分类层
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    backbone.to(device).eval()

    # 预处理
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"模型初始化完成，使用设备: {device}")

    # --- 2. 开始循环 ---
    any_fail = False
    for subj, trials in MISSING.items():
        for t in trials:
            # Step 1: 抽帧
            ok_frames = extract_frames_if_needed(subj, t)
            if not ok_frames:
                any_fail = True
                continue  # 没帧就没法提特征，跳过

            # Step 2: 提特征 (传入全局模型)
            ok_feat = extract_feat_if_needed(subj, t, backbone, tfm, device)
            if not ok_feat:
                any_fail = True

    print("\n=== DONE ===")
    if any_fail:
        print("提示：部分条目处理失败（可能源视频缺失或帧读取错误）。")
    else:
        print("所有任务完成。")


if __name__ == "__main__":
    main()