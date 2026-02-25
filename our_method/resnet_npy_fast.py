import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

INPUT_ROOT = Path("/root/autodl-tmp/eeg/data/DEAP/face_frames")
OUTPUT_ROOT = Path("/root/autodl-tmp/eeg/data/DEAP/facial_ResNet50_features")

# --- 配置参数 ---
BATCH_SIZE = 128  # 显存够大可调至 256，显存小改 64
NUM_WORKERS = 8  # CPU 核心数，AutoDL 机器通常可以设为 4-16


# --- 定义简单的 Dataset 用于并行加载 ---
class FrameDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # 即使某张图坏了也尽量不崩，返回全0作为替代（或者你可以选择报错）
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回一个空的 tensor 防止 DataLoader 崩溃
            return torch.zeros(3, 224, 224)


# ResNet50 特征提取
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.to(device)
resnet.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 处理所有被试
# 使用 list 直接获取所有被试目录，方便看进度
subj_dirs = sorted([d for d in INPUT_ROOT.iterdir() if d.is_dir()])

for subj_path in tqdm(subj_dirs, desc="Subjects"):
    subj_id = subj_path.name
    out_dir = OUTPUT_ROOT / subj_id
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_dirs = sorted([d for d in subj_path.iterdir() if d.is_dir()])

    for trial_path in trial_dirs:
        # --- [优化点 1] 断点续传检测 ---
        out_path = out_dir / f"{trial_path.name}.npy"
        if out_path.exists():
            print(f"Skipping {trial_path.name}, already exists.") # 嫌啰嗦可以注释掉
            continue

        frame_files = sorted([
            f for f in trial_path.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if len(frame_files) == 0:
            continue

        # --- [优化点 2] 使用 DataLoader 批量处理 ---
        dataset = FrameDataset(frame_files, transform)
        # pin_memory=True 加速数据从 CPU 传到 GPU
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        features_list = []

        with torch.no_grad():
            # 使用 tqdm 显示当前 trial 的 batch 进度
            for batch_imgs in loader:
                batch_imgs = batch_imgs.to(device)

                # ResNet 输出是 [B, 2048, 1, 1]
                outputs = resnet(batch_imgs)

                # 展平为 [B, 2048] 并转为 numpy
                outputs = outputs.reshape(outputs.shape[0], -1).cpu().numpy()
                features_list.append(outputs)

        if len(features_list) > 0:
            # 拼接所有 batch 的结果
            all_features = np.concatenate(features_list, axis=0)
            np.save(out_path, all_features)
            # print(f"Saved {out_path} -> shape {all_features.shape}")