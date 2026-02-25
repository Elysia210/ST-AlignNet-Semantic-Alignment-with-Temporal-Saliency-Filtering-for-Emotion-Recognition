import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path

INPUT_ROOT = Path("/root/autodl-tmp/eeg/data/DEAP/face_frames")
OUTPUT_ROOT = Path("/root/autodl-tmp/eeg/data/DEAP/facial_ResNet50_features")

# ResNet50 特征提取
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
for subj_path in sorted(INPUT_ROOT.iterdir()):
    if not subj_path.is_dir():
        continue
    subj_id = subj_path.name

    for trial_path in sorted(subj_path.iterdir()):
        if not trial_path.is_dir():
            continue

        frame_files = sorted([
            f for f in trial_path.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if len(frame_files) == 0:
            print(f" No frames found in {trial_path}")
            continue

        features = []
        for frame_file in tqdm(frame_files, desc=f"{subj_id}/{trial_path.name}"):
            try:
                img = Image.open(frame_file).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = resnet(img_tensor).squeeze().cpu().numpy()
                features.append(feat)
            except Exception as e:
                print(f" Failed on {frame_file}: {e}")

        features = np.stack(features, axis=0)
        out_dir = OUTPUT_ROOT / subj_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{trial_path.name}.npy"
        np.save(out_path, features)
        print(f" Saved {out_path} → shape {features.shape}")
