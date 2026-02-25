# data_preprocessing.py (修复维度错误版)
import numpy as np
import pickle
import os
import cv2
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import sys

# 尝试导入
try:
    from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights
except ImportError:
    pass


class DEAPPreprocessor:
    def __init__(self, data_path, video_path=None, frames_cache_path=None,
                 window_size=2.0, overlap=0.5, sampling_rate=128,
                 feature_model='mobilenet', batch_size=256):
        self.data_path = data_path
        self.video_path = video_path
        self.frames_cache_path = frames_cache_path
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.feature_model = feature_model
        self.batch_size = batch_size

        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_facial_extractor()

        print(f"⚡ 配置: Model={feature_model}, BatchSize={batch_size}, Device={self.device}")

    def _init_facial_extractor(self):
        # 初始化模型
        if self.feature_model == 'mobilenet':
            weights = MobileNet_V2_Weights.DEFAULT
            base_model = mobilenet_v2(weights=weights)
            # MobileNetV2 的 features 部分输出是 (1280, 7, 7)
            self.feature_extractor = base_model.features
            self.feature_dim = 1280
        else:
            weights = ResNet50_Weights.DEFAULT
            base_model = resnet50(weights=weights)
            # ResNet 去掉最后两层 (AvgPool 和 FC)，输出 (2048, 7, 7)
            self.feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # 定义全局平均池化层，将 (C, 7, 7) 变成 (C, 1, 1)
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

        # 预处理保持简单
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_eeg_data(self, subject_id):
        filename = f"s{subject_id:02d}.dat"
        file_path = os.path.join(self.data_path, filename)
        if not os.path.exists(file_path): return None, None
        with open(file_path, 'rb') as f:
            content = pickle.load(f, encoding='latin1')
        return content['data'][:, 0:32, :], content['labels']

    def create_sliding_windows(self, eeg_data):
        eeg_data = eeg_data[:, :, 384:]  # 去除 3s 基线
        n_trials, n_channels, n_samples = eeg_data.shape
        all_windows, window_trial_indices = [], []
        for trial_idx in range(n_trials):
            trial_data = eeg_data[trial_idx]
            n_windows = (n_samples - self.window_samples) // self.step_samples + 1
            for win_idx in range(n_windows):
                start = win_idx * self.step_samples
                end = start + self.window_samples
                if end <= n_samples:
                    all_windows.append(trial_data[:, start:end])
                    window_trial_indices.append(trial_idx)
        return np.array(all_windows), np.array(window_trial_indices)

    def normalize_eeg(self, eeg_data):
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        return (eeg_data - mean) / (std + 1e-8)

    def load_frames(self, subject_id, trial_id):
        """优先加载缓存"""
        if self.frames_cache_path:
            cache_file = os.path.join(self.frames_cache_path, f's{subject_id:02d}_trial{trial_id:02d}.npy')
            if os.path.exists(cache_file):
                return np.load(cache_file)
        return None

    def extract_facial_features_optimized(self, subject_id, trial_id, num_windows):
        all_frames = self.load_frames(subject_id, trial_id)
        if all_frames is None or len(all_frames) == 0:
            return np.zeros((num_windows, 16, self.feature_dim))

        fps = len(all_frames) / 63.0
        baseline_frames = int(3.0 * fps)
        valid_frames = all_frames[baseline_frames:]

        # 1. 收集所有需要的帧
        batch_frames = []
        for win_idx in range(num_windows):
            start_time = win_idx * self.step_samples / self.sampling_rate
            end_time = start_time + self.window_size
            f_start = int(start_time * fps)
            f_end = int(end_time * fps)

            f_end = min(f_end, len(valid_frames))
            f_start = min(f_start, f_end)

            raw_segment = valid_frames[f_start:f_end]

            if len(raw_segment) >= 16:
                indices = np.linspace(0, len(raw_segment) - 1, 16, dtype=int)
                sampled = raw_segment[indices]
            elif len(raw_segment) > 0:
                padding = [raw_segment[-1]] * (16 - len(raw_segment))
                sampled = np.concatenate([raw_segment, np.array(padding)])
            else:
                sampled = np.zeros((16, 224, 224, 3), dtype=np.uint8)

            batch_frames.append(sampled)

        # 转换为大 Tensor: (num_windows * 16, 224, 224, 3)
        all_inputs = np.concatenate(batch_frames, axis=0)

        # 2. 批量推理
        dataset = TensorDataset(torch.from_numpy(all_inputs).permute(0, 3, 1, 2))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        all_features = []

        with torch.no_grad():
            for (imgs,) in loader:
                imgs = imgs.float().to(self.device) / 255.0
                imgs = (imgs - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                       torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

                # 使用新的 amp API
                with torch.amp.autocast('cuda'):
                    feat = self.feature_extractor(imgs)  # (B, C, 7, 7)
                    feat = self.pooling(feat)  # (B, C, 1, 1) <-- 关键修复
                    feat = feat.flatten(1)  # (B, C)

                all_features.append(feat.cpu().numpy())

        # 3. 重组
        total_feats = np.concatenate(all_features, axis=0)
        final_output = total_feats.reshape(num_windows, 16, self.feature_dim)

        return final_output

    def create_labels(self, labels):
        valence = labels[:, 0]
        arousal = labels[:, 1]
        return (valence >= 5.0).astype(np.int64), (arousal >= 5.0).astype(np.int64)


def prepare_deap_data(args):
    print(f"\n{'=' * 60}\nProcessing DEAP (Optimized)\n{'=' * 60}")

    preprocessor = DEAPPreprocessor(
        args.data_path, args.video_path, args.frames_cache_path,
        args.window_size, args.overlap, feature_model=args.feature_model,
        batch_size=args.batch_size
    )

    all_data = {'eeg': [], 'facial': [], 'val': [], 'aro': [], 'sub': [], 'trial': [], 'val_s': [], 'aro_s': []}

    for subject_id in tqdm(range(1, 33), desc="Subjects"):
        eeg_data, labels = preprocessor.load_eeg_data(subject_id)
        if eeg_data is None: continue

        eeg_windows, window_trial_indices = preprocessor.create_sliding_windows(eeg_data)
        eeg_windows = preprocessor.normalize_eeg(eeg_windows)

        if args.modality in ['eeg', 'both']:
            all_data['eeg'].append(eeg_windows)

        if args.modality in ['facial', 'both']:
            subject_facial = []

            # 使用 tqdm 显示 trial 进度，避免静默太久
            for trial_id in tqdm(range(1, 41), desc=f"S{subject_id:02d} Video", leave=False):
                num_trial_windows = np.sum(window_trial_indices == (trial_id - 1))
                if num_trial_windows > 0:
                    feats = preprocessor.extract_facial_features_optimized(subject_id, trial_id, num_trial_windows)
                    subject_facial.append(feats)

            if len(subject_facial) > 0:
                all_data['facial'].append(np.concatenate(subject_facial, axis=0))
            else:
                all_data['facial'].append(np.zeros((len(eeg_windows), 16, preprocessor.feature_dim)))

        val_bin, aro_bin = preprocessor.create_labels(labels)
        all_data['val'].append(val_bin[window_trial_indices])
        all_data['aro'].append(aro_bin[window_trial_indices])
        all_data['val_s'].append(labels[window_trial_indices, 0])
        all_data['aro_s'].append(labels[window_trial_indices, 1])
        all_data['sub'].extend([subject_id] * len(eeg_windows))
        all_data['trial'].extend(window_trial_indices + 1)

    os.makedirs(args.output_path, exist_ok=True)
    print("\nSaving data...")

    if args.modality in ['eeg', 'both']:
        np.save(os.path.join(args.output_path, 'eeg_data.npy'), np.concatenate(all_data['eeg']))

    if args.modality in ['facial', 'both']:
        np.save(os.path.join(args.output_path, 'facial_data.npy'), np.concatenate(all_data['facial']))

    np.save(os.path.join(args.output_path, 'valence_labels.npy'), np.concatenate(all_data['val']))
    np.save(os.path.join(args.output_path, 'arousal_labels.npy'), np.concatenate(all_data['aro']))
    np.save(os.path.join(args.output_path, 'valence_scores.npy'), np.concatenate(all_data['val_s']))
    np.save(os.path.join(args.output_path, 'arousal_scores.npy'), np.concatenate(all_data['aro_s']))
    np.save(os.path.join(args.output_path, 'subjects.npy'), np.array(all_data['sub']))
    np.save(os.path.join(args.output_path, 'trials.npy'), np.array(all_data['trial']))

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/root/autodl-tmp/eeg/data/DEAP/data_preprocessed_python')
    parser.add_argument('--video_path', default='/root/autodl-tmp/eeg/data/DEAP/face_video')
    parser.add_argument('--frames_cache_path', default='/root/autodl-tmp/eeg/data/DEAP/frames_cache')
    parser.add_argument('--output_path', default='/root/autodl-tmp/eeg/data/DEAP/processed')
    parser.add_argument('--modality', default='both')
    parser.add_argument('--window_size', type=float, default=2.0)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--feature_model', default='mobilenet')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    prepare_deap_data(args)