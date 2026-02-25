# data_preprocessing_mahnob.py
"""
数据预处理脚本 - MAHNOB-HCI 终极修复版
修复内容：
1. Trial ID 生成：解析文件夹名，确保 Subject-Dependent CV 能正确划分折数。
2. 视频文件选择：优先锁定 'C1' (面部摄像头) 视频，防止读取到屏幕录像。
3. 标签清洗：严格剔除 XML 缺失或标签无效的数据。
4. 通道匹配：放宽 EEG 通道匹配策略，保留更多有效数据。
"""

import numpy as np
import os
import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
import argparse
import glob
import xml.etree.ElementTree as ET
import warnings
import sys

# 忽略 MNE 读取 BDF 时的警告
warnings.filterwarnings("ignore")

try:
    import mne
except ImportError:
    print("Warning: 'mne' library not found. MAHNOB preprocessing will fail.")


class BasePreprocessor:
    """预处理基类：视觉特征提取"""

    def __init__(self, data_path, output_path, window_size=2.0, overlap=0.5,
                 sampling_rate=128, feature_model='mobilenet', batch_size=256, device=None):
        self.data_path = data_path
        self.output_path = output_path
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.feature_model = feature_model
        self.batch_size = batch_size

        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_facial_extractor()

    def _init_facial_extractor(self):
        print(f"⚡ 初始化视觉模型: {self.feature_model} on {self.device}")
        if self.feature_model == 'mobilenet':
            weights = MobileNet_V2_Weights.DEFAULT
            base_model = mobilenet_v2(weights=weights)
            self.feature_extractor = base_model.features
            self.feature_dim = 1280
        else:
            weights = ResNet50_Weights.DEFAULT
            base_model = resnet50(weights=weights)
            self.feature_extractor = torch.nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

    def batch_inference(self, batch_frames):
        if len(batch_frames) == 0:
            return None

        all_inputs = np.concatenate(batch_frames, axis=0)
        # 归一化已经在读取时完成，这里转 Tensor
        dataset = TensorDataset(torch.from_numpy(all_inputs).permute(0, 3, 1, 2))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        all_features = []
        with torch.no_grad():
            for (imgs,) in loader:
                imgs = imgs.float().to(self.device) / 255.0
                # ImageNet 标准归一化
                imgs = (imgs - torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)) / \
                       torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

                with torch.amp.autocast('cuda'):
                    feat = self.feature_extractor(imgs)
                    feat = self.pooling(feat)
                    feat = feat.flatten(1)
                all_features.append(feat.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def normalize_eeg(self, eeg_data):
        # Z-score normalization per channel
        mean = np.mean(eeg_data, axis=-1, keepdims=True)
        std = np.std(eeg_data, axis=-1, keepdims=True)
        return (eeg_data - mean) / (std + 1e-8)


class MahnobPreprocessor(BasePreprocessor):
    """MAHNOB-HCI 专用处理器"""

    def __init__(self, data_path, video_path=None, **kwargs):
        super().__init__(data_path, **kwargs)
        # MAHNOB 标准 32 通道
        self.target_channels = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
            'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
        ]
        self.root_dir = os.path.join(data_path, 'Sessions')
        self.session_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])
        print(f"🔍 发现 {len(self.session_dirs)} 个 Session 目录")

    def parse_session_xml(self, xml_path):
        """解析 XML 标签，增加严格校验"""
        if not os.path.exists(xml_path):
            return None, "XML missing"
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            subject_node = root.find('subject')
            if subject_node is None:
                return None, "Subject tag missing"
            subject_id = int(subject_node.attrib.get('id', -1))

            # MAHNOB 标签有时会有不同的 Attribute key，这里主要取 feltVlnc/feltArsl
            valence = root.attrib.get('feltVlnc')
            arousal = root.attrib.get('feltArsl')

            if valence is None or arousal is None:
                return None, "Labels None"

            # 转换 int 失败或值为 -1 都视为无效
            try:
                v_score = int(valence)
                a_score = int(arousal)
            except ValueError:
                return None, "Labels non-int"

            # 过滤未打分的 Session (通常 -1 或 0 代表无效)
            if v_score <= 0 or a_score <= 0:
                return None, "Invalid scores (<=0)"

            return (subject_id, v_score, a_score), "Success"
        except Exception as e:
            return None, f"Parse Error: {str(e)}"

    def process_bdf(self, bdf_path):
        """读取 BDF 并对齐通道"""
        if not os.path.exists(bdf_path): return None

        try:
            raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
        except Exception:
            return None

        # 通道名称清洗 (去除 '1Fp1' 中的 '1')
        available_chans = raw.info['ch_names']
        mapping = {}
        for ch in available_chans:
            clean_name = ch.lstrip('0123456789')
            if clean_name in self.target_channels:
                mapping[ch] = clean_name

        if mapping:
            raw.rename_channels(mapping)

        # 数据提取与补全
        final_data = []
        current_data = raw.get_data()
        current_chans = raw.info['ch_names']
        ch_to_idx = {name: i for i, name in enumerate(current_chans)}

        found_count = 0
        for target in self.target_channels:
            if target in ch_to_idx:
                idx = ch_to_idx[target]
                final_data.append(current_data[idx])
                found_count += 1
            else:
                # 缺失通道用 0 填充
                final_data.append(np.zeros(current_data.shape[1]))

        # 如果缺失通道太多 (>12个缺失)，丢弃该数据
        if found_count < 20:
            return None

        data = np.array(final_data)

        # 重采样
        if int(raw.info['sfreq']) != self.sampling_rate:
            # 简单的手动重采样或使用 mne resample
            # 为了速度和兼容性，这里使用 MNE 对象进行重采样
            info = mne.create_info(ch_names=self.target_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
            raw_temp = mne.io.RawArray(data, info, verbose=False)
            raw_temp.resample(self.sampling_rate, verbose=False)
            raw_temp.filter(1., 45., verbose=False)  # 带通滤波
            data = raw_temp.get_data()

        return data

    def extract_video_path(self, session_dir):
        """优先选择 C1 (Face) 摄像头的视频"""
        all_avis = glob.glob(os.path.join(session_dir, "*.avi"))
        if not all_avis: return None

        # 1. 优先找 C1 (Camera 1 - Face)
        c1_files = [f for f in all_avis if 'C1' in f]
        if c1_files: return c1_files[0]

        # 2. 其次找不带 BW (Black & White) 的彩色视频
        color_files = [f for f in all_avis if 'BW' not in f]
        if color_files: return color_files[0]

        # 3. 最后随便返回一个 (由于上面逻辑很强，通常不会走到这)
        return all_avis[0]

    def process_one_session(self, session_dir):
        xml_path = os.path.join(session_dir, "session.xml")
        bdf_files = glob.glob(os.path.join(session_dir, "*.bdf"))

        meta, msg = self.parse_session_xml(xml_path)
        if meta is None:
            return None

        subject_id, valence_score, arousal_score = meta

        # === 修复: 从文件夹名获取 Trial ID ===
        try:
            trial_id = int(os.path.basename(session_dir))
        except:
            # 如果文件夹名不是数字，使用 hash 生成唯一 ID
            trial_id = abs(hash(os.path.basename(session_dir))) % 100000

        if not bdf_files: return None

        # 处理 EEG
        eeg_raw = self.process_bdf(bdf_files[0])
        if eeg_raw is None: return None

        n_channels, n_samples = eeg_raw.shape
        # 计算滑动窗口数量
        n_windows = (n_samples - self.window_samples) // self.step_samples + 1

        if n_windows <= 0: return None

        # 切分 EEG
        eeg_windows = []
        for i in range(n_windows):
            start = i * self.step_samples
            end = start + self.window_samples
            eeg_windows.append(eeg_raw[:, start:end])
        eeg_windows = np.array(eeg_windows)
        eeg_windows = self.normalize_eeg(eeg_windows)

        # 处理 Facial (如果需要)
        facial_feats = np.zeros((n_windows, 16, self.feature_dim))
        video_path = self.extract_video_path(session_dir)

        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = cap.get(cv2.CAP_PROP_FPS)
                if vid_fps <= 0: vid_fps = 60.0

                batch_frames = []
                # 仅处理与 EEG 对齐的窗口
                for i in range(n_windows):
                    start_sec = i * self.step_samples / self.sampling_rate
                    end_sec = start_sec + self.window_size

                    f_start = int(start_sec * vid_fps)
                    f_end = int(end_sec * vid_fps)

                    # 在时间窗口内均匀采样 16 帧
                    indices = np.linspace(f_start, min(f_end, total_frames) - 1, 16).astype(int)

                    win_frames = []
                    current_pos = -1
                    for idx in indices:
                        if idx >= total_frames: idx = total_frames - 1
                        if idx < 0: idx = 0

                        # 优化 seek：如果连续读取不需要 set
                        if idx != current_pos + 1:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

                        ret, frame = cap.read()
                        current_pos = idx

                        if ret:
                            frame = cv2.resize(frame, (224, 224))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            win_frames.append(frame)
                        else:
                            # 视频读取失败，补黑帧
                            win_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

                    batch_frames.append(np.array(win_frames))

                cap.release()

                # 批量推理
                if len(batch_frames) > 0:
                    facial_feats = self.batch_inference(batch_frames)
                    facial_feats = facial_feats.reshape(n_windows, 16, self.feature_dim)

        return {
            'eeg': eeg_windows,
            'facial': facial_feats,
            'sub': subject_id,
            'trial': trial_id,  # 关键修复
            'val': valence_score,
            'aro': arousal_score
        }

    def run(self):
        all_data = {'eeg': [], 'facial': [], 'val': [], 'aro': [],
                    'sub': [], 'trial': [], 'val_s': [], 'aro_s': []}

        valid_count = 0
        valid_trials_set = set()

        for session_dir in tqdm(self.session_dirs, desc="Processing MAHNOB Sessions"):
            res = self.process_one_session(session_dir)
            if res is None: continue

            valid_count += 1
            n_wins = len(res['eeg'])
            valid_trials_set.add(res['trial'])

            all_data['eeg'].append(res['eeg'])
            all_data['facial'].append(res['facial'])

            all_data['sub'].extend([res['sub']] * n_wins)
            all_data['trial'].extend([res['trial']] * n_wins)
            all_data['val_s'].extend([res['val']] * n_wins)
            all_data['aro_s'].extend([res['aro']] * n_wins)

            # 标签二值化 (Threshold = 5)
            # MAHNOB 1-9: 1-4 Low (0), 5-9 High (1)
            val_bin = 1 if res['val'] >= 5 else 0
            aro_bin = 1 if res['aro'] >= 5 else 0
            all_data['val'].extend([val_bin] * n_wins)
            all_data['aro'].extend([aro_bin] * n_wins)

        if valid_count == 0:
            print("❌ Error: No valid sessions processed.")
            sys.exit(1)

        print(f"\n✅ Processed {valid_count} sessions.")

        # === 核心诊断：输出标签分布 (防止 5% 准确率重演) ===
        val_arr = np.array(all_data['val'])
        aro_arr = np.array(all_data['aro'])

        print(f"\n📊 Label Distribution Diagnostics:")
        print(f"  Total Windows: {len(val_arr)}")
        print(f"  Valence (0/1): {np.sum(val_arr == 0)} vs {np.sum(val_arr == 1)} (Ratio 1: {np.mean(val_arr):.2f})")
        print(f"  Arousal (0/1): {np.sum(aro_arr == 0)} vs {np.sum(aro_arr == 1)} (Ratio 1: {np.mean(aro_arr):.2f})")
        print(f"  Unique Trials: {len(valid_trials_set)} (Example: {list(valid_trials_set)[:5]})")

        if np.sum(aro_arr == 1) == 0 or np.sum(aro_arr == 0) == 0:
            print(
                "\n⚠️⚠️⚠️ CRITICAL WARNING: Arousal labels contain ONLY ONE CLASS. Accuracy will be 0% or 100%. Check XML parsing!")

        print(f"\n💾 Saving to {self.output_path}...")
        os.makedirs(self.output_path, exist_ok=True)

        np.save(os.path.join(self.output_path, 'eeg_data.npy'), np.concatenate(all_data['eeg']))
        np.save(os.path.join(self.output_path, 'facial_data.npy'), np.concatenate(all_data['facial']))
        np.save(os.path.join(self.output_path, 'valence_labels.npy'), val_arr)
        np.save(os.path.join(self.output_path, 'arousal_labels.npy'), aro_arr)
        np.save(os.path.join(self.output_path, 'valence_scores.npy'), np.array(all_data['val_s']))
        np.save(os.path.join(self.output_path, 'arousal_scores.npy'), np.array(all_data['aro_s']))
        np.save(os.path.join(self.output_path, 'subjects.npy'), np.array(all_data['sub']))
        np.save(os.path.join(self.output_path, 'trials.npy'), np.array(all_data['trial']))

        print("✅ MAHNOB Processing Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mahnob', choices=['deap', 'mahnob'])
    parser.add_argument('--data_path', default='/root/autodl-tmp/eeg/data/MAHNOB/mahnob')
    parser.add_argument('--output_path', default='/root/autodl-tmp/eeg/data/MAHNOB/processed')
    parser.add_argument('--feature_model', default='mobilenet', help="resnet or mobilenet")
    args = parser.parse_args()

    if args.dataset == 'deap':
        print("Use original DEAP script.")
    elif args.dataset == 'mahnob':
        processor = MahnobPreprocessor(
            args.data_path, output_path=args.output_path, feature_model=args.feature_model
        )
        processor.run()


if __name__ == "__main__":
    main()