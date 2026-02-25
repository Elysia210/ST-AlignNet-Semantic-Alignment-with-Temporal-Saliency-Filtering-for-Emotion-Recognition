# -*- coding: utf-8 -*-
"""
按被试独立（LOSO）划分生成数据包 - Python格式版本
- 读取DEAP数据集的 .dat 文件（Python pickle格式）
- 每个 fold 留出 1 个 subject 作为 test
- 其余 subject 合并后再按 9:1 切分出 train / val
- 每个 fold 保存为 fold{idx}_loso.pkl，内容为 dict: {'train','val','test'}

使用方法：
    python make_pkl_ind_python.py --eeg_root /path/to/data_preprocessed_python \
                                   --facial_root /path/to/Facial_CNN_Features \
                                   --save_root /path/to/make_data_ind \
                                   --max_facial_len 128

DEAP Python格式说明：
- 文件名：s01.dat ~ s32.dat (32个被试)
- 数据结构：
  - data: [40 trials, 40 channels, 8064 timepoints]
  - labels: [40 trials, 4] (valence, arousal, dominance, liking)
"""
import os
import argparse
import pickle
import numpy as np

# ======== 默认路径（可用命令行覆盖） ========
DEFAULT_EEG_ROOT = r"/root/autodl-tmp/eeg/data/DEAP/data_preprocessed_python"
DEFAULT_FACIAL_ROOT = r"/root/autodl-tmp/eeg/data/DEAP/facial_ResNet50_features"
DEFAULT_SAVE_ROOT = r"/root/autodl-tmp/eeg/data/DEAP/aligned_independent_data"

DEFAULT_MAX_FACIAL_LEN = 128
DEFAULT_SUBJECT_IDS = [f"s{str(i + 1).zfill(2)}" for i in range(32)]  # s01 ~ s32


def parse_args():
    p = argparse.ArgumentParser(description="生成LOSO数据包（Python格式）")
    p.add_argument("--eeg_root", type=str, default=DEFAULT_EEG_ROOT,
                   help="EEG .dat 文件所在目录")
    p.add_argument("--facial_root", type=str, default=DEFAULT_FACIAL_ROOT,
                   help="Facial .npy 特征文件所在目录")
    p.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT,
                   help="输出 .pkl 文件保存目录")
    p.add_argument("--max_facial_len", type=int, default=DEFAULT_MAX_FACIAL_LEN,
                   help="Facial序列最大长度（pad/truncate）")
    p.add_argument("--eeg_channels", type=int, default=32,
                   help="使用的EEG通道数（从前往后取）")
    p.add_argument("--eeg_timesteps", type=int, default=128,
                   help="使用的EEG时间步数（从前往后取）")
    p.add_argument("--num_subjects", type=int, default=32,
                   help="被试总数（DEAP标准为32）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    return p.parse_args()


def load_trials_from_subject(subject_id, eeg_root, facial_root,
                             max_facial_len=128, eeg_channels=32, eeg_timesteps=128):
    """
    读取单个 subject 的 40 个 trial（自动跳过缺失 facial .npy）并做预处理：
    - EEG：逐通道 z-score，形状 [eeg_channels, eeg_timesteps]
    - Facial：按时间维 pad/trunc 到 max_facial_len，形状 [max_facial_len, 2048]
    - Label：取前4个维度 [val, aro, dom, lik]

    Args:
        subject_id: 被试ID，如 's01'
        eeg_root: EEG .dat 文件目录
        facial_root: Facial .npy 文件目录
        max_facial_len: Facial序列最大长度
        eeg_channels: 使用的EEG通道数
        eeg_timesteps: 使用的EEG时间步数

    Returns:
        list[dict]: 每个元素形如 {'eeg', 'facial', 'label'}
    """
    # 1. 加载EEG数据（Python格式 .dat）
    eeg_path = os.path.join(eeg_root, f"{subject_id}.dat")

    if not os.path.exists(eeg_path):
        print(f"[WARN] EEG .dat missing for {subject_id}: {eeg_path}")
        return []

    # 加载pickle文件（需要encoding='latin1'因为可能是Python2保存的）
    try:
        with open(eeg_path, 'rb') as f:
            subject_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"[ERROR] Failed to load {eeg_path}: {e}")
        return []

    # DEAP数据结构：
    # subject_data['data']: [40, 40, 8064]
    # subject_data['labels']: [40, 4]
    eeg_data = subject_data['data'][:, :eeg_channels, :eeg_timesteps]  # [40, channels, timesteps]
    labels = subject_data['labels'][:, :4]  # [40, 4] - valence, arousal, dominance, liking

    # 2. 处理Facial数据
    facial_dir = os.path.join(facial_root, subject_id)

    trials = []
    missing = 0

    for i in range(40):  # DEAP有40个trial
        # Facial特征文件命名
        npy_name = f"{subject_id}_trial{str(i + 1).zfill(2)}.npy"
        facial_path = os.path.join(facial_dir, npy_name)

        if not os.path.exists(facial_path):
            missing += 1
            continue

        # 3. EEG预处理：逐通道 z-score
        eeg = eeg_data[i].astype(np.float32)  # [channels, timesteps]
        eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-6)

        # 4. Facial预处理：pad/truncate 到固定长度
        facial = np.load(facial_path).astype(np.float32)  # [T, 2048]

        if facial.shape[0] > max_facial_len:
            facial = facial[:max_facial_len]
        elif facial.shape[0] < max_facial_len:
            pad = np.zeros((max_facial_len - facial.shape[0], facial.shape[1]), dtype=np.float32)
            facial = np.concatenate([facial, pad], axis=0)

        # 5. 标签（valence, arousal, dominance, liking）
        label = labels[i].astype(np.float32)  # [4]

        trials.append({
            "eeg": eeg,  # [channels, timesteps]
            "facial": facial,  # [max_facial_len, 2048]
            "label": label  # [4]
        })

    print(f"[{subject_id}] valid={len(trials):2d} | missing facial={missing:2d}")
    return trials


def build_subject_trials(subject_ids, eeg_root, facial_root, max_facial_len,
                         eeg_channels, eeg_timesteps):
    """
    返回 dict: {sid: list_of_trials}
    """
    subj_trials = {}
    total = 0

    for sid in subject_ids:
        trials = load_trials_from_subject(
            sid, eeg_root, facial_root, max_facial_len, eeg_channels, eeg_timesteps
        )
        subj_trials[sid] = trials
        total += len(trials)

    print(f"\n[INFO] Collected total {total} trials from {len(subject_ids)} subjects.")
    return subj_trials


def split_save_loso(subj_trials, save_dir, seed=42):
    """
    为每个 subject 生成一个 LOSO 折：
      - test = 该 subject 的全部 trial
      - train/val = 其余 subject 的 trial 合并后，按 9:1 随机切分
    """
    os.makedirs(save_dir, exist_ok=True)
    subject_ids = list(subj_trials.keys())
    rng = np.random.default_rng(seed)

    fold_count = 0

    for test_sid in subject_ids:
        test_data = subj_trials[test_sid]

        # 跳过没有有效数据的被试
        if len(test_data) == 0:
            print(f"[SKIP] No test data for {test_sid}.")
            continue

        # 合并其他被试的数据
        train_val = []
        for sid in subject_ids:
            if sid == test_sid:
                continue
            train_val.extend(subj_trials[sid])

        if len(train_val) == 0:
            print(f"[SKIP] No train/val data when testing on {test_sid}.")
            continue

        # 随机划分train/val（9:1）
        idx = rng.permutation(len(train_val))
        split = int(0.9 * len(train_val))
        train_idx, val_idx = idx[:split], idx[split:]

        train_data = [train_val[i] for i in train_idx]
        val_data = [train_val[i] for i in val_idx]

        # 保存
        pack = {"train": train_data, "val": val_data, "test": test_data}

        out_path = os.path.join(save_dir, f"fold{fold_count}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(pack, f)

        print(f"[SAVE] fold{fold_count}_loso.pkl | test={test_sid:>3s} | "
              f"train={len(train_data):4d} | val={len(val_data):4d} | test={len(test_data):3d}")

        fold_count += 1

    print(f"\n[DONE] Total LOSO folds saved: {fold_count}")


def main():
    args = parse_args()

    print("=" * 60)
    print("LOSO Data Packer (DEAP Python Format)")
    print("=" * 60)
    print(f"EEG Root    : {args.eeg_root}")
    print(f"Facial Root : {args.facial_root}")
    print(f"Save To     : {args.save_root}")
    print(f"Max Facial  : {args.max_facial_len}")
    print(f"EEG Shape   : [{args.eeg_channels}, {args.eeg_timesteps}]")
    print(f"Num Subjects: {args.num_subjects}")
    print("=" * 60)
    print()

    # 生成被试ID列表
    subject_ids = [f"s{str(i + 1).zfill(2)}" for i in range(args.num_subjects)]

    # 加载所有被试的trial数据
    subj_trials = build_subject_trials(
        subject_ids,
        args.eeg_root,
        args.facial_root,
        args.max_facial_len,
        args.eeg_channels,
        args.eeg_timesteps
    )

    # 划分并保存LOSO折
    split_save_loso(subj_trials, args.save_root, seed=args.seed)


if __name__ == "__main__":
    main()
