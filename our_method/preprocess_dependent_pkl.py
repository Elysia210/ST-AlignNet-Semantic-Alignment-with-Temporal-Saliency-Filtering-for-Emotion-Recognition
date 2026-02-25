import os
import numpy as np
import pickle
from sklearn.model_selection import KFold

# ==== 路径配置 ====
# 注意：DEAP 的 Python 版数据通常叫 data_preprocessed_python
# 请确保 eeg_root 指向包含 s01.dat, s02.dat ... 的文件夹
eeg_root = r"/root/autodl-tmp/eeg/data/DEAP/data_preprocessed_python"
facial_root = r"/root/autodl-tmp/eeg/data/DEAP/facial_ResNet50_features"
save_root = '/root/autodl-tmp/eeg/data/DEAP/aligned_dependent_data'

max_facial_len = 128
subject_ids = [f's{str(i + 1).zfill(2)}' for i in range(22)]  # s01 ~ s22


# 单个 subject 处理
def load_trials_from_subject(subject_id):
    # DEAP Python 格式文件后缀通常是 .dat
    eeg_path = os.path.join(eeg_root, f'{subject_id}.dat')
    facial_dir = os.path.join(facial_root, subject_id)

    # --- 修改部分开始：加载 Python 格式数据 ---
    if not os.path.exists(eeg_path):
        print(f"[Error] EEG file not found: {eeg_path}")
        return []

    with open(eeg_path, 'rb') as f:
        # DEAP 数据是用 Python 2 生成的，在 Python 3 中加载必须加 encoding='latin1'
        content = pickle.load(f, encoding='latin1')

    # content 是一个字典：
    # 'data'   shape: (40, 40, 8064) -> (trials, channels, data)
    # 'labels' shape: (40, 4)        -> (trials, labels)

    # 保持你的切片逻辑：取前32个通道(EEG)，前128个时间点
    # 注意：原始 DEAP 采样率 128Hz，长度 8064。你这里取 128 代表只取了前 1 秒
    eeg_data = content['data'][:, :32, :128]
    labels = content['labels'][:, :3]
    # --- 修改部分结束 ---

    trial_list = []
    for i in range(40):
        # 你的 facial 特征文件命名逻辑
        npy_name = f'{subject_id}_trial{str(i + 1).zfill(2)}.npy'
        facial_path = os.path.join(facial_dir, npy_name)

        # 跳过缺失 .npy
        if not os.path.exists(facial_path):
            print(f" Missing facial file: {facial_path} → skipping trial {i + 1}")
            continue

        eeg = eeg_data[i].astype(np.float32)  # [32, 128]
        try:
            facial = np.load(facial_path).astype(np.float32)  # [T, 2048]
        except Exception as e:
            print(f" Error loading {facial_path}: {e}")
            continue

        # EEG 标准化（逐通道 z-score）
        # 加上 1e-6 防止除以 0
        eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-6)

        # Facial padding / truncating
        if facial.shape[0] > max_facial_len:
            facial = facial[:max_facial_len]
        elif facial.shape[0] < max_facial_len:
            pad_len = max_facial_len - facial.shape[0]
            # 补零
            pad = np.zeros((pad_len, facial.shape[1]), dtype=np.float32)
            facial = np.concatenate([facial, pad], axis=0)

        label = labels[i]  # [3]
        trial_list.append({
            'eeg': eeg,
            'facial': facial,
            'label': label.astype(np.float32)
        })

    print(f" Processed {subject_id}: {len(trial_list)} valid trials.")
    return trial_list


# ==== 汇总所有 subject 的 trial 数据 ====
def build_all_trials():
    all_trials = []
    for sid in subject_ids:
        trials = load_trials_from_subject(sid)
        all_trials.extend(trials)
    print(f"\n Total collected trials: {len(all_trials)}")
    return all_trials


# ==== 划分 10 折并保存 ====
def split_and_save_folds(trial_list, save_dir, n_folds=10):
    if not trial_list:
        print("No trials to split!")
        return

    os.makedirs(save_dir, exist_ok=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 必须转为 object array 才能让 sklearn 切分包含 dict 的 list
    trial_array = np.array(trial_list, dtype=object)

    for fold, (_, val_idx) in enumerate(kf.split(trial_array)):
        fold_data = trial_array[val_idx].tolist()
        save_path = os.path.join(save_dir, f'fold{fold}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(fold_data, f)
        print(f"Saved fold{fold}.pkl → {len(fold_data)} trials.")


# ==== 主函数 ====
if __name__ == "__main__":
    print(" Loading all trial-level data from EEG (Python format) + Facial...")
    all_data = build_all_trials()
    print("\n Splitting into 10 folds...")
    split_and_save_folds(all_data, save_dir=save_root)
    print("\n Done.")