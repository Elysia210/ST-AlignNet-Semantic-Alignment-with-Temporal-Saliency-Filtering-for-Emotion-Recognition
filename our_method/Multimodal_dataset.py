import os, pickle, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        eeg = torch.tensor(s['eeg'], dtype=torch.float32)        # [B,32,128] 格式按你预处理保持
        facial = torch.tensor(s['facial'], dtype=torch.float32)  # [B,T,2048]
        label = torch.tensor(s['label'], dtype=torch.float32)    # [3] (val, aro, lik)
        val = label[0].item()
        y = 1 if val > 5.0 else 0
        return {'eeg': eeg, 'facial': facial, 'label_cls': torch.tensor(y, dtype=torch.long)}

def build_loaders_binary(fold, config):
    # test
    with open(os.path.join(config["data_path"], f"fold{fold}.pkl"), "rb") as f:
        test_data = pickle.load(f)

    # train = other 9 folds
    train_data = []
    for i in range(10):
        if i == fold: continue
        with open(os.path.join(config["data_path"], f"fold{i}.pkl"), "rb") as f:
            train_data += pickle.load(f)

    np.random.seed(42)
    idx = np.random.permutation(len(train_data))
    split = int(0.9 * len(train_data))
    train_set = [train_data[i] for i in idx[:split]]
    val_set   = [train_data[i] for i in idx[split:]]

    train_ds = MultimodalDataset(train_set)
    val_ds   = MultimodalDataset(val_set)
    test_ds  = MultimodalDataset(test_data)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=config["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    return {"train": train_loader, "val": val_loader, "test": test_loader}

# ==== 新增：更稳的二分类数据集（可调阈值）====
import torch
from torch.utils.data import Dataset

# models/Multimodal_dataset.py

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, items, bin_thr=5.0, facial_mean=None, facial_std=None, eps=1e-6, target_dim=0):
        self.items = items
        self.thr = float(bin_thr)
        self.facial_mean = None if facial_mean is None else torch.tensor(facial_mean, dtype=torch.float32)
        self.facial_std  = None if facial_std  is None else torch.tensor(facial_std,  dtype=torch.float32)
        self.eps = eps
        # target_dim=0: Valence, target_dim=1: Arousal, target_dim=2: Liking
        self.target_dim = target_dim

    def __len__(self):
        return len(self.items)  # ★ 必须有

    def __getitem__(self, idx):
        s = self.items[idx]
        eeg    = torch.tensor(s['eeg'],    dtype=torch.float32)      # [32,128]（你原来已做逐通道 z-score）
        facial = torch.tensor(s['facial'], dtype=torch.float32)      # [T,2048]

        # ★ 跨被试标准化：三份 split 统一用 train 的 mean/std
        if self.facial_mean is not None and self.facial_std is not None:
            facial = (facial - self.facial_mean) / (self.facial_std + self.eps)

        # v = float(s['label'][0])         # valence
        v = float(s['label'][self.target_dim])  # <--- 改这里
        y = 1 if v > self.thr else 0
        return {'eeg': eeg, 'facial': facial, 'label_cls': torch.tensor(y, dtype=torch.long)}

# ==== 新增：LOSO 的二分类 loaders（含 class-balance 采样）====
import os, pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

def _fit_facial_stats(samples):
    # samples: list of dict，里边有 'facial' : [T,2048] 的 numpy 或 torch
    feats = []
    for s in samples:
        x = s["facial"]
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        feats.append(x.reshape(-1, x.shape[-1]))  # [T,2048] -> [T,2048]
    X = np.concatenate(feats, axis=0)  # [N_frames, 2048]
    mean = X.mean(axis=0, keepdims=True)      # [1,2048]
    std  = X.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def _apply_facial_norm_inplace(samples, mean, std):
    for s in samples:
        x = s["facial"]
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        s["facial"] = ((x - mean) / std).astype(np.float32)

def _make_weights_for_items(items, thr):
    ys = np.array([1 if float(s['label'][0]) > thr else 0 for s in items], dtype=int)
    cnt = np.bincount(ys, minlength=2).astype(float)
    # 经典反频率权重：total/(K*count[y])
    w_per_class = (ys.size) / (2.0 * np.maximum(cnt, 1.0))
    w = w_per_class[ys]
    return w.tolist(), ys.tolist(), cnt.tolist()

def _facial_mean_std(items):
    # items: list of dicts with key 'facial' -> [T,2048]
    # 统计所有时间步的 2048 维特征
    cnt, mean, M2 = 0, None, None
    for s in items:
        x = s['facial']  # np.ndarray [T,2048]
        x = x.reshape(-1, x.shape[-1])  # [N,2048]
        if mean is None:
            mean = x.mean(axis=0)
            M2   = ((x - mean)**2).sum(axis=0)
            cnt  = x.shape[0]
        else:
            # 合并两个样本集的并行算法（避免一次性堆所有帧）
            n = x.shape[0]
            new_cnt = cnt + n
            delta = x.mean(axis=0) - mean
            mean = mean + delta * (n / new_cnt)
            M2   = M2 + ((x - mean)**2).sum(axis=0) + (delta**2) * cnt * n / new_cnt
            cnt  = new_cnt
    std = np.sqrt(M2 / max(cnt - 1, 1))
    return mean.astype('float32'), std.astype('float32')

import numpy as np

def _facial_mean_std(items):
    # items: list[dict]，每个 dict['facial'] 形状 [T,2048] (np.float32)
    cnt, mean, M2 = 0, None, None
    for s in items:
        x = s['facial']
        x = x.reshape(-1, x.shape[-1])            # [N,2048]
        if mean is None:
            mean = x.mean(axis=0)
            M2   = ((x - mean)**2).sum(axis=0)
            cnt  = x.shape[0]
        else:
            n     = x.shape[0]
            new_n = cnt + n
            delta = x.mean(axis=0) - mean
            mean  = mean + delta * (n / new_n)
            M2    = M2 + ((x - mean)**2).sum(axis=0) + (delta**2) * cnt * n / new_n
            cnt   = new_n
    std = np.sqrt(M2 / max(cnt - 1, 1))
    return mean.astype('float32'), std.astype('float32')

def build_loaders_loso_binary(fold, config):
    pkl = os.path.join(config["data_path"], f"fold{fold}.pkl")
    with open(pkl, "rb") as f:
        pack = pickle.load(f)   # {'train','val','test'}

    thr = float(config.get("binary_threshold", 5.0))

    # 先算 facial 的 train 统计量
    f_mean, f_std = _facial_mean_std(pack["train"])  # ★ 只用 train 统计

    train_ds = BinaryDataset(pack["train"], bin_thr=thr, facial_mean=f_mean, facial_std=f_std, target_dim=config["target_dim"])
    val_ds = BinaryDataset(pack["val"], bin_thr=thr, facial_mean=f_mean, facial_std=f_std, target_dim=config["target_dim"])
    test_ds = BinaryDataset(pack["test"], bin_thr=thr, facial_mean=f_mean, facial_std=f_std, target_dim=config["target_dim"])

    print(f"[build] sizes | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # ⚠️ 先移除 WeightedRandomSampler，避免校准崩坏
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"], drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"],
                             shuffle=False, num_workers=config["num_workers"])

    # 如仍想看类分布：
    ys = [1 if float(s['label'][0]) > thr else 0 for s in pack["train"]]
    cnt0, cnt1 = ys.count(0), ys.count(1)
    print(f"[Fold {fold}] train class_cnt={[cnt0, cnt1]}, thr={thr}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}