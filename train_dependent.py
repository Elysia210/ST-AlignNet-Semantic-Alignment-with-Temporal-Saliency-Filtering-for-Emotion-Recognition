# train_facial_fixed.py
"""
训练脚本 - 修复版
保留了用户所有的 CV 逻辑和参数解析。
修复内容：
1. Dataset: 移除 eeg.unsqueeze(0)，解决维度冲突。
2. Optimizer: 使用 AdamW 替代 Adam。
3. Scheduler: 引入 OneCycleLR。
4. Loss: 引入 Label Smoothing。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import argparse
import os
import json
from tqdm import tqdm
import yaml
import sys
import torch.nn.functional as F

# Import encoders
try:
    from models.eeg_encoders import (
        EEGNet, DGCNN, LGGNet, TSception, CCNN, BiHDM, GCBNet
    )
    from models.facial_encoders import (
        C3D, SlowFast, VideoSwin, FormerDFER, LOGOFormer, EST
    )
except ImportError:
    pass


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for Emotion Recognition
    支持 memory-mapped numpy array
    """

    def __init__(self, eeg_data, facial_data, valence_labels, arousal_labels,
                 valence_scores, arousal_scores, modality='both'):
        self.eeg_data = eeg_data
        self.facial_data = facial_data
        self.valence_labels = torch.LongTensor(valence_labels)
        self.arousal_labels = torch.LongTensor(arousal_labels)
        self.valence_scores = torch.FloatTensor(valence_scores)
        self.arousal_scores = torch.FloatTensor(arousal_scores)
        self.modality = modality

    def __len__(self):
        return len(self.valence_labels)

    def __getitem__(self, idx):
        item = {
            'valence_label': self.valence_labels[idx],
            'arousal_label': self.arousal_labels[idx],
            'valence_score': self.valence_scores[idx],
            'arousal_score': self.arousal_scores[idx]
        }

        # 动态读取：只有在需要时才从磁盘(mmap)加载数据并转为Tensor
        if self.modality in ['eeg', 'both'] and self.eeg_data is not None:
            data = np.array(self.eeg_data[idx])
            # [Fix]: 移除 unsqueeze(0)，因为模型内部会处理，或者数据已经是 (32, 256)
            item['eeg'] = torch.from_numpy(data).float()

        if self.modality in ['facial', 'both'] and self.facial_data is not None:
            data = np.array(self.facial_data[idx])
            item['facial'] = torch.from_numpy(data).float()

        return item


class ClassificationModel(nn.Module):
    def __init__(self, encoder, encoder_output_dim, n_classes=2):
        super(ClassificationModel, self).__init__()
        self.encoder = encoder
        self.valence_classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_classes)
        )
        self.arousal_classifier = nn.Sequential(
            nn.Linear(encoder_output_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.valence_classifier(features), self.arousal_classifier(features)


class RegressionModel(nn.Module):
    def __init__(self, encoder, encoder_output_dim):
        super(RegressionModel, self).__init__()
        self.encoder = encoder
        self.valence_regressor = nn.Sequential(
            nn.Linear(encoder_output_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1)
        )
        self.arousal_regressor = nn.Sequential(
            nn.Linear(encoder_output_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.valence_regressor(features), self.arousal_regressor(features)


def create_encoder(args):
    """根据参数创建encoder"""
    facial_input_dim = 1280

    if args.modality == 'eeg':
        # EEG encoders - 256时间点
        if args.encoder == 'eegnet':
            encoder = EEGNet(n_channels=32, n_timepoints=256)
        elif args.encoder == 'dgcnn':
            encoder = DGCNN(n_channels=32, n_timepoints=256)
        elif args.encoder == 'lggnet':
            encoder = LGGNet(n_channels=32, n_timepoints=256)
        elif args.encoder == 'tsception':
            encoder = TSception(n_channels=32, n_timepoints=256)
        elif args.encoder == 'ccnn':
            encoder = CCNN(n_channels=32, n_timepoints=256)
        elif args.encoder == 'bihdm':
            encoder = BiHDM(n_channels=32, n_timepoints=256)
        elif args.encoder == 'gcbnet':
            encoder = GCBNet(n_channels=32, n_timepoints=256)
        else:
            raise ValueError(f"Unknown EEG encoder: {args.encoder}")

    elif args.modality == 'facial':
        if args.encoder == 'c3d':
            encoder = C3D(n_timepoints=16, input_dim=facial_input_dim)
        elif args.encoder == 'slowfast':
            encoder = SlowFast(n_timepoints=16, input_dim=facial_input_dim)
        elif args.encoder == 'videoswin':
            encoder = VideoSwin(n_timepoints=16, input_dim=facial_input_dim)
        elif args.encoder == 'formerdfer':
            encoder = FormerDFER(n_timepoints=16, input_dim=facial_input_dim)
        elif args.encoder == 'logoformer':
            encoder = LOGOFormer(n_timepoints=16, input_dim=facial_input_dim)
        elif args.encoder == 'est':
            encoder = EST(n_timepoints=16, input_dim=facial_input_dim)
        else:
            raise ValueError(f"Unknown Facial encoder: {args.encoder}")
    else:
        raise ValueError("Multimodal not implemented in this script. Use modality='eeg' or 'facial'.")

    return encoder


def train_one_epoch_classification(model, dataloader, optimizer, device, modality, scheduler=None, pbar=None):
    model.train()
    total_loss = 0
    # [Fix]: 使用 Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for batch in dataloader:
        inputs = batch[modality].to(device)
        valence_labels = batch['valence_label'].to(device)
        arousal_labels = batch['arousal_label'].to(device)

        optimizer.zero_grad()
        valence_logits, arousal_logits = model(inputs)

        loss = criterion(valence_logits, valence_labels) + \
               criterion(arousal_logits, arousal_labels)

        loss.backward()

        # [Fix]: 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # [Fix]: Scheduler step (OneCycleLR 需要每个 step 更新)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if pbar:
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def train_one_epoch_regression(model, dataloader, optimizer, device, modality, scheduler=None, pbar=None):
    model.train()
    total_loss = 0
    criterion = nn.MSELoss()

    for batch in dataloader:
        inputs = batch[modality].to(device)
        valence_scores = batch['valence_score'].to(device).float()
        arousal_scores = batch['arousal_score'].to(device).float()

        optimizer.zero_grad()
        valence_pred, arousal_pred = model(inputs)

        valence_scores_norm = (valence_scores - 1) / 8
        arousal_scores_norm = (arousal_scores - 1) / 8

        loss = criterion(valence_pred.squeeze(), valence_scores_norm) + \
               criterion(arousal_pred.squeeze(), arousal_scores_norm)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if pbar:
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate_classification(model, dataloader, device, modality):
    model.eval()
    valence_preds, arousal_preds = [], []
    valence_trues, arousal_trues = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[modality].to(device)
            valence_logits, arousal_logits = model(inputs)
            valence_preds.extend(valence_logits.argmax(dim=1).cpu().numpy())
            arousal_preds.extend(arousal_logits.argmax(dim=1).cpu().numpy())
            valence_trues.extend(batch['valence_label'].numpy())
            arousal_trues.extend(batch['arousal_label'].numpy())

    return {
        'valence_acc': accuracy_score(valence_trues, valence_preds),
        'arousal_acc': accuracy_score(arousal_trues, arousal_preds),
        'valence_f1': f1_score(valence_trues, valence_preds, average='binary', zero_division=0),
        'arousal_f1': f1_score(arousal_trues, arousal_preds, average='binary', zero_division=0)
    }


def evaluate_regression(model, dataloader, device, modality):
    model.eval()
    valence_preds, arousal_preds = [], []
    valence_trues, arousal_trues = [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[modality].to(device)
            valence_pred, arousal_pred = model(inputs)
            valence_preds.extend(valence_pred.squeeze().cpu().numpy() * 8 + 1)
            arousal_preds.extend(arousal_pred.squeeze().cpu().numpy() * 8 + 1)
            valence_trues.extend(batch['valence_score'].numpy())
            arousal_trues.extend(batch['arousal_score'].numpy())

    return {
        'valence_mae': np.mean(np.abs(np.array(valence_preds) - np.array(valence_trues))),
        'arousal_mae': np.mean(np.abs(np.array(arousal_preds) - np.array(arousal_trues)))
    }


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def subject_independent_cv(args, device):
    print(f"\n{'=' * 60}")
    print(f"Subject-Independent CV")
    print(f"Using Device: {device}")
    print(f"{'=' * 60}\n")

    print("Loading data...")
    eeg_data = np.load(os.path.join(args.data_path, 'eeg_data.npy'), mmap_mode='r') if args.modality in ['eeg',
                                                                                                         'both'] else None
    facial_data = np.load(os.path.join(args.data_path, 'facial_data.npy'), mmap_mode='r') if args.modality in ['facial',
                                                                                                               'both'] else None

    valence_labels = np.load(os.path.join(args.data_path, 'valence_labels.npy'))
    arousal_labels = np.load(os.path.join(args.data_path, 'arousal_labels.npy'))
    valence_scores = np.load(os.path.join(args.data_path, 'valence_scores.npy'))
    arousal_scores = np.load(os.path.join(args.data_path, 'arousal_scores.npy'))
    subjects = np.load(os.path.join(args.data_path, 'subjects.npy'))

    unique_subjects = np.unique(subjects)
    all_fold_results = []

    for fold, test_subject in enumerate(unique_subjects):
        if test_subject not in (1, 8): continue  # Debug 仅跑部分
        print(f"\n[{fold + 1}/{len(unique_subjects)}] Test Subject: {test_subject}")

        train_idx, test_idx = subjects != test_subject, subjects == test_subject

        train_dataset = EmotionDataset(
            eeg_data[train_idx] if eeg_data is not None else None,
            facial_data[train_idx] if facial_data is not None else None,
            valence_labels[train_idx], arousal_labels[train_idx],
            valence_scores[train_idx], arousal_scores[train_idx], modality=args.modality
        )
        test_dataset = EmotionDataset(
            eeg_data[test_idx] if eeg_data is not None else None,
            facial_data[test_idx] if facial_data is not None else None,
            valence_labels[test_idx], arousal_labels[test_idx],
            valence_scores[test_idx], arousal_scores[test_idx], modality=args.modality
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

        encoder = create_encoder(args)
        model_cls = ClassificationModel if args.task == 'classification' else RegressionModel
        model = model_cls(encoder, encoder.feature_dim).to(device)

        # [Fix]: AdamW
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # [Fix]: Scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
        )

        total_steps = args.epochs * len(train_loader)
        with tqdm(total=total_steps, desc=f"Subject {test_subject} Training") as pbar:
            for epoch in range(args.epochs):
                pbar.set_description(f"Sub {test_subject} [Ep {epoch + 1}/{args.epochs}]")

                if args.task == 'classification':
                    train_one_epoch_classification(model, train_loader, optimizer, device, args.modality, scheduler,
                                                   pbar)
                else:
                    train_one_epoch_regression(model, train_loader, optimizer, device, args.modality, scheduler, pbar)

        if args.task == 'classification':
            res = evaluate_classification(model, test_loader, device, args.modality)
            print(f"  Result: Val Acc={res['valence_acc']:.4f}, Aro Acc={res['arousal_acc']:.4f}")
        else:
            res = evaluate_regression(model, test_loader, device, args.modality)
            print(f"  Result: Val MAE={res['valence_mae']:.4f}, Aro MAE={res['arousal_mae']:.4f}")

        all_fold_results.append(res)
        os.makedirs(os.path.join(args.output_path, 'checkpoints'), exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.output_path, 'checkpoints', f'subject_{test_subject}_best.pth'))

    # Average Results
    if len(all_fold_results) > 0:
        avg_results = {k: float(np.mean([r[k] for r in all_fold_results])) for k in all_fold_results[0].keys()}
        print(f"\n === Final Average Results ===")
        print(avg_results)


def subject_dependent_cv(args, device):
    print(f"\nSubject-Dependent CV (10-Fold)\nUsing Device: {device}\n")

    eeg_data = np.load(os.path.join(args.data_path, 'eeg_data.npy'), mmap_mode='r') if args.modality in ['eeg',
                                                                                                         'both'] else None
    facial_data = np.load(os.path.join(args.data_path, 'facial_data.npy'), mmap_mode='r') if args.modality in ['facial',
                                                                                                               'both'] else None
    valence_labels = np.load(os.path.join(args.data_path, 'valence_labels.npy'))
    arousal_labels = np.load(os.path.join(args.data_path, 'arousal_labels.npy'))
    valence_scores = np.load(os.path.join(args.data_path, 'valence_scores.npy'))
    arousal_scores = np.load(os.path.join(args.data_path, 'arousal_scores.npy'))
    subjects = np.load(os.path.join(args.data_path, 'subjects.npy'))
    trials = np.load(os.path.join(args.data_path, 'trials.npy'))

    unique_subjects = np.unique(subjects)
    all_subject_results = []

    for subject in unique_subjects:
        if subject not in (3, 5): continue  # Debug
        print(f"\nProcessing Subject {subject}...")
        subject_idx = np.where(subjects == subject)[0]
        subject_trials = trials[subject_idx]
        unique_trials = np.unique(subject_trials)

        np.random.seed(args.seed)
        shuffled_trials = unique_trials[np.random.permutation(len(unique_trials))]

        subject_fold_results = []
        n_folds = 10
        fold_size = len(unique_trials) // n_folds

        for fold in range(n_folds):
            test_start, test_end = fold * fold_size, (fold + 1) * fold_size if fold < n_folds - 1 else len(
                unique_trials)
            test_trials_ids = shuffled_trials[test_start:test_end]
            test_mask = np.isin(subject_trials, test_trials_ids)
            train_mask = ~test_mask

            test_idx_global, train_idx_global = subject_idx[test_mask], subject_idx[train_mask]

            train_dataset = EmotionDataset(
                eeg_data[train_idx_global] if eeg_data is not None else None,
                facial_data[train_idx_global] if facial_data is not None else None,
                valence_labels[train_idx_global], arousal_labels[train_idx_global],
                valence_scores[train_idx_global], arousal_scores[train_idx_global], modality=args.modality
            )
            test_dataset = EmotionDataset(
                eeg_data[test_idx_global] if eeg_data is not None else None,
                facial_data[test_idx_global] if facial_data is not None else None,
                valence_labels[test_idx_global], arousal_labels[test_idx_global],
                valence_scores[test_idx_global], arousal_scores[test_idx_global], modality=args.modality
            )

            if len(test_dataset) == 0: continue

            train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

            encoder = create_encoder(args)
            model_cls = ClassificationModel if args.task == 'classification' else RegressionModel
            model = model_cls(encoder, encoder.feature_dim).to(device)

            # [Fix]: AdamW
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # [Fix]: OneCycleLR
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
            )

            total_steps = args.epochs * len(train_loader)
            with tqdm(total=total_steps, desc=f"Sub {subject} Fold {fold + 1}") as pbar:
                for epoch in range(args.epochs):
                    pbar.set_description(f"Sub {subject} Fold {fold + 1} [Ep {epoch + 1}/{args.epochs}]")
                    if args.task == 'classification':
                        train_one_epoch_classification(model, train_loader, optimizer, device, args.modality, scheduler,
                                                       pbar)
                    else:
                        train_one_epoch_regression(model, train_loader, optimizer, device, args.modality, scheduler,
                                                   pbar)

            if args.task == 'classification':
                res = evaluate_classification(model, test_loader, device, args.modality)
                tqdm.write(
                    f"  Fold {fold + 1}: Val Acc={res['valence_acc']:.3f}, Aro Acc={res['arousal_acc']:.3f}, Val F1={res['valence_f1']:.3f}, Aro F1={res['arousal_f1']:.3f}")
                subject_fold_results.append(res)
            else:
                subject_fold_results.append(evaluate_regression(model, test_loader, device, args.modality))

        subject_avg = {k: float(np.mean([r[k] for r in subject_fold_results])) for k in subject_fold_results[0].keys()}
        all_subject_results.append(subject_avg)
        print(f"  Avg: {subject_avg}")

    if len(all_subject_results) > 0:
        final_avg = {k: float(np.mean([r[k] for r in all_subject_results])) for k in all_subject_results[0].keys()}
        final_std = {k: float(np.std([r[k] for r in all_subject_results])) for k in all_subject_results[0].keys()}

        print("\n dependent:")
        for k, v in final_avg.items(): print(f"{k}: {v:.4f} ± {final_std[k]:.4f}")

        with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
            json.dump({'average': final_avg, 'std': final_std, 'per_subject': all_subject_results}, f, indent=2)


def print_colored_config(args):
    print(f"\n{'=' * 60}")
    print(f"Experiment Configuration")
    print(f"{'=' * 60}")
    print(f"    • Dataset:      {args.dataset}")
    print(f"    • Modality:     {args.modality}")
    print(f"    • Data Path:    {args.data_path}")
    print(f"    • Encoder:      {args.encoder}")
    print(f"    • Epochs:       {args.epochs}")
    print(f"    • Batch Size:   {args.batch_size}")
    print(f"    • Learning Rate:{args.lr}")
    print(f"    • Weight Decay: {args.weight_decay}")
    print(f"    • Device:       {args.device}")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotion Recognition Training')

    # 数据和任务参数
    parser.add_argument('--dataset', type=str, default='deap', choices=['deap', 'mahnob'])
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'])
    parser.add_argument('--eval_mode', type=str, default='subject_dependent',
                        choices=['subject_dependent', 'subject_independent'])
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/eeg/data/DEAP/processed')  # 默认指向新路径
    parser.add_argument('--output_path', type=str, default='./results/exp_fixed')

    # 模型参数
    parser.add_argument('--modality', type=str, default='eeg', choices=['eeg', 'facial', 'both'])
    # --- 这里使用了你要求的详细 help ---
    parser.add_argument('--encoder', type=str, default='est',
                        choices=[
                            # EEG Models
                            'eegnet', 'dgcnn', 'lggnet', 'tsception', 'ccnn', 'bihdm', 'gcbnet',
                            # Facial Models
                            'c3d', 'slowfast', 'videoswin', 'formerdfer', 'logoformer', 'est'
                        ],
                        help='Encoder architecture selection.\n'
                             'EEG options: [eegnet, dgcnn, lggnet, tsception, ccnn, bihdm, gcbnet]\n'
                             'Facial options: [c3d, slowfast, videoswin, formerdfer, logoformer, est]\n'
                             '(default: eegnet)')

    # 训练参数
    # [Fix]: 默认参数优化
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')  # 调高LR配合Scheduler
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')

    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    print_colored_config(args)
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    if args.eval_mode == 'subject_independent':
        results = subject_independent_cv(args, device)
    else:
        results = subject_dependent_cv(args, device)

    print(f"\n✅ Results saved to: {args.output_path}")