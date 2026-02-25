# train_fusion.py
"""
多模态情感识别训练脚本 - EEG + Facial 融合
支持8种融合方法和4种融合时机

修改自 train_facial.py，主要改动：
1. 同时加载EEG和Facial数据
2. 创建两个encoder + fusion module
3. 支持选择融合方法
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

# Import encoders
from models.eeg_encoders import (
    EEGNet, DGCNN, LGGNet, TSception, CCNN, BiHDM, GCBNet
)
from models.facial_encoders import (
    C3D, SlowFast, VideoSwin, FormerDFER, LOGOFormer, EST
)
from models.fusion_modules import create_fusion_module


class EmotionDataset(Dataset):
    """多模态Dataset - 同时加载EEG和Facial"""
    def __init__(self, eeg_data, facial_data, valence_labels, arousal_labels,
                 valence_scores, arousal_scores):
        self.eeg_data = eeg_data
        self.facial_data = facial_data
        self.valence_labels = torch.LongTensor(valence_labels)
        self.arousal_labels = torch.LongTensor(arousal_labels)
        self.valence_scores = torch.FloatTensor(valence_scores)
        self.arousal_scores = torch.FloatTensor(arousal_scores)

    def __len__(self):
        return len(self.valence_labels)

    def __getitem__(self, idx):
        eeg = torch.from_numpy(np.array(self.eeg_data[idx])).float()
        facial = torch.from_numpy(np.array(self.facial_data[idx])).float()
        
        return {
            'eeg': eeg,
            'facial': facial,
            'valence_label': self.valence_labels[idx],
            'arousal_label': self.arousal_labels[idx],
            'valence_score': self.valence_scores[idx],
            'arousal_score': self.arousal_scores[idx]
        }


class MultiModalModel(nn.Module):
    """多模态模型 = EEG_Encoder + Facial_Encoder + Fusion + Classifier"""
    def __init__(self, eeg_encoder, facial_encoder, fusion_module, 
                 fused_dim, n_classes=2, task='classification'):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.facial_encoder = facial_encoder
        self.fusion = fusion_module
        self.task = task
        
        if task == 'classification':
            self.valence_head = nn.Sequential(
                nn.Linear(fused_dim, 128), nn.ReLU(), 
                nn.Dropout(0.5), nn.Linear(128, n_classes)
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(fused_dim, 128), nn.ReLU(), 
                nn.Dropout(0.5), nn.Linear(128, n_classes)
            )
        else:  # regression
            self.valence_head = nn.Sequential(
                nn.Linear(fused_dim, 128), nn.ReLU(), 
                nn.Dropout(0.5), nn.Linear(128, 1)
            )
            self.arousal_head = nn.Sequential(
                nn.Linear(fused_dim, 128), nn.ReLU(), 
                nn.Dropout(0.5), nn.Linear(128, 1)
            )

    def forward(self, eeg, facial):
        # 编码
        feat_eeg = self.eeg_encoder(eeg)
        feat_facial = self.facial_encoder(facial)
        
        # 融合
        fused = self.fusion(feat_eeg, feat_facial)
        
        # 预测
        val_out = self.valence_head(fused)
        aro_out = self.arousal_head(fused)
        
        return val_out, aro_out


def create_encoders_and_fusion(args):
    """创建EEG encoder, Facial encoder和Fusion module"""
    # EEG encoder
    eeg_encoders = {
        'eegnet': EEGNet, 'dgcnn': DGCNN, 'lggnet': LGGNet,
        'tsception': TSception, 'ccnn': CCNN, 'bihdm': BiHDM, 'gcbnet': GCBNet
    }
    eeg_encoder = eeg_encoders[args.eeg_encoder](n_channels=32, n_timepoints=256)
    
    # Facial encoder
    facial_input_dim = 1280  # MobileNet特征
    facial_encoders = {
        'c3d': C3D, 'slowfast': SlowFast, 'videoswin': VideoSwin,
        'formerdfer': FormerDFER, 'logoformer': LOGOFormer, 'est': EST
    }
    facial_encoder = facial_encoders[args.facial_encoder](
        n_timepoints=16, input_dim=facial_input_dim
    )
    
    # Fusion module
    fusion_module = create_fusion_module(
        fusion_type=args.fusion_type,
        dim_eeg=eeg_encoder.feature_dim,
        dim_facial=facial_encoder.feature_dim,
        output_dim=args.fusion_dim
    )
    
    return eeg_encoder, facial_encoder, fusion_module


def train_one_epoch(model, dataloader, optimizer, device, task, pbar=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
    
    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        facial = batch['facial'].to(device)
        val_labels = batch['valence_label' if task == 'classification' else 'valence_score'].to(device)
        aro_labels = batch['arousal_label' if task == 'classification' else 'arousal_score'].to(device)
        
        optimizer.zero_grad()
        val_out, aro_out = model(eeg, facial)
        
        if task == 'classification':
            loss = criterion(val_out, val_labels) + criterion(aro_out, aro_labels)
        else:
            val_norm = (val_labels - 1) / 8
            aro_norm = (aro_labels - 1) / 8
            loss = criterion(val_out.squeeze(), val_norm) + criterion(aro_out.squeeze(), aro_norm)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if pbar:
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, task):
    """评估模型"""
    model.eval()
    val_preds, aro_preds = [], []
    val_trues, aro_trues = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            facial = batch['facial'].to(device)
            val_out, aro_out = model(eeg, facial)
            
            if task == 'classification':
                val_preds.extend(val_out.argmax(dim=1).cpu().numpy())
                aro_preds.extend(aro_out.argmax(dim=1).cpu().numpy())
                val_trues.extend(batch['valence_label'].numpy())
                aro_trues.extend(batch['arousal_label'].numpy())
            else:
                val_preds.extend((val_out.squeeze().cpu().numpy() * 8 + 1).tolist())
                aro_preds.extend((aro_out.squeeze().cpu().numpy() * 8 + 1).tolist())
                val_trues.extend(batch['valence_score'].numpy())
                aro_trues.extend(batch['arousal_score'].numpy())
    
    if task == 'classification':
        return {
            'valence_acc': accuracy_score(val_trues, val_preds),
            'arousal_acc': accuracy_score(aro_trues, aro_preds),
            'valence_f1': f1_score(val_trues, val_preds, average='binary'),
            'arousal_f1': f1_score(aro_trues, aro_preds, average='binary')
        }
    else:
        return {
            'valence_mae': np.mean(np.abs(np.array(val_preds) - np.array(val_trues))),
            'arousal_mae': np.mean(np.abs(np.array(aro_preds) - np.array(aro_trues)))
        }


def subject_independent_cv(args, device):
    """Subject-Independent交叉验证"""
    print(f"\n{'='*60}")
    print(f"Subject-Independent CV - Fusion: {args.fusion_type}")
    print(f"EEG: {args.eeg_encoder}, Facial: {args.facial_encoder}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 加载数据
    print("Loading data...")
    eeg_data = np.load(os.path.join(args.data_path, 'eeg_data.npy'), mmap_mode='r')
    facial_data = np.load(os.path.join(args.data_path, 'facial_data.npy'), mmap_mode='r')
    valence_labels = np.load(os.path.join(args.data_path, 'valence_labels.npy'))
    arousal_labels = np.load(os.path.join(args.data_path, 'arousal_labels.npy'))
    valence_scores = np.load(os.path.join(args.data_path, 'valence_scores.npy'))
    arousal_scores = np.load(os.path.join(args.data_path, 'arousal_scores.npy'))
    subjects = np.load(os.path.join(args.data_path, 'subjects.npy'))
    
    unique_subjects = np.unique(subjects)
    all_fold_results = []
    
    for fold, test_subject in enumerate(unique_subjects):
        print(f"\n[{fold+1}/{len(unique_subjects)}] Test Subject: {test_subject}")
        
        train_idx = subjects != test_subject
        test_idx = subjects == test_subject
        
        train_dataset = EmotionDataset(
            eeg_data[train_idx], facial_data[train_idx],
            valence_labels[train_idx], arousal_labels[train_idx],
            valence_scores[train_idx], arousal_scores[train_idx]
        )
        test_dataset = EmotionDataset(
            eeg_data[test_idx], facial_data[test_idx],
            valence_labels[test_idx], arousal_labels[test_idx],
            valence_scores[test_idx], arousal_scores[test_idx]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        # 创建模型
        eeg_encoder, facial_encoder, fusion_module = create_encoders_and_fusion(args)
        model = MultiModalModel(eeg_encoder, facial_encoder, fusion_module,
                               fused_dim=args.fusion_dim, task=args.task).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 训练
        total_steps = args.epochs * len(train_loader)
        with tqdm(total=total_steps, desc=f"Subject {test_subject}") as pbar:
            for epoch in range(args.epochs):
                pbar.set_description(f"Sub {test_subject} [Ep {epoch+1}/{args.epochs}]")
                train_one_epoch(model, train_loader, optimizer, device, args.task, pbar)
        
        # 评估
        res = evaluate(model, test_loader, device, args.task)
        if args.task == 'classification':
            print(f"  Val Acc={res['valence_acc']:.4f}, Aro Acc={res['arousal_acc']:.4f}")
        else:
            print(f"  Val MAE={res['valence_mae']:.4f}, Aro MAE={res['arousal_mae']:.4f}")
        
        all_fold_results.append(res)
        
        # 保存模型
        os.makedirs(os.path.join(args.output_path, 'checkpoints'), exist_ok=True)
        torch.save(model.state_dict(),
                  os.path.join(args.output_path, 'checkpoints', f'subject_{test_subject}_best.pth'))
    
    # 平均结果
    avg_results = {k: float(np.mean([r[k] for r in all_fold_results])) 
                   for k in all_fold_results[0].keys()}
    std_results = {k: float(np.std([r[k] for r in all_fold_results])) 
                   for k in all_fold_results[0].keys()}
    
    print(f"\n{'='*60}")
    print(f"Final Results - Fusion: {args.fusion_type}")
    print(f"{'='*60}")
    for k, v in avg_results.items():
        print(f"{k}: {v:.4f} ± {std_results[k]:.4f}")
    
    # 保存结果
    with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
        json.dump({
            'average': avg_results,
            'std': std_results,
            'folds': all_fold_results,
            'config': vars(args)
        }, f, indent=2)
    
    return avg_results


def subject_dependent_cv(args, device):
    """Subject-Dependent交叉验证 (10-Fold)"""
    print(f"\n{'='*60}")
    print(f"Subject-Dependent CV (10-Fold) - Fusion: {args.fusion_type}")
    print(f"{'='*60}\n")
    
    # 加载数据
    eeg_data = np.load(os.path.join(args.data_path, 'eeg_data.npy'), mmap_mode='r')
    facial_data = np.load(os.path.join(args.data_path, 'facial_data.npy'), mmap_mode='r')
    valence_labels = np.load(os.path.join(args.data_path, 'valence_labels.npy'))
    arousal_labels = np.load(os.path.join(args.data_path, 'arousal_labels.npy'))
    valence_scores = np.load(os.path.join(args.data_path, 'valence_scores.npy'))
    arousal_scores = np.load(os.path.join(args.data_path, 'arousal_scores.npy'))
    subjects = np.load(os.path.join(args.data_path, 'subjects.npy'))
    trials = np.load(os.path.join(args.data_path, 'trials.npy'))
    
    unique_subjects = np.unique(subjects)
    all_subject_results = []
    
    for subject in unique_subjects:
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
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(unique_trials)
            test_trials_ids = shuffled_trials[test_start:test_end]
            test_mask = np.isin(subject_trials, test_trials_ids)
            train_mask = ~test_mask
            
            test_idx_global = subject_idx[test_mask]
            train_idx_global = subject_idx[train_mask]
            
            train_dataset = EmotionDataset(
                eeg_data[train_idx_global], facial_data[train_idx_global],
                valence_labels[train_idx_global], arousal_labels[train_idx_global],
                valence_scores[train_idx_global], arousal_scores[train_idx_global]
            )
            test_dataset = EmotionDataset(
                eeg_data[test_idx_global], facial_data[test_idx_global],
                valence_labels[test_idx_global], arousal_labels[test_idx_global],
                valence_scores[test_idx_global], arousal_scores[test_idx_global]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)),
                                     shuffle=True, num_workers=args.num_workers, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)),
                                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
            
            # 创建模型
            eeg_encoder, facial_encoder, fusion_module = create_encoders_and_fusion(args)
            model = MultiModalModel(eeg_encoder, facial_encoder, fusion_module,
                                   fused_dim=args.fusion_dim, task=args.task).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # 训练
            total_steps = args.epochs * len(train_loader)
            with tqdm(total=total_steps, desc=f"Sub {subject} Fold {fold+1}") as pbar:
                for epoch in range(args.epochs):
                    pbar.set_description(f"Sub {subject} F{fold+1} [Ep {epoch+1}/{args.epochs}]")
                    train_one_epoch(model, train_loader, optimizer, device, args.task, pbar)
            
            # 评估
            subject_fold_results.append(evaluate(model, test_loader, device, args.task))
        
        # Subject平均
        subject_avg = {k: float(np.mean([r[k] for r in subject_fold_results])) 
                      for k in subject_fold_results[0].keys()}
        all_subject_results.append(subject_avg)
        print(f"  Subject {subject} Avg: {subject_avg}")
    
    # 最终平均
    final_avg = {k: float(np.mean([r[k] for r in all_subject_results])) 
                for k in all_subject_results[0].keys()}
    final_std = {k: float(np.std([r[k] for r in all_subject_results])) 
                for k in all_subject_results[0].keys()}
    
    print(f"\n{'='*60}")
    print("Final Average Results:")
    for k, v in final_avg.items():
        print(f"{k}: {v:.4f} ± {final_std[k]:.4f}")
    
    with open(os.path.join(args.output_path, 'results.json'), 'w') as f:
        json.dump({
            'average': final_avg,
            'std': final_std,
            'per_subject': all_subject_results,
            'config': vars(args)
        }, f, indent=2)
    
    return final_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multimodal Emotion Recognition - Fusion')
    
    # 数据和任务
    parser.add_argument('--dataset', type=str, default='deap', choices=['deap', 'mahnob'])
    parser.add_argument('--task', type=str, default='classification', 
                       choices=['classification', 'regression'])
    parser.add_argument('--eval_mode', type=str, default='subject_independent',
                       choices=['subject_dependent', 'subject_independent'])
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/eeg/data/DEAP/processed')
    parser.add_argument('--output_path', type=str, default='./results/fusion_exp1')
    
    # Encoder选择
    parser.add_argument('--eeg_encoder', type=str, default='gcbnet',
                       choices=['eegnet', 'dgcnn', 'lggnet', 'tsception', 'ccnn', 'bihdm', 'gcbnet'])
    parser.add_argument('--facial_encoder', type=str, default='est',
                       choices=['c3d', 'slowfast', 'videoswin', 'formerdfer', 'logoformer', 'est'])
    
    # 融合设置
    parser.add_argument('--fusion_type', type=str, default='concat',
                       choices=['concat', 'f1', 'sum', 'f2', 'product', 'f3',
                               'gated', 'f4', 'mlp', 'f5', 'bilinear', 'f6',
                               'cross_attn', 'f7', 'co_attn', 'f8'],
                       help='Fusion method: concat/sum/product/gated/mlp/bilinear/cross_attn/co_attn')
    parser.add_argument('--fusion_dim', type=int, default=256, help='Fused feature dimension')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # 打印配置
    print(f"\n{'='*60}")
    print(f"🔥 Multimodal Fusion Experiment")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Eval Mode: {args.eval_mode}")
    print(f"EEG Encoder: {args.eeg_encoder}")
    print(f"Facial Encoder: {args.facial_encoder}")
    print(f"Fusion Type: {args.fusion_type}")
    print(f"Fusion Dim: {args.fusion_dim}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 保存配置
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    # 运行交叉验证
    if args.eval_mode == 'subject_independent':
        results = subject_independent_cv(args, device)
    else:
        results = subject_dependent_cv(args, device)
    
    print(f"\n✅ Results saved to: {args.output_path}")
