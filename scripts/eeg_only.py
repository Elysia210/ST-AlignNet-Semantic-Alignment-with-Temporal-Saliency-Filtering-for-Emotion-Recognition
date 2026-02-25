"""
Experiment 1.1a: EEG-only emotion recognition
Tests different EEG encoder architectures

Usage:
    python scripts/eeg_only.py --encoder eegnet --data_dir data/processed/DEAP --gpu 0
    python scripts/eeg_only.py --encoder dgat --data_dir data/processed/DEAP --gpu 0
    python scripts/eeg_only.py --encoder de_cnn --data_dir data/processed/DEAP --gpu 0
    python scripts/eeg_only.py --encoder tsception --data_dir data/processed/DEAP --gpu 0
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from datasets.deap_dataset import DEAPDataset
from models.eeg_encoders import get_eeg_encoder
from models.task_heads import MultiTaskHead
from utils.metrics import compute_metrics, AverageMeter


class EEGOnlyModel(nn.Module):
    """Complete EEG-only model with encoder and task heads"""
    def __init__(self, encoder_type, encoder_config, num_classes=2,
                 num_tasks=2, pooling='mean', output_dim=128):
        super(EEGOnlyModel, self).__init__()
        
        # EEG encoder
        self.encoder = get_eeg_encoder(encoder_type, **encoder_config)
        
        # Task heads
        self.task_head = MultiTaskHead(
            input_dim=output_dim,
            num_classes=num_classes,
            num_cls_tasks=num_tasks,
            num_reg_tasks=num_tasks,
            pooling=pooling,
            hidden_dim=256,
            dropout=0.5
        )
        
    def forward(self, eeg):
        """
        Args:
            eeg: [batch, channels, timesteps]
        Returns:
            cls_outputs: List of classification logits
            reg_outputs: Regression predictions
        """
        # Encode
        features = self.encoder(eeg)  # [batch, timesteps_out, dim]
        
        # Task-specific predictions
        cls_outputs, reg_outputs = self.task_head(features)
        
        return cls_outputs, reg_outputs


def train_epoch(model, dataloader, optimizer, criterion_cls, criterion_reg,
                device, task='both'):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        eeg = batch['eeg'].to(device)
        cls_labels = batch['cls_labels'].to(device)  # [batch, 2]
        reg_labels = batch['reg_labels'].to(device)  # [batch, 2]
        
        # Forward
        cls_outputs, reg_outputs = model(eeg)
        
        # Compute losses
        loss = 0
        if task in ['classification', 'both']:
            cls_loss = sum([criterion_cls(cls_outputs[i], cls_labels[:, i])
                           for i in range(len(cls_outputs))]) / len(cls_outputs)
            loss += cls_loss
            cls_losses.update(cls_loss.item(), eeg.size(0))
        
        if task in ['regression', 'both']:
            reg_loss = criterion_reg(reg_outputs, reg_labels)
            loss += reg_loss
            reg_losses.update(reg_loss.item(), eeg.size(0))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), eeg.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'cls_loss': f'{cls_losses.avg:.4f}',
            'reg_loss': f'{reg_losses.avg:.4f}'
        })
    
    return {
        'loss': losses.avg,
        'cls_loss': cls_losses.avg,
        'reg_loss': reg_losses.avg
    }


def evaluate(model, dataloader, criterion_cls, criterion_reg, device, task='both'):
    """Evaluate the model"""
    model.eval()
    
    all_cls_preds = [[] for _ in range(2)]  # valence, arousal
    all_cls_labels = [[] for _ in range(2)]
    all_reg_preds = []
    all_reg_labels = []
    
    losses = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            eeg = batch['eeg'].to(device)
            cls_labels = batch['cls_labels'].to(device)
            reg_labels = batch['reg_labels'].to(device)
            
            # Forward
            cls_outputs, reg_outputs = model(eeg)
            
            # Compute loss
            loss = 0
            if task in ['classification', 'both']:
                cls_loss = sum([criterion_cls(cls_outputs[i], cls_labels[:, i])
                               for i in range(len(cls_outputs))]) / len(cls_outputs)
                loss += cls_loss
            
            if task in ['regression', 'both']:
                reg_loss = criterion_reg(reg_outputs, reg_labels)
                loss += reg_loss
            
            losses.update(loss.item(), eeg.size(0))
            
            # Collect predictions
            for i in range(2):
                preds = torch.argmax(cls_outputs[i], dim=1)
                all_cls_preds[i].extend(preds.cpu().numpy())
                all_cls_labels[i].extend(cls_labels[:, i].cpu().numpy())
            
            all_reg_preds.append(reg_outputs.cpu().numpy())
            all_reg_labels.append(reg_labels.cpu().numpy())
    
    # Compute metrics
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_reg_labels = np.concatenate(all_reg_labels, axis=0)
    
    metrics = compute_metrics(
        cls_preds=all_cls_preds,
        cls_labels=all_cls_labels,
        reg_preds=all_reg_preds,
        reg_labels=all_reg_labels,
        task_names=['valence', 'arousal']
    )
    
    metrics['loss'] = losses.avg
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='EEG-only emotion recognition')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing processed data')
    parser.add_argument('--dataset', type=str, default='deap',
                       choices=['deap', 'mahnob'])
    parser.add_argument('--num_folds', type=int, default=10,
                       help='Number of cross-validation folds')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, required=True,
                       choices=['eegnet', 'dgat', 'de_cnn', 'tsception'],
                       help='EEG encoder type')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'attention'],
                       help='Temporal pooling method')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--task', type=str, default='both',
                       choices=['classification', 'regression', 'both'],
                       help='Task type')
    
    # Other arguments
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f'eeg_{args.encoder}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Cross-validation
    all_results = []
    
    for fold in range(args.num_folds):
        print(f"\n{'='*80}")
        print(f"Fold {fold+1}/{args.num_folds}")
        print(f"{'='*80}")
        
        # Create datasets
        train_dataset = DEAPDataset(
            data_dir=args.data_dir,
            fold=fold,
            split='train',
            modality='eeg',
            target_dims=['valence', 'arousal']
        )
        
        test_dataset = DEAPDataset(
            data_dir=args.data_dir,
            fold=fold,
            split='test',
            modality='eeg',
            target_dims=['valence', 'arousal']
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        encoder_config = {
            'num_channels': 32,
            'num_timesteps': 128,
            'output_dim': args.hidden_dim
        }
        
        model = EEGOnlyModel(
            encoder_type=args.encoder,
            encoder_config=encoder_config,
            num_classes=2,
            num_tasks=2,
            pooling=args.pooling,
            output_dim=args.hidden_dim
        ).to(device)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
        criterion_cls = nn.CrossEntropyLoss()
        criterion_reg = nn.L1Loss()
        
        # Training loop
        best_val_score = 0
        best_metrics = None
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Alternate between tasks (optional)
            if args.task == 'both':
                task = 'classification' if epoch % 2 == 0 else 'regression'
            else:
                task = args.task
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer,
                criterion_cls, criterion_reg, device, task
            )
            
            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                test_metrics = evaluate(
                    model, test_loader,
                    criterion_cls, criterion_reg, device, 'both'
                )
                
                print(f"\nTest Results:")
                print(f"  Valence - Acc: {test_metrics['valence_acc']:.4f}, "
                      f"MAE: {test_metrics['valence_mae']:.4f}")
                print(f"  Arousal - Acc: {test_metrics['arousal_acc']:.4f}, "
                      f"MAE: {test_metrics['arousal_mae']:.4f}")
                
                # Save best model
                val_score = (test_metrics['valence_acc'] + test_metrics['arousal_acc']) / 2
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_metrics = test_metrics
                    torch.save(model.state_dict(),
                             os.path.join(output_dir, f'best_model_fold{fold}.pth'))
        
        # Save fold results
        all_results.append(best_metrics)
        
        print(f"\n{'='*80}")
        print(f"Fold {fold+1} Best Results:")
        print(f"  Valence - Acc: {best_metrics['valence_acc']:.4f}, "
              f"MAE: {best_metrics['valence_mae']:.4f}")
        print(f"  Arousal - Acc: {best_metrics['arousal_acc']:.4f}, "
              f"MAE: {best_metrics['arousal_mae']:.4f}")
        print(f"{'='*80}")
    
    # Compute average across folds
    avg_results = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results]
        avg_results[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'all_folds': all_results,
            'average': avg_results
        }, f, indent=2)
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS (Average across all folds)")
    print(f"{'='*80}")
    print(f"Valence Classification Accuracy: "
          f"{avg_results['valence_acc']['mean']:.4f} ± {avg_results['valence_acc']['std']:.4f}")
    print(f"Arousal Classification Accuracy: "
          f"{avg_results['arousal_acc']['mean']:.4f} ± {avg_results['arousal_acc']['std']:.4f}")
    print(f"Valence Regression MAE: "
          f"{avg_results['valence_mae']['mean']:.4f} ± {avg_results['valence_mae']['std']:.4f}")
    print(f"Arousal Regression MAE: "
          f"{avg_results['arousal_mae']['mean']:.4f} ± {avg_results['arousal_mae']['std']:.4f}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
