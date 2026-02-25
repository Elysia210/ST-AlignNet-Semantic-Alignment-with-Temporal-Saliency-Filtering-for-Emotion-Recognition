# main.py
import os, argparse, torch, pandas as pd, torch.nn.functional as F
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from Multimodal_dataset import build_loaders_binary, MultimodalDataset
from fusion import MultimodalFusionModel_Binary
from evaluate import evaluate_binary_acc

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--label_smoothing', type=float, default=0.0)  # 只用acc时，默认关掉更稳
    p.add_argument('--data_path', type=str, default='/root/autodl-tmp/eeg/data/DEAP/aligned_dependent_data')
    p.add_argument('--save_dir', type=str, default='/root/autodl-tmp/eeg/results/deap')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--folds', nargs='+', type=int, default=list(range(10)))
    p.add_argument('--target_dim', type=int, default=0, help='Valence, target_dim=1: Arousal')
    return vars(p.parse_args())

def train_one_fold(fold, cfg, model, loaders, device):
    writer = SummaryWriter(log_dir=f"runs_binary/fold_{fold}")
    os.makedirs(cfg["save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    # === 按当前二分类训练集统计类权重（应对类别不平衡） ===
    labels = []
    for b in loaders["train"]:
        labels += b["label_cls"].tolist()
    cnt = Counter(labels)
    total = sum(cnt.values())
    class_weights = [total / (2 * cnt.get(i, 1)) for i in range(2)]
    w = torch.tensor(class_weights, dtype=torch.float, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_acc = 0.0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, right, seen = 0.0, 0, 0

        for batch in loaders["train"]:
            eeg = batch["eeg"].to(device)
            facial = batch["facial"].to(device)
            y = batch["label_cls"].to(device)

            logits = model(eeg, facial)["logits"]
            loss = F.cross_entropy(logits, y, weight=w, label_smoothing=cfg["label_smoothing"])

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            right += (logits.argmax(1) == y).sum().item()
            seen  += y.size(0)

        train_acc = right / max(seen, 1)
        val_acc = evaluate_binary_acc(model, loaders["val"], device)

        writer.add_scalar("Loss/train", loss_sum, ep)
        writer.add_scalar("Acc/train", train_acc, ep)
        writer.add_scalar("Acc/val",   val_acc, ep)

        print(f"[Fold {fold}] Ep{ep:02d} | Loss {loss_sum:.3f} | TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f}")

        # === 仅以 ValAcc 作为保存标准 ===
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ save best (ValAcc={best_acc:.4f}) -> {save_path}")

    writer.close()
    return best_acc

def test_one_fold(fold, cfg, device):
    """用 best ckpt 在该 fold 的【真正 test 集】上评估 Acc。"""
    # 复用 build_loaders_binary 的同一套划分，避免与 val 重合
    loaders = build_loaders_binary(fold, cfg)
    test_loader = loaders.get("test", None)
    if test_loader is None:
        print(f"[Fold {fold}] [WARN] 构建器未提供 test 集，临时用 val 代替（仅供调试，不能当最终测试）")
        test_loader = loaders["val"]

    model = MultimodalFusionModel_Binary().to(device)
    ckpt_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    acc = evaluate_binary_acc(model, test_loader, device)
    print(f"[Fold {fold}] TEST | ACC {acc:.4f}")
    return {"fold": fold, "acc": acc}

def sanity_check_splits(loaders, fold):
    def _ids_from_loader(loader):
        ds = loader.dataset
        # 如果是 Subset，拿到底层 dataset 和索引
        if hasattr(ds, "dataset") and hasattr(ds, "indices"):
            base, idxs = ds.dataset, ds.indices
            # 以对象地址作为唯一ID（不会跨split重复）
            return set(id(base[i]) for i in idxs)
        # 如果是你现在这种自定义Dataset，直接用 data 列表里元素的对象地址
        if hasattr(ds, "data"):
            return set(id(x) for x in ds.data)
        # 兜底：不用 "dsrow_i" 这种会重复的伪ID
        return set()

    tr_ids = _ids_from_loader(loaders["train"])
    va_ids = _ids_from_loader(loaders["val"])
    te = loaders.get("test")
    te_ids = _ids_from_loader(te) if te is not None else set()

    print(f"[Fold {fold}] sizes: train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}")
    print(f"[Fold {fold}] overlaps: "
          f"train∩val={len(tr_ids & va_ids)} "
          f"train∩test={len(tr_ids & te_ids)} "
          f"val∩test={len(va_ids & te_ids)}")

    assert not (tr_ids & va_ids or tr_ids & te_ids or va_ids & te_ids), \
        f"[Fold {fold}] 数据泄漏：split 之间存在重复样本！"


def main():
    cfg = get_config()
    os.makedirs(cfg["save_dir"], exist_ok=True)
    all_rows = []

    for fold in cfg["folds"]:
        print(f"\n=== Fold {fold} (Binary) ===")
        loaders = build_loaders_binary(fold, cfg)
        sanity_check_splits(loaders, fold)

        model = MultimodalFusionModel_Binary().to(cfg["device"])

        _ = train_one_fold(fold, cfg, model, loaders, cfg["device"])
        all_rows.append(test_one_fold(fold, cfg, cfg["device"]))

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(cfg["save_dir"], "fold_results_binary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"AVG ACC: {df['acc'].mean():.4f}")

if __name__ == "__main__":
    main()
