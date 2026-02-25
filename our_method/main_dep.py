import os, argparse, torch, pandas as pd, torch.nn.functional as F
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

# 你的工程内模块（按你当前结构）
from models.Multimodal_dataset import build_loaders_binary, MultimodalDataset
from models.fusion import MultimodalFusionModel_Binary
from models.evaluate import evaluate_binary_acc


def get_config():
    p = argparse.ArgumentParser("Dependent (chain) training for binary classification")
    # 常规
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--data_path', type=str, default='./make_data')
    p.add_argument('--save_dir', type=str, default='./output/binary_dep')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--folds', nargs='+', type=int, default=list(range(10)))

    # 依赖训练相关
    p.add_argument('--init_ckpt', type=str, default='', help='可选：预训练权重路径（fold0 从此处加载）')
    p.add_argument('--freeze_epochs', type=int, default=0, help='前 K 个 epoch 只训分类头，之后解冻两个 encoder')
    return vars(p.parse_args())


def train_one_fold(fold, cfg, model, loaders, device, freeze_epochs=0):
    """单 fold 训练：ValAcc 选优；支持前K轮冻结→解冻"""
    log_dir = os.path.join("runs_binary_dep", f"fold_{fold}")
    writer = SummaryWriter(log_dir=log_dir)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    # 统计训练集标签分布 → 类权重（应对不平衡）
    labels = []
    for b in loaders["train"]:
        labels += b["label_cls"].tolist()
    cnt = Counter(labels); total = sum(cnt.values())
    class_w = torch.tensor([ total / (2 * cnt.get(i, 1)) for i in range(2) ],
                           dtype=torch.float, device=device)

    # 冻结编码器（若需要）
    if freeze_epochs > 0:
        for p in model.eeg_encoder.parameters():    p.requires_grad = False
        for p in model.facial_encoder.parameters(): p.requires_grad = False

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_acc = 0.0

    for ep in range(1, cfg["epochs"] + 1):
        # 到达解冻点：解冻并重建优化器
        if freeze_epochs > 0 and ep == freeze_epochs + 1:
            for p in model.eeg_encoder.parameters():    p.requires_grad = True
            for p in model.facial_encoder.parameters(): p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            print(f"[Fold {fold}] Unfreeze encoders at epoch {ep}")

        model.train()
        loss_sum, right, seen = 0.0, 0, 0

        for batch in loaders["train"]:
            eeg = batch["eeg"].to(device)
            facial = batch["facial"].to(device)
            y = batch["label_cls"].to(device)

            logits = model(eeg, facial)["logits"]  # [B,2]
            loss = F.cross_entropy(logits, y, weight=class_w, label_smoothing=cfg["label_smoothing"])

            opt.zero_grad(); loss.backward(); opt.step()

            loss_sum += loss.item()
            right += (logits.argmax(1) == y).sum().item()
            seen  += y.size(0)

        train_acc = right / max(seen, 1)
        val_acc   = evaluate_binary_acc(model, loaders["val"], device)

        writer.add_scalar("Loss/train", loss_sum, ep)
        writer.add_scalar("Acc/train",  train_acc, ep)
        writer.add_scalar("Acc/val",    val_acc, ep)

        print(f"[Fold {fold}] Ep{ep:02d} | Loss {loss_sum:.3f} | TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ save best (ValAcc={best_acc:.4f}) -> {save_path}")

    writer.close()
    return best_acc


def test_one_fold(fold, cfg, device):
    """加载 best_fold{fold}.pt，在该 fold 的测试集上评估 ACC"""
    from torch.utils.data import DataLoader
    import pickle

    with open(os.path.join(cfg["data_path"], f"fold{fold}.pkl"), "rb") as f:
        data = pickle.load(f)
    split = int(len(data) * 0.8)
    test_set = MultimodalDataset(data[split:])
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
    )

    model = MultimodalFusionModel_Binary().to(device)
    ckpt = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    acc = evaluate_binary_acc(model, test_loader, device)
    print(f"[Fold {fold}] TEST | ACC {acc:.4f}")
    return {"fold": fold, "acc": acc}


def main():
    cfg = get_config()
    os.makedirs(cfg["save_dir"], exist_ok=True)

    all_rows = []
    prev_best = None

    for i, fold in enumerate(cfg["folds"]):
        print(f"\n=== Fold {fold} (Binary, Dependent) ===")
        loaders = build_loaders_binary(fold, cfg)

        # 构建模型
        model = MultimodalFusionModel_Binary().to(cfg["device"])

        # 1) 若提供了 init_ckpt，则优先加载（fold0 也会加载）
        if cfg["init_ckpt"]:
            if os.path.exists(cfg["init_ckpt"]):
                print(f"Load init_ckpt: {cfg['init_ckpt']}")
                model.load_state_dict(torch.load(cfg["init_ckpt"], map_location=cfg["device"]), strict=True)
            else:
                print(f"[Warn] init_ckpt not found: {cfg['init_ckpt']} (ignored)")

        # 2) 如果是链式且存在上一折 best，则覆盖当前初始化（让知识延续）
        if prev_best and os.path.exists(prev_best):
            print(f"Warm-start from prev fold: {prev_best}")
            model.load_state_dict(torch.load(prev_best, map_location=cfg["device"]), strict=True)

        # 训练本 fold
        _ = train_one_fold(
            fold=fold,
            cfg=cfg,
            model=model,
            loaders=loaders,
            device=cfg["device"],
            freeze_epochs=cfg["freeze_epochs"]
        )

        # 评估 & 记录
        all_rows.append(test_one_fold(fold, cfg, cfg["device"]))

        # 更新“上一折 best”
        prev_best = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    # 汇总 CSV
    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(cfg["save_dir"], "fold_results_binary_dep.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"AVG ACC: {df['acc'].mean():.4f}")


if __name__ == "__main__":
    main()
