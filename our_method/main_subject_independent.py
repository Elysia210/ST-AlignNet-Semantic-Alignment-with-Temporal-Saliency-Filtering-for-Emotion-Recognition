# main.py
import os, argparse
import torch
import pandas as pd
import torch.nn.functional as F
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import math
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from models.Multimodal_dataset import build_loaders_loso_binary as build_loaders
from models.fusion import MultimodalFusionModel_Binary
from models.evaluate import (
    evaluate_binary_acc,
    pick_best_calibrator,  # 新接口
    _collect_binlogits_labels,
    evaluate_loso_binary,
    auc_with_T,  # 新的收集函数名
)


import torch.nn as nn

def freeze_bn(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

def replace_bn_with_gn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            setattr(module, name, nn.GroupNorm(1, child.num_features, affine=True))
        else:
            replace_bn_with_gn(child)

@torch.no_grad()
def peek_val_stats(model, loader, device, k=5):
    """看一眼验证集概率分布，判断是否‘全 0.5’或‘全部一侧’。"""
    model.eval()
    ps, ys = [], []
    for b in loader:
        p1 = model(b['eeg'].to(device), b['facial'].to(device))['logits'].softmax(1)[:,1]
        ps.append(p1.cpu()); ys.append(b['label_cls'].cpu())
    if not ys:
        print("[VAL] empty"); return
    import torch as _t
    ps = _t.cat(ps).numpy(); ys = _t.cat(ys).numpy()
    print(f"[VAL] p1 mean={ps.mean():.3f}, std={ps.std():.3f}, min={ps.min():.3f}, max={ps.max():.3f}")
    # 简单看 top-K 最靠近两端的样本
    idx_hi = ps.argsort()[-k:][::-1]; idx_lo = ps.argsort()[:k]
    print("[VAL] top-hi p1:", ps[idx_hi])
    print("[VAL] top-lo p1:", ps[idx_lo])

def print_val_class_dist(loader):
    from collections import Counter
    c = Counter()
    for b in loader:
        c.update(b["label_cls"].tolist())
    print(f"[VAL] class_dist: {dict(c)}")

def get_config():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--data_path', type=str, default='./make_data_ind')
    p.add_argument('--save_dir', type=str, default='./output/binary_ind1')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--folds', nargs='+', type=int, default=list(range(22)))
    p.add_argument('--binary_threshold', type=float, default=5.0)
    return vars(p.parse_args())

def train_one_fold(fold, cfg, model, loaders, device):
    writer = SummaryWriter(log_dir=f"runs_binary/fold_{fold}")
    os.makedirs(cfg["save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    opt = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

    def build_warmup_cosine(optimizer, warmup_epochs, max_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / float(max(1, warmup_epochs))
            t = (epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = build_warmup_cosine(opt, warmup_epochs=5, max_epochs=cfg['epochs'])
    best_metric = -1.0  # 用 Val-BA 选最优
    bad = 0

    # --- epoch loop ---
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, right, seen = 0.0, 0, 0

        for batch in loaders["train"]:
            eeg = batch["eeg"].to(device)
            facial = batch["facial"].to(device)
            y = batch["label_cls"].to(device)

            logits = model(eeg, facial)["logits"]
            loss = F.cross_entropy(logits, y, weight=None, label_smoothing=cfg["label_smoothing"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += float(loss.item())
            right += int((logits.argmax(1) == y).sum().item())
            seen  += int(y.size(0))

        # 学习率调度在 epoch 末
        scheduler.step()

        # ===== 验证 =====
        # 1) 先拿 bin-logit
        val_bl, val_y = _collect_binlogits_labels(model, loaders['val'], device)
        # 2) 选择校准器并在验证集按 BA 搜 τ*
        cal, tau_star, val_bal_acc = pick_best_calibrator(
            val_bl, val_y, methods=('none', 'platt', 'temp'), metric='ece'
        )
        calib_tag = cal.method if isinstance(cal.method, str) else str(cal.method)

        # 3) 固定 0.5 的 Val-ACC
        val_acc_05 = evaluate_binary_acc(model, loaders["val"], device)

        # 4) Val AUC（不依赖阈值；用未校准的 logits->softmax 即可）
        #    你也可以用 cal 后的概率来算，AUC（单调）通常不变。
        val_auc = auc_with_T(model, loaders["val"], device, T=1.0)

        # ===== 打印 =====
        train_acc  = right / max(seen, 1)
        train_loss = loss_sum / max(1, len(loaders["train"]))
        print(
            f"[Fold {fold}] Ep{ep:02d} | Loss {train_loss:.3f} | TrainAcc {train_acc:.3f} | "
            f"Val@0.5 {val_acc_05:.3f} | Val@τ*Bal {val_bal_acc:.3f} "
            f"(τ*={tau_star:.3f}, cal={calib_tag})"
        )

        # ===== 早停 + 保存 =====
        if val_bal_acc > best_metric:
            best_metric = val_bal_acc
            bad = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ save best (ValBalancedAcc={best_metric:.4f}, τ*={tau_star:.3f}) -> {save_path}")
        else:
            bad += 1
            if bad >= 8:
                print(f"[ES] stop at epoch {ep}, best ValBalancedAcc={best_metric:.4f}")
                break

        # ===== TensorBoard =====
        writer.add_scalar("Train/Loss",       train_loss, ep)
        writer.add_scalar("Train/Acc",        train_acc,  ep)
        writer.add_scalar("Val/BA@tau*",      val_bal_acc, ep)
        writer.add_scalar("Val/ACC@0.5",      val_acc_05,  ep)
        writer.add_scalar("Val/AUC",          val_auc,     ep)
        writer.add_scalar("Val/tau_star",     tau_star,    ep)
        _cal2id = {"none": 0, "platt": 1, "temp": 2}
        writer.add_scalar("Val/calibrator_id", _cal2id.get(calib_tag, -1), ep)

    writer.close()
    return best_metric

def test_one_fold(fold, cfg, device):
    loaders = build_loaders(fold, cfg)   # 你现有的函数
    val_loader, test_loader = loaders["val"], loaders["test"]

    model = MultimodalFusionModel_Binary()
    replace_bn_with_gn(model)
    model = model.to(device)
    ckpt = os.path.join(cfg["save_dir"], f"best_fold{fold}.pt")

    sd = torch.load(ckpt, map_location=device)
    msg = model.load_state_dict(sd, strict=True)   # ← 关键：strict=False
    print("[load]", msg)                            # 打印下 missing/unexpected keys 以便核对
    model.eval()

    # 核心评估：在 val 上按 ACC 选 τ*，用 τ* 评 test（不做 platt/temp）
    out = evaluate_loso_binary(model, val_loader, test_loader, device,
     use_prior_correction=False,        # 先不开 EM
     calibrator_metric="nll"            # NLL/ECE 都行，保持和训练一致
 )
    # ……前面加载 ckpt 并得到 out 之后：
    print(
        f"[Fold {fold}] TEST | ACC {out['acc']:.4f} | AUC {out['auc']:.3f} "
        f"| τ*={out['thr_star']:.3f} | τ={out['thr']:.3f} | calib={out['calib']}"
    )

    writer = SummaryWriter(f"runs_binary/fold_{fold}")  # 若训练阶段的 writer 已可用，就复用
    writer.add_scalar("Test/ACC", out["acc"], fold)
    writer.add_scalar("Test/AUC", out["auc"], fold)
    writer.add_scalar("Test/tau_used", out["thr"], fold)  # 最终用的阈值（τ*或τ_π）
    writer.add_scalar("Test/tau_star", out["thr_star"], fold)  # 验证集上选出的 τ*
    writer.add_scalar("Test/ValAUC", out["val_auc"], fold)  # 为对照也记录一下 Val AUC

    writer.close()
    # 如果你的下游CSV还想保留旧列名 'tau'，就把 τ* 映射回去
    return {
        "fold": fold,
        "test_acc": out["acc"],
        "auc": out["auc"],
        "tau": out["thr_star"],  # 旧名=τ*（也可以换成 out['thr'] 看你想记录哪个）
        "calib": out["calib"],
    }


def sanity_check_splits(loaders, fold):
    def _ids_from_loader(loader):
        ds = loader.dataset
        if hasattr(ds, "data"): return set(id(x) for x in ds.data)
        if hasattr(ds, "dataset") and hasattr(ds, "indices"):
            base, idxs = ds.dataset, ds.indices
            return set(id(base[i]) for i in idxs)
        return set()
    tr = _ids_from_loader(loaders["train"])
    va = _ids_from_loader(loaders["val"])
    te = _ids_from_loader(loaders["test"])
    print(f"[Fold {fold}] sizes: train={len(tr)} val={len(va)} test={len(te)}")
    print(f"[Fold {fold}] overlaps: train∩val={len(tr&va)} train∩test={len(tr&te)} val∩test={len(va&te)}")
    assert not (tr & va or tr & te or va & te), f"[Fold {fold}] 数据泄漏：split 之间存在重复样本！"

def main():
    cfg = get_config()
    os.makedirs(cfg["save_dir"], exist_ok=True)
    all_rows = []

    for fold in cfg["folds"]:
        print(f"\n=== Fold {fold} (Binary) ===")
        loaders = build_loaders(fold, cfg)
        sanity_check_splits(loaders, fold)

        model = MultimodalFusionModel_Binary()
        replace_bn_with_gn(model)  # ★ 训练时也替换
        model = model.to(cfg["device"])  # ★ 再 to(device)
        _ = train_one_fold(fold, cfg, model, loaders, cfg["device"])
        all_rows.append(test_one_fold(fold, cfg, cfg["device"]))

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(cfg["save_dir"], "fold_results_binary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"AVG ACC: {df['test_acc'].mean():.4f}")

if __name__ == "__main__":
    main()
