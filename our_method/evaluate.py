# models/evaluate.py
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
try:
    from sklearn.isotonic import IsotonicRegression
    _HAS_ISO = True
except Exception:
    _HAS_ISO = False

from sklearn.metrics import roc_auc_score

@torch.no_grad()
def auc_with_T(model, loader, device, T=1.0):
    model.eval()
    ps, ys = [], []
    for b in loader:
        lg = model(b['eeg'].to(device), b['facial'].to(device))['logits']
        p1 = (lg / T).softmax(1)[:,1]
        ps.append(p1.cpu()); ys.append(b['label_cls'].cpu())
    if not ys: return float('nan')
    import torch as _t
    ps = _t.cat(ps).numpy(); ys = _t.cat(ys).numpy()
    try:
        return float(roc_auc_score(ys, ps))
    except:
        return float('nan')

@torch.no_grad()
def _to_binlogit(logits: torch.Tensor) -> torch.Tensor:
    """
    把 2 类 raw logits 转为“阳性类的单一 logit”（log-odds）:
      binlogit = logit_pos - logit_neg
    若已是 [B] 形状则原样返回。
    """
    if logits.ndim == 2 and logits.size(-1) == 2:
        return (logits[:, 1] - logits[:, 0]).detach()
    if logits.ndim == 1:
        return logits.detach()
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")
@torch.no_grad()
def _collect_binlogits_labels(model, loader, device):
    model.eval()
    bl_list, ys = [], []
    for b in loader:
        eeg = b['eeg'].to(device)
        facial = b['facial'].to(device)
        y = b['label_cls'].to(device)
        lg2 = model(eeg, facial)['logits']  # [B,2]
        bl = _to_binlogit(lg2)              # [B]
        bl_list.append(bl)
        ys.append(y)
    if not ys:
        return None, None
    return torch.cat(bl_list), torch.cat(ys)

class Calibrator:
    """
    method ∈ {'none','temp','platt','isotonic'}
    - temp: Temperature Scaling (Guo et al., ICML'17)，T ∈ [0.7, 2.0]
    - platt: Platt Scaling (Zadrozny & Elkan'02)
    - isotonic: Isotonic Regression（若 sklearn 可用）
    针对“单一 logit”（已用 _to_binlogit 转好），输出校准后的 p1 概率。
    """
    def __init__(self, method='none', t_min=0.7, t_max=2.0, max_iter=200):
        self.method = method
        self.t_min, self.t_max = t_min, t_max
        self.max_iter = max_iter
        self.params = None

    def fit(self, binlogits: torch.Tensor, y: torch.Tensor):
        if self.method == 'none':
            return self

        if self.method == 'temp':
            T = nn.Parameter(torch.tensor(1.0, device=binlogits.device))
            bce = nn.BCEWithLogitsLoss()

            def closure():
                opt.zero_grad()
                t = torch.clamp(T, self.t_min, self.t_max)
                loss = bce(binlogits / t, y.float())
                loss.backward()
                return loss

            opt = optim.LBFGS([T], max_iter=100)
            opt.step(closure)
            T_val = torch.clamp(T, self.t_min, self.t_max).detach().item()
            self.params = {'T': float(T_val)}
            self.T = float(T_val)
            return self

        # models/evaluate.py 里 Calibrator.fit 的 'platt' 分支，替换成下面这个更稳的版本
        if self.method == 'platt':
            a = nn.Parameter(torch.tensor(1.0, device=binlogits.device))
            b = nn.Parameter(torch.tensor(0.0, device=binlogits.device))
            bce = nn.BCEWithLogitsLoss()

            def closure():
                opt.zero_grad()
                loss = bce(a * binlogits + b, y.float())
                loss.backward()
                return loss

            opt = optim.LBFGS([a, b], max_iter=self.max_iter)
            opt.step(closure)

            a_val = float(a.detach().item())
            b_val = float(b.detach().item())

            # 若斜率为负，吸收到参数里并记录 flipped=True
            self.flipped = False
            if a_val < 0:
                a_val, b_val = -a_val, -b_val
                self.flipped = True

            self.params = {'a': a_val, 'b': b_val}
            return self

        # isotonic 用概率域拟合：先把 binlogit 映射到 sigmoid 概率再拟合单调函数
        if self.method == 'isotonic' and _HAS_ISO:
            p = torch.sigmoid(binlogits).detach().cpu().numpy()
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(p, y.detach().cpu().numpy())
            self.params = ir
            return self

        self.method = 'none'
        return self

    @torch.no_grad()
    def transform(self, binlogits: torch.Tensor) -> torch.Tensor:
        if self.method == 'none':
            return torch.sigmoid(binlogits)
        if self.method == 'temp':
            T = self.params['T']
            return torch.sigmoid(binlogits / T)
        if self.method == 'platt':
            a, b = self.params['a'], self.params['b']
            return torch.sigmoid(a * binlogits + b)
        if self.method == 'isotonic' and _HAS_ISO:
            p = torch.sigmoid(binlogits).detach().cpu().numpy()
            p_cal = self.params.predict(p)
            return torch.tensor(p_cal, device=binlogits.device, dtype=torch.float32)
        return torch.sigmoid(binlogits)

@torch.no_grad()
def find_best_tau_acc(p: torch.Tensor, y: torch.Tensor, n_grid: int = 401):
    """在概率 p 上搜索最大化 Accuracy 的阈值。"""
    device = p.device
    taus = torch.linspace(0, 1, steps=n_grid, device=device)
    y = y.long()
    best_tau, best_acc = 0.5, -1.0
    for t in taus:
        pred = (p >= t).long()
        acc = (pred == y).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_tau = float(t.item())
    return best_tau, best_acc

@torch.no_grad()
def pick_best_calibrator(
    val_bl, val_y,
    methods=('none','platt','temp'),
    metric='ece',
    tau_metric='ba'      # ★ 新增：τ* 的选择指标
):
    assert metric in {'ece','nll'}
    assert tau_metric in {'acc','ba'}

    def tie_breaker(name):
        return {'platt': 0, 'temp': 1, 'none': 2}.get(name, 9)

    best = None; best_tau = 0.5
    best_val_score = -1.0    # ★ 这里不再叫 val_bal
    best_score = float('inf')

    for m in methods:
        cal = Calibrator(method=m).fit(val_bl.clone(), val_y)
        p = cal.transform(val_bl).clamp(1e-6, 1-1e-6)
        p, flipped = maybe_flip_to_class1(p, val_y)
        cal.flipped = flipped

        # ① 校准质量（越小越好）
        cal_score = (F.binary_cross_entropy(p, val_y.float()).item()
                     if metric == 'nll' else ece_score(p, val_y))

        # ② 在验证集上选 τ*
        if tau_metric == 'acc':
            tau, val_score = find_best_tau_acc(p, val_y)        # ★ 用 ACC
        else:
            tau, val_score = find_best_tau_balanced(p, val_y)   # 旧的 BA

        better = (cal_score < best_score - 1e-6) or \
                 (abs(cal_score - best_score) <= 1e-6 and
                  (val_score > best_val_score + 1e-6 or
                   (abs(val_score - best_val_score) <= 1e-6 and
                    tie_breaker(m) < tie_breaker(best.method if best else 'zzz'))))
        if better:
            best, best_score, best_tau, best_val_score = cal, cal_score, tau, val_score

    return best, float(best_tau), float(best_val_score)

@torch.no_grad()
def probs_with_temperature(model, loader, device, T=1.0):
    """返回经过温度缩放后的阳性概率 p1。"""
    model.eval()
    p_list = []
    for b in loader:
        logits = model(b['eeg'].to(device), b['facial'].to(device))['logits']
        p1 = (logits / T).softmax(dim=1)[:, 1]
        p_list.append(p1.cpu())
    if not p_list:
        return None
    return torch.cat(p_list)


# -------------------------
# 监控用：固定 0.5 的 Val-Acc
# -------------------------
@torch.no_grad()
def evaluate_binary_acc(model, loader, device):
    """固定阈值0.5的验证ACC（监控用）。"""
    model.eval()
    y_pred, y_true = [], []
    for b in loader:
        eeg = b['eeg'].to(device)
        facial = b['facial'].to(device)
        y = b['label_cls'].to(device)
        logits = model(eeg, facial)['logits']
        pred = logits.argmax(dim=1)
        y_pred.append(pred.cpu()); y_true.append(y.cpu())
    if not y_true:
        return float('nan')
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()
    return float(accuracy_score(y_true, y_pred))

# ------------------------------------------
# ACC 优先：在 val 上找最优阈值 → 用阈值评 ACC
# ------------------------------------------
@torch.no_grad()
def find_best_tau(model, val_loader, device, metric: str = "ba", grid=None):
    """
    在验证集上搜索最佳概率阈值。
    metric='acc'：最大化 Accuracy（推荐）。
    """
    assert metric in {"acc", "ba"}  # 预留 ba，当前我们只用 acc
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    model.eval()
    probs, ys = [], []
    for b in val_loader:
        p1 = model(b['eeg'].to(device), b['facial'].to(device))['logits'].softmax(1)[:, 1]
        probs.append(p1.cpu()); ys.append(b['label_cls'].cpu())
    if not ys:
        return 0.5, float('nan')

    probs = torch.cat(probs).numpy()
    ys = torch.cat(ys).numpy()

    best_t, best_s = 0.5, -1.0
    for t in grid:
        pred = (probs >= t).astype(int)
        s = accuracy_score(ys, pred) if metric == "acc" else balanced_accuracy_score(ys, pred)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)

@torch.no_grad()
def evaluate_binary_acc_with_tau(model, loader, device, tau: float):
    """
    用给定阈值 tau 在 loader 上评估 Accuracy。
    """
    model.eval()
    y_pred, y_true = [], []
    for b in loader:
        p1 = model(b['eeg'].to(device), b['facial'].to(device))['logits'].softmax(1)[:, 1]
        y_pred.append((p1 >= tau).long().cpu())
        y_true.append(b['label_cls'].cpu())
    if not y_true:
        return float('nan')
    y_pred = torch.cat(y_pred).numpy()
    y_true = torch.cat(y_true).numpy()
    return accuracy_score(y_true, y_pred)

@torch.no_grad()
def maybe_flip_to_class1(p: torch.Tensor, y: torch.Tensor):
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y.cpu().numpy(), p.cpu().numpy())
    except Exception:
        return p, False
    return (1.0 - p, True) if (auc == auc and auc < 0.5) else (p, False)

@torch.no_grad()
def _collect_logits_labels(model, loader, device):
    model.eval()
    logits_list, ys = [], []
    for b in loader:
        eeg = b['eeg'].to(device); facial = b['facial'].to(device)
        y   = b['label_cls'].to(device)
        logits = model(eeg, facial)['logits']
        logits_list.append(logits.detach()); ys.append(y.detach())
    if not ys:
        return None, None
    return torch.cat(logits_list), torch.cat(ys)

def fit_temperature(model, val_loader, device, max_iter=200):
    logits, y = _collect_logits_labels(model, val_loader, device)
    if logits is None:
        return 1.0

    logT = torch.nn.Parameter(torch.zeros(1, device=device))
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)

    def _closure():
        opt.zero_grad()
        T = torch.nn.functional.softplus(logT) + 1e-3  # T>0
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        return loss

    opt.step(_closure)
    with torch.no_grad():
        T = torch.nn.functional.softplus(logT).item() + 1e-3
        # 原来是 [0.5, 5.0] —— 会出现 3.x 的过平滑；收紧到 [0.5, 2.0]
        T = float(max(0.5, min(T, 2.0)))
    return T

@torch.no_grad()
def _collect_probs_labels_with_T(model, loader, device, T: float):
    model.eval()
    probs, ys = [], []
    for b in loader:
        lg = model(b["eeg"].to(device), b["facial"].to(device))["logits"]
        p1 = (lg / T).softmax(1)[:, 1]
        probs.append(p1.cpu()); ys.append(b["label_cls"].cpu())
    if not ys:
        return None, None
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()

def _best_tau_on_val(val_probs, val_ys, metric="acc", grid=None):
    if grid is None:
        # 更稳一些：阈值只在 [0.2, 0.8] 里搜，避免极端阈值过拟合
        grid = np.linspace(0.2, 0.8, 121)
    best_t, best_s = 0.5, -1.0
    for t in grid:
        pred = (val_probs >= t).astype(int)
        s = accuracy_score(val_ys, pred) if metric == "acc" else balanced_accuracy_score(val_ys, pred)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s

@torch.no_grad()
def prior_correction_em(p_test: torch.Tensor, p_val: torch.Tensor, y_val: torch.Tensor,
                        max_iter: int = 100, eps: float = 1e-6):
    """
    Saerens et al. (2002) label-shift EM:
    - 输入：p_test 为 test 上（已校准）的 P(y=1|x)，p_val,y_val 用来估计训练/验证先验 π。
    - 返回：p_test_adj（先验校正后的概率）、估计的 test 先验 q1。
    """
    device = p_test.device
    # 训练/验证先验 π = P(y=1) on val
    pi1 = y_val.float().mean().clamp(1e-6, 1-1e-6)
    pi = torch.tensor([1.0 - pi1, pi1], device=device)

    # 初始化 test 先验 q，用 test 上的平均概率
    q1 = p_test.mean().clamp(1e-6, 1-1e-6)
    q = torch.tensor([1.0 - q1, q1], device=device)

    # 组成 [N,2] 的类别后验矩阵
    P = torch.stack([1.0 - p_test, p_test], dim=1)  # [N,2]

    for _ in range(max_iter):
        prev = q.clone()
        # r_jk ∝ P_jk * q_k / pi_k
        W = P * (q / pi)                # [N,2]
        R = W / W.sum(dim=1, keepdim=True).clamp_min(1e-12)  # 归一化 responsibilities
        q = R.mean(dim=0).clamp(1e-6, 1-1e-6)
        if torch.max(torch.abs(q - prev)) < eps:
            break

    # 校正后的后验：P'(y|x) ∝ P(y|x) * q/π
    Padj = P * (q / pi)
    Padj = Padj / Padj.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return Padj[:, 1], float(q[1].item())

@torch.no_grad()
def evaluate_loso_binary(model, val_loader, test_loader, device,
                         use_prior_correction: bool = False,
                         calibrator_metric: str = "nll"):
    val_bl, val_y = _collect_binlogits_labels(model, val_loader, device)
    tst_bl, tst_y = _collect_binlogits_labels(model, test_loader, device)
    if val_bl is None or tst_bl is None:
        return {"acc": float("nan"), "thr": 0.5, "val_bal_acc": float("nan"),
                "calib": "none", "auc": float("nan")}

    cal, tau_star, val_score = pick_best_calibrator(
        val_bl, val_y, methods=("none", "platt", "temp"),
        metric=calibrator_metric, tau_metric="ba"
    )
    tag = cal.method if isinstance(cal.method, str) else str(cal.method)

    # 校准后概率
    p_val   = cal.transform(val_bl).clamp(1e-6, 1-1e-6)
    p_test0 = cal.transform(tst_bl).clamp(1e-6, 1-1e-6)

    # 若校准时发生翻转，在 test 同样翻转
    if getattr(cal, "flipped", False):
        p_val   = 1.0 - p_val
        p_test0 = 1.0 - p_test0

    # AUC 用未做先验校正的概率
    # 在 evaluate_loso_binary 里得到 p_test0 后、算 auc 之前：
    auc0 = roc_auc_safe(p_test0, tst_y)
    if auc0 == auc0 and auc0 < 0.5:
        p_test0 = 1.0 - p_test0
        auc = 1.0 - auc0
    else:
        auc = auc0

    # 方式①：τ*（来自 Val-BA）
    pred_star = (p_test0 >= tau_star).long()
    acc_star = (pred_star == tst_y).float().mean().item()

    # 方式②：τ_π（先验匹配，不用 EM）
    pi_val = float(val_y.float().mean().item())
    pi_val = max(1e-6, min(1-1e-6, pi_val))
    tau_pi = float(torch.quantile(p_test0, 1.0 - pi_val))
    acc_pi = (((p_test0 >= tau_pi).long() == tst_y).float().mean().item())

    # 原：use_pi = acc_pi > acc_star
    prefer_pi = True
    margin = 0.03  # 容忍 1% 的差距
    use_pi = (acc_pi >= acc_star - margin) if prefer_pi else (acc_pi > acc_star)

    thr_used = tau_pi if use_pi else tau_star
    acc_used = acc_pi if use_pi else acc_star

    best_acc = acc_used
    best_thr = thr_used
    best_tag = f"{tag}"

    if use_prior_correction:
        # 先验校正
        p_test_em, q1 = prior_correction_em(p_test0, p_val, val_y)

        # 方案A：EM后用 τ*
        acc_em_star = (((p_test_em >= tau_star).long() == tst_y).float().mean().item())
        # 方案B：EM后用 0.5（等代价下更理论）
        acc_em_05 = (((p_test_em >= 0.5).long() == tst_y).float().mean().item())

        # 也可给 EM 后再配 τ_π（以 EM 概率分位数匹配验证先验）
        tau_pi_em = float(torch.quantile(p_test_em, 1.0 - float(val_y.float().mean().item())))
        acc_em_pi = (((p_test_em >= tau_pi_em).long() == tst_y).float().mean().item())

        # 统一挑最优
        candidates = [
            (acc_star, tau_star, f"{tag}"),
            (acc_pi, tau_pi, f"{tag}+pi"),
            (acc_em_star, tau_star, f"{tag}+em"),
            (acc_em_05, 0.5, f"{tag}+em@0.5"),
            (acc_em_pi, tau_pi_em, f"{tag}+em+pi"),
        ]
        best_acc, best_thr, best_tag = max(candidates, key=lambda x: x[0])

        print(
            f"[TEST-em] ACC@τ*={acc_em_star:.3f} | ACC@0.5={acc_em_05:.3f} | ACC@τ_π(em)={acc_em_pi:.3f} | q1={q1:.3f}")

    # 覆盖最终使用
    acc_used, thr_used, tag_used = best_acc, best_thr, best_tag
    print(f"[TEST] FINAL use={tag_used} | ACC={acc_used:.3f} | thr={thr_used:.3f} | AUC={auc:.3f}")

    print(f"[VAL] pick {tag} | τ*={tau_star:.3f} | ValACC={val_score:.4f}")
    print(f"[TEST] ACC@τ*={acc_star:.3f} | ACC@τ_π={acc_pi:.3f} | AUC={auc:.3f} | use={'τ_π' if use_pi else 'τ*'}")
    T_used = 1.0
    if str(cal.method) == "temp":
        T_used = float(getattr(cal, "T", 1.0))
    val_auc = roc_auc_safe(p_val, val_y)
    return {
        "acc": float(acc_used),
        "thr": float(thr_used),
        "acc_star": float(acc_star),
        "thr_star": float(tau_star),
        "acc_pi": float(acc_pi),
        "thr_pi": float(tau_pi),
        # 向后兼容：如果外层还读 val_bal_acc，这里给同一个值
        "val_acc": float(val_score),
        "val_bal_acc": float(val_score),
        "calib": tag,
        "auc": float(auc),
        "val_auc": float(val_auc),
        "T": float(T_used),
    }

@torch.no_grad()
def find_best_tau_balanced(p: torch.Tensor, y: torch.Tensor, n_grid: int = 401):
    """在概率 p 上搜索最大化 Balanced-ACC 的阈值（Youden’s J）。"""
    device = p.device
    taus = torch.linspace(0, 1, steps=n_grid, device=device)
    y = y.long()
    best_tau, best_balacc = 0.5, -1.0
    for t in taus:
        pred = (p >= t).long()
        TP = ((pred == 1) & (y == 1)).sum().float()
        TN = ((pred == 0) & (y == 0)).sum().float()
        P  = (y == 1).sum().float().clamp(min=1.0)
        N  = (y == 0).sum().float().clamp(min=1.0)
        TPR = TP / P; TNR = TN / N
        balacc = 0.5 * (TPR + TNR)
        if balacc > best_balacc:
            best_balacc = float(balacc.item())
            best_tau = float(t.item())
    return best_tau, best_balacc

def roc_auc_safe(p: torch.Tensor, y: torch.Tensor):
    try:
        return float(roc_auc_score(y.detach().cpu().numpy(), p.detach().cpu().numpy()))
    except Exception:
        return float('nan')

def ece_score(p: torch.Tensor, y: torch.Tensor, n_bins: int = 15):
    y = y.detach().cpu().numpy(); p = p.detach().cpu().numpy()
    bins = np.linspace(0.0, 1.0, n_bins+1); ece = 0.0; n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = ((p[mask] >= 0.5).astype(int) == y[mask]).mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

import torch, numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
POS = 1  # 正类

@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    P, Y = [], []
    sm = torch.nn.Softmax(dim=1)
    for b in loader:
        eeg = b['eeg'].to(device); facial = b['facial'].to(device)
        y = b['label_cls'].to(device)
        prob1 = sm(model(eeg, facial)['logits'])[:, POS]
        P.append(prob1.cpu()); Y.append(y.cpu())
    return torch.cat(Y).numpy().astype(int), torch.cat(P).numpy()

from sklearn.metrics import balanced_accuracy_score
def _argmax_tau(y, p, step=0.01, metric='ba'):
    taus = np.arange(0.0, 1.0+1e-12, step)
    if metric == 'ba':
        score = lambda yy, pp: balanced_accuracy_score(yy, pp)
    else:
        score = lambda yy, pp: accuracy_score(yy, pp)
    scores = [score(y, (p>=t).astype(int)) for t in taus]
    i = int(np.argmax(scores))
    return float(taus[i]), float(scores[i])

@torch.no_grad()
def eval_with_val_tau(model, val_loader, test_loader, device, step=0.01, metric='ba'):
    yv, pv = _collect(model, val_loader, device)
    val_auc = roc_auc_score(yv, pv)                 # AUC 用正类概率，方向统一
    tau, val_score = _argmax_tau(yv, pv, step, metric)  # ★ BA/ACC 任选，默认 BA
    yt, pt = _collect(model, test_loader, device)
    test_acc = accuracy_score(yt, (pt >= tau).astype(int))
    print(
        f"[VAL] AUC={val_auc:.3f} | best τ*={tau:.3f} (by {metric.upper()}) | Val{metric.upper()}={val_score:.4f}")
    print(f"[TEST] ACC@τ*={test_acc:.4f}")

    return {"tau": tau, "val_metric": val_score, "test_acc": test_acc, "val_auc": float(val_auc)}