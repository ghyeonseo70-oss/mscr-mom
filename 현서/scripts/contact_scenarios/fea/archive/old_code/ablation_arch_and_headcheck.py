"""캐시된 15만개 single-probe 데이터(segment_bfield_singleprobe_4seg.npz)를 재사용해서
FEA 재계산 없이 두 가지를 빠르게 검증:

1) 구간분류 head가 정말 필요한가 - 이미 학습된 CNN의 "연속값 s 예측을 그냥 20mm로 이진화"한
   것만으로 별도 분류 head와 비슷한 정확도가 나오는지 비교.
2) CNN이 최선인가 - 5x5짜리 초소형 그리드는 CNN의 핵심 이점(위치 무관 가중치 공유)이
   딱히 의미가 없어 보여서(센서 25개가 전부 고정된 서로 다른 위치), 같은 데이터/같은
   학습설정으로 순수 MLP(flatten 75 -> MLP)를 새로 학습시켜 CNN과 성능을 비교.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")

BIN_WIDTH_MM = 20.0
N_CLASSES = 4
N_PROBES = 1
N_EPOCHS = 60


def s_to_bin_vec(s_mm):
    return np.clip((s_mm / BIN_WIDTH_MM).astype(int), 0, N_CLASSES - 1)


def balanced_acc(pred, true, n_classes):
    recalls = []
    for c in range(n_classes):
        mask = true == c
        recalls.append((pred[mask] == c).mean() if mask.sum() > 0 else 0.0)
    return float(np.mean(recalls))


class SingleProbeClassifier(nn.Module):
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2, n_config=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        self.config_head = nn.Linear(128, n_config)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


class PlainMLPClassifier(nn.Module):
    """5x5 센서그리드는 CNN의 위치-무관 가중치공유가 의미 없어 보여서(고정된 25개 위치),
    같은 파라미터 예산으로 순수 MLP(입력 75개 flatten)를 비교용으로 학습."""
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2, n_config=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_probes * 3 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        self.config_head = nn.Linear(128, n_config)

    def forward(self, x):
        h = self.net(x.reshape(x.shape[0], -1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


def train_and_eval(model, X_norm, y_all, f_norm, f_all, f_mean, f_std, s_norm, s_all, s_mean, s_std,
                    c_norm, c_all, c_mean, c_std, train_idx, val_idx, device, n_epochs=N_EPOCHS, tag=""):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_all[train_idx]).long(),
                      torch.tensor(f_norm[train_idx]).float(), torch.tensor(s_norm[train_idx]).float(),
                      torch.tensor(c_norm[train_idx]).float()),
        batch_size=256, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float().to(device)
    val_y = torch.tensor(y_all[val_idx]).long().to(device)
    val_f = torch.tensor(f_norm[val_idx]).float().to(device)
    val_s = torch.tensor(s_norm[val_idx]).float().to(device)
    val_c = torch.tensor(c_norm[val_idx]).float().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    seg_criterion = nn.CrossEntropyLoss()
    force_criterion = nn.MSELoss()
    s_criterion = nn.MSELoss()
    config_criterion = nn.MSELoss()

    t0 = time.time()
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        for bx, by, bf, bs, bc in train_loader:
            bx, by, bf, bs, bc = bx.to(device), by.to(device), bf.to(device), bs.to(device), bc.to(device)
            optimizer.zero_grad()
            seg_logits, force_pred, s_pred, config_pred = model(bx)
            loss = (seg_criterion(seg_logits, by) + force_criterion(force_pred, bf)
                    + s_criterion(s_pred, bs) + config_criterion(config_pred, bc))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_seg_logits, val_force_pred, val_s_pred, val_config_pred = model(val_X)
            val_loss = (seg_criterion(val_seg_logits, val_y).item() + force_criterion(val_force_pred, val_f).item()
                        + s_criterion(val_s_pred, val_s).item() + config_criterion(val_config_pred, val_c).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  [{tag}] epoch {epoch+1}/{n_epochs} val_loss={val_loss:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_seg_logits, val_force_pred, val_s_pred, val_config_pred = model(val_X)
        val_pred = val_seg_logits.argmax(dim=1).cpu().numpy()
        val_force_phys = val_force_pred.cpu().numpy() * f_std + f_mean
        val_s_phys = val_s_pred.cpu().numpy() * s_std + s_mean
        val_c_phys = val_config_pred.cpu().numpy() * c_std + c_mean
    val_y_np = y_all[val_idx]
    acc = (val_pred == val_y_np).mean()
    bacc = balanced_acc(val_pred, val_y_np, N_CLASSES)

    def r2(pred, true):
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"[{tag}] 학습시간 {time.time()-t0:.0f}s")
    print(f"[{tag}] seg acc={acc*100:.1f}%  balanced={bacc*100:.1f}%")
    print(f"[{tag}] s: R2={r2(val_s_phys, s_all[val_idx]):.3f} MAE={np.mean(np.abs(val_s_phys-s_all[val_idx])):.2f}mm")
    print(f"[{tag}] Fx: R2={r2(val_force_phys[:,0], f_all[val_idx][:,0]):.3f}  Fy: R2={r2(val_force_phys[:,1], f_all[val_idx][:,1]):.3f}")
    print(f"[{tag}] L_M: R2={r2(val_c_phys[:,0], c_all[val_idx][:,0]):.3f}  phi: R2={r2(val_c_phys[:,1], c_all[val_idx][:,1]):.3f}")
    return dict(pred=val_pred, s_phys=val_s_phys, acc=acc, bacc=bacc)


if __name__ == "__main__":
    data = np.load(os.path.join(FEA_DATA_DIR, "segment_bfield_singleprobe_4seg.npz"))
    X_all, y_all, f_all, s_all, c_all = data["X"], data["y"], data["f"], data["s"], data["c"]
    print(f"데이터 로드: {len(y_all)}개")

    X_mean2, X_std2 = X_all.mean(), X_all.std()
    X_norm = (X_all - X_mean2) / X_std2

    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(X_norm))
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    fxy_all = f_all[:, :2]
    f_mean, f_std = fxy_all.mean(axis=0), fxy_all.std(axis=0)
    f_std[f_std < 1e-12] = 1.0
    f_norm = (fxy_all - f_mean) / f_std

    s_mean, s_std = s_all.mean(), s_all.std()
    s_norm = (s_all - s_mean) / s_std

    c_mean, c_std = c_all.mean(axis=0), c_all.std(axis=0)
    c_std[c_std < 1e-12] = 1.0
    c_norm = (c_all - c_mean) / c_std

    device = torch.device("cpu")

    # ── 검증 1) 구간분류 head vs 연속값 s 이진화 (기존 학습된 CNN 재사용, 재학습 없음) ──
    print("\n=== 검증 1: 분류 head vs 연속값 이진화 (기존 학습된 CNN 모델 사용) ===")
    ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_4seg.pth"),
                       map_location="cpu", weights_only=False)
    cnn_model = SingleProbeClassifier()
    cnn_model.load_state_dict(ckpt["state_dict"])
    cnn_model.eval()
    val_X = torch.tensor(X_norm[val_idx]).float()
    with torch.no_grad():
        seg_logits, _, s_pred, _ = cnn_model(val_X)
    val_pred_cls = seg_logits.argmax(dim=1).numpy()
    val_s_pred_phys = s_pred.numpy() * s_std + s_mean
    val_pred_froms = s_to_bin_vec(val_s_pred_phys)
    val_y_np = y_all[val_idx]

    acc_cls = (val_pred_cls == val_y_np).mean()
    bacc_cls = balanced_acc(val_pred_cls, val_y_np, N_CLASSES)
    acc_froms = (val_pred_froms == val_y_np).mean()
    bacc_froms = balanced_acc(val_pred_froms, val_y_np, N_CLASSES)
    print(f"  분류 head:      acc={acc_cls*100:.1f}%  balanced={bacc_cls*100:.1f}%")
    print(f"  연속값 이진화:   acc={acc_froms*100:.1f}%  balanced={bacc_froms*100:.1f}%")

    # ── 검증 2) CNN vs 순수 MLP (같은 데이터/설정으로 재학습해서 비교) ──
    print("\n=== 검증 2: CNN vs 순수 MLP (같은 15만개 데이터로 재학습, 60 epoch) ===")
    mlp_model = PlainMLPClassifier().to(device)
    train_and_eval(mlp_model, X_norm, y_all, f_norm, f_all, f_mean, f_std, s_norm, s_all, s_mean, s_std,
                   c_norm, c_all, c_mean, c_std, train_idx, val_idx, device, tag="MLP")

    print("\n[참고: CNN 원본 결과] seg balanced=71.3%  s R2=0.760  Fx R2=0.953  Fy R2=0.953  L_M R2=0.969  phi R2=0.972")
