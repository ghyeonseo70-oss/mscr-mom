"""
이미 만들어둔 15만개(segment_bfield_multiprobe_150k_11probe_force.npz)를 재사용해서,
더 큰 네트워크 + 코사인 LR 스케줄 + 더 많은 에폭으로 재학습만 다시 시도.
데이터 재생성(100분+) 없이 학습만(몇 분) 다시 하는 거라 빠르게 결과 확인 가능.
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

N_CLASSES, N_PROBES = 5, 11
BIN_WIDTH_MM = 20.0

print("데이터 로드 중...")
d = np.load(os.path.join(FEA_DATA_DIR, "segment_bfield_multiprobe_150k_11probe_force.npz"))
X_all, y_all, f_all = d["X"], d["y"], d["f"]
print(f"n={len(y_all)}")

fxy_all = f_all[:, :2]
f_mean, f_std = fxy_all.mean(0), fxy_all.std(0)
f_std[f_std < 1e-12] = 1.0
f_norm = (fxy_all - f_mean) / f_std

X_mean2, X_std2 = X_all.mean(), X_all.std()
X_norm = (X_all - X_mean2) / X_std2

rng = np.random.default_rng(0)
idx = rng.permutation(len(X_norm))
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_all[train_idx]).long(),
                  torch.tensor(f_norm[train_idx]).float()),
    batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_X = torch.tensor(X_norm[val_idx]).float().to(device)
val_y = torch.tensor(y_all[val_idx]).long().to(device)
val_f = torch.tensor(f_norm[val_idx]).float().to(device)
val_f_phys = fxy_all[val_idx]


class MultiProbeClassifierBig(nn.Module):
    """기존(16/32채널, trunk 128) 대비 더 크게(32/64채널, trunk 256) + conv 1개 추가."""
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 5 * 5, 128), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 * n_probes, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
        )
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h)


EPOCHS = 150
model = MultiProbeClassifierBig().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
seg_criterion = nn.CrossEntropyLoss()
force_criterion = nn.MSELoss()

print(f"학습 시작 (device={device}, epochs={EPOCHS}, 큰 네트워크 + 코사인 LR 스케줄)...")
t0 = time.time()
best_val_loss = float("inf")
best_state = None
best_epoch = -1
for epoch in range(EPOCHS):
    model.train()
    for bx, by, bf in train_loader:
        bx, by, bf = bx.to(device, non_blocking=True), by.to(device, non_blocking=True), bf.to(device, non_blocking=True)
        optimizer.zero_grad()
        seg_logits, force_pred = model(bx)
        loss = seg_criterion(seg_logits, by) + force_criterion(force_pred, bf)
        loss.backward()
        optimizer.step()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        val_seg_logits, val_force_pred = model(val_X)
        val_seg_loss = seg_criterion(val_seg_logits, val_y).item()
        val_force_loss = force_criterion(val_force_pred, val_f).item()
        val_loss = val_seg_loss + val_force_loss
        val_acc = (val_seg_logits.argmax(dim=1) == val_y).float().mean().item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1
    if (epoch + 1) % 10 == 0:
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] ValAcc {val_acc*100:5.1f}%  SegLoss {val_seg_loss:.4f}  "
              f"ForceLoss {val_force_loss:.4f}  LR {lr_now:.5f}")
print(f"학습 완료 ({time.time()-t0:.0f}s), 최적 epoch={best_epoch}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    val_seg_logits, val_force_pred = model(val_X)
    val_pred = val_seg_logits.argmax(dim=1)
    val_force_pred_phys = val_force_pred.cpu().numpy() * f_std + f_mean
final_acc = (val_pred == val_y).float().mean().item()
conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.int32)
for t, p in zip(val_y.tolist(), val_pred.tolist()):
    conf[t, p] += 1
bin_labels = [f"{int(i*BIN_WIDTH_MM)}-{int((i+1)*BIN_WIDTH_MM)}mm" for i in range(N_CLASSES)]
per_class_recall = [conf[i, i].item() / max(1, conf[i].sum().item()) for i in range(N_CLASSES)]

print(f"\n=== 최종 검증 정확도: {final_acc*100:.1f}% (n_val={len(val_idx)}), 무작위 기준선=20.0% ===")
print(f"balanced accuracy: {np.mean(per_class_recall)*100:.1f}%")
print("구간별 recall:", {bin_labels[i]: f"{per_class_recall[i]*100:.1f}%" for i in range(N_CLASSES)})
print(conf.numpy())

force_names = ["Fx_board_N", "Fy_board_N"]
for i, name in enumerate(force_names):
    pred_i, true_i = val_force_pred_phys[:, i], val_f_phys[:, i]
    r2 = 1 - np.sum((pred_i - true_i) ** 2) / np.sum((true_i - true_i.mean()) ** 2)
    print(f"  {name}: R^2={r2:.3f}")
pred_fmag = np.sqrt(val_force_pred_phys[:, 0] ** 2 + val_force_pred_phys[:, 1] ** 2)
true_fmag = np.sqrt(val_f_phys[:, 0] ** 2 + val_f_phys[:, 1] ** 2)
r2 = 1 - np.sum((pred_fmag - true_fmag) ** 2) / np.sum((true_fmag - true_fmag.mean()) ** 2)
print(f"  F_mag(유도): R^2={r2:.3f}")

os.makedirs(MODELS_DIR, exist_ok=True)
torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
            "f_mean": f_mean, "f_std": f_std, "bin_width_mm": BIN_WIDTH_MM, "n_classes": N_CLASSES},
           os.path.join(MODELS_DIR, "position_segment_classifier_bigger.pth"))
print(f"저장: {MODELS_DIR}/position_segment_classifier_bigger.pth")
print(f"총 소요시간: {(time.time()-t0)/60:.1f}분")
