"""
폭(width) 연속값 추정은 실패(R2=0.02)했으니, "좁다/넓다" 이진분류로 낮춰서 재시도.
기존에 생성해둔 surface_contact_features_15k.npz를 그대로 재사용(데이터 재생성 없음).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data", "contact_scenarios")

d = np.load(os.path.join(DATA_DIR, "surface_contact_features_15k.npz"))
X, y, targets = d["X"], d["y"], list(d["targets"])
tcol = {c: i for i, c in enumerate(targets)}
width = y[:, tcol["width"]]

threshold = np.median(width)
label = (width > threshold).astype(np.float32)
print(f"기준선(중앙값): {threshold:.2f}mm -> 좁음 {int((1-label).sum())}개 / 넓음 {int(label.sum())}개")

X_mean, X_std = X.mean(0), X.std(0)
X_norm = (X - X_mean) / (X_std + 1e-9)

rng = np.random.default_rng(0)
idx = rng.permutation(len(X_norm))
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(label[train_idx]).float()),
    batch_size=64, shuffle=True)
val_X = torch.tensor(X_norm[val_idx]).float()
val_y = torch.tensor(label[val_idx]).float()
val_width = width[val_idx]


class WidthClassifier(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = WidthClassifier(X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print("\n폭 이진분류(좁음/넓음) 학습 시작...")
for epoch in range(150):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 15 == 0:
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(val_X)) > 0.5).float()
            acc = (pred == val_y).float().mean().item()
        print(f"Epoch [{epoch+1:3d}/150] Val Accuracy {acc*100:.1f}%")

model.eval()
with torch.no_grad():
    logits = model(val_X)
    pred = (torch.sigmoid(logits) > 0.5).float().numpy()
true = val_y.numpy()

acc = (pred == true).mean()
tpr = (pred[true == 1] == 1).mean()  # 넓음을 넓다고 맞춘 비율
tnr = (pred[true == 0] == 0).mean()  # 좁음을 좁다고 맞춘 비율
print(f"\n[결과] 정확도={acc*100:.1f}% (넓음탐지율={tpr*100:.1f}%, 좁음탐지율={tnr*100:.1f}%)")
print(f"(참고: 그냥 항상 다수클래스로 찍으면 정확도=50%대 - 그것보다 얼마나 나은지가 진짜 성능)")

# 기준선에서 멀수록(아주 좁거나 아주 넓거나) 더 잘 맞히는지 확인
correct = (pred == true)
bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
print("\n[폭 구간별 정확도]")
for lo, hi in bins:
    m = (val_width >= lo) & (val_width < hi)
    if m.sum() > 5:
        print(f"  {lo}-{hi}mm (n={m.sum()}): {correct[m].mean()*100:.1f}%")

np.savez(os.path.join(DATA_DIR, "width_classification_result.npz"),
          pred=pred, true=true, width=val_width, threshold=threshold, acc=acc)
print(f"\n저장: width_classification_result.npz")
