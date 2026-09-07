"""
능동 탐색(다중 phi 스냅샷) 기반 접촉 위치/힘 추정 모델. train_contact_model.py(단일 스냅샷)는
검증 정확도가 사실상 평균값 수준이었는데, test_active_sensing.py에서 phi를 바꿔가며 보면
"헷갈리는 쌍"이 구별된다는 걸 확인했으니, 이번엔 phi 3곳(-90/0/90도)에서 관측한 delta B를
전부 입력으로 준다.

구조: 각 프로브(phi 하나)마다 같은 CNN(가중치 공유)으로 특징을 뽑고, 3개 프로브의 특징을
이어붙인 뒤 회귀 헤드로 s,F_mag,Fx,Fy를 예측 - 가중치 공유라 프로브 순서/개수가 늘어나도
파라미터 수가 크게 안 늘어남.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_DIR, '..', '..', 'data', 'contact_scenarios')
MODELS_DIR = os.path.join(_DIR, '..', '..', 'models')

d = np.load(os.path.join(DATA_DIR, 'contact_multiprobe_5probe_15k_unbiased.npz'))
B_free, B_load, y_all, y_cols = d['B_free'], d['B_load'], d['y'], list(d['y_columns'])
ycol = {c: i for i, c in enumerate(y_cols)}
n_probes = B_free.shape[1]
print(f"n={len(B_free)}, probes={n_probes}, phi_probes={d['phi_probes']}")

delta = B_load - B_free  # (n, n_probes, 25, 3)
X = delta.reshape(len(delta), n_probes, 5, 5, 3).transpose(0, 1, 4, 2, 3)  # (n, probes, 3, 5, 5)

TARGETS = ['s', 'F_mag', 'Fx', 'Fy']
y = y_all[:, [ycol[t] for t in TARGETS]]

X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std
y_mean, y_std = y.mean(axis=0), y.std(axis=0)
y_norm = (y - y_mean) / y_std

rng = np.random.default_rng(0)
idx = rng.permutation(len(X_norm))
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_norm[train_idx]).float()),
    batch_size=64, shuffle=True)
val_X = torch.tensor(X_norm[val_idx]).float()
val_y = torch.tensor(y_norm[val_idx]).float()


class MultiProbeEstimator(nn.Module):
    """프로브마다 같은 CNN(가중치 공유)으로 임베딩을 뽑고 이어붙여서 회귀."""

    def __init__(self, n_probes, n_targets):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_targets)
        )

    def forward(self, x):  # x: (B, n_probes, 3, 5, 5)
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        return self.regressor(torch.cat(embeds, dim=1))


model = MultiProbeEstimator(n_probes, len(TARGETS))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
print("\n능동탐색(다중 프로브) 접촉 추정 모델 학습 시작...")
for epoch in range(200):
    model.train()
    batch_losses = []
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))

    model.eval()
    with torch.no_grad():
        val_losses.append(criterion(model(val_X), val_y).item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1:3d}/200] Train {train_losses[-1]:.4f}  Val {val_losses[-1]:.4f}")

model.eval()
with torch.no_grad():
    pred_norm = model(val_X).numpy()
true_norm = val_y.numpy()
pred = pred_norm * y_std + y_mean
true = true_norm * y_std + y_mean

print("\n[검증셋 오차 - 실제 단위] (단일 스냅샷 결과와 비교: s MAE는 21.67mm, F_mag MAE는 0.00496N 였음)")
for i, t in enumerate(TARGETS):
    mae = np.mean(np.abs(pred[:, i] - true[:, i]))
    unit = 'mm' if t == 's' else 'N'
    r2 = 1 - np.sum((pred[:, i] - true[:, i])**2) / np.sum((true[:, i] - true[:, i].mean())**2)
    print(f"  {t}: MAE={mae:.5f} {unit}  R2={r2:.3f}  (타깃 표준편차={y_std[i]:.5f} {unit})")

torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'contact_multiprobe_estimator_15k_unbiased.pth'))
np.savez(os.path.join(DATA_DIR, 'multiprobe_train_history_15k_unbiased.npz'),
         train_losses=train_losses, val_losses=val_losses,
         val_pred=pred, val_true=true, targets=np.array(TARGETS))
print("\n저장 완료: models/contact_multiprobe_estimator_15k_unbiased.pth, data/contact_scenarios/multiprobe_train_history_15k_unbiased.npz")
