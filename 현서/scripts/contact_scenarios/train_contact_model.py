"""
contact_bfield_dataset.npz(물리모델 기반 3000개 - B_free/B_load/타깃)로 접촉(충돌) 추정
모델을 학습. cnn_model/cnn_ai.py(위치추정 CNN)와 같은 구조(5x5x3 -> Conv -> 회귀)를 재사용.

입력은 B_load 자체가 아니라 delta = B_load - B_free (접촉 없을 때 대비 자기장 변화량)를 씀 -
실제 시스템에서도 "지금 명령한 L_M/phi로 기대되는 자기장(B_free, 계산 가능)" 대비 "실제로
측정된 자기장(B_load)"의 차이가 접촉을 알려주는 핵심 신호이기 때문.

타깃은 4개로 좁힘: 접촉위치 s, 힘 크기 F_mag, Fx, Fy. (L_M/phi_deg는 이미 구동 시 알고
있는 명령값이라 "추정"할 필요가 없어서 제외.)
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

# 1. 데이터 로드 및 전처리
d = np.load(os.path.join(DATA_DIR, 'contact_bfield_dataset.npz'))
B_free, B_load, y_all, y_cols = d['B_free'], d['B_load'], d['y'], list(d['y_columns'])
ycol = {c: i for i, c in enumerate(y_cols)}

delta = B_load - B_free  # (n, 25, 3)
X = delta.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)  # (n, 3, 5, 5), cnn_ai.py와 동일 레이아웃

TARGETS = ['s', 'F_mag', 'Fx', 'Fy']
y = y_all[:, [ycol[t] for t in TARGETS]]

# 입력/타깃 모두 스케일 차이가 커서(자기장 uT vs 힘 N) 정규화
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


class ContactEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, len(TARGETS))
        )

    def forward(self, x):
        return self.regressor(self.features(x))


model = ContactEstimator()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
print("접촉 추정 모델 학습 시작 (n_train=%d, n_val=%d)..." % (len(train_idx), len(val_idx)))
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

# 2. 검증셋 실제 단위 오차 (정규화 해제)
model.eval()
with torch.no_grad():
    pred_norm = model(val_X).numpy()
true_norm = val_y.numpy()
pred = pred_norm * y_std + y_mean
true = true_norm * y_std + y_mean

print("\n[검증셋 오차 - 실제 단위]")
for i, t in enumerate(TARGETS):
    mae = np.mean(np.abs(pred[:, i] - true[:, i]))
    unit = 'mm' if t == 's' else 'N'
    print(f"  {t}: MAE={mae:.5f} {unit}  (타깃 표준편차={y_std[i]:.5f} {unit})")

# 3. 저장
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'contact_estimator.pth'))
np.savez(os.path.join(DATA_DIR, 'contact_train_history.npz'),
         train_losses=train_losses, val_losses=val_losses,
         val_pred=pred, val_true=true, targets=np.array(TARGETS),
         X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
print("\n저장 완료: models/contact_estimator.pth, data/contact_scenarios/contact_train_history.npz")
