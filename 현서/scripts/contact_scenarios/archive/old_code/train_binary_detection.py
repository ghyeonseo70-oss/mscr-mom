"""
목표를 낮춘 첫 시도: "접촉이 있었나 없었나"만 판별하는 이진분류.

contact_force_scenarios.npz의 3000개 구동상태(L_M,phi)마다, 무외력(free)/접촉시(load) 두
형상에 대해 자기장을 새로 계산한다(generate_contact_bfield.py와 같은 자석/센서 모델 재사용).
중요한 차이: 여기서는 "관측값 - 기대값(무외력 예측)"을 만들 때, 무외력 케이스와 접촉 케이스
각각에 독립적인 노이즈를 새로 뽑아서 입힌다 - 실제로는:
  - 무외력 상황: 센서가 (기대되는 무외력 자기장 + 측정노이즈)를 읽음 -> 기대값과의 차이 = 노이즈뿐
  - 접촉 상황: 센서가 (실제 접촉시 자기장 + 측정노이즈)를 읽음 -> 차이 = 진짜 변화 + 노이즈
이렇게 3000(무접촉) + 3000(접촉) = 6000개 균형잡힌 이진분류 데이터셋을 만든다.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'force_model'))
sys.path.insert(0, HERE)
import force_model as fm
from generate_contact_bfield import compute_B  # 자석/센서 모델 재사용

DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'contact_scenarios')
MODELS_DIR = os.path.join(HERE, '..', '..', 'models')

d = np.load(os.path.join(DATA_DIR, "contact_force_scenarios.npz"), allow_pickle=True)
data, cols = d["data"], list(d["columns"])
col = {c: i for i, c in enumerate(cols)}
n = len(data)
print(f"입력: {n}개 구동상태")

rng = np.random.default_rng(123)


def noisy(B, frac=0.05):
    return B + rng.normal(0, np.max(np.abs(B)) * frac, B.shape)


X_list, y_list, Fmag_list = [], [], []
for i in range(n):
    row = data[i]
    B_free_true = compute_B(row[col["xLM_free"]], row[col["yLM_free"]], row[col["thLM_free"]],
                             row[col["xL_free"]], row[col["yL_free"]], row[col["thL_free"]])
    B_load_true = compute_B(row[col["xLM_load"]], row[col["yLM_load"]], row[col["thLM_load"]],
                             row[col["xL_load"]], row[col["yL_load"]], row[col["thL_load"]])

    obs_no_contact = noisy(B_free_true)      # 무접촉: 기대값 + 노이즈만
    obs_contact = noisy(B_load_true)         # 접촉: 실제 변화 + 노이즈

    X_list.append((obs_no_contact - B_free_true).flatten())
    y_list.append(0)
    Fmag_list.append(0.0)

    X_list.append((obs_contact - B_free_true).flatten())
    y_list.append(1)
    Fmag_list.append(row[col["F_mag"]])

    if (i + 1) % 500 == 0:
        print(f"{i + 1}/{n}", flush=True)

X = np.array(X_list).reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
y = np.array(y_list)
F_mag_arr = np.array(Fmag_list)

X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std

rng2 = np.random.default_rng(0)
idx = rng2.permutation(len(X_norm))
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y[train_idx]).float()),
    batch_size=64, shuffle=True)
val_X = torch.tensor(X_norm[val_idx]).float()
val_y = torch.tensor(y[val_idx]).float()
val_Fmag = F_mag_arr[val_idx]


class ContactDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(-1)


model = ContactDetector()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

print("\n이진분류(접촉 유무) 학습 시작...")
for epoch in range(100):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_pred = (torch.sigmoid(val_logits) > 0.5).float()
            acc = (val_pred == val_y).float().mean().item()
        print(f"Epoch [{epoch+1:3d}/100] Val Accuracy: {acc*100:.1f}%")

# 최종 평가 + F_mag(힘 크기)별 정확도 - "얼마나 약한 접촉까지 감지되는지" 확인
model.eval()
with torch.no_grad():
    val_logits = model(val_X)
    val_pred = (torch.sigmoid(val_logits) > 0.5).float()
final_acc = (val_pred == val_y).float().mean().item()
print(f"\n최종 검증 정확도: {final_acc*100:.2f}%")

contact_mask = val_y.numpy() == 1
correct_contact = (val_pred.numpy()[contact_mask] == 1)
f_vals = val_Fmag[contact_mask] * 1000  # mN
bins = [0, 2, 5, 10, 15, 20]
print("\n[힘 크기 구간별 접촉 탐지율]")
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (f_vals >= lo) & (f_vals < hi)
    if m.sum() > 0:
        print(f"  F={lo}~{hi}mN (n={m.sum()}): 탐지율 {correct_contact[m].mean()*100:.1f}%")

no_contact_mask = val_y.numpy() == 0
false_positive_rate = (val_pred.numpy()[no_contact_mask] == 1).mean()
print(f"\n오탐율(접촉 없는데 있다고 판단): {false_positive_rate*100:.2f}%")

torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'contact_binary_detector.pth'))
np.savez(os.path.join(DATA_DIR, 'binary_detection_result.npz'),
         val_pred=val_pred.numpy(), val_true=val_y.numpy(), val_Fmag=val_Fmag,
         final_acc=final_acc, false_positive_rate=false_positive_rate)
print("\n저장 완료: models/contact_binary_detector.pth, data/contact_scenarios/binary_detection_result.npz")
