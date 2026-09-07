"""
2단계 NN 접근법의 이상적 상한선 테스트.
NN1(자기장->위치추정)이 완벽하다고 가정하고(노이즈 없는 진짜 x,y,theta 사용), L_M까지 입력에
포함해서 NN2(위치+L_M -> s_f,F)를 학습시켜 얼마나 정확한지 확인.

이전 시도(같은 조건, 물리모델을 수치최적화로 역산)는 R2가 음수로 완전히 실패했었음(느리고
국소최솟값에 갇힘). 이번엔 최적화 대신 신경망으로 학습시켜서 그 문제를 피할 수 있는지 확인.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "force_model"))
import force_model as fm

DATA_DIR = os.path.join(HERE, "..", "..", "data", "contact_scenarios")
PHI_PROBES = [-120.0, -60.0, 0.0, 60.0, 120.0]

d = np.load(os.path.join(DATA_DIR, "contact_multiprobe_5probe_15k_unbiased.npz"))
y_all_full = d["y"]
y_cols = list(d["y_columns"])
ycol = {c: i for i, c in enumerate(y_cols)}

# 3000개 부분표본에서 이미 방향성 확인됨(s R2=0.90) - 이번엔 15000개 전체로 스케일업
y_all = y_all_full
n = len(y_all)
print(f"n={n} (전체), 위치(x,y,theta) 재계산 중...")

# ── 저장된 (L_M,s,Fx,Fy)로부터 진짜(노이즈 없는) x,y,theta를 각 프로브에서 재계산 ──
X_list = []
targets = ['s', 'F_mag', 'Fx', 'Fy']
y = y_all[:, [ycol[t] for t in targets]]
L_M_arr = y_all[:, ycol['L_M']]

n_fail = 0
valid_idx = []
for i in range(n):
    L_M = y_all[i, ycol['L_M']]
    s = y_all[i, ycol['s']]
    Fx = y_all[i, ycol['Fx']]
    Fy = y_all[i, ycol['Fy']]

    feat = [L_M]
    ok = True
    for phi in PHI_PROBES:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        try:
            r_load = fm.solve_shape_robust(L_M=L_M, phi_deg=phi,
                                            loads=[{"type": "point", "s": s, "Fx": Fx, "Fy": Fy}],
                                            theta_L_hint_deg=r_free["theta_L_deg"])
        except RuntimeError:
            ok = False
            break
        # delta descriptor (loaded - free), CNN의 delta_B와 같은 취지
        feat.extend([
            r_load["x_L"] - r_free["x_L"], r_load["y_L"] - r_free["y_L"],
            r_load["theta_L_deg"] - r_free["theta_L_deg"],
            r_load["x_LM"] - r_free["x_LM"], r_load["y_LM"] - r_free["y_LM"],
            r_load["theta_LM_deg"] - r_free["theta_LM_deg"],
        ])
    if not ok:
        n_fail += 1
        continue
    X_list.append(feat)
    valid_idx.append(i)

    if (len(X_list)) % 1000 == 0:
        print(f"{len(X_list)}/{n}", flush=True)

X = np.array(X_list)  # (n_valid, 31) = L_M(1) + 5probes*6
y = y[valid_idx]
print(f"완료: {len(X)}개 유효 (실패 {n_fail}건)")

# 재사용 위해 저장 (다음에 NN1+NN2 실제 파이프라인 만들 때 이 재계산을 또 안 하려고)
np.savez(os.path.join(DATA_DIR, "shape_to_contact_features_15k.npz"),
         X=X, y=y, targets=np.array(targets), valid_idx=np.array(valid_idx))
print("저장: shape_to_contact_features_15k.npz")

X_mean, X_std = X.mean(0), X.std(0)
X_norm = (X - X_mean) / (X_std + 1e-9)
y_mean, y_std = y.mean(0), y.std(0)
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


class ShapeToContactNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


model = ShapeToContactNet(X.shape[1], len(targets))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nNN2(좌표+L_M -> s_f,F) 학습 시작...")
for epoch in range(200):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            vloss = criterion(model(val_X), val_y).item()
        print(f"Epoch [{epoch+1:3d}/200] Val Loss {vloss:.4f}")

model.eval()
with torch.no_grad():
    pred_norm = model(val_X).numpy()
true_norm = val_y.numpy()
pred = pred_norm * y_std + y_mean
true = true_norm * y_std + y_mean

print("\n[결과 - 이상적 상한선(위치추정 노이즈 없음 가정), n=15000]")
for i, t in enumerate(targets):
    mae = np.mean(np.abs(pred[:, i] - true[:, i]))
    r2 = 1 - np.sum((pred[:, i] - true[:, i])**2) / np.sum((true[:, i] - true[:, i].mean())**2)
    unit = 'mm' if t == 's' else 'N'
    print(f"  {t}: R2={r2:.3f}  MAE={mae:.5f}{unit}")
print("\n(비교: 3천개 부분표본 - s R2=0.902, F_mag R2=0.706)")
print("(비교: 기존 CNN 직접추정 - s R2=0.750, F_mag R2=0.335)")
print("(비교: 물리모델 최적화역산 - s R2=-0.100, F_mag R2=-1.315, 완전실패)")

MODELS_DIR = os.path.join(HERE, "..", "..", "models")
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "shape_to_contact_nn2_15k.pth"))
np.savez(os.path.join(DATA_DIR, "shape_to_contact_train_history_15k.npz"),
         val_pred=pred, val_true=true, targets=np.array(targets),
         X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
print("저장: models/shape_to_contact_nn2_15k.pth")
