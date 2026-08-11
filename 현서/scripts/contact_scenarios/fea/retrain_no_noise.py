"""resume_from_surrogate.py가 이미 만들어둔 5만개 합성 시나리오(fea_contact_force_scenarios.npz)를
재사용해서, 노이즈 없이(원인 후보 1 검증) 자기장 계산 + 최종 CNN(ContactEstimator) 학습만 다시
돌린다. 5만개 시나리오 생성(1.8시간)은 건너뛰므로 몇 분이면 끝남.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios")
FEA_DATA_DIR = os.path.join(DATA_DIR, "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")

LOG = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
sys.path.insert(0, os.path.join(HERE, ".."))
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.36
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_local, yLM_local, thLM_deg, xL_local, yL_local, thL_deg):
    xLM_b, yLM_b = fm.to_board_frame(xLM_local, yLM_local)
    xL_b, yL_b = fm.to_board_frame(xL_local, yL_local)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler('z', -thLM_deg, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler('z', -thL_deg, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


LOG("캐시된 5만개 합성 시나리오 로드")
cache = np.load(os.path.join(FEA_DATA_DIR, "fea_contact_force_scenarios.npz"), allow_pickle=True)
data_arr = cache["data"]
columns = list(cache["columns"])
col = {c: i for i, c in enumerate(columns)}
n = len(data_arr)
LOG(f"{n}개 로드 완료")

LOG("자기장(B) 계산 시작 (노이즈 없음)")
B_free_all = np.zeros((n, 25, 3))
B_load_all = np.zeros((n, 25, 3))
for i in range(n):
    row = data_arr[i]
    B_free_all[i] = compute_B(row[col["xLM_free"]], row[col["yLM_free"]], row[col["thLM_free"]],
                               row[col["xL_free"]], row[col["yL_free"]], row[col["thL_free"]])
    B_load_all[i] = compute_B(row[col["xLM_load"]], row[col["yLM_load"]], row[col["thLM_load"]],
                               row[col["xL_load"]], row[col["yL_load"]], row[col["thL_load"]])
    if (i + 1) % 10000 == 0:
        LOG(f"자기장 계산 {i+1}/{n}")

# 노이즈 없음(원인 후보1 검증) - add_noise 호출 자체를 생략
B_free_noisy = B_free_all
B_load_noisy = B_load_all

y_all = data_arr[:, [col["s"], col["F_mag"], col["Fx"], col["Fy"]]]
np.savez(os.path.join(FEA_DATA_DIR, "fea_contact_bfield_dataset_nonoise.npz"),
         B_free=B_free_noisy, B_load=B_load_noisy, y=y_all,
         y_columns=np.array(["s", "F_mag", "Fx", "Fy"]))
LOG("자기장 데이터셋(노이즈 없음) 저장 완료")

LOG("CNN(ContactEstimator) 학습 시작 (노이즈 없음)")
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

TARGETS2 = ["s", "F_mag", "Fx", "Fy"]
delta = B_load_noisy - B_free_noisy
Xc = delta.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
yc = y_all

Xc_mean, Xc_std = Xc.mean(), Xc.std()
Xc_norm = (Xc - Xc_mean) / Xc_std
yc_mean, yc_std = yc.mean(axis=0), yc.std(axis=0)
yc_norm = (yc - yc_mean) / yc_std

rng3 = np.random.default_rng(0)
idx = rng3.permutation(len(Xc_norm))
split = int(0.8 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

train_loader = DataLoader(
    TensorDataset(torch.tensor(Xc_norm[train_idx]).float(), torch.tensor(yc_norm[train_idx]).float()),
    batch_size=64, shuffle=True)
val_X = torch.tensor(Xc_norm[val_idx]).float()
val_y = torch.tensor(yc_norm[val_idx]).float()


class ContactEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, len(TARGETS2))
        )

    def forward(self, x):
        return self.regressor(self.features(x))


cnn_model = ContactEstimator()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
for epoch in range(200):
    cnn_model.train()
    batch_losses = []
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(cnn_model(bx), by)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    train_losses.append(np.mean(batch_losses))
    cnn_model.eval()
    with torch.no_grad():
        val_losses.append(criterion(cnn_model(val_X), val_y).item())
    if (epoch + 1) % 20 == 0:
        LOG(f"CNN Epoch [{epoch+1:3d}/200] Train {train_losses[-1]:.4f}  Val {val_losses[-1]:.4f}")

cnn_model.eval()
with torch.no_grad():
    pred_norm = cnn_model(val_X).numpy()
true_norm = val_y.numpy()
pred = pred_norm * yc_std + yc_mean
true = true_norm * yc_std + yc_mean

LOG("=== 최종 CNN 검증셋 오차 (노이즈 없음, 실제 단위) ===")
for i, t in enumerate(TARGETS2):
    mae = np.mean(np.abs(pred[:, i] - true[:, i]))
    r2 = r2_score(true[:, i], pred[:, i])
    unit = 'mm' if t == 's' else 'N'
    LOG(f"  {t}: MAE={mae:.5f} {unit}, R^2={r2:.3f} (타깃 표준편차={yc_std[i]:.5f})")

torch.save(cnn_model.state_dict(), os.path.join(MODELS_DIR, "fea_contact_estimator_nonoise.pth"))
LOG("=== 노이즈 없음 재학습 완료 ===")
