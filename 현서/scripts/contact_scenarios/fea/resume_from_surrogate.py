"""
master_pipeline.py가 5만개 생성 도중(캐시 버그로) 죽어서, 이미 끝난 재시도/대체모델 학습은
다시 안 하고 저장된 대체모델(models/fea_surrogate_mlp.pth)을 불러와 생성 단계부터 재개.
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


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


ckpt = torch.load(os.path.join(MODELS_DIR, "fea_surrogate_mlp.pth"), weights_only=False)
FEATURES = ckpt["features"]
TARGETS = ckpt["targets"]
X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
surrogate = SurrogateMLP(len(FEATURES), len(TARGETS))
surrogate.load_state_dict(ckpt["state_dict"])
surrogate.eval()
LOG(f"대체모델 로드 완료: {FEATURES} -> {TARGETS}")


def predict_surrogate(L_M, phi, beta, s, depth):
    x = np.array([[L_M, phi, beta, s, depth]])
    xn = (x - X_mean) / X_std
    with torch.no_grad():
        pn = surrogate(torch.tensor(xn, dtype=torch.float32)).numpy()[0]
    p = pn * y_std + y_mean
    return dict(zip(TARGETS, p))


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


N_SAMPLES = 50000
rng = np.random.default_rng(42)
L_M_range = (0.0, 100.0)
phi_range = (-120.0, 120.0)
beta_range = (0.0, 360.0)
s_range = (10.0, 90.0)
depth_range = (0.02, 0.20)

LOG(f"{N_SAMPLES}개 합성 시나리오 생성 시작")
records = []
free_cache = {}
t_start = time.time()
for i in range(N_SAMPLES):
    L_M = rng.uniform(*L_M_range)
    phi = rng.uniform(*phi_range)
    beta = rng.uniform(*beta_range)
    s = rng.uniform(*s_range)
    depth = rng.uniform(*depth_range)

    key = (round(L_M, 1), round(phi, 1))
    if key in free_cache:
        r_free = free_cache[key]
    else:
        try:
            r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        except Exception:
            continue
        free_cache[key] = r_free
        if len(free_cache) > 5000:
            free_cache.clear()

    pred = predict_surrogate(L_M, phi, beta, s, depth)
    d_xL_local = pred["tip_uy_avg_mm"]
    d_yL_local = pred["tip_ux_avg_mm"]
    d_thL = -pred["tip_theta_deg_board"]

    frac = L_M / 100.0
    d_xLM_local = d_xL_local * frac
    d_yLM_local = d_yL_local * frac
    d_thLM = d_thL * frac

    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]

    xL_load = xL_free + d_xL_local
    yL_load = yL_free + d_yL_local
    thL_load = thL_free + d_thL
    xLM_load = xLM_free + d_xLM_local
    yLM_load = yLM_free + d_yLM_local
    thLM_load = thLM_free + d_thLM

    Fx, Fy, F_mag = pred["Fx_total_N"], pred["Fy_total_N"], pred["F_mag_N"]
    F_ang_deg = float(np.degrees(np.arctan2(Fy, Fx)))

    records.append((L_M, phi, s, F_mag, F_ang_deg, Fx, Fy,
                     xL_free, yL_free, thL_free, xLM_free, yLM_free, thLM_free,
                     xL_load, yL_load, thL_load, xLM_load, yLM_load, thLM_load))

    if (i + 1) % 5000 == 0:
        LOG(f"합성 시나리오 {i+1}/{N_SAMPLES} 생성됨 ({time.time()-t_start:.0f}s)")

columns = ["L_M", "phi_deg", "s", "F_mag", "F_ang_deg", "Fx", "Fy",
           "xL_free", "yL_free", "thL_free", "xLM_free", "yLM_free", "thLM_free",
           "xL_load", "yL_load", "thL_load", "xLM_load", "yLM_load", "thLM_load"]
data_arr = np.array(records)
np.savez(os.path.join(FEA_DATA_DIR, "fea_contact_force_scenarios.npz"),
         data=data_arr, columns=np.array(columns))
LOG(f"합성 시나리오 저장 완료: {len(records)}개 -> fea_contact_force_scenarios.npz")

LOG("자기장(B) 계산 시작")
col = {c: i for i, c in enumerate(columns)}
n = len(data_arr)
B_free_all = np.zeros((n, 25, 3))
B_load_all = np.zeros((n, 25, 3))
for i in range(n):
    row = data_arr[i]
    B_free_all[i] = compute_B(row[col["xLM_free"]], row[col["yLM_free"]], row[col["thLM_free"]],
                               row[col["xL_free"]], row[col["yL_free"]], row[col["thL_free"]])
    B_load_all[i] = compute_B(row[col["xLM_load"]], row[col["yLM_load"]], row[col["thLM_load"]],
                               row[col["xL_load"]], row[col["yL_load"]], row[col["thL_load"]])
    if (i + 1) % 5000 == 0:
        LOG(f"자기장 계산 {i+1}/{n}")

rng2 = np.random.default_rng(42)
def add_noise(B, frac=0.05):
    return B + rng2.normal(0, np.max(np.abs(B)) * frac, B.shape)

B_free_noisy = add_noise(B_free_all)
B_load_noisy = add_noise(B_load_all)

y_all = data_arr[:, [col["s"], col["F_mag"], col["Fx"], col["Fy"]]]
np.savez(os.path.join(FEA_DATA_DIR, "fea_contact_bfield_dataset.npz"),
         B_free=B_free_noisy, B_load=B_load_noisy, y=y_all,
         y_columns=np.array(["s", "F_mag", "Fx", "Fy"]))
LOG("자기장 데이터셋 저장 완료: fea_contact_bfield_dataset.npz")

# ── 최종 역방향 CNN(ContactEstimator) 학습 ────────────────────────────
LOG("최종 CNN(ContactEstimator) 학습 시작")
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

LOG("=== 최종 CNN 검증셋 오차 (실제 단위) ===")
for i, t in enumerate(TARGETS2):
    mae = np.mean(np.abs(pred[:, i] - true[:, i]))
    r2 = r2_score(true[:, i], pred[:, i])
    unit = 'mm' if t == 's' else 'N'
    LOG(f"  {t}: MAE={mae:.5f} {unit}, R^2={r2:.3f} (타깃 표준편차={yc_std[i]:.5f})")

torch.save(cnn_model.state_dict(), os.path.join(MODELS_DIR, "fea_contact_estimator.pth"))
np.savez(os.path.join(FEA_DATA_DIR, "fea_contact_train_history.npz"),
          train_losses=train_losses, val_losses=val_losses,
          val_pred=pred, val_true=true, targets=np.array(TARGETS2))
LOG(f"저장 완료: {MODELS_DIR}/fea_contact_estimator.pth")
LOG("=== 전체 파이프라인 완료 ===")
