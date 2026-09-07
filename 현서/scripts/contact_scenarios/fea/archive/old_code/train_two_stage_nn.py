"""2단계 NN 파이프라인 (PROJECT_STATUS.md에 기록된, 예전에 가장 잘 됐던 접근 - 이상적 상한선
R^2=0.87 - 을 지금의 FEA 기반 노이즈 없는 데이터로 재현).

NN1 (CNN): delta-B(5x5x3, 노이즈 없음) -> 팁/MOM 위치·각도 변화량 6개
           (d_xL, d_yL, d_thL, d_xLM, d_yLM, d_thLM) - test_shape_to_contact_nn.py가
           "이상적(노이즈 없는 위치추정 가정)"으로 썼던 것과 같은 서술자.
NN2 (MLP): NN1이 추정한 6개 값 + L_M(알고 있는 값) -> 접촉정보(s, F_mag, Fx, Fy)
           (ShapeToContactNet과 동일 구조)

두 단계를 각각 학습한 뒤, "NN2에 진짜(ground truth) 6개 값을 넣었을 때"(이상적 상한선)와
"NN1이 예측한 6개 값을 넣었을 때"(실제 엔드투엔드) 둘 다 평가해서 NN1의 오차가 얼마나
전파되는지 확인한다.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
LOG = lambda msg: print(msg, flush=True)

cache = np.load(os.path.join(FEA_DATA_DIR, "fea_contact_force_scenarios.npz"), allow_pickle=True)
data_arr = cache["data"]
columns = list(cache["columns"])
col = {c: i for i, c in enumerate(columns)}

bfield = np.load(os.path.join(FEA_DATA_DIR, "fea_contact_bfield_dataset_nonoise.npz"))
B_free, B_load = bfield["B_free"], bfield["B_load"]
n = len(data_arr)
LOG(f"n={n}")

# ── NN1 타깃: 팁/MOM 위치·각도 변화량 6개 (test_shape_to_contact_nn.py와 동일한 서술자) ──
shift = np.stack([
    data_arr[:, col["xL_load"]] - data_arr[:, col["xL_free"]],
    data_arr[:, col["yL_load"]] - data_arr[:, col["yL_free"]],
    data_arr[:, col["thL_load"]] - data_arr[:, col["thL_free"]],
    data_arr[:, col["xLM_load"]] - data_arr[:, col["xLM_free"]],
    data_arr[:, col["yLM_load"]] - data_arr[:, col["yLM_free"]],
    data_arr[:, col["thLM_load"]] - data_arr[:, col["thLM_free"]],
], axis=1)
SHIFT_NAMES = ["d_xL", "d_yL", "d_thL", "d_xLM", "d_yLM", "d_thLM"]

L_M = data_arr[:, col["L_M"]]
y_contact = data_arr[:, [col["s"], col["F_mag"], col["Fx"], col["Fy"]]]
CONTACT_NAMES = ["s", "F_mag", "Fx", "Fy"]

delta_B = B_load - B_free
Xc = delta_B.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)

# ── 공통 train/val 분할 (NN1/NN2 같은 분할 써야 누수 없이 엔드투엔드 평가 가능) ──
rng = np.random.default_rng(0)
idx = rng.permutation(n)
split = int(0.8 * n)
train_idx, val_idx = idx[:split], idx[split:]


def normalize(a, mean=None, std=None):
    if mean is None:
        mean, std = a.mean(0), a.std(0)
        std = np.where(std < 1e-9, 1.0, std)
    return (a - mean) / std, mean, std


Xc_mean, Xc_std = Xc.mean(), Xc.std()
Xc_norm = (Xc - Xc_mean) / Xc_std
shift_norm, shift_mean, shift_std = normalize(shift)

# ══════════════════════════════════════════════════════════════════════
# NN1: CNN, delta-B -> 6개 위치/각도 변화량
# ══════════════════════════════════════════════════════════════════════
class PositionShiftCNN(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.regressor(self.features(x))


nn1 = PositionShiftCNN(len(SHIFT_NAMES))
opt1 = optim.Adam(nn1.parameters(), lr=0.001)
crit = nn.MSELoss()

train_loader1 = DataLoader(
    TensorDataset(torch.tensor(Xc_norm[train_idx]).float(), torch.tensor(shift_norm[train_idx]).float()),
    batch_size=64, shuffle=True)
val_X1 = torch.tensor(Xc_norm[val_idx]).float()
val_y1 = torch.tensor(shift_norm[val_idx]).float()

LOG("=== NN1 (CNN: deltaB -> 위치/각도 변화량 6개) 학습 시작 ===")
for epoch in range(200):
    nn1.train()
    for bx, by in train_loader1:
        opt1.zero_grad()
        loss = crit(nn1(bx), by)
        loss.backward()
        opt1.step()
    if (epoch + 1) % 40 == 0:
        nn1.eval()
        with torch.no_grad():
            vloss = crit(nn1(val_X1), val_y1).item()
        LOG(f"  NN1 Epoch [{epoch+1:3d}/200] Val Loss {vloss:.4f}")

nn1.eval()
with torch.no_grad():
    shift_pred_val_norm = nn1(val_X1).numpy()
    shift_pred_train_norm = nn1(torch.tensor(Xc_norm[train_idx]).float()).numpy()
shift_pred_val = shift_pred_val_norm * shift_std + shift_mean
shift_true_val = shift[val_idx]

LOG("\n[NN1 결과 - 위치/각도 변화량 예측 정확도]")
for i, name in enumerate(SHIFT_NAMES):
    r2 = r2_score(shift_true_val[:, i], shift_pred_val[:, i])
    mae = np.mean(np.abs(shift_pred_val[:, i] - shift_true_val[:, i]))
    LOG(f"  {name}: R2={r2:.3f}  MAE={mae:.5f}")

# ══════════════════════════════════════════════════════════════════════
# NN2: MLP, (위치/각도 변화량 6개 + L_M) -> 접촉정보(s,F_mag,Fx,Fy)
# ══════════════════════════════════════════════════════════════════════
X2_true = np.concatenate([shift, L_M[:, None]], axis=1)
X2_mean, X2_std = X2_true.mean(0), X2_true.std(0)
X2_std = np.where(X2_std < 1e-9, 1.0, X2_std)
X2_true_norm = (X2_true - X2_mean) / X2_std

y2_mean, y2_std = y_contact.mean(0), y_contact.std(0)
y2_norm = (y_contact - y2_mean) / y2_std

train_loader2 = DataLoader(
    TensorDataset(torch.tensor(X2_true_norm[train_idx]).float(), torch.tensor(y2_norm[train_idx]).float()),
    batch_size=64, shuffle=True)
val_X2_true = torch.tensor(X2_true_norm[val_idx]).float()
val_y2 = torch.tensor(y2_norm[val_idx]).float()


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


nn2 = ShapeToContactNet(X2_true.shape[1], len(CONTACT_NAMES))
opt2 = optim.Adam(nn2.parameters(), lr=0.001)

LOG("\n=== NN2 (MLP: 위치/각도변화량(진짜값)+L_M -> 접촉정보) 학습 시작 [이상적 상한선] ===")
for epoch in range(200):
    nn2.train()
    for bx, by in train_loader2:
        opt2.zero_grad()
        loss = crit(nn2(bx), by)
        loss.backward()
        opt2.step()
    if (epoch + 1) % 40 == 0:
        nn2.eval()
        with torch.no_grad():
            vloss = crit(nn2(val_X2_true), val_y2).item()
        LOG(f"  NN2 Epoch [{epoch+1:3d}/200] Val Loss {vloss:.4f}")

nn2.eval()
with torch.no_grad():
    pred_ideal_norm = nn2(val_X2_true).numpy()
pred_ideal = pred_ideal_norm * y2_std + y2_mean
true_contact = y_contact[val_idx]

LOG("\n[NN2 결과 - 이상적 상한선(진짜 위치/각도변화량 사용)]")
for i, name in enumerate(CONTACT_NAMES):
    r2 = r2_score(true_contact[:, i], pred_ideal[:, i])
    mae = np.mean(np.abs(pred_ideal[:, i] - true_contact[:, i]))
    unit = "mm" if name == "s" else "N"
    LOG(f"  {name}: R2={r2:.3f}  MAE={mae:.5f}{unit}")

# ── 엔드투엔드: NN1이 예측한 값을 NN2에 넣기 (실제 배포 시나리오) ──
X2_e2e_val = np.concatenate([shift_pred_val, L_M[val_idx, None]], axis=1)
X2_e2e_val_norm = (X2_e2e_val - X2_mean) / X2_std
with torch.no_grad():
    pred_e2e_norm = nn2(torch.tensor(X2_e2e_val_norm).float()).numpy()
pred_e2e = pred_e2e_norm * y2_std + y2_mean

LOG("\n[엔드투엔드 결과 - NN1 예측값을 NN2에 그대로 넣음 (실제 배포 시나리오)]")
for i, name in enumerate(CONTACT_NAMES):
    r2 = r2_score(true_contact[:, i], pred_e2e[:, i])
    mae = np.mean(np.abs(pred_e2e[:, i] - true_contact[:, i]))
    unit = "mm" if name == "s" else "N"
    LOG(f"  {name}: R2={r2:.3f}  MAE={mae:.5f}{unit}")

torch.save(nn1.state_dict(), os.path.join(MODELS_DIR, "fea_nn1_position_shift.pth"))
torch.save(nn2.state_dict(), os.path.join(MODELS_DIR, "fea_nn2_shape_to_contact.pth"))
np.savez(os.path.join(FEA_DATA_DIR, "fea_two_stage_train_history.npz"),
         pred_ideal=pred_ideal, pred_e2e=pred_e2e, true_contact=true_contact,
         contact_names=np.array(CONTACT_NAMES))
LOG("\n저장 완료: fea_nn1_position_shift.pth, fea_nn2_shape_to_contact.pth")
