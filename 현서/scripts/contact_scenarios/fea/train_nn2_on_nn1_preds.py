"""NN2를 "진짜 정답" 대신 "NN1이 실제로 예측한 값"으로 재학습해서 exposure bias를 줄여본다.
(train_two_stage_nn.py가 저장해둔 NN1 가중치를 재사용, NN1은 다시 학습 안 함 - 몇 분이면 끝남)

가설: NN2가 훈련 내내 오차 없는 입력만 보다가 실전(NN1의 살짝 틀린 예측)을 받으면 낯설어해서
성능이 떨어졌다(엔드투엔드 s R2=0.257, 이상적상한선 R2=0.682). NN2를 처음부터 NN1의 실제
예측값(=NN1의 진짜 오차 패턴이 실린 값)으로 학습시키면 이 문제가 줄어드는지 확인.
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
Xc_norm = (Xc - Xc.mean()) / Xc.std()
shift_mean, shift_std = shift.mean(0), shift.std(0)
shift_std = np.where(shift_std < 1e-9, 1.0, shift_std)
shift_norm_gt = (shift - shift_mean) / shift_std

rng = np.random.default_rng(0)  # train_two_stage_nn.py와 동일한 분할
idx = rng.permutation(n)
split = int(0.8 * n)
train_idx, val_idx = idx[:split], idx[split:]


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


LOG("NN1(이미 학습됨) 로드해서 train/val 양쪽에 실제 예측값 뽑기...")
nn1 = PositionShiftCNN(len(SHIFT_NAMES))
nn1.load_state_dict(torch.load(os.path.join(MODELS_DIR, "fea_nn1_position_shift.pth")))
nn1.eval()
with torch.no_grad():
    shift_pred_train_norm = nn1(torch.tensor(Xc_norm[train_idx]).float()).numpy()
    shift_pred_val_norm = nn1(torch.tensor(Xc_norm[val_idx]).float()).numpy()
shift_pred_train = shift_pred_train_norm * shift_std + shift_mean
shift_pred_val = shift_pred_val_norm * shift_std + shift_mean

for i, name in enumerate(SHIFT_NAMES):
    r2 = r2_score(shift[train_idx, i], shift_pred_train[:, i])
    LOG(f"  (참고) NN1 train셋 자체 R2 {name}: {r2:.3f}")

# ── NN2를 "NN1의 실제 예측값"으로 학습 (exposure bias 완화 시도) ──────────────
X2_train_pred = np.concatenate([shift_pred_train, L_M[train_idx, None]], axis=1)
X2_val_pred = np.concatenate([shift_pred_val, L_M[val_idx, None]], axis=1)

X2_mean, X2_std = X2_train_pred.mean(0), X2_train_pred.std(0)
X2_std = np.where(X2_std < 1e-9, 1.0, X2_std)
X2_train_norm = (X2_train_pred - X2_mean) / X2_std
X2_val_norm = (X2_val_pred - X2_mean) / X2_std

y2_mean, y2_std = y_contact[train_idx].mean(0), y_contact[train_idx].std(0)
y2_train_norm = (y_contact[train_idx] - y2_mean) / y2_std
y2_val_norm = (y_contact[val_idx] - y2_mean) / y2_std

train_loader = DataLoader(
    TensorDataset(torch.tensor(X2_train_norm).float(), torch.tensor(y2_train_norm).float()),
    batch_size=64, shuffle=True)
val_X2 = torch.tensor(X2_val_norm).float()
val_y2 = torch.tensor(y2_val_norm).float()


class ShapeToContactNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, len(CONTACT_NAMES)),
        )

    def forward(self, x):
        return self.net(x)


nn2v2 = ShapeToContactNet(X2_train_pred.shape[1], len(CONTACT_NAMES))
opt = optim.Adam(nn2v2.parameters(), lr=0.001)
crit = nn.MSELoss()

LOG("\n=== NN2 (NN1의 실제 예측값으로 학습) 시작 ===")
for epoch in range(200):
    nn2v2.train()
    for bx, by in train_loader:
        opt.zero_grad()
        loss = crit(nn2v2(bx), by)
        loss.backward()
        opt.step()
    if (epoch + 1) % 40 == 0:
        nn2v2.eval()
        with torch.no_grad():
            vloss = crit(nn2v2(val_X2), val_y2).item()
        LOG(f"  Epoch [{epoch+1:3d}/200] Val Loss {vloss:.4f}")

nn2v2.eval()
with torch.no_grad():
    pred_norm = nn2v2(val_X2).numpy()
pred = pred_norm * y2_std + y2_mean
true = y_contact[val_idx]

LOG("\n[결과 - NN2를 NN1 실제 예측값으로 재학습한 엔드투엔드]")
for i, name in enumerate(CONTACT_NAMES):
    r2 = r2_score(true[:, i], pred[:, i])
    mae = np.mean(np.abs(pred[:, i] - true[:, i]))
    unit = "mm" if name == "s" else "N"
    LOG(f"  {name}: R2={r2:.3f}  MAE={mae:.5f}{unit}")

LOG("\n(비교: 기존 방식 - NN2를 정답으로 학습 - 의 엔드투엔드: s=0.257, F_mag=0.615, Fx=0.864, Fy=0.883)")
LOG("(비교: direct CNN(노이즈 없음): s=0.417, F_mag=0.686, Fx=0.898, Fy=0.905)")

torch.save(nn2v2.state_dict(), os.path.join(MODELS_DIR, "fea_nn2_trained_on_nn1_preds.pth"))
np.savez(os.path.join(FEA_DATA_DIR, "fea_nn2_on_nn1_preds_history.npz"),
         pred=pred, true=true, targets=np.array(CONTACT_NAMES))
LOG("저장 완료")
