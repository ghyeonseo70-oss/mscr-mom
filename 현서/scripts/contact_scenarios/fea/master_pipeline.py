"""
전체 파이프라인 무인 실행 스크립트.
1) 4단계(각도 보강, 이미 실행 중인 프로세스) 끝날 때까지 대기
2) 1~2단계에서 실패했던 케이스(failed_cases.json)를 동시성 낮춰서 재시도
3) 전체 FEA 결과 병합
4) 대체모델(MLP, 팁 변위/회전 + 힘까지 8개 타깃) 재학습
5) 그 대체모델로 5만개 합성 시나리오 생성 -> magpylib 자기장 계산 (기존
   generate_contact_bfield.py와 동일한 센서/자석 모델 재사용)
6) 최종 역방향 CNN(ContactEstimator, train_contact_model.py와 동일 구조) 학습

중간에 개별 케이스가 실패해도 전체가 멈추지 않도록 각 단계에서 예외를 잡아 넘어간다.
"""
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios")
FEA_DATA_DIR = os.path.join(DATA_DIR, "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")

STAGE4_PID = 3619042
LOG = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_pid(pid, poll=30):
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(poll)


# ── 1) 4단계 완료 대기 ──────────────────────────────────────────────
LOG("4단계(각도 보강) 완료 대기 시작")
wait_pid(STAGE4_PID)
LOG("4단계 완료 확인됨")

# ── 2) 실패 케이스 재시도 (동시성 6, 스레드 10) ──────────────────────
LOG("실패 케이스 재시도 시작")
failed = json.load(open(os.path.join(HERE, "failed_cases.json")))

s1_by_s = defaultdict(list)
for s, d in failed["stage1"]:
    s1_by_s[s].append(d)

s2_by_tag = defaultdict(lambda: {"L_M": None, "phi": None, "pairs": []})
for lm, phi, tag, s, d in failed["stage2"]:
    s2_by_tag[tag]["L_M"] = lm
    s2_by_tag[tag]["phi"] = phi
    s2_by_tag[tag]["pairs"].append((s, d))

retry_cmds = []
for s, depths in s1_by_s.items():
    depths_str = ",".join(str(d) for d in depths)
    retry_cmds.append([sys.executable, "-u", "retry_stage1_worker.py",
                        "--s", str(s), "--depths", depths_str, "--threads", "10"])
for tag, info in s2_by_tag.items():
    pairs_str = ",".join(f"{s}:{d}" for s, d in info["pairs"])
    retry_cmds.append([sys.executable, "-u", "retry_stage2_worker.py",
                        "--L_M", str(info["L_M"]), "--phi", str(info["phi"]),
                        "--tag", tag, "--pairs", pairs_str, "--threads", "10"])

LOG(f"재시도 작업 {len(retry_cmds)}개 (동시성 6)")
os.makedirs(os.path.join(HERE, "retry_logs"), exist_ok=True)
running = []
idx = 0
while idx < len(retry_cmds) or running:
    while len(running) < 6 and idx < len(retry_cmds):
        cmd = retry_cmds[idx]
        tag = cmd[cmd.index("--tag") + 1] if "--tag" in cmd else f"s{cmd[cmd.index('--s')+1]}"
        logf = open(os.path.join(HERE, "retry_logs", f"retry_{tag}.log"), "w")
        p = subprocess.Popen(cmd, cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
        running.append((p, logf))
        idx += 1
    time.sleep(15)
    still = []
    for p, logf in running:
        if p.poll() is None:
            still.append((p, logf))
        else:
            logf.close()
    running = still
    if idx % 5 == 0:
        LOG(f"재시도 진행: {idx}/{len(retry_cmds)} 작업 시작됨, {len(running)}개 실행중")

LOG("실패 케이스 재시도 전체 완료")

# ── 3) 전체 FEA 결과 병합 ────────────────────────────────────────────
def merge(pattern, sort_key, out_name):
    rows = []
    for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, pattern))):
        rows.extend(json.load(open(f)))
    rows.sort(key=sort_key)
    out = os.path.join(FEA_DATA_DIR, out_name)
    json.dump(rows, open(out, "w"), indent=2, ensure_ascii=False)
    return rows

rows1 = merge("fea_bent_contact_sweep_s*.json", lambda r: (r["contact_s_mm"], r["push_depth_mm"]),
              "fea_bent_contact_sweep.json")
rows2 = merge("fea_geom_sweep_*.json", lambda r: (r["L_M_mm"], r["phi_deg"], r["contact_s_mm"], r["push_depth_mm"]),
              "fea_geom_sweep_all.json")
rows3 = merge("fea_angle_sweep_beta*.json", lambda r: (r["beta_deg"], r["contact_s_mm"], r["push_depth_mm"]),
              "fea_angle_sweep_all.json")
LOG(f"최종 병합: 1단계 {len(rows1)}, 2단계 {len(rows2)}, 3~4단계(각도) {len(rows3)}")

# ── 4) 대체모델 재학습 (팁 변위/회전 + 힘까지 8개 타깃) ────────────────
LOG("대체모델(MLP) 재학습 시작")
sys.path.insert(0, HERE)
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

all_rows = []
for fname in ["fea_bent_contact_sweep.json", "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]:
    for r in json.load(open(os.path.join(FEA_DATA_DIR, fname))):
        row = dict(DEFAULTS)
        row.update(r)
        all_rows.append(row)
LOG(f"대체모델 학습 데이터: {len(all_rows)}개")

X = np.array([[r[f] for f in FEATURES] for r in all_rows])
y = np.array([[r[t] for t in TARGETS] for r in all_rows])
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std < 1e-9] = 1.0
y_mean, y_std = y.mean(axis=0), y.std(axis=0)
y_std[y_std < 1e-9] = 1.0
Xn = (X - X_mean) / X_std
yn = (y - y_mean) / y_std


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


def train_mlp(X_tr, y_tr, X_val, epochs=1000, lr=1e-3, weight_decay=1e-3):
    model = SurrogateMLP(X_tr.shape[1], y_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    return pred, model


kf = KFold(n_splits=5, shuffle=True, random_state=0)
preds = np.zeros_like(yn)
for tr_idx, val_idx in kf.split(Xn):
    pred, _ = train_mlp(Xn[tr_idx], yn[tr_idx], Xn[val_idx])
    preds[val_idx] = pred

LOG("=== 최종 대체모델 5-fold R^2 ===")
for i, t in enumerate(TARGETS):
    LOG(f"  {t}: R^2={r2_score(yn[:, i], preds[:, i]):.3f}")

_, final_model = train_mlp(Xn, yn, Xn)
os.makedirs(MODELS_DIR, exist_ok=True)
torch.save({"state_dict": final_model.state_dict(), "X_mean": X_mean, "X_std": X_std,
            "y_mean": y_mean, "y_std": y_std, "features": FEATURES, "targets": TARGETS},
           os.path.join(MODELS_DIR, "fea_surrogate_mlp.pth"))
LOG(f"저장: {MODELS_DIR}/fea_surrogate_mlp.pth")

# ── 5) 5만개 합성 시나리오 생성 + magpylib 자기장 계산 ─────────────────
LOG("5만개 합성 데이터 생성 시작")
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


# 대체모델 로드해서 예측 함수로 사용
surrogate = SurrogateMLP(len(FEATURES), len(TARGETS))
surrogate.load_state_dict(final_model.state_dict())
surrogate.eval()


def predict_surrogate(L_M, phi, beta, s, depth):
    x = np.array([[L_M, phi, beta, s, depth]])
    xn = (x - X_mean) / X_std
    with torch.no_grad():
        pn = surrogate(torch.tensor(xn, dtype=torch.float32)).numpy()[0]
    p = pn * y_std + y_mean
    return dict(zip(TARGETS, p))


N_SAMPLES = 50000
rng = np.random.default_rng(42)
L_M_range = (0.0, 100.0)
phi_range = (-120.0, 120.0)
beta_range = (0.0, 360.0)
s_range = (10.0, 90.0)
depth_range = (0.02, 0.20)

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
    # 보드좌표계(FEA 계산 결과) -> 로컬좌표계 변환: to_board_frame이 (x_b=BASE+y_l, y_b=x_l)인
    # 좌표축 교환이라, 델타도 dx_local=dy_board, dy_local=dx_board로 맞바꿔야 함
    # (make_bent_contact_scene.py의 로컬->보드 접선 변환 주석 참고). 각도는 보드각=90-로컬각
    # 관계라 델타 부호가 반대로 뒤집힘.
    d_xL_local = pred["tip_uy_avg_mm"]
    d_yL_local = pred["tip_ux_avg_mm"]
    d_thL = -pred["tip_theta_deg_board"]

    # MOM(L_M) 지점 변위는 FEA로 직접 측정 안 했음(N_TIP만 측정) - 팁 변위를 s_LM/L 비율로
    # 선형 스케일링해서 근사 (연속 탄성체라 급격한 불연속은 없을 거라는 가정, 검증 안 된 근사임 -
    # 나중에 L_M 위치에도 별도 절점그룹 추가해서 직접 측정하는 게 더 정확함).
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

# ── magpylib B_free/B_load 계산 ──────────────────────────────────────
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

# ── 6) 최종 역방향 CNN(ContactEstimator) 학습 ─────────────────────────
LOG("최종 CNN(ContactEstimator) 학습 시작")
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
