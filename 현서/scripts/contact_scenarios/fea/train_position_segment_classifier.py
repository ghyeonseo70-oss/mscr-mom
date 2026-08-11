"""
교수님 코멘트: 정확한 접촉위치(s)를 회귀로 구할 필요 없이, 곡선을 몇 구간으로 나눠서
"어느 구간에 접촉했는지"만 잘 맞히면 됨.

159개(LM x phi x s 그리드)만으로는 클래스당 샘플이 너무 적어서(최대 33, 최소 3개) 검증
정확도가 22%로 낮게 나왔음 -> 같은 스키마(tip 변위/회전 포함)로 이미 완료된 이전 FEA
스윕들(fea_bent_contact_sweep.json, fea_geom_sweep_all.json, fea_angle_sweep_all.json)도
전부 합쳐서 샘플 수를 늘리고(총 500개대), s를 정확히 매칭하는 대신 20mm 폭 구간 5개로
좀 더 거칠게 나눠서(교수님이 원하는 수준에 더 가까움) 재학습한다.

파이프라인(master_pipeline.py와 동일한 방식 재사용):
1) FEA 결과(L_M,phi,s -> tip 변위/회전)로 무외력(free)/접촉시(load) 자석 위치를 만들고
   magpylib으로 5x5 홀센서 보드가 볼 자기장(B) 계산
2) delta = B_load - B_free 를 입력으로, contact_s_mm이 속한 구간 인덱스를 타깃으로
   train_contact_model.py의 ContactEstimator(CNN)를 회귀 대신 분류로 학습
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm  # noqa: E402
import magpylib as magpy  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios")
FEA_DATA_DIR = os.path.join(DATA_DIR, "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")

BIN_WIDTH_MM = 20.0
N_CLASSES = 5  # [0,20) [20,40) [40,60) [60,80) [80,100] mm


def s_to_bin(s_mm):
    return min(N_CLASSES - 1, int(s_mm / BIN_WIDTH_MM))

# ── 센서/자석 모델 (generate_contact_bfield.py, master_pipeline.py와 동일) ─────────
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
    mom.orientation = Rotation.from_euler("z", -thLM_deg, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL_deg, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6  # T -> uT


def add_noise(B, rng, frac=0.05):
    return B + rng.normal(0, np.max(np.abs(B)) * frac, B.shape)


# ── 1) FEA 결과 로드(여러 스윕 병합) + free/load 자석 위치 -> B 계산 ────────────
SOURCES = ["fea_lm_phi_pos_sweep_all.json", "fea_bent_contact_sweep.json",
           "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0}

rows = []
for fname in SOURCES:
    path = os.path.join(FEA_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    for r in json.load(open(path)):
        row = dict(DEFAULTS)
        row.update(r)
        rows.append(row)
print(f"입력 FEA 케이스(병합): {len(rows)}개 ({', '.join(SOURCES)})")

rng = np.random.default_rng(0)
free_cache = {}
B_delta_all = []
labels = []
skipped = 0
for r in rows:
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    s_mm = r["contact_s_mm"]
    label = s_to_bin(s_mm)

    key = (round(L_M, 1), round(phi, 1))
    if key not in free_cache:
        try:
            free_cache[key] = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        except Exception as e:
            print(f"  free-shape 실패 (L_M={L_M}, phi={phi}): {e}")
            continue
    r_free = free_cache[key]

    # master_pipeline.py와 동일한 보드<->로컬 축 교환 변환 (make_bent_contact_scene.py 주석 참고)
    d_xL_local = r["tip_uy_avg_mm"]
    d_yL_local = r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    frac = L_M / 100.0
    d_xLM_local = d_xL_local * frac
    d_yLM_local = d_yL_local * frac
    d_thLM = d_thL * frac

    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]

    B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    B_load = compute_B(
        xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
        xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL,
    )
    B_free = add_noise(B_free, rng)
    B_load = add_noise(B_load, rng)
    B_delta_all.append((B_load - B_free).reshape(5, 5, 3))
    labels.append(label)

print(f"B-field 계산 완료: {len(labels)}개 (스킵 {skipped}개)")
if len(labels) < 30:
    print("데이터가 너무 적어서 분류기 학습을 건너뜁니다.")
    sys.exit(1)

X = np.array(B_delta_all).transpose(0, 3, 1, 2)  # (n, 3, 5, 5)
y = np.array(labels, dtype=np.int64)
print("구간별 샘플 수:", {int(c): int((y == c).sum()) for c in range(N_CLASSES)})

X_mean, X_std = X.mean(), X.std()
X_norm = (X - X_mean) / X_std


# ── 2) 분류기 (train_contact_model.py의 ContactEstimator를 회귀->분류로 변경, 500개
#     규모 데이터에 맞춰 채널/파라미터 수를 줄이고 정규화를 강하게 함) ─────────────
class SegmentClassifier(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_fold(train_idx, val_idx, epochs=150):
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y[train_idx]).long()),
        batch_size=32, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float()
    val_y = torch.tensor(y[val_idx]).long()

    model = SegmentClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    # 구간별 샘플 수가 60/224/152/109/27로 불균형이라(모델이 다수 구간만 찍는 문제 확인됨),
    # 클래스별 가중치(적은 구간일수록 손실에 더 크게 반영)로 보정
    class_counts = np.bincount(y[train_idx], minlength=N_CLASSES)
    class_weights = torch.tensor(len(train_idx) / (N_CLASSES * np.maximum(class_counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc, best_val_loss, best_pred = 0.0, float("inf"), None
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y).item()
            val_pred = val_logits.argmax(dim=1)
        if val_loss < best_val_loss:  # val loss 기준 조기종료(과적합 방지)
            best_val_loss = val_loss
            best_val_acc = (val_pred == val_y).float().mean().item()
            best_pred = val_pred
    return best_val_acc, best_pred, val_y


from sklearn.model_selection import KFold  # noqa: E402

kf = KFold(n_splits=5, shuffle=True, random_state=0)
fold_accs = []
conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.int32)
print(f"\n5-fold 교차검증 시작 (n={len(X_norm)}, {N_CLASSES}구간, 무작위 기준선={100/N_CLASSES:.1f}%)...")
for k, (tr_idx, va_idx) in enumerate(kf.split(X_norm)):
    acc, pred, true = run_fold(tr_idx, va_idx)
    fold_accs.append(acc)
    for t, p in zip(true.tolist(), pred.tolist()):
        conf[t, p] += 1
    print(f"  fold {k+1}: ValAcc {acc*100:5.1f}% (n_val={len(va_idx)})")

fold_accs = np.array(fold_accs)
print(f"\n=== 5-fold 평균 검증 정확도: {fold_accs.mean()*100:.1f}% (±{fold_accs.std()*100:.1f}%p), "
      f"무작위 추정 기준선={100/N_CLASSES:.1f}% ===")

adjacent_ok = sum(conf[i, max(0, i-1):i+2].sum().item() for i in range(N_CLASSES))
print(f"±1구간 오차 허용 시 정확도: {adjacent_ok / conf.sum().item() * 100:.1f}%")

bin_labels = [f"{int(i*BIN_WIDTH_MM)}-{int((i+1)*BIN_WIDTH_MM)}mm" for i in range(N_CLASSES)]
print(f"\n혼동행렬(5-fold 합산, 행=실제 구간, 열=예측 구간, {bin_labels}):")
print(conf.numpy())

# 불균형 때문에 전체 정확도만으로는 "다수 구간만 찍어도 높게 나오는" 착시가 생길 수 있어서,
# 구간별 recall을 평균한 balanced accuracy도 같이 확인(모든 구간을 고르게 잘 맞히는지)
per_class_recall = [conf[i, i].item() / max(1, conf[i].sum().item()) for i in range(N_CLASSES)]
print("\n구간별 recall:", {bin_labels[i]: f"{per_class_recall[i]*100:.1f}%" for i in range(N_CLASSES)})
print(f"balanced accuracy(구간별 recall 평균): {np.mean(per_class_recall)*100:.1f}%")
