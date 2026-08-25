"""가설 검증: phi=90~150 s격자 조밀화(114개 추가, 404->518) 이후 Fx_board(=로컬Fy) 실측
홀드아웃 R^2가 0.729->0.462로 떨어진 게 (a) 모델이 진짜 더 나빠져서인지, 아니면 (b) 홀드아웃
구성 자체가 바뀌어서인지(실측 홀드아웃은 rng(42).permutation(len(all_rows))로 뽑는데,
all_rows 길이가 404->518로 바뀌면 같은 시드라도 뽑히는 행 자체가 달라짐 - 이번 홀드아웃(n=103)에
phi=90~150 신규 데이터가 얼마나 섞여 들어갔는지, 그 신규 데이터만 유독 나쁜지를 직접 확인.

방법: 현재(518개) 데이터로 train_segment_classifier_singleprobe_beta0180_4seg.py와 완전히
동일한 로직(rng_holdout seed=42)으로 real_holdout_rows(n=103)를 재구성한 뒤, 저장된 최신
체크포인트로 예측해서 Fx_board R^2를 (1) 전체 (2) 기존 404개에 있던 행만 (3) 이번에 새로
추가된 114개 중 홀드아웃에 뽑힌 행만, 그리고 phi<90 vs phi>=90으로 나눠서 비교."""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")
sys.path.insert(0, FORCE_MODEL_DIR)
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_CLASSES = 4


def row_key(r):
    return (r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"])


# ---- 현재(518개) all_rows + 옛날(404개, sdensify 스윕 직전) 행 목록 로드 ----
all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)
print(f"현재 전체 데이터: {len(all_rows)}개")

OLD_JSON = os.path.join(HERE, "_diag_old_404_snapshot.json")
old_keys = set()
if os.path.exists(OLD_JSON):
    for r in json.load(open(OLD_JSON, encoding="utf-8")):
        row = dict(DEFAULTS)
        row.update(r)
        old_keys.add(row_key(row))
    print(f"이전(sdensify 전) 스냅샷: {len(old_keys)}개")
else:
    print("경고: 이전 스냅샷 파일 없음 - old/new 구분 없이 phi 구간별로만 분석함")

# ---- 학습 스크립트와 완전히 동일한 real_holdout 재구성 ----
# 2026-08-25: train_segment_classifier_singleprobe_beta0180_4seg.py와 동일한 해시기반
# 고정 홀드아웃으로 교체함(이 스크립트가 찾아낸 문제의 수정판 - 상세 이유는 그 파일 참고)
def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (h % 10000) < int(frac * 10000)


holdout_idx = [i for i, r in enumerate(all_rows) if is_holdout_row(r)]
real_holdout_rows = [all_rows[i] for i in holdout_idx]
print(f"실측 홀드아웃 재구성: n={len(real_holdout_rows)}")

n_new_in_holdout = sum(1 for r in real_holdout_rows if row_key(r) not in old_keys)
n_old_in_holdout = len(real_holdout_rows) - n_new_in_holdout
n_high_phi = sum(1 for r in real_holdout_rows if abs(r["phi_deg"]) >= 90)
n_low_phi = len(real_holdout_rows) - n_high_phi
print(f"  - 이전 스냅샷에 없던(신규) 행: {n_new_in_holdout}개, 기존 행: {n_old_in_holdout}개")
print(f"  - |phi|>=90: {n_high_phi}개, |phi|<90: {n_low_phi}개")

# ---- 모델 로드 (518개로 재학습된 최신 체크포인트) ----
ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                   map_location="cpu", weights_only=False)


class SingleProbeClassifier(nn.Module):
    def __init__(self, n_probes=1, n_classes=N_CLASSES, n_force=2, n_config=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        self.config_head = nn.Linear(128, n_config)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


cnn = SingleProbeClassifier()
cnn.load_state_dict(ckpt["state_dict"])
cnn.eval()
X_mean2, X_std2 = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]
force_names = ckpt["force_names"]
print("force_names:", force_names)

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.4
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


real_X, real_f, meta = [], [], []
for r in real_holdout_rows:
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    except Exception:
        continue
    d_xL_local, d_yL_local = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    frac = L_M / 100.0
    d_xLM_local, d_yLM_local, d_thLM = d_xL_local * frac, d_yL_local * frac, d_thL * frac
    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
    B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                        xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
    real_X.append((B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1))
    real_f.append([r["Fy_total_N"], r["Fx_total_N"]])
    meta.append({"is_new": row_key(r) not in old_keys, "high_phi": abs(phi) >= 90, "phi": phi, "L_M": L_M})

real_X = np.array(real_X, dtype=np.float32)
real_f_arr = np.array(real_f, dtype=np.float32)
real_X_norm = (real_X - X_mean2) / X_std2

with torch.no_grad():
    rX = torch.tensor(real_X_norm[:, None]).float()
    _, r_force_pred, _, _ = cnn(rX)
    r_force_phys = r_force_pred.numpy() * f_std + f_mean

print(f"\nfree-shape 계산 성공: {len(real_X)}/{len(real_holdout_rows)}")


def r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def report(mask, label):
    n = mask.sum()
    if n < 3:
        print(f"  [{label}] n={n} - 너무 적어서 R^2 생략")
        return
    for i, name in enumerate(force_names):
        print(f"  [{label}] n={n}  {name}: R^2={r2(r_force_phys[mask, i], real_f_arr[mask, i]):.3f}")


is_new = np.array([m["is_new"] for m in meta])
high_phi = np.array([m["high_phi"] for m in meta])

print("\n=== 전체 ===")
report(np.ones(len(meta), dtype=bool), "전체")
print("\n=== 신규(sdensify) 행 vs 기존 행 ===")
report(~is_new, "기존행")
report(is_new, "신규행(phi=90~150 s조밀화)")
print("\n=== phi 구간별 (신규/기존 안 가리고) ===")
report(~high_phi, "|phi|<90")
report(high_phi, "|phi|>=90")
print("\n=== 교차: 기존행 중 phi 구간별 (신규 데이터 자체가 원인인지, 기존행마저 나빠졌는지 확인) ===")
report(~is_new & ~high_phi, "기존행 & |phi|<90")
report(~is_new & high_phi, "기존행 & |phi|>=90 (원래도 홀드아웃에 있었을 수 있는 고각도 기존행)")
report(is_new & high_phi, "신규행 & |phi|>=90 (전부 새 phi=90~150 데이터)")
