"""force_model.py의 정밀 엘라스티카(점하중) 해가 실측 FEA 변위와 얼마나 맞는지 확인.
서로게이트(MLP, 145개로 학습)가 변위(ux,uy) 예측을 못하는 게 병목이었는데, force_model.py는
데이터 학습이 필요 없는 닫힌 물리모델이라 - 이게 실측과 잘 맞으면 서로게이트를 변위 예측에서
아예 빼고 힘(Fx,Fy)만 예측하게 해서 문제의 근원을 없앨 수 있음. 36개 홀드아웃 전부에서
실측 Fx,Fy를 그대로 force_model.solve_shape의 하중으로 넣어보고 변위를 비교."""
import json
import os
import sys

import numpy as np
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")
sys.path.insert(0, FORCE_MODEL_DIR)
import force_model as fm

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)

# train_segment_classifier_singleprobe_beta0180_4seg.py와 동일한 홀드아웃(시드42) - 36개
rng_holdout = np.random.default_rng(42)
perm = rng_holdout.permutation(len(all_rows))
n_holdout = max(20, int(len(all_rows) * 0.2))
holdout_idx = perm[:n_holdout]
holdout_rows = [all_rows[i] for i in holdout_idx]

real_ux, real_uy, real_theta = [], [], []
fm_ux, fm_uy, fm_theta = [], [], []
n_fail = 0
for r in holdout_rows:
    L_M, phi, s = r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]
    beta = r["beta_deg"]
    Fx_local, Fy_local = r["Fx_total_N"], r["Fy_total_N"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        r_load = fm.solve_shape(L_M=L_M, phi_deg=phi,
                                 loads=[{"type": "point", "s": s, "Fx": Fx_local, "Fy": Fy_local}],
                                 theta_L_hint_deg=r_free["theta_L_deg"])
    except Exception as e:
        n_fail += 1
        continue
    fm_ux.append(r_load["x_L"] - r_free["x_L"])
    fm_uy.append(r_load["y_L"] - r_free["y_L"])
    fm_theta.append(r_load["theta_L_deg"] - r_free["theta_L_deg"])
    real_ux.append(r["tip_ux_avg_mm"])
    real_uy.append(r["tip_uy_avg_mm"])
    real_theta.append(r["tip_theta_deg_board"])

real_ux, real_uy, real_theta = np.array(real_ux), np.array(real_uy), np.array(real_theta)
fm_ux, fm_uy, fm_theta = np.array(fm_ux), np.array(fm_uy), np.array(fm_theta)

print(f"성공 {len(real_ux)}/{len(holdout_rows)} (실패 {n_fail}개)")
print(f"\n=== force_model.py(정밀 엘라스티카, 실측 Fx,Fy를 하중으로 사용) vs 실측 FEA 변위 ===")
print(f"tip_ux: R^2={r2_score(real_ux, fm_ux):.3f}  MAE={np.mean(np.abs(real_ux-fm_ux)):.3f}mm")
print(f"tip_uy: R^2={r2_score(real_uy, fm_uy):.3f}  MAE={np.mean(np.abs(real_uy-fm_uy)):.3f}mm")
print(f"tip_theta: R^2={r2_score(real_theta, fm_theta):.3f}  MAE={np.mean(np.abs(real_theta-fm_theta)):.3f}deg")
print(f"\n(참고: 서로게이트 단독 홀드아웃 R^2는 tip_ux=0.521, tip_uy=0.676, tip_theta=0.984였음)")
