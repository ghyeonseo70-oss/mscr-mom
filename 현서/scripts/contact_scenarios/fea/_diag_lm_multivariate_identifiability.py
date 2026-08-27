"""14번: L_M=0 다변량 식별가능성 직접 테스트.

11번 테스트는 phi/s를 고정한 "짝지어진" 비교라 L_M 하나만 봤을 때는 신호가 있음을
보여줬지만, 실제 학습에서는 phi/s가 L_M과 동시에 자유롭게 변함 - phi/s가 만드는 신호
변동이 L_M=0의 (원래도 약한) 신호를 덮어버려서 실제로는 구분이 안 될 수 있음.

방법: L_M=0 그룹과 L_M=12.5 그룹의 실측 delta-B(phi/s는 자연 분포 그대로, 짝짓지 않음)를
모아서, 이 두 그룹을 구분하는 간단한 분류기(로지스틱 회귀)를 k-fold 교차검증으로 학습.
정확도가 50%(우연)에 가까우면 다변량으로 확인 불가능하다는 뜻, 90%+ 근처면 신호는
있는데 최종 CNN이 못 배운다는 뜻(학습 문제로 좁혀짐). 비교군으로 "잘 되는" L_M=25 vs
37.5 쌍도 같이 확인해서 분류기 자체의 기준 성능을 가늠함."""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")
sys.path.insert(0, FORCE_MODEL_DIR)
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
main_magnet = magpy.magnet.Cylinder(polarization=(0, 0.4, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -0.4, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


def delta_B_real(r):
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    r_free = fm.solve_shape(L_M=max(L_M, 0.5), phi_deg=phi, loads=[])
    d_xL, d_yL = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    if "mom_ux_avg_mm" in r:
        d_xLM, d_yLM, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    else:
        frac = L_M / 100.0
        d_xLM, d_yLM, d_thLM = d_xL * frac, d_yL * frac, d_thL * frac
    xL, yL, thL = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM, yLM, thLM = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
    Bf = compute_B(xLM, yLM, thLM, xL, yL, thL)
    Bl = compute_B(xLM + d_xLM, yLM + d_yLM, thLM + d_thLM, xL + d_xL, yL + d_yL, thL + d_thL)
    return (Bl - Bf).flatten()


rows = json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8"))


def test_pair(lm_a, lm_b, label):
    rows_a = [r for r in rows if r["L_M_mm"] == lm_a and r["beta_deg"] == 0.0]
    rows_b = [r for r in rows if r["L_M_mm"] == lm_b and r["beta_deg"] == 0.0]
    X = np.array([delta_B_real(r) for r in rows_a + rows_b])
    y = np.array([0] * len(rows_a) + [1] * len(rows_b))
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(clf, Xn, y, cv=cv, scoring="accuracy")
    print(f"[{label}] n_a={len(rows_a)}, n_b={len(rows_b)} (phi/s 자연분포, 안 맞춤)")
    print(f"  5-fold 교차검증 정확도: {scores.mean()*100:.1f}% (±{scores.std()*100:.1f}%)  "
          f"(50%=우연, 100%=완벽구분)")
    return scores.mean()


print("=== 문제 구간: L_M=0 vs 12.5mm ===")
acc_problem = test_pair(0.0, 12.5, "L_M=0 vs 12.5")

print("\n=== 비교 기준(잘 되는 구간): L_M=25 vs 37.5mm ===")
acc_good1 = test_pair(25.0, 37.5, "L_M=25 vs 37.5")

print("\n=== 비교 기준2: L_M=50 vs 62.5mm ===")
acc_good2 = test_pair(50.0, 62.5, "L_M=50 vs 62.5")

print("\n=== 결론 ===")
print(f"문제구간(0 vs 12.5) 정확도={acc_problem*100:.1f}% vs 정상구간 평균={((acc_good1+acc_good2)/2)*100:.1f}%")
if acc_problem < 0.65:
    print("-> phi/s가 같이 변하는 실제 분포에서는 L_M=0이 다변량으로 헷갈림(가설 확인) - 최종 CNN이 학습을 못 하는 게 당연함")
else:
    print("-> phi/s가 같이 변해도 여전히 잘 구분됨 - 다변량 식별 문제는 아님, 학습/구조 문제로 좁혀짐")
