"""2026-09-22(37번): "자기장만 보고 충돌 세기(깊이)를 알 수 있는가?" 식별가능성 테스트.

배경: 깊이를 랜덤화하니 위치 정확도가 떨어졌고(s MAE 5.34->9.01mm), 원인을 "깊이가
관측 불가능한 숨은 변수라 위치/세기가 혼동된다"로 추정했음. 그런데 delta-B는 숫자 하나가
아니라 75개(센서 5x5 x 3축)라, 패턴까지 보면 원리상 구분 가능할 수도 있음.
- 구분 가능하다 -> 모호성 문제가 아니라 학습/데이터 문제. 깊이 보강이나 깊이 출력 추가로 해결.
- 구분 불가능하다 -> 정보가 애초에 없음. 범위를 좁히거나 고정하는 수밖에 없음.

34번에서 확인한 물리를 놓고 예측해보면: 변형의 "모양"은 깊이와 무관하고 깊이는 "크기"만
조절하므로, delta-B ~= (진폭) x (그 조합 고유의 고정 패턴) 형태일 것. 그렇다면
  - 패턴 모양  -> 접촉 위치(조합)를 알려주고
  - 크기       -> 진폭(=깊이 x 조합별 이득)을 알려줌
이 되어, 둘을 순서대로 풀면 분리 가능해야 함. 이 가설을 세 가지 입력으로 나눠 검증:
  (A) 전체 delta-B + known(L_M,phi)   : 패턴+크기 다 봄
  (B) 크기 제거한 패턴만 + known      : 패턴만 (가설대로면 깊이 정보 거의 없어야 함)
  (C) 크기(놈)만 + known              : 크기만 (위치와 혼동되어 중간 정도)

⚠️ 같은 (L_M,phi,s) 조합이 train/test에 나뉘어 들어가면 "이 조합의 이 크기=0.20mm"를
외워버릴 수 있으므로 **조합 단위 GroupKFold**로 분리함.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "force_model"))
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

sensors = magpy.Collection([magpy.Sensor(position=(x, y, 15))
                            for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)])
main_magnet = magpy.magnet.Cylinder(polarization=(0, 0.4, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -0.4, 0), dimension=(1, 8))
robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(robot, sensors) * 1e6


rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    rows.append(row)

dB, meta = [], []
for r in rows:
    try:
        rf = fm.solve_shape(L_M=r["L_M_mm"], phi_deg=r["phi_deg"], loads=[])
    except Exception:
        continue
    d_xL, d_yL, d_thL = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"], -r["tip_theta_deg_board"]
    if "mom_ux_avg_mm" in r:
        d_xLM, d_yLM, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    else:
        fr = r["L_M_mm"] / 100.0
        d_xLM, d_yLM, d_thLM = d_xL * fr, d_yL * fr, d_thL * fr
    B0 = compute_B(rf["x_LM"], rf["y_LM"], rf["theta_LM_deg"], rf["x_L"], rf["y_L"], rf["theta_L_deg"])
    B1 = compute_B(rf["x_LM"] + d_xLM, rf["y_LM"] + d_yLM, rf["theta_LM_deg"] + d_thLM,
                    rf["x_L"] + d_xL, rf["y_L"] + d_yL, rf["theta_L_deg"] + d_thL)
    dB.append((B1 - B0).ravel())
    meta.append(r)

dB = np.array(dB, dtype=np.float64)
depth = np.array([round(r.get("push_depth_mm", 0.1), 3) for r in meta])
known = np.array([[r["L_M_mm"], r["phi_deg"]] for r in meta], dtype=np.float64)
groups = np.array([f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}" for r in meta])

norm = np.linalg.norm(dB, axis=1, keepdims=True)
norm[norm < 1e-12] = 1.0
INPUTS = [
    ("(A) 전체 delta-B + known", np.hstack([dB, known])),
    ("(B) 크기 제거한 패턴만 + known", np.hstack([dB / norm, known])),
    ("(C) 크기(놈)만 + known", np.hstack([norm, known])),
    # ⚠️ 필수 대조군: 깊이 스윕이 특정 (L_M,phi) 조합만 골라 돌렸기 때문에, 분류기가
    # 자기장이 아니라 "이 조합은 스윕 대상이었다"는 표본 설계를 외워서 맞힐 수 있음.
    # 자기장을 아예 안 주고 known만 줬을 때의 점수가 그 '부정행위 상한선'이고,
    # 위 (A)~(C)는 이 값보다 유의미하게 높아야 진짜 신호를 쓴 것임.
    ("(D) [대조군] known만, 자기장 없음", known),
]

classes, counts = np.unique(depth, return_counts=True)
# sklearn 분류기는 연속값 라벨을 거부하므로 깊이를 클래스 인덱스로 변환
y = np.searchsorted(classes, depth)
chance = 1.0 / len(classes)
print(f"데이터 {len(depth)}행, 깊이 클래스 {dict(zip(classes, counts))}")
print(f"조합(그룹) 수 {len(np.unique(groups))} - 같은 조합은 train/test에 안 걸치게 GroupKFold")
print(f"무작위 기준선(balanced) = {chance*100:.1f}%\n")

for label, Xin in INPUTS:
    preds = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(Xin, y, groups):
        clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced", n_jobs=-1)
        clf.fit(Xin[tr], y[tr])
        preds[te] = clf.predict(Xin[te])
    bal = balanced_accuracy_score(y, preds)
    print(f"{label:32s} balanced acc = {bal*100:5.1f}%")
    cm = confusion_matrix(y, preds, labels=np.arange(len(classes)))
    for c, rowc in zip(classes, cm):
        print(f"      실제 {c:.2f}mm -> 예측 " + "  ".join(f"{cls:.2f}:{n:3d}" for cls, n in zip(classes, rowc)))
    print()

print("해석: (A)가 높으면 자기장에 깊이 정보가 있다는 뜻 -> 모호성이 아니라 학습/데이터 문제.")
print("      (B)가 낮고 (C)가 중간이면 34번 예측대로 '패턴=위치, 크기=세기' 구조라는 뜻.")
