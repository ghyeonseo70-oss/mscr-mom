"""
빔이론 단순모델(force_model.py) 기준으로, 같은 구동상태(L_M=50,phi=60 - bent_centerline.json과
동일)에서 접촉위치 s는 많이 다른데 결과 팁 위치는 거의 똑같이 나오는 "헷갈리는 쌍"을 찾는다.
이 쌍을 실제 FEA로 각각 돌려서(run_bent_tip_test.py), 진짜 연속탄성체도 이 둘을 구별 못 하는지
(=근본적 물리한계) 아니면 단순모델만 정보를 버린 건지(=FEA 데이터가 실제로 도움됨) 확인하기 위함.

힘 방향은 FEA에서 실제로 미는 방향(그 위치의 접선에 수직, 바깥쪽)과 맞춰야 비교 의미가 있어서,
접촉위치 s에서의 로컬 접선각도로부터 법선방향을 계산해서 그 방향으로만 민다(수직 하중).
"""
import os
import sys
import numpy as np
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

L_M, PHI_DEG = 50.0, 60.0
S_GRID = np.arange(10.0, 91.0, 10.0)
F_GRID = np.array([0.002, 0.005, 0.008, 0.012, 0.016, 0.02])

r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG, loads=[], return_curve=True)
order = np.argsort(r_free["curve_s_mm"])
s_arr = r_free["curve_s_mm"][order]
th_arr = r_free["curve_theta_deg"][order]


def normal_at_s(s):
    """make_bent_contact_scene._point_and_normal_at_s와 같은 로컬->보드 변환 논리로 유도한
    로컬프레임 법선(바깥쪽, FEA가 미는 방향)."""
    th = np.radians(np.interp(s, s_arr, th_arr))
    return np.array([np.sin(th), -np.cos(th)])


rows = []
for s in S_GRID:
    n = normal_at_s(s)
    for F in F_GRID:
        Fx, Fy = F * n[0], F * n[1]
        try:
            r = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG,
                                loads=[{"type": "point", "s": s, "Fx": Fx, "Fy": Fy}],
                                theta_L_hint_deg=r_free["theta_L_deg"])
        except RuntimeError:
            continue
        rows.append({"s": s, "F": F, "x_L": r["x_L"], "y_L": r["y_L"], "theta_L_deg": r["theta_L_deg"],
                     "x_LM": r["x_LM"], "y_LM": r["y_LM"], "theta_LM_deg": r["theta_LM_deg"]})

print(f"{len(rows)}개 (s,F) 조합 계산 완료")

# 6D 기술자(descriptor) 정규화 후 최근접(단, s가 20mm 이상 차이나는 쌍만) 탐색
desc = np.array([[r["x_L"], r["y_L"], r["theta_L_deg"], r["x_LM"], r["y_LM"], r["theta_LM_deg"]] for r in rows])
desc_n = (desc - desc.mean(0)) / desc.std(0)

best = None
for i, j in itertools.combinations(range(len(rows)), 2):
    if abs(rows[i]["s"] - rows[j]["s"]) < 20:
        continue
    d = np.linalg.norm(desc_n[i] - desc_n[j])
    if best is None or d < best[0]:
        best = (d, i, j)

d, i, j = best
print(f"\n가장 헷갈리는 쌍 (정규화거리={d:.4f}):")
for k in (i, j):
    r = rows[k]
    print(f"  s={r['s']:.0f}mm, F={r['F']*1000:.1f}mN -> "
          f"x_L={r['x_L']:.3f}, y_L={r['y_L']:.3f}, th_L={r['theta_L_deg']:.3f}, "
          f"x_LM={r['x_LM']:.3f}, y_LM={r['y_LM']:.3f}, th_LM={r['theta_LM_deg']:.3f}")

xL_diff = abs(rows[i]["x_L"] - rows[j]["x_L"])
yL_diff = abs(rows[i]["y_L"] - rows[j]["y_L"])
tip_dist = np.hypot(xL_diff, yL_diff)
print(f"\n팁 위치 차이(단순모델 기준): {tip_dist:.4f}mm  (이게 FEA에서는 얼마나 다르게 나오는지가 관건)")

import json
out = {"L_M": L_M, "phi_deg": PHI_DEG,
       "pair": [rows[i], rows[j]], "normalized_distance": float(d), "tip_dist_mm": float(tip_dist)}
with open(os.path.join(HERE, "confusable_pair.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\n저장: {os.path.join(HERE, 'confusable_pair.json')}")
