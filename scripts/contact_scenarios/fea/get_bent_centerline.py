"""force_model.py로 특정 구동상태(L_M, phi)의 굽은 중심선 좌표를 뽑아서
Gmsh 스윕(파이프)용 스플라인 포인트 목록으로 저장.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm

# 대표 구동상태 (기존 force_model.py __main__ 예시와 동일한 값 사용)
L_M = 50.0
PHI_DEG = 60.0
N_SPLINE_POINTS = 40  # 곡선을 스플라인으로 근사할 때 쓸 점 개수(너무 많으면 gmsh가 느려짐)

r = fm.solve_shape(L_M=L_M, phi_deg=PHI_DEG, loads=[], return_curve=True)

import numpy as np

# force_model의 shoot()은 팁(s=L)->베이스(s=0) 방향으로 적분해서 curve_s_mm이 내림차순으로
# 나옴. np.interp는 오름차순 x좌표를 요구하는데 이걸 그냥 넣으면 에러 없이 조용히 엉뚱한 값을
# 반환해버려서(실제로 겪음: 첫 점이 팁 좌표로 나옴) 오름차순으로 정렬해서 씀.
order = np.argsort(r["curve_s_mm"])
s = r["curve_s_mm"][order]
x = r["curve_x_mm"][order]
y = r["curve_y_mm"][order]
theta = r["curve_theta_deg"][order]

# s(호길이) 기준으로 균등 간격 N_SPLINE_POINTS개를 뽑음(끝점 포함)
s_target = np.linspace(s.min(), s.max(), N_SPLINE_POINTS)
x_i = np.interp(s_target, s, x)
y_i = np.interp(s_target, s, y)
theta_i = np.interp(s_target, s, theta)  # 접선각도(deg) - 참고용(로컬 좌표계 기준)

# ── 홀센서 보드(하드웨어) 좌표계로 변환 ──────────────────────────────
# force_model.py의 to_board_frame과 동일한 규칙(베이스=(90,0), 전역 y=자라나는 방향)에
# 높이 z=3mm를 더함. 그러면 카테터 베이스(s=0)가 정확히 (90,0,3)에 위치하게 됨.
# 로컬(x,y) 평면에서의 굽힘이 그대로 보드의 수평면(x,y)에서 일어나고, z는 보드 위 높이로
# 일정하게 유지된다고 가정.
BOARD_BASE_X = 90.0
BOARD_Z = 3.0
x_board = BOARD_BASE_X + y_i
y_board = x_i
z_board = np.full_like(x_i, BOARD_Z)

points = [{"s": float(si), "x": float(xb), "y": float(yb), "z": float(zb), "theta_deg": float(ti)}
          for si, xb, yb, zb, ti in zip(s_target, x_board, y_board, z_board, theta_i)]

# 팁/MOM 위치도 같은 보드좌표로 변환 (points와 일관되게)
x_LM_board = BOARD_BASE_X + r["y_LM"]
y_LM_board = r["x_LM"]
x_L_board = BOARD_BASE_X + r["y_L"]
y_L_board = r["x_L"]

print(f"L_M={L_M}, phi={PHI_DEG}deg")
print(f"베이스(보드좌표): ({BOARD_BASE_X}, 0, {BOARD_Z})")
print(f"팁 위치(보드좌표): x_L={x_L_board:.2f}, y_L={y_L_board:.2f}, z=3.00, theta_L={r['theta_L_deg']:.2f}deg")
print(f"MOM 위치(보드좌표): x_LM={x_LM_board:.2f}, y_LM={y_LM_board:.2f}, z=3.00, theta_LM={r['theta_LM_deg']:.2f}deg")
print(f"스플라인 점 {len(points)}개")

out_path = os.path.join(HERE, "bent_centerline.json")
with open(out_path, "w") as f:
    json.dump({"L_M": L_M, "phi_deg": PHI_DEG, "points": points,
               "board_base_x": BOARD_BASE_X, "board_z": BOARD_Z,
               "x_LM": x_LM_board, "y_LM": y_LM_board, "theta_LM_deg": r["theta_LM_deg"],
               "x_L": x_L_board, "y_L": y_L_board, "theta_L_deg": r["theta_L_deg"]}, f, indent=2)
print(f"저장: {out_path}")
