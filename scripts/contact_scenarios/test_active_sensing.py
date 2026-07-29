"""
능동 탐색(active sensing) 아이디어 검증: 자석/센서를 늘릴 수 없으니, 대신 로봇 구동상태
(L_M, phi)를 이것저것 바꿔가며 여러 스냅샷을 보면 "헷갈리는 쌍"이 구별되는지 확인.

find_confusable_pair.py가 L_M=50,phi=60 에서 찾은 헷갈리는 쌍:
  s=10mm, F=16mN  vs  s=30mm, F=2mN  (그 상태에서는 결과 형상이 거의 동일)
여기서는 이 "물리적 접촉 조건"(로컬 좌표계 기준 s와 Fx,Fy는 고정)은 그대로 두고, 로봇을
다른 L_M,phi로 움직였을 때도 여전히 두 형상이 구별 안 되는지, 아니면 특정 구동상태에서는
뚜렷이 달라지는지를 본다. force_model만 쓰므로 FEA 없이 빠르게 확인 가능.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "force_model"))
import force_model as fm

# ── 원래 헷갈리는 쌍을 찾았던 상태에서, 그때 쓴 힘 벡터(로컬 Fx,Fy)를 재현 ──────
L_M0, PHI0 = 50.0, 60.0
r_free0 = fm.solve_shape(L_M=L_M0, phi_deg=PHI0, loads=[], return_curve=True)
order = np.argsort(r_free0["curve_s_mm"])
s_arr0 = r_free0["curve_s_mm"][order]
th_arr0 = r_free0["curve_theta_deg"][order]


def normal_at_s0(s):
    th = np.radians(np.interp(s, s_arr0, th_arr0))
    return np.array([np.sin(th), -np.cos(th)])


S_A, F_A = 10.0, 0.016   # 16 mN
S_B, F_B = 30.0, 0.002   # 2 mN
nA, nB = normal_at_s0(S_A), normal_at_s0(S_B)
FxA, FyA = F_A * nA[0], F_A * nA[1]
FxB, FyB = F_B * nB[0], F_B * nB[1]
print(f"고정된 물리적 힘벡터(로컬프레임): A(s={S_A})=({FxA:.5f},{FyA:.5f}), "
      f"B(s={S_B})=({FxB:.5f},{FyB:.5f})")

# ── 원래 상태에서 재확인 ──────────────────────────────
rA0 = fm.solve_shape(L_M=L_M0, phi_deg=PHI0, loads=[{"type": "point", "s": S_A, "Fx": FxA, "Fy": FyA}],
                      theta_L_hint_deg=r_free0["theta_L_deg"])
rB0 = fm.solve_shape(L_M=L_M0, phi_deg=PHI0, loads=[{"type": "point", "s": S_B, "Fx": FxB, "Fy": FyB}],
                      theta_L_hint_deg=r_free0["theta_L_deg"])
d0 = np.hypot(rA0["x_L"] - rB0["x_L"], rA0["y_L"] - rB0["y_L"])
print(f"\n[기준상태 L_M={L_M0},phi={PHI0}] 팁 위치 차이: {d0:.4f}mm (헷갈리는 정도)")

# ── 다른 구동상태들로 로봇을 움직였을 때 같은 두 물리적 힘이 여전히 헷갈리는지 ──────
print("\n다른 구동상태에서 재확인:")
print(f"{'L_M':>6} {'phi':>6} | {'팁위치차이(mm)':>14} | {'구별가능?':>8}")
test_states = [(50, 60), (30, 60), (70, 60), (50, 0), (50, -60), (50, 120),
               (20, 30), (80, 90), (35, -30), (65, 100)]
results = []
for L_M, phi in test_states:
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], return_curve=True)
        rA = fm.solve_shape(L_M=L_M, phi_deg=phi,
                             loads=[{"type": "point", "s": S_A, "Fx": FxA, "Fy": FyA}],
                             theta_L_hint_deg=r_free["theta_L_deg"])
        rB = fm.solve_shape(L_M=L_M, phi_deg=phi,
                             loads=[{"type": "point", "s": S_B, "Fx": FxB, "Fy": FyB}],
                             theta_L_hint_deg=r_free["theta_L_deg"])
        d = np.hypot(rA["x_L"] - rB["x_L"], rA["y_L"] - rB["y_L"])
        distinguishable = "O" if d > 3.0 else ("애매" if d > 1.0 else "X")
        print(f"{L_M:6.0f} {phi:6.0f} | {d:14.4f} | {distinguishable:>8}")
        results.append({"L_M": L_M, "phi": phi, "tip_diff_mm": float(d)})
    except RuntimeError as e:
        print(f"{L_M:6.0f} {phi:6.0f} | 계산 실패: {e}")

import json
with open(os.path.join(HERE, "active_sensing_test.json"), "w") as f:
    json.dump({"base_state": {"L_M": L_M0, "phi": PHI0, "tip_diff_mm": float(d0)},
                "fixed_loads": {"A": {"s": S_A, "F_mN": F_A*1000}, "B": {"s": S_B, "F_mN": F_B*1000}},
                "other_states": results}, f, indent=2)
print(f"\n저장: {os.path.join(HERE, 'active_sensing_test.json')}")
print("\n판정기준: 3mm 넘게 벌어지면 노이즈보다 확실히 큰 차이라 구별 가능(기준: B-필드 노이즈로 인한 "
      "위치추정 오차가 대략 mm 단위인 점 감안).")
