"""
force_model.py 검증 스크립트.

검증 1: 외력=0일 때, ODE+슈팅법 결과가 논문 원래 닫힌형 수식(7)-(8)을
        직접 fsolve로 푼 "기준 해"와 일치하는지 (독립적인 두 계산 경로 비교)
검증 2: 위치가 물리적으로 말이 되는지 (베이스-팁 직선거리 <= 전체 길이 L)
"""
import numpy as np
from scipy.optimize import fsolve
import force_model as fm


def reference_closed_form(L_M, phi_deg):
    """논문 식(7),(8)을 직접 2변수 fsolve로 풀어서 theta_LM, theta_L을 구하는 독립 계산."""
    phi = np.radians(phi_deg)
    l1 = L_M / 1000.0 - fm.H_M / 1000.0 / 2
    l2 = fm.L / 1000.0 - L_M / 1000.0 - fm.H_M / 1000.0 / 2

    def tau1(th_lm):
        return fm.M1 * fm.B_FIELD * np.sin(phi - th_lm - np.pi)

    def tau2(th_l):
        return fm.M2 * fm.B_FIELD * np.sin(phi - th_l)

    def equations(vars):
        th_lm, th_l = vars
        eq1 = fm.K1 * th_lm - (tau1(th_lm) + tau2(th_l)) * l1
        eq2 = fm.K2 * (th_l - th_lm) - tau2(th_l) * l2
        return [eq1, eq2]

    best = None
    for th0 in np.radians(np.arange(-170, 171, 20)):
        sol, info, ier, msg = fsolve(equations, [th0, th0], full_output=True)
        if ier == 1:
            th_l = sol[1]
            dist = abs(((th_l - phi + np.pi) % (2 * np.pi)) - np.pi)
            if best is None or dist < best[0]:
                best = (dist, sol)
    if best is None:
        raise RuntimeError("기준 해 못 찾음")
    return best[1]  # theta_LM, theta_L (rad)


def get_position_from_angles(L_M, theta_LM, theta_L):
    """theta_LM, theta_L이 주어졌을 때 논문 식(9)-(12)로 직접 위치 계산 (닫힌형)."""
    l1 = L_M - fm.H_M / 2
    l2 = fm.L - L_M - fm.H_M / 2
    t1, dt = theta_LM, theta_L - theta_LM
    R1 = l1 / (t1 if abs(t1) > 1e-9 else 1e-9)
    R2 = l2 / (dt if abs(dt) > 1e-9 else 1e-9)
    x_lm = R1 * np.sin(t1) + (fm.H_M / 2) * np.cos(t1)
    y_lm = R1 * (1 - np.cos(t1)) + (fm.H_M / 2) * np.sin(t1)
    x_l = x_lm + R2 * (np.sin(theta_L) - np.sin(t1)) + (fm.H_M / 2) * np.cos(t1)
    y_l = y_lm + R2 * (np.cos(t1) - np.cos(theta_L)) + (fm.H_M / 2) * np.sin(t1)
    return x_lm, y_lm, x_l, y_l


print("=" * 70)
print("검증 1: 외력=0, ODE+슈팅법 vs 논문 닫힌형 수식(7)-(9) 직접 계산")
print("=" * 70)
test_cases = [(30, 30), (50, 60), (70, -45), (20, 90), (80, 120)]
for L_M, phi_deg in test_cases:
    ode_result = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, s_t=50, Fx=0, Fy=0)

    th_lm_ref, th_l_ref = reference_closed_form(L_M, phi_deg)
    x_lm_ref, y_lm_ref, x_l_ref, y_l_ref = get_position_from_angles(L_M, th_lm_ref, th_l_ref)

    print(f"\nL_M={L_M}, phi={phi_deg}deg")
    print(f"  ODE법   : theta_LM={ode_result['theta_LM_deg']:7.2f}  theta_L={ode_result['theta_L_deg']:7.2f}  "
          f"(x_LM,y_LM)=({ode_result['x_LM']:6.1f},{ode_result['y_LM']:6.1f})  "
          f"(x_L,y_L)=({ode_result['x_L']:6.1f},{ode_result['y_L']:6.1f})")
    print(f"  닫힌형법 : theta_LM={np.degrees(th_lm_ref):7.2f}  theta_L={np.degrees(th_l_ref):7.2f}  "
          f"(x_LM,y_LM)=({x_lm_ref:6.1f},{y_lm_ref:6.1f})  (x_L,y_L)=({x_l_ref:6.1f},{y_l_ref:6.1f})")

    ang_diff_lm = abs(ode_result['theta_LM_deg'] - np.degrees(th_lm_ref))
    ang_diff_l = abs(ode_result['theta_L_deg'] - np.degrees(th_l_ref))
    pos_diff = np.hypot(ode_result['x_L'] - x_l_ref, ode_result['y_L'] - y_l_ref)
    status = "OK" if (ang_diff_lm < 1.0 and ang_diff_l < 1.0 and pos_diff < 1.0) else "MISMATCH"
    print(f"  차이: theta_LM {ang_diff_lm:.3f}deg, theta_L {ang_diff_l:.3f}deg, tip위치 {pos_diff:.3f}mm  -> {status}")

print("\n" + "=" * 70)
print("검증 2: 위치가 물리적으로 타당한지 (베이스-팁 직선거리 <= L=100mm)")
print("=" * 70)
for L_M, phi_deg in test_cases:
    r = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, s_t=50, Fx=0, Fy=0)
    dist = np.hypot(r['x_L'], r['y_L'])
    status = "OK" if dist <= 100.5 else "위반!"
    print(f"  L_M={L_M},phi={phi_deg}: base-tip 거리={dist:.1f}mm (<=100mm 여야 함) -> {status}")
