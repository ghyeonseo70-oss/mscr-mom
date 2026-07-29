"""K1, K2, M1/M2 비율(자유도 3개)로 확장 피팅. M1=M2 가정이 안 맞을 수 있어서 추가.
여러 시작점(multi-start)으로 국소최적해를 피함."""
import json
import numpy as np
from scipy.optimize import fsolve, least_squares
import force_model as fm

with open("../../data/force_model/fig3_digitized.json", encoding="utf-8") as f:
    digitized = json.load(f)

targets = []
for ratio_str, points in digitized.items():
    ratio = float(ratio_str)
    L_M = max(ratio * fm.L, 0.5)
    for phi_str, (yL_cm, thL_deg, sz) in points.items():
        phi = int(phi_str)
        targets.append((L_M, phi, yL_cm * 10, thL_deg))

M2_FIXED = fm.M2  # M2는 고정(스케일 기준), M1만 비율로 자유화


def closed_form(L_M, phi_deg, K1, K2, M1, hint_deg):
    phi = np.radians(phi_deg)
    l1 = L_M / 1000.0 - fm.H_M / 1000.0 / 2
    l2 = fm.L / 1000.0 - L_M / 1000.0 - fm.H_M / 1000.0 / 2

    def eqs(vars):
        th_lm, th_l = vars
        tau1 = M1 * fm.B_FIELD * np.sin(phi - th_lm - np.pi)
        tau2 = M2_FIXED * fm.B_FIELD * np.sin(phi - th_l)
        return [K1 * th_lm - (tau1 + tau2) * l1, K2 * (th_l - th_lm) - tau2 * l2]

    hint = np.radians(hint_deg)
    sol, info, ier, msg = fsolve(eqs, [hint * 0.5, hint], full_output=True)
    if ier != 1 or abs(sol[0]) > np.pi or abs(sol[1]) > 2 * np.pi:
        return None
    return sol


def position(L_M, theta_LM, theta_L):
    l1 = L_M - fm.H_M / 2
    l2 = fm.L - L_M - fm.H_M / 2
    t1, dt = theta_LM, theta_L - theta_LM
    R1 = l1 / (t1 if abs(t1) > 1e-9 else 1e-9)
    R2 = l2 / (dt if abs(dt) > 1e-9 else 1e-9)
    x_lm = R1 * np.sin(t1) + (fm.H_M / 2) * np.cos(t1)
    y_lm = R1 * (1 - np.cos(t1)) + (fm.H_M / 2) * np.sin(t1)
    x_l = x_lm + R2 * (np.sin(theta_L) - np.sin(t1)) + (fm.H_M / 2) * np.cos(t1)
    y_l = y_lm + R2 * (np.cos(t1) - np.cos(theta_L)) + (fm.H_M / 2) * np.sin(t1)
    return x_l, y_l


def residuals(params):
    logK1, logK2, logMratio = params
    K1, K2 = np.exp(logK1), np.exp(logK2)
    M1 = np.exp(logMratio) * M2_FIXED
    res = []
    for L_M, phi, yL_t, thL_t in targets:
        sol = closed_form(L_M, phi, K1, K2, M1, thL_t)
        if sol is None:
            res += [8.0, 8.0]
            continue
        th_lm, th_l = sol
        _, y_l = position(L_M, th_lm, th_l)
        res.append((y_l - yL_t) / 10.0)
        res.append((np.degrees(th_l) - thL_t) / 30.0)
    return np.array(res)


bounds_lo = [np.log(1e-8), np.log(1e-8), np.log(0.05)]
bounds_hi = [np.log(1e-3), np.log(1e-3), np.log(20.0)]

best = None
starts = [
    (np.log(3.8257e-06), np.log(9.8875e-07), np.log(1.290)),  # 이전 최적점 근처 정밀화
    (np.log(2e-6), np.log(1e-6), np.log(1.2)),
    (np.log(5e-6), np.log(8e-7), np.log(1.4)),
    (np.log(3e-6), np.log(1.2e-6), np.log(1.1)),
    (np.log(4e-6), np.log(9e-7), np.log(1.5)),
    (np.log(1e-6), np.log(1e-6), np.log(1.0)),
]
for i, x0 in enumerate(starts):
    result = least_squares(residuals, x0, method="trf", bounds=(bounds_lo, bounds_hi),
                            xtol=1e-10, ftol=1e-10, max_nfev=500)
    rms = np.sqrt(np.mean(result.fun ** 2))
    K1, K2, Mr = np.exp(result.x)
    print(f"start{i}: K1={K1:.3e} K2={K2:.3e} M1/M2={Mr:.3f}  rms={rms:.4f}", flush=True)
    if best is None or rms < best[0]:
        best = (rms, K1, K2, Mr)

print(f"\n최적: rms={best[0]:.4f}  K1={best[1]:.4e}  K2={best[2]:.4e}  K1/K2={best[1]/best[2]:.3f}  M1/M2={best[3]:.3f}")

with open("../../data/force_model/k1k2_fit_result.json", "w") as f:
    json.dump({"K1": best[1], "K2": best[2], "K1_K2_ratio": best[1] / best[2],
               "M1_over_M2": best[3], "rms_residual": best[0]}, f, indent=2)
print("저장: ../../data/force_model/k1k2_fit_result.json")
