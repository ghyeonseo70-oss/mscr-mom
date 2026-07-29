"""MOM 패널(LM/L=0.25,0.5,0.75)에서 theta_LM, theta_L을 모두 안다고 가정하고(디지털화값),
식(7)(8)을 각각 독립적으로 뒤집어 K1, K2를 데이터점별로 역산 -> 일관성 확인."""
import json
import numpy as np
import force_model as fm

with open("../../data/force_model/fig3_digitized.json", encoding="utf-8") as f:
    digitized = json.load(f)

print(f"M1={fm.M1:.4e}, M2={fm.M2:.4e}, B={fm.B_FIELD}\n")

for ratio_str in ["0.25", "0.5", "0.75"]:
    ratio = float(ratio_str)
    L_M = ratio * fm.L
    l1 = (L_M - fm.H_M / 2) / 1000.0
    l2 = (fm.L - L_M - fm.H_M / 2) / 1000.0
    pts = digitized[ratio_str]
    print(f"=== LM/L={ratio}  (l1={l1*1000:.1f}mm, l2={l2*1000:.1f}mm) ===")
    print(f"{'phi':>6} {'th_LM':>8} {'th_L':>8} {'K2_역산(eq8)':>14} {'K1_역산(eq7)':>14}")
    for phi_str, (yL_cm, thL_deg, sz) in sorted(pts.items(), key=lambda kv: int(kv[0])):
        phi = int(phi_str)
        # theta_LM은 직접 안 주어짐 -> yL,thetaL로부터 역산해야 하지만 일단 근사: 두 미지수 중
        # theta_L은 알고, theta_LM은 모름. 그러나 eq8은 theta_LM도 필요.
        # 여기서는 yL(위치)까지 이용해 theta_LM을 별도로 구해야 하므로 스킵하고,
        # 대신 eq8만으로는 안 되니 이 스크립트에서는 위치식(9)-(12)까지 fsolve로 theta_LM 역산.
        from scipy.optimize import fsolve
        thL_rad = np.radians(thL_deg)

        def eqs(th_lm):
            th_lm = th_lm[0]
            dt = thL_rad - th_lm
            R1 = l1 / (th_lm if abs(th_lm) > 1e-9 else 1e-9)
            R2 = l2 / (dt if abs(dt) > 1e-9 else 1e-9)
            y_lm = R1 * (1 - np.cos(th_lm)) + (fm.H_M / 2000) * np.sin(th_lm)
            y_l = y_lm + R2 * (np.cos(th_lm) - np.cos(thL_rad)) + (fm.H_M / 2000) * np.sin(th_lm)
            return [y_l * 1000 - yL_cm * 10]  # mm

        sol = fsolve(eqs, [thL_rad * 0.3], full_output=False)
        th_lm = sol[0]

        tau1 = fm.M1 * fm.B_FIELD * np.sin(np.radians(phi) - th_lm - np.pi)
        tau2 = fm.M2 * fm.B_FIELD * np.sin(np.radians(phi) - thL_rad)
        # 부호 뒤집어서 테스트: CMSCR 검증에서 음수 없는 쪽이 물리적으로 맞았음
        K1_est = (tau1 + tau2) * l1 / th_lm if abs(th_lm) > 1e-6 else float("nan")
        dt = thL_rad - th_lm
        K2_est = tau2 * l2 / dt if abs(dt) > 1e-6 else float("nan")
        print(f"{phi:6d} {np.degrees(th_lm):8.2f} {thL_deg:8.2f} {K2_est:14.4e} {K1_est:14.4e}")
    print()
