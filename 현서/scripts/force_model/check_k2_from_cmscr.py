"""CMSCR(LM/L=0)은 K2 하나로만 결정됨(K1,M1 무관). 각 phi 데이터점에서 K2를 독립적으로
역산해서, 서로 다른 phi에서 얼마나 일관된 K2가 나오는지 확인 -> 모델/디지털화 검증."""
import json
import numpy as np
import force_model as fm

with open("../../data/force_model/fig3_digitized.json", encoding="utf-8") as f:
    digitized = json.load(f)

pts = digitized["0.0"]
print(f"M2={fm.M2:.4e}, B={fm.B_FIELD}, L={fm.L}mm")
print(f"{'phi':>6} {'yL_cm':>8} {'thetaL_deg':>10} {'K2_역산':>12}")
for phi_str, (yL_cm, thL_deg, sz) in sorted(pts.items(), key=lambda kv: int(kv[0])):
    phi = int(phi_str)
    thL_rad = np.radians(thL_deg)
    if abs(thL_rad) < 1e-6:
        continue
    tau2 = fm.M2 * fm.B_FIELD * np.sin(np.radians(phi - thL_deg))
    # K2 * thetaL = -tau2 * (-L) => 논문 부호규약 확인용으로 두 부호 다 계산
    K2_a = tau2 * (fm.L / 1000.0) / thL_rad
    print(f"{phi:6d} {yL_cm:8.3f} {thL_deg:10.2f} {K2_a:12.4e}")
