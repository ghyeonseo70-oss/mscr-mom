import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

with open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8") as f:
    rows = json.load(f)

success = set()
for r in rows:
    success.add((r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"]))

LM_LIST_FULL = [0.0, 25.0, 50.0, 75.0]
PHI_LIST_FULL = [0.0, 30.0, -30.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0]
S_LIST = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

attempted = []
for lm in LM_LIST_FULL:
    for phi in PHI_LIST_FULL:
        for s in S_LIST:
            attempted.append((lm, phi, 0.0, s))

failed = [t for t in attempted if t not in success]
print("총 시도(beta=0만):", len(attempted), " 성공:", len(attempted) - len(failed), " 실패:", len(failed))

fail_by_phi = Counter(t[1] for t in failed)
attempt_by_phi = Counter(t[1] for t in attempted)
print()
for phi in PHI_LIST_FULL:
    print(f"phi={phi:>6}: 실패 {fail_by_phi[phi]:>2}/{attempt_by_phi[phi]} ({fail_by_phi[phi]/attempt_by_phi[phi]*100:.0f}%)")

print("\n=== 파일럿 테스트용 실패 케이스 12개 (phi=90/120/150 위주, L_M/s 다양하게) ===")
pilot = [t for t in failed if abs(t[1]) >= 90][:12]
for lm, phi, beta, s in pilot:
    print(f"  L_M={lm}, phi={phi}, s={s}")
