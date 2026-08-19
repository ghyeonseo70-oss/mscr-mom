"""phi=+-30 실패 케이스 중 STABILIZE/명시적 최소증분 파일럿 테스트용 12개 뽑기.
_diag_failure_by_phi.py와 같은 로직, phi 필터만 90->30으로 바꿈."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

with open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8") as f:
    rows = json.load(f)

success = set()
for r in rows:
    success.add((r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"]))

LM_LIST_FULL = [0.0, 25.0, 50.0, 75.0]
S_LIST = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

attempted = []
for lm in LM_LIST_FULL:
    for phi in (30.0, -30.0):
        for s in S_LIST:
            attempted.append((lm, phi, 0.0, s))

failed = [t for t in attempted if t not in success]
print(f"phi=+-30 총 시도: {len(attempted)}, 실패: {len(failed)}")

print("\n=== 파일럿 12개 (L_M별로 3개씩 골고루) ===")
pilot = []
for lm in LM_LIST_FULL:
    lm_fails = [t for t in failed if t[0] == lm]
    pilot.extend(lm_fails[:3])
pilot = pilot[:12]
for lm, phi, beta, s in pilot:
    print(f"  L_M={lm}, phi={phi}, s={s}")
