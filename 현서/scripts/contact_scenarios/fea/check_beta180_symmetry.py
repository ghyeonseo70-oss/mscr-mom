"""run_lm_phi_position_matv2_sweep.sh 실행 후 돌릴 것.

beta=180의 90개 표본이 대응하는 beta=0 케이스(같은 L_M,phi,s)의 부호반전(Fx,Fy,tip_ux,
tip_uy 전부 -1배)과 얼마나 일치하는지 확인. fea_beta_fine_resolution.json(옛 재료값)에서
이미 확인된 대칭성이 새 재료값에서도 유지되는지가 관건 - 유지되면 beta=180을 90개 이상
더 돌릴 필요 없음(전체 360개를 만들 필요 없이 beta=0 데이터에 부호만 뒤집어 재사용).

판정: 상대오차(F_mag 기준) 중앙값이 5% 이내면 "대칭성 유지"로 보고 그대로 사용 권장.
그 이상이면 beta=180도 전체 360개(LM 4 x phi 9 x s 10) 스윕 필요.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

with open(os.path.join(FEA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8") as f:
    rows = json.load(f)

beta0 = {(r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]): r for r in rows if r["beta_deg"] == 0.0}
beta180 = [r for r in rows if r["beta_deg"] == 180.0]

if not beta180:
    raise SystemExit("beta=180 데이터가 없음 - 스윕이 아직 안 끝났거나 실패한 것으로 보임")

errs = []
print(f"{'L_M':>6} {'phi':>6} {'s':>6}  {'Fmag(b=0)':>10} {'Fmag(b=180)':>11}  {'상대오차':>8}")
for r180 in beta180:
    key = (r180["L_M_mm"], r180["phi_deg"], r180["contact_s_mm"])
    r0 = beta0.get(key)
    if r0 is None:
        print(f"  [경고] 대응하는 beta=0 케이스 없음: {key}")
        continue
    f0, f180 = r0["F_mag_N"], r180["F_mag_N"]
    rel_err = abs(f180 - f0) / max(f0, 1e-12)
    errs.append(rel_err)
    print(f"{key[0]:>6.0f} {key[1]:>6.0f} {key[2]:>6.0f}  {f0*1000:>9.4f}mN {f180*1000:>10.4f}mN  {rel_err*100:>7.2f}%")

errs.sort()
median_err = errs[len(errs) // 2]
print()
print(f"F_mag 상대오차 중앙값: {median_err*100:.2f}%  (표본 {len(errs)}개)")
if median_err < 0.05:
    print("판정: 대칭성 유지 -> beta=180 추가 스윕 불필요, beta=0 데이터를 부호반전해서 재사용 권장.")
else:
    print("판정: 대칭성 어긋남 -> beta=180도 전체 360케이스(run_lm_phi_position_matv2_sweep.sh의 "
          "LM_LIST_FULL/PHI_LIST_FULL을 beta=180에도 적용) 스윕 필요.")
