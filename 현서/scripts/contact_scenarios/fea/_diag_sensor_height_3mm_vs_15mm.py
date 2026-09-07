"""센서 높이(z) 15mm vs 3mm 비교 - "신호 증폭 vs 45mm 격자의 공간 해상도 부족" 트레이드오프를
정량적으로 확인. 두 가지 독립적 테스트:

(A) 신호 증폭 + 구분가능성: 실측 FEA 데이터의 접촉 전/후 delta-B를 z=15/z=3에서 각각 계산해서
    (1) 절대 크기가 실제로 얼마나 커지는지, (2) L_M 매칭쌍 구분신호/자체크기 비율이 커지는지도
    같이 확인(_diag_lm_bfield_distinguishability.py와 같은 방법론) - 신호가 커져도 "상대적
    구분력"까지 좋아지는지는 별개 질문이라 따로 봐야 함.

(B) 공간 해상도(에일리어싱) 점검: 자석 하나를 대표 위치에 놓고 보드 위 1mm 간격 촘촘한 격자로
    Bz를 스캔해서, 자석 바로 위 피크의 반치폭(FWHM)을 z=15/z=3에서 측정. FWHM이 45mm(센서
    간격)보다 훨씬 좁으면 그 피크가 인접 센서 사이로 "빠질" 위험이 커서 정보 손실(에일리어싱)
    가능성이 실제로 있다는 뜻.
"""
import json
import os
import sys

import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")
sys.path.insert(0, FORCE_MODEL_DIR)
import force_model as fm

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

main_magnet = magpy.magnet.Cylinder(polarization=(0, 0.4, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -0.4, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def make_sensors(height_mm):
    positions = [(x, y, height_mm) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
    return magpy.Collection([magpy.Sensor(position=p) for p in positions])


def compute_B(sensors, xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6  # uT


# ============================================================
# (A) 신호 증폭 + 구분가능성 (실측 FEA 접촉 전/후 delta-B)
# ============================================================
all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)

rng = np.random.default_rng(0)
sample_idx = rng.choice(len(all_rows), size=min(120, len(all_rows)), replace=False)

sensors_15 = make_sensors(15.0)
sensors_3 = make_sensors(3.0)

mags_15, mags_3 = [], []
for i in sample_idx:
    r = all_rows[i]
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    except Exception:
        continue
    d_xL_local, d_yL_local = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    if "mom_ux_avg_mm" in r:
        d_xLM_local, d_yLM_local, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    else:
        frac = L_M / 100.0
        d_xLM_local, d_yLM_local, d_thLM = d_xL_local * frac, d_yL_local * frac, d_thL * frac
    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]

    args_free = (xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    args_load = (xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                 xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)

    delta_15 = compute_B(sensors_15, *args_load) - compute_B(sensors_15, *args_free)
    delta_3 = compute_B(sensors_3, *args_load) - compute_B(sensors_3, *args_free)
    mags_15.append(np.abs(delta_15).mean())
    mags_3.append(np.abs(delta_3).mean())

mags_15 = np.array(mags_15)
mags_3 = np.array(mags_3)
print("=== (A) 접촉 delta-B 절대 크기 (n={}) ===".format(len(mags_15)))
print(f"  z=15mm: 평균 |delta-B| = {mags_15.mean():.4f} uT")
print(f"  z=3mm : 평균 |delta-B| = {mags_3.mean():.4f} uT")
print(f"  증폭 배율 = {mags_3.mean() / mags_15.mean():.1f}배 (이론 역세제곱 예측치: {(15/3)**3:.0f}배)")

# ---- L_M 매칭쌍 구분가능성 (자체 델타-B 신호, _diag_lm_bfield_distinguishability.py와 같은 방식) ----
print("\n=== (A-2) L_M 매칭쌍 구분가능성 (phi,beta,s 고정, L_M만 다른 쌍, 자유형상 기준 B) ===")


def free_B(sensors, L_M, phi):
    r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    return compute_B(sensors, r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"],
                      r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"])


by_key = {}
for r in all_rows:
    key = (r["phi_deg"], r["beta_deg"], r["contact_s_mm"])
    by_key.setdefault(key, {})[r["L_M_mm"]] = r

lm_vals = sorted(set(r["L_M_mm"] for r in all_rows))
for h_label, sensors in [("z=15mm", sensors_15), ("z=3mm", sensors_3)]:
    print(f"  --- {h_label} ---")
    for a, b in zip(lm_vals[:-1], lm_vals[1:]):
        diffs, mags = [], []
        for key, d in by_key.items():
            if a in d and b in d:
                phi = key[0]
                Ba = free_B(sensors, a, phi)
                Bb = free_B(sensors, b, phi)
                diffs.append(np.abs(Ba - Bb).mean())
                mags.append((np.abs(Ba).mean() + np.abs(Bb).mean()) / 2)
        if diffs:
            diffs, mags = np.array(diffs), np.array(mags)
            print(f"    L_M {a}-{b}mm: 매칭쌍={len(diffs)}  구분신호={diffs.mean():.3f}uT  "
                  f"자체크기={mags.mean():.3f}uT  상대비율={diffs.mean()/mags.mean():.3f}")

# ============================================================
# (B) 공간 해상도(에일리어싱) 점검 - Bz 피크 반치폭(FWHM) vs 45mm 격자
# ============================================================
print("\n=== (B) 자석 바로 위 Bz 피크 반치폭(FWHM) vs 센서 간격(45mm) ===")
# 대표 위치: 보드 중앙 부근에 메인자석 하나만 배치(로봇 전체 대신 단일 자석으로 "피크가
# 얼마나 뾰족한지"만 순수하게 확인 - MOM까지 포함하면 두 피크가 겹쳐 해석이 복잡해짐)
main_magnet.position = (90, 90, 0)
main_magnet.orientation = Rotation.from_euler("z", 0, degrees=True)
mom.position = (1000, 1000, -1000)  # 멀리 치워서 기여 무시

x_fine = np.linspace(60, 120, 601)  # 0.1mm 간격
for h_label, h in [("z=15mm", 15.0), ("z=3mm", 3.0)]:
    scan_sensors = magpy.Collection([magpy.Sensor(position=(x, 90, h)) for x in x_fine])
    Bz = magpy.getB(main_magnet, scan_sensors)[:, 2] * 1e6
    peak = np.abs(Bz).max()
    half = peak / 2
    above = np.abs(Bz) >= half
    idx = np.where(above)[0]
    fwhm = x_fine[idx[-1]] - x_fine[idx[0]] if len(idx) > 1 else float("nan")
    print(f"  {h_label}: 피크 |Bz|={peak:.2f}uT, FWHM={fwhm:.2f}mm "
          f"({'45mm 격자보다 훨씬 좁음 - 에일리어싱 위험' if fwhm < 45 else '45mm 격자로 충분히 커버됨'})")
