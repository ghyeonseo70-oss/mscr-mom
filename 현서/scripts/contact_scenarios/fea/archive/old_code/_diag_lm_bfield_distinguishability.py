"""2026-08-26: L_M=0 실측 R^2 저조(MOM 재확보로도 미해결) 원인이 "학습 문제"가 아니라
"B-field 자체가 L_M을 물리적으로 구분 못 함"인지 확인하는 진단(추가 FEA 불필요).

방법: 실측 FEA 데이터에서 (phi,beta,s)가 일치하는 L_M 인접쌍(0-12.5, 12.5-25, ..., 75-87.5,
전부 12.5mm 등간격)을 찾아, 각 행의 실측 tip_*/mom_* 변위로 CNN이 실제로 보는 입력과 동일한
방식(B_load - B_free, train_segment_classifier_singleprobe_beta0180_4seg.py의 compute_B와
동일 로직)의 차분 B-field를 계산한다. 그 다음 인접한 두 L_M의 차분 B-field끼리 얼마나
다른지(distinguishing signal = L2 norm of difference)를 L_M 구간별로 비교한다.

가설이 맞다면(0-12.5mm 구간이 물리적으로 구분 안 됨): 0-12.5 구간의 distinguishing signal이
다른 구간(12.5-25, 25-37.5, ...)보다 뚜렷이 작아야 한다. 만약 비슷하거나 오히려 크다면
물리적 신호는 있는데 모델/데이터 쪽에서 못 배우고 있다는 뜻이므로 원인을 다시 좁혀야 한다.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")
sys.path.insert(0, FORCE_MODEL_DIR)
import force_model as fm  # noqa: E402
import magpylib as magpy  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.4
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


free_cache = {}


def get_free(L_M, phi):
    key = (round(L_M, 1), phi)
    if key not in free_cache:
        free_cache[key] = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    return free_cache[key]


def diff_B_for_row(r):
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    r_free = get_free(L_M, phi)
    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]

    if "mom_ux_avg_mm" in r:
        mom_ux, mom_uy, mom_theta = r["mom_ux_avg_mm"], r["mom_uy_avg_mm"], r["mom_theta_deg_board"]
    else:
        frac = L_M / 100.0
        mom_ux, mom_uy, mom_theta = r["tip_ux_avg_mm"] * frac, r["tip_uy_avg_mm"] * frac, r["tip_theta_deg_board"] * frac

    d_xL_local = r["tip_uy_avg_mm"]
    d_yL_local = r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    d_xLM_local = mom_uy
    d_yLM_local = mom_ux
    d_thLM = -mom_theta

    B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                        xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
    return (B_load - B_free).reshape(-1)


def main():
    rows = json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8"))
    by_lm_phi_beta_s = {}
    for r in rows:
        by_lm_phi_beta_s[(r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"])] = r

    lm_values = sorted(set(r["L_M_mm"] for r in rows))
    print("L_M values:", lm_values)

    results = defaultdict(list)  # (lm_a, lm_b) -> list of (||diff_a - diff_b||, ||diff_a||, ||diff_b||)
    for i in range(len(lm_values) - 1):
        lm_a, lm_b = lm_values[i], lm_values[i + 1]
        rows_a = [r for r in rows if r["L_M_mm"] == lm_a]
        for ra in rows_a:
            key_b = (lm_b, ra["phi_deg"], ra["beta_deg"], ra["contact_s_mm"])
            rb = by_lm_phi_beta_s.get(key_b)
            if rb is None:
                continue
            try:
                diff_a = diff_B_for_row(ra)
                diff_b = diff_B_for_row(rb)
            except Exception as e:
                print(f"  skip {lm_a}/{lm_b} phi={ra['phi_deg']} s={ra['contact_s_mm']}: {e}")
                continue
            dist = np.linalg.norm(diff_a - diff_b)
            results[(lm_a, lm_b)].append((dist, np.linalg.norm(diff_a), np.linalg.norm(diff_b)))

    print(f"\n{'L_M 구간':<15} {'매칭쌍 수':>8} {'구분신호 평균(uT)':>18} {'자체크기 평균(uT)':>18} {'상대비율':>10}")
    for (lm_a, lm_b), vals in sorted(results.items()):
        dists = np.array([v[0] for v in vals])
        selfmags = np.array([(v[1] + v[2]) / 2 for v in vals])
        rel = dists / np.maximum(selfmags, 1e-9)
        print(f"{lm_a:>5.1f}-{lm_b:<7.1f} {len(vals):>8d} {dists.mean():>18.4f} {selfmags.mean():>18.4f} {rel.mean():>10.3f}")


if __name__ == "__main__":
    main()
