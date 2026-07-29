"""
능동 탐색(active sensing) 학습 데이터 생성. test_active_sensing.py에서 확인한 것처럼, 접촉
하나(s,F)를 놓고 외부자기장 방향(phi)을 몇 가지로 바꿔가며 반응을 같이 보면 "헷갈리는 쌍"이
구별됐음 - 그래서 한 번의 스냅샷이 아니라, 서로 다른 phi 3곳에서 관측한 B-필드 3세트를
입력으로 주는 데이터셋을 만든다.

시나리오: 로봇이 L_M 위치에 있고, s 지점에 힘(Fx,Fy)이 걸린 접촉이 발생 - 이 물리적 상황은
탐색 내내 고정. 외부자기장 방향만 PHI_PROBES 3곳으로 순서대로 바꿔가며(실제로도 자기장
방향은 외부 코일로 빠르게 바꿀 수 있어 하드웨어 추가 없이 가능) 각각의 B-필드를 측정한다고
가정.
"""
import os
import sys
import numpy as np
import magpylib as magpy
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "force_model"))
import force_model as fm

DATA_DIR = os.path.join(HERE, "..", "..", "data", "contact_scenarios")

N_SAMPLES = 3000
L_M_RANGE = (20.0, 80.0)
S_MARGIN = 5.0
F_MAX = 0.02
PHI_PROBES = [-90.0, 0.0, 90.0]  # deg, 능동탐색으로 순서대로 걸어볼 외부자기장 방향 3곳
SEED = 7

# ── cnn_data_generated.py와 동일한 센서/자석 모델 ──────────────────────────
SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.36
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_local, yLM_local, thLM_deg, xL_local, yL_local, thL_deg):
    xLM_b, yLM_b = fm.to_board_frame(xLM_local, yLM_local)
    xL_b, yL_b = fm.to_board_frame(xL_local, yL_local)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler('z', -thLM_deg, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler('z', -thL_deg, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


def add_noise(B, rng, frac=0.05):
    return B + rng.normal(0, np.max(np.abs(B)) * frac, B.shape)


def free_shape_closed_form(L_M, phi_deg):
    """contact_force_sweep.py와 동일 로직(논문 식(7)-(9) 닫힌형)."""
    phi = np.radians(phi_deg)
    l1 = (L_M - fm.H_M / 2) / 1000.0
    l2 = (fm.L - L_M - fm.H_M / 2) / 1000.0

    def eqs(vars):
        th_lm, th_l = vars
        tau1 = fm.M1 * fm.B_FIELD * np.sin(phi - th_lm - np.pi)
        tau2 = fm.M2 * fm.B_FIELD * np.sin(phi - th_l)
        return [fm.K1 * th_lm - (tau1 + tau2) * l1, fm.K2 * (th_l - th_lm) - tau2 * l2]

    best = None
    for th0 in np.radians(np.arange(-170, 171, 20)):
        sol, info, ier, msg = fsolve(eqs, [th0, th0], full_output=True)
        if ier == 1:
            dist = abs(((sol[1] - phi + np.pi) % (2 * np.pi)) - np.pi)
            if best is None or dist < best[0]:
                best = (dist, sol)
    if best is None:
        return None
    th_lm, th_l = best[1]
    l1_mm, l2_mm = L_M - fm.H_M / 2, fm.L - L_M - fm.H_M / 2
    R1 = l1_mm / (th_lm if abs(th_lm) > 1e-9 else 1e-9)
    dt = th_l - th_lm
    R2 = l2_mm / (dt if abs(dt) > 1e-9 else 1e-9)
    x_lm = R1 * np.sin(th_lm) + (fm.H_M / 2) * np.cos(th_lm)
    y_lm = R1 * (1 - np.cos(th_lm)) + (fm.H_M / 2) * np.sin(th_lm)
    x_l = x_lm + R2 * (np.sin(th_l) - np.sin(th_lm)) + (fm.H_M / 2) * np.cos(th_lm)
    y_l = y_lm + R2 * (np.cos(th_lm) - np.cos(th_l)) + (fm.H_M / 2) * np.sin(th_lm)
    return {"theta_LM_deg": np.degrees(th_lm), "x_LM": x_lm, "y_LM": y_lm,
            "theta_L_deg": np.degrees(th_l), "x_L": x_l, "y_L": y_l}


rng = np.random.default_rng(SEED)
noise_rng = np.random.default_rng(SEED + 1)

rows_y = []
B_free_all = np.zeros((N_SAMPLES, len(PHI_PROBES), 25, 3))
B_load_all = np.zeros((N_SAMPLES, len(PHI_PROBES), 25, 3))

n_done, n_fail = 0, 0
while n_done < N_SAMPLES:
    L_M = rng.uniform(*L_M_RANGE)
    s = rng.uniform(S_MARGIN, fm.L - S_MARGIN)
    F_mag = rng.uniform(0.0, F_MAX)
    F_ang = rng.uniform(0, 2 * np.pi)
    Fx, Fy = F_mag * np.cos(F_ang), F_mag * np.sin(F_ang)

    ok = True
    probe_B_free, probe_B_load = [], []
    for phi_deg in PHI_PROBES:
        r_free = free_shape_closed_form(L_M, phi_deg)
        if r_free is None:
            ok = False
            break
        try:
            r_load = fm.solve_shape(L_M=L_M, phi_deg=phi_deg,
                                     loads=[{"type": "point", "s": s, "Fx": Fx, "Fy": Fy}],
                                     theta_L_hint_deg=r_free["theta_L_deg"])
        except RuntimeError:
            ok = False
            break
        Bf = compute_B(r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"],
                        r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"])
        Bl = compute_B(r_load["x_LM"], r_load["y_LM"], r_load["theta_LM_deg"],
                        r_load["x_L"], r_load["y_L"], r_load["theta_L_deg"])
        probe_B_free.append(add_noise(Bf, noise_rng))
        probe_B_load.append(add_noise(Bl, noise_rng))

    if not ok:
        n_fail += 1
        continue

    B_free_all[n_done] = np.stack(probe_B_free)
    B_load_all[n_done] = np.stack(probe_B_load)
    rows_y.append([s, F_mag, np.degrees(F_ang), Fx, Fy, L_M])
    n_done += 1
    if n_done % 500 == 0:
        print(f"{n_done}/{N_SAMPLES} (실패 {n_fail}건)", flush=True)

y = np.array(rows_y)
y_cols = ["s", "F_mag", "F_ang_deg", "Fx", "Fy", "L_M"]

out_path = os.path.join(DATA_DIR, "contact_multiprobe_bfield.npz")
np.savez(out_path, B_free=B_free_all, B_load=B_load_all, y=y, y_columns=np.array(y_cols),
         phi_probes=np.array(PHI_PROBES))
print(f"\n저장: {out_path}  (n={n_done}, probes={PHI_PROBES}, 실패 {n_fail}건)")
