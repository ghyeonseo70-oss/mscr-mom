"""
접촉(외력) 시나리오(contact_force_scenarios.npz, 3000개)에 대해, 실제 홀센서 보드가
관측하게 될 자기장(B)을 시뮬레이션해서 학습용 입력(X)을 만든다.

cnn_data_generated.py와 같은 2-자석(magpylib) 모델·같은 센서 배치(5x5, 15mm 높이,
실측 자석 세기 0.36T)를 그대로 재사용한다. 다만 자석 위치/각도는 (그 스크립트의
닫힌형 근사가 아니라) contact_force_sweep.py가 이미 계산해둔, force_model ODE
슈팅법 결과(xL,yL,thL,xLM,yLM,thLM)를 force_model.to_board_frame()으로 보드
좌표계 변환해서 그대로 쓴다 - 두 스크립트의 각도(theta) 정의가 같은 로컬 각도
표기라는 걸 to_board_frame의 arc 공식과 cnn_data_generated.get_magnet_positions()
공식을 대조해서 확인함 (H_M/2 오프셋 항까지 정확히 일치).

무외력(free)과 접촉시(load) 두 형상 각각에 대해 B를 계산해서 같이 저장한다 -
"접촉이 없을 때 대비 자기장이 얼마나/어떻게 변했는지(B_load - B_free)"가 접촉을
감지하는 핵심 단서가 되기 때문에, 학습 시 둘 다 활용할 수 있게 남겨둠.
"""
import os
import sys
import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'force_model'))
import force_model as fm

DATA_DIR = os.path.join(HERE, '..', '..', 'data', 'contact_scenarios')

# ── cnn_data_generated.py와 동일한 센서/자석 모델 (하드웨어 실측 기반 상수) ──
SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])

MAGNET_BR_TESLA = 0.36
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_local, yLM_local, thLM_deg, xL_local, yL_local, thL_deg):
    """force_model 로컬좌표 -> 보드좌표 변환 후 자기장(uT, 노이즈 없음) 계산."""
    xLM_b, yLM_b = fm.to_board_frame(xLM_local, yLM_local)
    xL_b, yL_b = fm.to_board_frame(xL_local, yL_local)

    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler('z', -thLM_deg, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler('z', -thL_deg, degrees=True)

    return magpy.getB(mscr_robot, sensors) * 1e6  # T -> uT (MLX90393 실측 단위)


def add_noise(B, rng, frac=0.05):
    return B + rng.normal(0, np.max(np.abs(B)) * frac, B.shape)


if __name__ == "__main__":
    src = os.path.join(DATA_DIR, "contact_force_scenarios.npz")
    d = np.load(src, allow_pickle=True)
    data, cols = d["data"], list(d["columns"])
    col = {c: i for i, c in enumerate(cols)}
    n = len(data)
    print(f"입력: {src} (n={n})")

    rng = np.random.default_rng(42)
    B_free_all = np.zeros((n, 25, 3))
    B_load_all = np.zeros((n, 25, 3))

    for i in range(n):
        row = data[i]
        B_free = compute_B(
            row[col["xLM_free"]], row[col["yLM_free"]], row[col["thLM_free"]],
            row[col["xL_free"]], row[col["yL_free"]], row[col["thL_free"]],
        )
        B_load = compute_B(
            row[col["xLM_load"]], row[col["yLM_load"]], row[col["thLM_load"]],
            row[col["xL_load"]], row[col["yL_load"]], row[col["thL_load"]],
        )
        B_free_all[i] = add_noise(B_free, rng)
        B_load_all[i] = add_noise(B_load, rng)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{n}", flush=True)

    # 학습 타깃 후보: 접촉위치(s), 힘 크기/방향, 그리고 참고용 구동상태(L_M, phi_deg)
    y_cols = ["s", "F_mag", "F_ang_deg", "Fx", "Fy", "L_M", "phi_deg"]
    y = data[:, [col[c] for c in y_cols]]

    out_path = os.path.join(DATA_DIR, "contact_bfield_dataset.npz")
    np.savez(
        out_path,
        B_free=B_free_all, B_load=B_load_all,
        y=y, y_columns=np.array(y_cols),
    )
    print(f"저장: {out_path}  (B_free/B_load shape={B_load_all.shape}, y shape={y.shape})")
