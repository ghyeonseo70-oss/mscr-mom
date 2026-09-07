"""find_safe_force_range.py와 완전히 같은 시나리오(임의방향 점하중)를 FEA로 재현해서,
해석모델이 "반전됐다"고 판정한 바로 그 조합에서 FEA는 실제로 어떻게 나오는지 직접 비교한다.
(공으로 누르는 접촉해석이 아니라, N_LOAD 절점들에 *CLOAD로 직접 힘을 걸고 그 절점들의 평균
변위를 읽어서, 해석모델의 solve_shape_robust(loads=[...])와 같은 조건으로 만든다.)

좌표계 변환: 이 프로젝트의 보드좌표는 항상 x_board=BASE+y_local, y_board=x_local인
축 교환이라(get_bent_centerline.py), 힘(자유벡터)도 같은 규칙으로 Fx_board=Fy_local,
Fy_board=Fx_local로 바꿔서 걸어야 함. 반전 판정 기준(법선)도 make_bent_contact_scene.py가
이미 쓰고 있는 보드좌표용 공식(-cos(theta_local), sin(theta_local))을 그대로 재사용.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import material_convert as mat  # noqa: E402
from run_contact import check_converged, run_ccx  # noqa: E402


def build_inp(Fx_board, Fy_board, n_load_nodes, inp_name, sets_name, job_name):
    fx_each = Fx_board / n_load_nodes
    fy_each = Fy_board / n_load_nodes
    inp_content = f"""*INCLUDE, INPUT={inp_name}
*INCLUDE, INPUT={sets_name}
**
*MATERIAL, NAME=SILICONE
*HYPERELASTIC, NEO HOOKE
{mat.C10_MM:.6E}, {mat.D1_MM:.6E}
**
*SOLID SECTION, ELSET=TUBE, MATERIAL=SILICONE
**
*BOUNDARY
N_FIXED, 1, 3
**
*STEP, NLGEOM, INC=100
*STATIC
0.02, 1.0
*CLOAD
N_LOAD, 1, {fx_each:.6E}
N_LOAD, 2, {fy_each:.6E}
*NODE PRINT, NSET=N_LOAD
U
*END STEP
"""
    inp_path = os.path.join(HERE, f"{job_name}.inp")
    with open(inp_path, "w") as f:
        f.write(inp_content)
    return inp_path


def parse_load_disp(job_name):
    # .dat 헤더 줄 바로 다음에 빈 줄이 하나 더 있어서(마커 다음 줄이 데이터가 아니라 공백줄),
    # 그걸 "데이터 끝"으로 오인해 즉시 break하는 바람에 처음엔 항상 NaN이 나왔음(실제로 겪음).
    # 그래서 "데이터 줄을 하나라도 읽기 시작한 뒤"에만 형식이 깨지면 종료하도록 고침.
    dat_path = os.path.join(HERE, f"{job_name}.dat")
    with open(dat_path, encoding="latin-1") as f:
        dat = f.read()
    idx = dat.rfind("displacements (vx,vy,vz) for set N_LOAD and time")
    chunk = dat[idx:]
    lines = chunk.splitlines()[1:]
    ux_sum = uy_sum = 0.0
    n = 0
    started = False
    for line in lines:
        parts = line.split()
        if len(parts) == 4:
            try:
                ux_sum += float(parts[1])
                uy_sum += float(parts[2])
                n += 1
                started = True
                continue
            except ValueError:
                pass
        if started:
            break
    return (ux_sum / n, uy_sum / n) if n else (float("nan"), float("nan"))


def run_case(case, work_dir=HERE, timeout=1800, n_threads=4):
    """case: dict with seed, L_M, phi, s, F_ang_deg, F_mag_mN, analytic_threshold_mN"""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
    import force_model as fm
    import make_point_load_scene as scene

    tag = f"pl_{case['seed']:03d}"
    centerline_path = os.path.join(work_dir, f"centerline_{tag}.json")
    r = fm.solve_shape(L_M=case["L_M"], phi_deg=case["phi"], loads=[], return_curve=True)
    order = np.argsort(r["curve_s_mm"])
    s_arr, x_arr, y_arr, th_arr = (r["curve_s_mm"][order], r["curve_x_mm"][order],
                                    r["curve_y_mm"][order], r["curve_theta_deg"][order])
    s_target = np.linspace(s_arr.min(), s_arr.max(), 40)
    x_i = np.interp(s_target, s_arr, x_arr)
    y_i = np.interp(s_target, s_arr, y_arr)
    th_i = np.interp(s_target, s_arr, th_arr)
    BASE_X, BOARD_Z = 90.0, 3.0
    points = [{"s": float(si), "x": float(BASE_X + yi), "y": float(xi), "z": BOARD_Z,
               "theta_deg": float(ti)}
              for si, xi, yi, ti in zip(s_target, x_i, y_i, th_i)]
    with open(centerline_path, "w") as f:
        json.dump({"L_M": case["L_M"], "phi_deg": case["phi"], "points": points}, f)

    inp_name = f"{tag}_mesh.inp"
    sets_name = f"{tag}_sets.inp"
    info = scene.build_mesh(centerline_path, contact_s=case["s"], inp_name=inp_name,
                              sets_name=sets_name, verbose=True)

    theta_local = np.radians(np.interp(case["s"], s_target, th_i))
    normal_board = np.array([-np.cos(theta_local), np.sin(theta_local)])

    F_mag = case["F_mag_mN"] / 1000.0  # N
    F_ang = np.radians(case["F_ang_deg"])
    Fx_local, Fy_local = F_mag * np.cos(F_ang), F_mag * np.sin(F_ang)
    Fx_board, Fy_board = Fy_local, Fx_local  # 축 교환

    job_name = f"{tag}_job"
    build_inp(Fx_board, Fy_board, info["n_load_nodes"], inp_name, sets_name, job_name)
    result = run_ccx(job_name, timeout=timeout, n_threads=n_threads)
    dat_path = os.path.join(work_dir, f"{job_name}.dat")
    if not os.path.exists(dat_path):
        return {**case, "fea_status": "no_dat", "stdout_tail": result.stdout[-1000:]}
    converged, last_time = check_converged(job_name)
    if not converged:
        return {**case, "fea_status": "not_converged", "last_time": last_time}

    ux, uy = parse_load_disp(job_name)
    disp_normal = np.dot([ux, uy], normal_board)
    push_normal = np.dot([Fx_board, Fy_board], normal_board)
    fea_reversed = bool(np.sign(disp_normal) != np.sign(push_normal))
    return {**case, "fea_status": "ok", "ux_board": ux, "uy_board": uy,
            "push_normal": float(push_normal), "disp_normal": float(disp_normal),
            "fea_reversed": fea_reversed}


if __name__ == "__main__":
    demo_case = {"seed": 0, "L_M": 58.22, "phi": 0.0, "s": 8.69, "F_ang_deg": 5.9, "F_mag_mN": 0.5,
                 "analytic_threshold_mN": 0.0}
    print(run_case(demo_case))
