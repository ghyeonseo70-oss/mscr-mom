"""
튜브 옆면을 강체 구(indenter)로 실제로 눌러서(변위제어) CalculiX 접촉해석 실행.
접촉쌍(slave=튜브 바깥면, master=구 표면), 압력-겹침(penalty) 접촉조건을 정의하고,
구의 모든 절점에 동일 변위를 직접 강제해서 강체 평행이동을 흉내낸다
(*RIGID BODY는 셸(S6) 절점과 함께 못 쓰는 CalculiX 제약이 있어서 이 방식을 씀).

sweep_contact.py에서 여러 시나리오(눌러넣는 깊이 등)를 반복 실행할 수 있도록 함수화함.
"""
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import material_convert as mat

E_RIGID_MM = 210000.0  # N/mm^2, 강체 근사(스틸 정도) - 변위를 직접 강제하므로 형식적인 값

# 2026-08-18 추가: 니티놀 와이어(사용자 실측값 - E=28GPa, 포아송비는 니티놀 전형값 0.33 사용).
# 소변형 선형탄성으로 근사(초탄성/초탄성 상변태 거동은 모델링 안 함 - 접촉으로 인한 국소변형은
# 작다는 전제). make_bent_contact_scene.py가 WIRE 물리그룹을 만들 때만(K1 구간 있을 때만)
# 존재하므로, ELSET=WIRE가 없는 메쉬(예: L_M=0)에서는 이 SOLID SECTION 줄이 CalculiX에서
# 무시되도록 존재 여부를 확인 후 조건부로 추가함.
E_NITINOL_MM = 28000.0  # N/mm^2 (28GPa = 28000 N/mm^2, 1GPa=1000 N/mm^2)
NU_NITINOL = 0.33
WIRE_R_MM = 0.05  # mm, make_bent_contact_scene.py의 WIRE_R와 반드시 같아야 함 (지름 100um)


def build_inp(push_depth, inp_name="contact_mesh.inp", sets_name="contact_node_sets.inp",
              job_name="contact_test", push_dir=(1.0, 0.0, 0.0), contact_stiffness=None,
              print_tip=False, inc=100, initial_inc=0.01):
    """push_dir: 접촉점에서 튜브 표면 바깥쪽을 향하는 법선(구가 원래 놓인 쪽) 방향.
    실제 변위는 이 방향의 반대(-push_dir)로 push_depth만큼 줘서 눌러 들어가게 함.
    직선 튜브는 구가 +X쪽에 있어서 기본값이 (1,0,0)이지만, 굽은 튜브는 접촉점마다 법선이
    달라서 반드시 make_bent_contact_scene.build_mesh()가 반환한 normal을 넣어야 함
    (방향을 잘못 넣으면 구가 튜브에서 멀어지는 쪽으로 밀려서 접촉력이 0으로 나옴 - 실제로
    두 번 겪은 버그: 처음엔 반대부호 기본값, 이후엔 일반화하며 부호가 또 꼬였었음)."""
    import numpy as np
    n = np.array(push_dir, dtype=float)
    n = n / np.linalg.norm(n)
    dx, dy, dz = (-n * push_depth).tolist()

    if contact_stiffness is None:
        # 이전엔 1000.0(N/mm^3)을 그냥 찍었는데, 재질 강성(E~1.34 N/mm^2)에 비해 100배 넘게
        # 커서 살짝만 파고들어도 압력이 폭발적으로 커져 수렴을 방해했을 가능성이 큼.
        # 물리적으로 더 말이 되는 값(E / 특성길이)으로 자동 추정: E_MM / 볼 메쉬크기
        contact_stiffness = mat.E * 1e-6 / 0.15  # N/mm^2 / mm = N/mm^3

    # make_bent_contact_scene.py가 K1 구간(베이스~MOM)이 있을 때만 WIRE_BEAM 물리그룹(=ELSET,
    # B32 1D 빔요소)을 만드므로(L_M=0이면 K1 구간이 없어 와이어도 없음), mesh include 파일에
    # ELSET=WIRE_BEAM이 실제로 있을 때만 니티놀 재질/빔단면/임베드 제약을 추가.
    # *EMBEDDED ELEMENT는 WIRE_BEAM 절점을 HOST ELSET(TUBE, 실리콘 솔리드)의 형상함수장에
    # 위치기반으로 묶어서(별도 접촉/TIE 없이) 두 재질이 같이 변형하게 만듦 - 콘크리트 속 철근
    # 모델링과 동일한 표준기법. 빔 단면은 원형(SECTION=CIRC), 법선(0,0,1)은 굽힘평면(board
    # z축 고정, make_bent_contact_scene.py 참고)과 항상 수직이라 빔축과 절대 평행하지 않음.
    with open(os.path.join(HERE, inp_name), encoding="latin-1") as f:
        has_wire = "ELSET=WIRE_BEAM" in f.read().upper()
    wire_block = f"""*MATERIAL, NAME=NITINOL
*ELASTIC
{E_NITINOL_MM}, {NU_NITINOL}
**
*BEAM SECTION, ELSET=WIRE_BEAM, MATERIAL=NITINOL, SECTION=CIRC
{WIRE_R_MM}
0.,0.,1.
**
*EMBEDDED ELEMENT, HOST ELSET=TUBE
WIRE_BEAM
**""" if has_wire else "**"

    inp_content = f"""*INCLUDE, INPUT={inp_name}
*INCLUDE, INPUT={sets_name}
**
*MATERIAL, NAME=SILICONE
*HYPERELASTIC, NEO HOOKE
{mat.C10_MM:.6E}, {mat.D1_MM:.6E}
**
*MATERIAL, NAME=RIGID_MAT
*ELASTIC
{E_RIGID_MM}, 0.3
**
{wire_block}
*SOLID SECTION, ELSET=TUBE, MATERIAL=SILICONE
*SOLID SECTION, ELSET=BALL, MATERIAL=RIGID_MAT
*SHELL SECTION, ELSET=BALL_SURF, MATERIAL=RIGID_MAT
0.01
**
*SURFACE, NAME=S_TUBE_OUTER, TYPE=NODE
N_TUBE_OUTER
*SURFACE, NAME=S_BALL, TYPE=ELEMENT
BALL_SURF, SPOS
**
*SURFACE INTERACTION, NAME=CONTACT1
*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR
{contact_stiffness:.4E}
**
*CONTACT PAIR, TYPE=NODE TO SURFACE, INTERACTION=CONTACT1
S_TUBE_OUTER, S_BALL
**
*BOUNDARY
N_FIXED, 1, 3
**
*STEP, NLGEOM, INC={inc}
*STATIC
{initial_inc}, 1.0
*BOUNDARY
N_BALL_ALL, 1, 1, {dx:.6E}
N_BALL_ALL, 2, 2, {dy:.6E}
N_BALL_ALL, 3, 3, {dz:.6E}
*NODE PRINT, NSET=N_BALL_ALL
U, RF
{"*NODE PRINT, NSET=N_TIP" + chr(10) + "U" if print_tip else "**"}
*NODE FILE
U
*EL FILE
S, E
*CONTACT FILE
CDIS, CSTR
*END STEP
"""
    inp_path = os.path.join(HERE, f"{job_name}.inp")
    with open(inp_path, "w") as f:
        f.write(inp_content)
    return inp_path


# CalculiX는 OMP_NUM_THREADS를 요소강성행렬 계산 등에 씀 - 8코어/16스레드 있는데 기본값이
# 안 정해져 있으면 1스레드로만 돎(실제 확인함). 접촉 스윕은 여러 케이스를 동시에 백그라운드로
# 돌리는 경우가 있어서(sweep_bent_contact.py + run_confusable_pair_test.py 등), 8을 다 주면
# 오히려 서로 경합할 수 있어 절반(4)로 설정 - 단독 실행이면 더 올려도 됨.
CCX_THREADS = 4


def run_ccx(job_name="contact_test", timeout=600, n_threads=CCX_THREADS):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)
    try:
        result = subprocess.run(
            ["ccx", job_name], capture_output=True, text=True, cwd=HERE, env=env, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        # Linux에서는 timeout 시 subprocess가 자식 프로세스(ccx)까지 함께 죽이므로
        # Windows판의 taskkill 정리 로직이 필요 없음.
        raise
    return result


def check_converged(job_name="contact_test", tol=1e-3):
    """.sta의 마지막 증분 TOTAL TIME이 1.0(=하중을 100% 다 걸었다)에 도달했는지 확인.
    깊게 누르는 케이스(s=10,depth=0.15mm)에서 접촉 수렴 실패로 중간(예: TOTAL TIME=0.14)에
    멈췄는데도 .dat 파일 자체는 존재해서, 기존엔 그 미완성 중간상태를 '결과'로 잘못 읽어
    힘=0,변위=0을 진짜 결과처럼 반환하는 버그가 있었음(s=10 케이스에서 실제로 겪음) -> 이 체크로 방지."""
    sta_path = os.path.join(HERE, f"{job_name}.sta")
    if not os.path.exists(sta_path):
        return False, 0.0
    last_total_time = 0.0
    with open(sta_path, encoding="latin-1") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 7 and parts[0].isdigit():
                try:
                    last_total_time = float(parts[4])
                except ValueError:
                    pass
    return last_total_time >= 1.0 - tol, last_total_time


def parse_results(job_name="contact_test", n_ball_nodes=None, print_tip=False):
    """마지막 증분(time=1.0)의 접촉반력 총합과 평균 x변위를 반환.
    (블록 경계를 '다음 마커 등장 지점'으로 정확히 잘라서 이전의 혼입 버그를 고침)"""
    dat_path = os.path.join(HERE, f"{job_name}.dat")
    with open(dat_path, encoding="latin-1") as f:
        dat = f.read()

    def last_block(marker):
        idx = dat.rfind(marker)
        if idx == -1:
            return None
        chunk = dat[idx + len(marker):]
        # 다음 블록 헤더('forces'나 'displacements' 등) 나오기 전까지만 자름
        for other in ["displacements (vx,vy,vz)", "forces (fx,fy,fz)"]:
            pos = chunk.find(other)
            if pos != -1:
                chunk = chunk[:pos]
        return chunk

    force_chunk = last_block("forces (fx,fy,fz) for set N_BALL_ALL and time")
    disp_chunk = last_block("displacements (vx,vy,vz) for set N_BALL_ALL and time")

    def sum_col(chunk, col):
        total, n = 0.0, 0
        for line in chunk.splitlines():
            parts = line.split()
            if len(parts) == 4:
                try:
                    total += float(parts[col])
                    n += 1
                except ValueError:
                    pass
        return total, n

    fx_total, n_f = sum_col(force_chunk, 1)
    fy_total, _ = sum_col(force_chunk, 2)
    fz_total, _ = sum_col(force_chunk, 3)
    ux_sum, n_u = sum_col(disp_chunk, 1)
    ux_avg = ux_sum / n_u if n_u else float("nan")
    f_mag = (fx_total ** 2 + fy_total ** 2 + fz_total ** 2) ** 0.5
    result = {"Fx_total_N": fx_total, "Fy_total_N": fy_total, "Fz_total_N": fz_total,
              "F_mag_N": f_mag, "n_force_nodes": n_f, "ux_avg_mm": ux_avg, "n_disp_nodes": n_u}

    if print_tip:
        # N_TIP(팁 캡 절점들)의 평균 변위 - 빔이론 단순모델이 예측하는 "팁 위치 이동량"과
        # 직접 비교하기 위한 값 (make_bent_contact_scene.py가 만든 TIP 물리그룹 필요).
        tip_chunk = last_block("displacements (vx,vy,vz) for set N_TIP and time")
        tux_sum, n_tu = sum_col(tip_chunk, 1)
        tuy_sum, _ = sum_col(tip_chunk, 2)
        tuz_sum, _ = sum_col(tip_chunk, 3)
        result["tip_ux_avg_mm"] = tux_sum / n_tu if n_tu else float("nan")
        result["tip_uy_avg_mm"] = tuy_sum / n_tu if n_tu else float("nan")
        result["tip_uz_avg_mm"] = tuz_sum / n_tu if n_tu else float("nan")
        result["n_tip_nodes"] = n_tu
    return result


def _parse_nset(sets_path, nset_name):
    """*NSET, NSET=<nset_name> 블록의 절점ID 목록(콤마구분, 여러 줄)을 파싱."""
    with open(sets_path) as f:
        text = f.read()
    marker = f"*NSET, NSET={nset_name}"
    idx = text.find(marker)
    if idx == -1:
        return []
    chunk = text[idx + len(marker):]
    end = chunk.find("*NSET")
    if end != -1:
        chunk = chunk[:end]
    ids = []
    for line in chunk.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        for tok in line.split(","):
            tok = tok.strip()
            if tok:
                ids.append(int(tok))
    return ids


def _parse_node_coords(inp_path, node_ids):
    """mesh .inp의 *NODE 블록에서 주어진 절점ID들의 원좌표(x,y,z)를 뽑음."""
    wanted = set(node_ids)
    coords = {}
    with open(inp_path, encoding="latin-1") as f:
        in_node_block = False
        for line in f:
            s = line.strip()
            if s.upper().startswith("*NODE"):
                in_node_block = True
                continue
            if s.startswith("*"):
                in_node_block = False
                continue
            if in_node_block and s:
                parts = s.split(",")
                if len(parts) >= 4:
                    nid = int(parts[0])
                    if nid in wanted:
                        coords[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return coords


def get_tip_rotation_deg(job_name, inp_name, sets_name):
    """접촉 전(원래 절점좌표) vs 접촉 후(원좌표+변위)의 N_TIP(팁 캡) 절점들을 강체회전으로
    피팅(Kabsch)해서, 팁 캡이 실제로 얼마나/어느 방향으로 회전했는지(deg) 추정.
    force_model.py의 theta(로컬 굽힘평면 xy 내 접선각) 정의와 비교 가능하도록,
    피팅된 3x3 회전행렬에서 z축 기준 평면내 회전성분만 뽑아 반환한다
    (굽힘이 z=일정 평면 내에서 일어난다는 가정 - board_z 고정, make_bent_contact_scene.py 참고).
    """
    dat_path = os.path.join(HERE, f"{job_name}.dat")
    sets_path = os.path.join(HERE, sets_name)
    inp_path = os.path.join(HERE, inp_name)

    tip_ids = _parse_nset(sets_path, "N_TIP")
    if not tip_ids:
        return {"tip_theta_deg": float("nan"), "tip_rotation_rmse_mm": float("nan")}
    orig = _parse_node_coords(inp_path, tip_ids)

    with open(dat_path, encoding="latin-1") as f:
        dat = f.read()
    marker = "displacements (vx,vy,vz) for set N_TIP and time"
    idx = dat.rfind(marker)
    if idx == -1:
        return {"tip_theta_deg": float("nan"), "tip_rotation_rmse_mm": float("nan")}
    chunk = dat[idx + len(marker):]
    for other in ["displacements (vx,vy,vz)", "forces (fx,fy,fz)"]:
        pos = chunk.find(other)
        if pos != -1:
            chunk = chunk[:pos]

    P, Q = [], []
    for line in chunk.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            nid = int(parts[0])
            ux, uy, uz = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if nid not in orig:
            continue
        x0, y0, z0 = orig[nid]
        P.append((x0, y0, z0))
        Q.append((x0 + ux, y0 + uy, z0 + uz))

    if len(P) < 3:
        return {"tip_theta_deg": float("nan"), "tip_rotation_rmse_mm": float("nan")}

    P = np.array(P)
    Q = np.array(Q)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T  # Pc를 Qc로 보내는 회전행렬 (Qc ~= R @ Pc)

    theta_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    fit_resid = Qc.T - R @ Pc.T
    rmse = float(np.sqrt(np.mean(np.sum(fit_resid ** 2, axis=0))))
    return {"tip_theta_deg": float(theta_deg), "tip_rotation_rmse_mm": rmse}


def run_case(push_depth, inp_name="contact_mesh.inp", sets_name="contact_node_sets.inp",
             job_name="contact_test", timeout=800, verbose=True, push_dir=(1.0, 0.0, 0.0),
             print_tip=False, n_threads=CCX_THREADS, inc=100, initial_inc=0.01):
    # 굽은 튜브 접촉해석에서 600초로 했다가 실제로는 100% 완료됐는데 그 직후 타임아웃에
    # 걸려서 결과를 놓칠 뻔한 적이 있어서(.sta로는 완료 확인, .dat도 무사히 써짐) 여유를 더 둠.
    build_inp(push_depth, inp_name, sets_name, job_name, push_dir=push_dir, print_tip=print_tip,
              inc=inc, initial_inc=initial_inc)
    if verbose:
        print(f"CalculiX 실행 중 (push_depth={push_depth}mm, push_dir={push_dir})...")
    result = run_ccx(job_name, timeout, n_threads=n_threads)
    dat_path = os.path.join(HERE, f"{job_name}.dat")
    if not os.path.exists(dat_path):
        if verbose:
            print("실패 - .dat 없음. stdout 마지막 부분:")
            print(result.stdout[-2000:])
        return None
    converged, last_time = check_converged(job_name)
    if not converged:
        if verbose:
            print(f"실패 - 수렴 안 됨 (TOTAL TIME={last_time:.4f} / 1.0에서 중단, 접촉이 너무 깊게 들어갔거나 "
                  f"해석이 발산했을 가능성)")
        return None
    res = parse_results(job_name, print_tip=print_tip)
    if print_tip:
        res.update(get_tip_rotation_deg(job_name, inp_name, sets_name))
    if verbose:
        print(f"  -> F_mag={res['F_mag_N']*1000:.4f} mN (Fx={res['Fx_total_N']*1000:.4f}, "
              f"Fy={res['Fy_total_N']*1000:.4f}, Fz={res['Fz_total_N']*1000:.4f} mN), "
              f"ux_avg={res['ux_avg_mm']:.5f}mm")
        if print_tip:
            print(f"  -> TIP shift: ({res['tip_ux_avg_mm']:.5f}, {res['tip_uy_avg_mm']:.5f}, "
                  f"{res['tip_uz_avg_mm']:.5f}) mm, TIP rotation(board frame)={res['tip_theta_deg']:.3f}deg "
                  f"(fit rmse={res['tip_rotation_rmse_mm']:.5f}mm)")
    return res


if __name__ == "__main__":
    r = run_case(push_depth=0.15)
    print(r)
