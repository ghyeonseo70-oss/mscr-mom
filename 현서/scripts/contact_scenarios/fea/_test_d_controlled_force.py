"""2026-10-07(44번 항목2, 미구현이던 부분): "힘만 걸기" 방식에서 접촉점의 법선방향 변위 d를
목표값으로 맞추는 시컨트(secant) 반복 루프. 44번에서 확인된 하중절점 선정 방식
(_test_s80_force_only.py: N_TUBE_OUTER 중 contact_point+normal*(D_OUT/2)에 가장 가까운
노드)을 재사용. .inp 메쉬 파일은 gitignore 대상이라 매번 scene.build_mesh로 재생성함
(gmsh가 결정적이라 같은 파라미터면 같은 메쉬/절점번호가 나옴 - 이미 여러 세션에서 확인된
전제).

사용법: python _test_d_controlled_force.py --s 30 --targets 2,5,10,20
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc
import material_convert as mat

L_M, PHI = 12.5, -120.0
BALL_R = 0.4
D_OUT = 2.0 * BALL_R  # 볼 지름


def build_case_mesh(s, tag):
    import subprocess
    centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    inp_name = f"matv2_mesh_{tag}.inp"
    sets_name = f"matv2_node_sets_{tag}.inp"
    scene_info = scene.build_mesh(
        contact_s=s, ball_r=BALL_R, verbose=False, centerline_path=centerline_path,
        beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
        contact_mesh_size=0.08, contact_refine_radius=BALL_R + 0.3,
    )
    try:
        os.remove(centerline_path)
    except OSError:
        pass
    normal = np.array(scene_info["normal"])
    contact_point = np.array(scene_info["contact_point"])

    outer_ids = rc._parse_nset(sets_name, "N_TUBE_OUTER")
    coords = rc._parse_node_coords(inp_name, outer_ids)
    surface_point = contact_point + normal * (D_OUT / 2.0)
    best_nid, best_d = None, 1e9
    for nid, (x, y, z) in coords.items():
        d = np.linalg.norm(np.array([x, y, z]) - surface_point)
        if d < best_d:
            best_d, best_nid = d, nid
    return inp_name, sets_name, normal, best_nid


def _wire_block(inp_name):
    with open(inp_name, encoding="latin-1") as f:
        has_wire = "ELSET=WIRE_BEAM" in f.read().upper()
    if not has_wire:
        return "**"
    return f"""*MATERIAL, NAME=NITINOL
*ELASTIC
{rc.E_NITINOL_MM}, {rc.NU_NITINOL}
**
*BEAM SECTION, ELSET=WIRE_BEAM, MATERIAL=NITINOL, SECTION=CIRC
{rc.WIRE_R_MM}
0.,0.,1.
**
*EMBEDDED ELEMENT, HOST ELSET=TUBE
WIRE_BEAM
**"""


def run_force_case(inp_name, sets_name, load_node, normal, f_mn, job_tag):
    """F(mN)로 힘만 걸기 실행, (converged, d_at_load_node, tip_ux, tip_uy, tip_mag) 반환.
    d_at_load_node = 하중절점 변위를 -normal(누르는 방향)에 투영한 값(양수=눌린 방향으로 이동)."""
    f_n = f_mn * 1e-3
    Fx, Fy, Fz = (-normal * f_n).tolist()
    job_name = f"matv2_DCTRL_{job_tag}_{int(round(f_mn * 1e5))}"
    inp_content = f"""*INCLUDE, INPUT={inp_name}
*INCLUDE, INPUT={sets_name}
**
*MATERIAL, NAME=SILICONE
*HYPERELASTIC, NEO HOOKE
{mat.C10_MM:.6E}, {mat.D1_MM:.6E}
**
*MATERIAL, NAME=RIGID_MAT
*ELASTIC
{rc.E_RIGID_MM}, 0.3
**
{_wire_block(inp_name)}
*SOLID SECTION, ELSET=TUBE, MATERIAL=SILICONE
*SOLID SECTION, ELSET=BALL, MATERIAL=RIGID_MAT
*SHELL SECTION, ELSET=BALL_SURF, MATERIAL=RIGID_MAT
0.01
**
*NSET, NSET=N_LOAD
{load_node}
**
*BOUNDARY
N_FIXED, 1, 3
N_BALL_ALL, 1, 3
**
*STEP, NLGEOM, INC=200
*STATIC
0.01, 1.0, 1e-10, 1.0
*CLOAD
{load_node}, 1, {Fx:.6E}
{load_node}, 2, {Fy:.6E}
{load_node}, 3, {Fz:.6E}
*NODE PRINT, NSET=N_LOAD
U
*NODE PRINT, NSET=N_TIP
U
*END STEP
"""
    with open(f"{job_name}.inp", "w") as fo:
        fo.write(inp_content)
    try:
        rc.run_ccx(job_name, timeout=1800, n_threads=4)
    except Exception as e:
        print(f"    (run_ccx 예외: {e})", flush=True)
        return False, None, None, None, None
    converged, _ = rc.check_converged(job_name)
    if not converged:
        return False, None, None, None, None
    with open(f"{job_name}.dat", encoding="latin-1") as fd:
        dat = fd.read()

    def _avg_u(nset_label):
        marker = f"displacements (vx,vy,vz) for set {nset_label} and time  0.1000000E+01"
        idx = dat.find(marker)
        if idx == -1:
            return None
        chunk = dat[idx + len(marker):]
        p = chunk.find("forces")
        if p != -1:
            chunk = chunk[:p]
        ux = uy = uz = 0.0
        n = 0
        for line in chunk.splitlines():
            parts = line.split()
            if len(parts) in (4, 5):
                try:
                    ux += float(parts[1]); uy += float(parts[2]); uz += float(parts[3]); n += 1
                except ValueError:
                    pass
        return None if not n else (ux / n, uy / n, uz / n)

    u_load = _avg_u("N_LOAD")
    u_tip = _avg_u("N_TIP")
    if u_load is None or u_tip is None:
        return False, None, None, None, None
    d_achieved = -np.dot(np.array(u_load), normal)  # -normal 방향(누르는 방향) 성분
    tip_ux, tip_uy = u_tip[0], u_tip[1]
    tip_mag = (tip_ux ** 2 + tip_uy ** 2) ** 0.5
    return True, d_achieved, tip_ux, tip_uy, tip_mag


def secant_find_force(inp_name, sets_name, load_node, normal, d_target, tag, max_iter=4):
    """d_target(mm)에 맞는 F(mN)를 시컨트법으로 찾음. 초기 추정은 알려진 F-vs-tip 스캔표
    (0.005mN->0.95mm, 0.05->8.80mm, 0.25->29.81mm, 1.0->51.12mm, 팁변위 기준이지만 자릿수
    맞추는 용도로 충분)를 거칠게 보간해서 잡음."""
    scan_f = [0.005, 0.05, 0.25, 1.0]
    scan_tip = [0.95, 8.80, 29.81, 51.12]
    f0 = float(np.interp(d_target, scan_tip, scan_f))
    f1 = f0 * 1.3

    history = []
    f_prev, f_cur = f0, f1
    d_prev = None
    for it in range(max_iter):
        f_try = f_cur
        ok, d, tip_ux, tip_uy, tip_mag = run_force_case(inp_name, sets_name, load_node, normal, f_try, tag)
        print(f"  [{tag} d_target={d_target}mm] iter{it}: F={f_try:.5f}mN -> "
              f"{'수렴' if ok else '실패'}" + (f", d={d:.3f}mm, 팁변위={tip_mag:.3f}mm" if ok else ""),
              flush=True)
        if not ok:
            history.append((f_try, None))
            # 실패하면 힘을 줄여서 재시도(한 단계만)
            f_cur = f_try * 0.7
            continue
        history.append((f_try, d))
        if abs(d - d_target) <= max(0.05 * d_target, 0.1):
            return dict(converged=True, F_mN=f_try, d=d, tip_ux=tip_ux, tip_uy=tip_uy,
                        tip_mag=tip_mag, iters=it + 1)
        if d_prev is None or f_prev == f_try:
            # 선형 비례로 다음값 추정(초기 1회)
            f_next = f_try * (d_target / max(d, 1e-6))
        else:
            denom = (d - d_prev)
            if abs(denom) < 1e-9:
                f_next = f_try * (d_target / max(d, 1e-6))
            else:
                f_next = f_try + (d_target - d) * (f_try - f_prev) / denom
        f_prev, d_prev = f_try, d
        f_cur = max(f_next, 1e-5)
    return dict(converged=False, F_mN=f_cur, d=history[-1][1] if history else None,
                tip_ux=None, tip_uy=None, tip_mag=None, iters=max_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=float, required=True)
    parser.add_argument("--targets", type=str, default="2,5,10,20")
    args = parser.parse_args()

    tag = f"DCTRL_s{int(args.s)}"
    print(f"=== 메쉬 생성: L_M={L_M}, phi={PHI}, s={args.s} ===", flush=True)
    t0 = time.time()
    inp_name, sets_name, normal, load_node = build_case_mesh(args.s, tag)
    print(f"  하중절점={load_node}, normal={normal}, 메쉬생성 {time.time()-t0:.1f}s", flush=True)

    results = {}
    for d_target in [float(x) for x in args.targets.split(",")]:
        res = secant_find_force(inp_name, sets_name, load_node, normal, d_target, tag)
        results[d_target] = res
        print(f"=== s={args.s}, d_target={d_target}mm 결과: {res} ===\n", flush=True)

    print("\n=== 전체 요약 ===")
    for d_target, res in results.items():
        if res["converged"]:
            print(f"s={args.s}, d_target={d_target}mm: F={res['F_mN']:.5f}mN, d_achieved={res['d']:.3f}mm, "
                  f"팁변위={res['tip_mag']:.3f}mm ({res['iters']}회 반복)")
        else:
            print(f"s={args.s}, d_target={d_target}mm: 수렴 실패({res['iters']}회 시도)")
