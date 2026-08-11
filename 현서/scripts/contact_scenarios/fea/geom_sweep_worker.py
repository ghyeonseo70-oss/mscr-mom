"""
L_M/phi(자기장 방향, 부호 포함)를 바꿔가며 굽힘 형상 자체를 바꾼 뒤, 각 형상에서
접촉위치(s) x 깊이(depth) 스윕을 돌리는 워커. sweep_bent_contact_worker.py의 지오메트리
확장판 — 형상 하나(L_M,phi 조합)를 통째로 맡아서 s 3곳 x depth 3단계 = 9케이스를 순차 처리.
여러 형상을 동시에(별도 프로세스로) 돌릴 수 있도록 모든 산출물 파일명을 tag로 구분.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

S_LIST = [25.0, 50.0, 75.0]            # mm, 중심선 호길이 (기존 직선/굽은 스윕과 동일 비율)
PUSH_DEPTH_LIST = [0.05, 0.10, 0.15]   # mm
BALL_R = 0.4                            # mm

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True, help="파일명용 접미사 (예: LM30_phiN60)")
args = parser.parse_args()

L_M, phi, tag = args.L_M, args.phi, args.tag

centerline_path = os.path.join(HERE, f"bent_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_geom_sweep_{tag}.json")
out_path = os.path.abspath(out_path)

results = []
total = len(S_LIST) * len(PUSH_DEPTH_LIST)
n = 0
for contact_s in S_LIST:
    inp_name = f"geom_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"geom_node_sets_{tag}_s{contact_s:.0f}.inp"
    scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                   centerline_path=centerline_path,
                                   inp_name=inp_name, sets_name=sets_name)
    normal = scene_info["normal"]
    for push_depth in PUSH_DEPTH_LIST:
        n += 1
        job_name = f"geomsweep_{tag}_{n:02d}"
        t0 = time.time()
        print(f"[{tag} {n}/{total}] L_M={L_M}, phi={phi}, s={contact_s}mm, "
              f"push_depth={push_depth}mm ...", flush=True)
        try:
            res = rc.run_case(
                push_depth, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
                timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=args.threads,
                print_tip=True,
            )
        except Exception as e:
            print(f"  [{tag}] 실패: {e}", flush=True)
            res = None
        dt = time.time() - t0
        if res is None:
            print(f"  [{tag}] [실패] ({dt:.1f}s)", flush=True)
            continue
        row = {
            "L_M_mm": L_M,
            "phi_deg": phi,
            "contact_s_mm": contact_s,
            "ball_r_mm": BALL_R,
            "push_depth_mm": push_depth,
            "normal": normal,
            "ball_center": scene_info["ball_center"],
            "Fx_total_N": res["Fx_total_N"],
            "Fy_total_N": res["Fy_total_N"],
            "Fz_total_N": res["Fz_total_N"],
            "F_mag_N": res["F_mag_N"],
            "ux_avg_mm": res["ux_avg_mm"],
            "tip_ux_avg_mm": res["tip_ux_avg_mm"],
            "tip_uy_avg_mm": res["tip_uy_avg_mm"],
            "tip_uz_avg_mm": res["tip_uz_avg_mm"],
            "tip_theta_deg_board": res["tip_theta_deg"],
            "tip_rotation_rmse_mm": res["tip_rotation_rmse_mm"],
            "wall_time_s": dt,
        }
        results.append(row)
        print(f"  [{tag}] F_mag={res['F_mag_N']*1000:.4f}mN  ({dt:.1f}s)", flush=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[{tag}] 완료: {len(results)}/{total} 성공", flush=True)
