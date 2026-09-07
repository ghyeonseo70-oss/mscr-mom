"""
beta 정밀분해능 스윕 워커: L_M,phi 고정, s=30mm/depth=0.10mm 고정, beta만 15도 간격으로
스캔해서 "정확히 몇 도에서 반응 방향이 뒤집히는지" 찾는다. 병렬화를 위해 beta 리스트를
쪼개서(--beta_list) 여러 프로세스가 나눠 맡음.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

S_FIXED = 30.0
DEPTH_FIXED = 0.10
BALL_R = 0.4

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--beta_list", type=str, required=True, help="콤마구분 beta값들")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()

L_M, phi, tag = args.L_M, args.phi, args.tag
beta_list = [float(x) for x in args.beta_list.split(",")]

centerline_path = os.path.join(HERE, f"betafine_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_beta_fine_resolution_{tag}.json")
out_path = os.path.abspath(out_path)


def cleanup(job_name, inp_name, sets_name):
    for pat in [f"{job_name}.*", inp_name, sets_name]:
        for f in glob.glob(os.path.join(HERE, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


results = []
total = len(beta_list)
for n, beta in enumerate(beta_list, 1):
    inp_name = f"betafine_mesh_{tag}_{n:02d}.inp"
    sets_name = f"betafine_node_sets_{tag}_{n:02d}.inp"
    job_name = f"betafinesweep_{tag}_{n:02d}"
    t0 = time.time()
    print(f"[{tag} {n}/{total}] L_M={L_M}, phi={phi}, beta={beta}, s={S_FIXED}mm, "
          f"depth={DEPTH_FIXED}mm ...", flush=True)
    try:
        scene_info = scene.build_mesh(contact_s=S_FIXED, ball_r=BALL_R, verbose=False,
                                       centerline_path=centerline_path, beta_deg=beta,
                                       inp_name=inp_name, sets_name=sets_name)
        normal = scene_info["normal"]
        res = rc.run_case(
            DEPTH_FIXED, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
            timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=args.threads,
            print_tip=True,
        )
    except Exception as e:
        print(f"  [{tag}] 실패: {e}", flush=True)
        res = None
        normal = None
        scene_info = None
    dt = time.time() - t0
    if res is None:
        print(f"  [{tag}] [실패] ({dt:.1f}s)", flush=True)
        cleanup(job_name, inp_name, sets_name)
        continue
    row = {
        "L_M_mm": L_M, "phi_deg": phi, "beta_deg": beta, "contact_s_mm": S_FIXED,
        "ball_r_mm": BALL_R, "push_depth_mm": DEPTH_FIXED,
        "normal": normal, "ball_center": scene_info["ball_center"],
        "Fx_total_N": res["Fx_total_N"], "Fy_total_N": res["Fy_total_N"],
        "Fz_total_N": res["Fz_total_N"], "F_mag_N": res["F_mag_N"],
        "ux_avg_mm": res["ux_avg_mm"],
        "tip_ux_avg_mm": res["tip_ux_avg_mm"], "tip_uy_avg_mm": res["tip_uy_avg_mm"],
        "tip_uz_avg_mm": res["tip_uz_avg_mm"], "tip_theta_deg_board": res["tip_theta_deg"],
        "tip_rotation_rmse_mm": res["tip_rotation_rmse_mm"], "wall_time_s": dt,
    }
    results.append(row)
    print(f"  [{tag}] F_mag={res['F_mag_N']*1000:.4f}mN  ({dt:.1f}s)", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    cleanup(job_name, inp_name, sets_name)

try:
    os.remove(centerline_path)
except OSError:
    pass
print(f"[{tag}] 완료: {len(results)}/{total} 성공", flush=True)
