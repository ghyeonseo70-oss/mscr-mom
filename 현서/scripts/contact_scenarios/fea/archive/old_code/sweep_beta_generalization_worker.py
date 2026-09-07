"""
beta(원주각) 효과가 L_M,phi에 상관없이 일반화되는지 검증용 워커.
기존 fea_angle_sweep_all.json(L_M=50,phi=60 딱 한 조합에서만 beta 스윕)을 다른 (L_M,phi)
조합에서도 재현해서, beta=45 vs 135(90도 차이)에서 팁변위 방향이 180도 차이나는(단순회전이
아닌) 패턴이 다른 형상에서도 나오는지 확인. 병렬화를 위해 depth 1개씩 맡아서
beta(7개) x s(2개) = 14케이스를 순차 처리.
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

BETA_LIST = [45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
S_LIST = [30.0, 60.0]
BALL_R = 0.4

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--depth", type=float, required=True)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()

L_M, phi, depth, tag = args.L_M, args.phi, args.depth, args.tag

centerline_path = os.path.join(HERE, f"betagen_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_beta_generalization_check_{tag}.json")
out_path = os.path.abspath(out_path)


def cleanup(job_name, inp_name, sets_name):
    for pat in [f"{job_name}.*", inp_name, sets_name]:
        for f in glob.glob(os.path.join(HERE, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


results = []
cases = [(beta, s) for beta in BETA_LIST for s in S_LIST]
total = len(cases)
for n, (beta, contact_s) in enumerate(cases, 1):
    inp_name = f"betagen_mesh_{tag}_{n:02d}.inp"
    sets_name = f"betagen_node_sets_{tag}_{n:02d}.inp"
    job_name = f"betagensweep_{tag}_{n:02d}"
    t0 = time.time()
    print(f"[{tag} {n}/{total}] L_M={L_M}, phi={phi}, beta={beta}, s={contact_s}mm, "
          f"depth={depth}mm ...", flush=True)
    try:
        scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                       centerline_path=centerline_path, beta_deg=beta,
                                       inp_name=inp_name, sets_name=sets_name)
        normal = scene_info["normal"]
        res = rc.run_case(
            depth, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
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
        "L_M_mm": L_M, "phi_deg": phi, "beta_deg": beta, "contact_s_mm": contact_s,
        "ball_r_mm": BALL_R, "push_depth_mm": depth,
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
