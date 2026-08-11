"""B) L_M=10mm 그리드 공백 검증용 워커. L_M 고정=10mm, phi는 인자로 받고, s는 대표값
3개(20,50,80mm)만, depth=0.10mm 고정, beta=0 고정. sweep_lm_phi_position_worker.py와
거의 동일하되 S_LIST만 축소."""
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

S_LIST = [20.0, 50.0, 80.0]
PUSH_DEPTH = 0.10
BETA_DEG = 0.0
BALL_R = 0.4
L_M = 10.0

parser = argparse.ArgumentParser()
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()

phi, tag = args.phi, args.tag

centerline_path = os.path.join(HERE, f"lm10gap_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_lm10_gap_check_{tag}.json")
out_path = os.path.abspath(out_path)


def cleanup(job_name, inp_name, sets_name):
    for pat in [f"{job_name}.*", inp_name, sets_name]:
        for f in glob.glob(os.path.join(HERE, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


results = []
total = len(S_LIST)
for n, contact_s in enumerate(S_LIST, 1):
    inp_name = f"lm10gap_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"lm10gap_node_sets_{tag}_s{contact_s:.0f}.inp"
    job_name = f"lm10gapsweep_{tag}_{n:02d}"
    t0 = time.time()
    print(f"[{tag} {n}/{total}] L_M={L_M}, phi={phi}, s={contact_s}mm, "
          f"push_depth={PUSH_DEPTH}mm ...", flush=True)
    try:
        scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                       centerline_path=centerline_path, beta_deg=BETA_DEG,
                                       inp_name=inp_name, sets_name=sets_name)
        normal = scene_info["normal"]
        res = rc.run_case(
            PUSH_DEPTH, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
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
        "L_M_mm": L_M, "phi_deg": phi, "beta_deg": BETA_DEG, "contact_s_mm": contact_s,
        "ball_r_mm": BALL_R, "push_depth_mm": PUSH_DEPTH,
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
