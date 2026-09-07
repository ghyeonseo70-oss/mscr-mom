"""
fea_lm_phi_pos_sweep_all.json에서 성공률 40% 미만이었던 s값들(60,70,80,90,100mm - s가 클수록
CalculiX 수렴 실패가 급증하는 게 확인됨: s=10 91% -> s=100 7%)을, 아직 안 채워진 (L_M,phi,s)
조합만 골라서 재시도. 단순 재시도가 아니라 수렴이 잘 되도록 INC(허용 증분 수)를 늘리고
초기 증분(initial_inc)을 더 잘게 잘라서(0.01->0.005) 재시도 성공률을 높이려는 시도.
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

BALL_R = 0.4
BETA_DEG = 0.0
PUSH_DEPTH = 0.10
INC = 400          # 기존 100 -> 400 (증분 한계로 실패하는 경우를 줄이려는 목적)
INITIAL_INC = 0.005  # 기존 0.01 -> 0.005 (초반을 더 잘게 쪼개서 시작)

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--s_list", type=str, required=True, help="콤마구분 s값들 (예: 60,70,90)")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()

L_M, phi, tag = args.L_M, args.phi, args.tag
s_list = [float(x) for x in args.s_list.split(",")]

centerline_path = os.path.join(HERE, f"retryls_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_retry_lowsucc_{tag}.json")
out_path = os.path.abspath(out_path)


def cleanup(job_name, inp_name, sets_name):
    for pat in [f"{job_name}.*", inp_name, sets_name]:
        for f in glob.glob(os.path.join(HERE, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


results = []
for n, contact_s in enumerate(s_list, 1):
    inp_name = f"retryls_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"retryls_node_sets_{tag}_s{contact_s:.0f}.inp"
    job_name = f"retryls_{tag}_{n:02d}"
    t0 = time.time()
    print(f"[{tag} {n}/{len(s_list)}] L_M={L_M}, phi={phi}, s={contact_s}mm ...", flush=True)
    try:
        scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                       centerline_path=centerline_path, beta_deg=BETA_DEG,
                                       inp_name=inp_name, sets_name=sets_name)
        normal = scene_info["normal"]
        res = rc.run_case(
            PUSH_DEPTH, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
            timeout=2400, verbose=False, push_dir=tuple(normal), n_threads=args.threads,
            print_tip=True, inc=INC, initial_inc=INITIAL_INC,
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
        "L_M_mm": L_M, "phi_deg": phi, "beta_deg": BETA_DEG,
        "contact_s_mm": contact_s, "ball_r_mm": BALL_R, "push_depth_mm": PUSH_DEPTH,
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
print(f"[{tag}] 완료: {len(results)}/{len(s_list)} 성공", flush=True)
