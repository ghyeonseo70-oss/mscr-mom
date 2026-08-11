"""2단계에서 실패했던 (s, depth) 조합만, 조합(L_M,phi)별로 재시도.
기존에 만들어둔 centerline 파일이 있으면 재사용, 없으면 새로 생성.
결과는 기존 fea_geom_sweep_{tag}.json에 병합."""
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

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--pairs", type=str, required=True, help="s:depth,s:depth,... 예: 25:0.05,50:0.1")
parser.add_argument("--threads", type=int, default=12)
args = parser.parse_args()

tag = args.tag
pairs = [(float(p.split(":")[0]), float(p.split(":")[1])) for p in args.pairs.split(",")]

centerline_path = os.path.join(HERE, f"bent_centerline_{tag}.json")
if not os.path.exists(centerline_path):
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(args.L_M), "--phi", str(args.phi), "--out", centerline_path],
        check=True, capture_output=True, text=True,
    )

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_geom_sweep_{tag}.json")
out_path = os.path.abspath(out_path)
existing = json.load(open(out_path)) if os.path.exists(out_path) else []

# s별로 묶어서 메쉬 재사용
from collections import defaultdict
by_s = defaultdict(list)
for s, d in pairs:
    by_s[s].append(d)

n = 0
for contact_s, depths in by_s.items():
    inp_name = f"retry_geom_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"retry_geom_node_sets_{tag}_s{contact_s:.0f}.inp"
    scene_info = scene.build_mesh(contact_s=contact_s, ball_r=0.4, verbose=False,
                                   centerline_path=centerline_path,
                                   inp_name=inp_name, sets_name=sets_name)
    normal = scene_info["normal"]
    for push_depth in depths:
        n += 1
        job_name = f"retry_geom_{tag}_{n:02d}"
        t0 = time.time()
        print(f"[retry {tag} {n}/{len(pairs)}] s={contact_s}mm, depth={push_depth}mm ...", flush=True)
        try:
            res = rc.run_case(
                push_depth, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
                timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=args.threads,
                print_tip=True,
            )
        except Exception as e:
            print(f"  [retry {tag}] 실패: {e}", flush=True)
            res = None
        dt = time.time() - t0
        if res is None:
            print(f"  [retry {tag}] [실패] ({dt:.1f}s)", flush=True)
            continue
        row = {
            "L_M_mm": args.L_M, "phi_deg": args.phi, "contact_s_mm": contact_s,
            "ball_r_mm": 0.4, "push_depth_mm": push_depth,
            "normal": normal, "ball_center": scene_info["ball_center"],
            "Fx_total_N": res["Fx_total_N"], "Fy_total_N": res["Fy_total_N"],
            "Fz_total_N": res["Fz_total_N"], "F_mag_N": res["F_mag_N"],
            "ux_avg_mm": res["ux_avg_mm"],
            "tip_ux_avg_mm": res["tip_ux_avg_mm"], "tip_uy_avg_mm": res["tip_uy_avg_mm"],
            "tip_uz_avg_mm": res["tip_uz_avg_mm"], "tip_theta_deg_board": res["tip_theta_deg"],
            "tip_rotation_rmse_mm": res["tip_rotation_rmse_mm"], "wall_time_s": dt,
        }
        existing.append(row)
        print(f"  [retry {tag}] F_mag={res['F_mag_N']*1000:.4f}mN  ({dt:.1f}s)", flush=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

print(f"[retry {tag}] 완료", flush=True)
