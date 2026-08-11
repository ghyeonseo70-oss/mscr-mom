"""1단계에서 실패했던 (s, depth) 조합만 재시도. 동시성을 줄이고(부하 완화) 스레드를 늘려서
같은 조건이라도 수렴 가능성을 높여본다. 결과는 기존 fea_bent_contact_sweep_s{s}.json에 병합."""
import argparse
import json
import os
import time

import make_bent_contact_scene as scene
import run_contact as rc

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--s", type=float, required=True)
parser.add_argument("--depths", type=str, required=True, help="콤마구분, 예: 0.08,0.10,0.12")
parser.add_argument("--threads", type=int, default=12)
args = parser.parse_args()

contact_s = args.s
tag = f"s{int(contact_s)}"
depths = [float(x) for x in args.depths.split(",")]

inp_name = f"retry_bent_contact_mesh_{tag}.inp"
sets_name = f"retry_bent_contact_node_sets_{tag}.inp"
out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_bent_contact_sweep_{tag}.json")
out_path = os.path.abspath(out_path)

existing = json.load(open(out_path)) if os.path.exists(out_path) else []

scene_info = scene.build_mesh(contact_s=contact_s, ball_r=0.4, verbose=False,
                               inp_name=inp_name, sets_name=sets_name)
normal = scene_info["normal"]

for i, push_depth in enumerate(depths, 1):
    job_name = f"retry_bent_{tag}_{i:02d}"
    t0 = time.time()
    print(f"[retry {tag} {i}/{len(depths)}] depth={push_depth}mm ...", flush=True)
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
        "contact_s_mm": contact_s, "ball_r_mm": 0.4, "push_depth_mm": push_depth,
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
