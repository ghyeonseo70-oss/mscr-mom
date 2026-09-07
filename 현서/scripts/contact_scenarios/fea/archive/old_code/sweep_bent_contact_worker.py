"""
sweep_bent_contact.py의 병렬 워커 버전. s값 하나를 맡아서 그 s의 메쉬를 한 번 만들고
PUSH_DEPTH_LIST 전체를 순차로 돌린 뒤, 자기 몫만 별도 JSON에 저장한다.
여러 s를 동시에(별도 프로세스로) 돌릴 수 있도록 inp_name/sets_name/job_name을 s별로 다르게 써서
공유 디렉터리(HERE) 안에서도 파일 충돌이 나지 않게 함.
"""
import argparse
import json
import os
import time

import make_bent_contact_scene as scene
import run_contact as rc

HERE = os.path.dirname(os.path.abspath(__file__))

PUSH_DEPTH_LIST = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]  # mm
BALL_R = 0.4  # mm

parser = argparse.ArgumentParser()
parser.add_argument("--s", type=float, required=True, help="contact_s (mm)")
parser.add_argument("--threads", type=int, default=6)
parser.add_argument("--tag", type=str, required=True, help="파일명 충돌 방지용 접미사 (예: s10)")
args = parser.parse_args()

contact_s = args.s
tag = args.tag
inp_name = f"bent_contact_mesh_{tag}.inp"
sets_name = f"bent_contact_node_sets_{tag}.inp"
out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_bent_contact_sweep_{tag}.json")
out_path = os.path.abspath(out_path)

scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                               inp_name=inp_name, sets_name=sets_name)
normal = scene_info["normal"]

results = []
for i, push_depth in enumerate(PUSH_DEPTH_LIST, 1):
    job_name = f"bent_sweep_{tag}_{i:02d}"
    t0 = time.time()
    print(f"[{tag} {i}/{len(PUSH_DEPTH_LIST)}] s={contact_s}mm, push_depth={push_depth}mm ...", flush=True)
    try:
        res = rc.run_case(
            push_depth,
            inp_name=inp_name,
            sets_name=sets_name,
            job_name=job_name,
            timeout=1800,
            verbose=False,
            push_dir=tuple(normal),
            n_threads=args.threads,
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

print(f"[{tag}] 완료: {len(results)}/{len(PUSH_DEPTH_LIST)} 성공", flush=True)
