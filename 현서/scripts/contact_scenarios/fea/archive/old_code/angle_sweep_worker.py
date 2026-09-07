"""
3단계: 접촉 각도(β, 원주방향) 스윕 워커. 기준 형상(L_M=50, phi=60) 고정, β 하나를 맡아서
s(3)xdepth(3)=9케이스를 순차 처리. β=0은 2단계의 LM50_phiP60 조합과 완전히 같은 조건이라
중복 계산을 피하려고 여기서는 β=90,180,270만 돌리면 됨(호출 시 --beta로 지정).
push_dir 공식(cos(beta)*평면내법선 + sin(beta)*binormal)은 make_bent_contact_scene.py 참고.
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

L_M = 50.0
PHI = 60.0
# 4단계(각도 보강): tip_uz 예측이 안 좋았던 게 beta 데이터 부족 때문이라 s/depth 그리드도 함께 확장
S_LIST = [10.0, 25.0, 50.0, 75.0, 90.0]          # mm
PUSH_DEPTH_LIST = [0.02, 0.05, 0.08, 0.10, 0.15]  # mm
BALL_R = 0.4                            # mm

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, required=True, help="원주방향 각도(deg): 0=바깥쪽,90=위,180=안쪽,270=아래")
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()

beta = args.beta
tag = f"beta{int(beta)}"

centerline_path = os.path.join(HERE, f"bent_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_angle_sweep_{tag}.json")
out_path = os.path.abspath(out_path)

results = []
total = len(S_LIST) * len(PUSH_DEPTH_LIST)
n = 0
for contact_s in S_LIST:
    inp_name = f"angle_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"angle_node_sets_{tag}_s{contact_s:.0f}.inp"
    scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                   centerline_path=centerline_path,
                                   inp_name=inp_name, sets_name=sets_name, beta_deg=beta)
    normal = scene_info["normal"]
    for push_depth in PUSH_DEPTH_LIST:
        n += 1
        job_name = f"anglesweep_{tag}_{n:02d}"
        t0 = time.time()
        print(f"[{tag} {n}/{total}] beta={beta}deg, s={contact_s}mm, "
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
            "phi_deg": PHI,
            "beta_deg": beta,
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
