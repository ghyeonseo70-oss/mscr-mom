"""
LM(자석 길이) x phi(자기구동 방향, 부호 포함) 그리드로 형상을 바꿔가며, 각 형상에서
접촉위치(s) 10곳을 스윕하는 워커. geom_sweep_worker.py의 변형판 - 교수님 코멘트대로
힘/깊이는 중요하지 않다고 보고 push_depth를 1개 값(고정)으로만 돌려서 케이스 수를 줄임.
형상 하나(L_M, phi 조합)를 통째로 맡아서 s 10곳 x depth 1개 = 10케이스를 순차 처리.
여러 형상을 동시에(별도 프로세스로) 돌릴 수 있도록 모든 산출물 파일명을 tag로 구분.

디스크 관리: 케이스 하나의 CalculiX 원본 산출물(.frd/.dat/.12d/.cvg/.sta/.inp)이 수백MB라
(기존 스윕에서 233GB까지 쌓였던 전례가 있음), 결과를 JSON으로 뽑아내자마자 그 케이스의
원본 파일을 바로 지운다. 필요하면 재계산 가능하고 .gitignore에도 원래 제외돼 있는 파일들.
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

S_LIST = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]  # mm, 10구간
PUSH_DEPTH = 0.10  # mm, 고정(힘/깊이는 중요하지 않다는 전제)
BETA_DEG = 0.0     # deg, 원주방향 접촉각 고정(바깥쪽)
BALL_R = 0.4       # mm

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True, help="파일명용 접미사 (예: LM25_phiN90)")
args = parser.parse_args()

L_M, phi, tag = args.L_M, args.phi, args.tag

centerline_path = os.path.join(HERE, f"lmphi_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_lm_phi_pos_sweep_{tag}.json")
out_path = os.path.abspath(out_path)


def cleanup(job_name, inp_name, sets_name):
    patterns = [f"{job_name}.*", inp_name, sets_name]
    for pat in patterns:
        for f in glob.glob(os.path.join(HERE, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


results = []
total = len(S_LIST)
for n, contact_s in enumerate(S_LIST, 1):
    inp_name = f"lmphi_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"lmphi_node_sets_{tag}_s{contact_s:.0f}.inp"
    job_name = f"lmphisweep_{tag}_{n:02d}"
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
        "L_M_mm": L_M,
        "phi_deg": phi,
        "beta_deg": BETA_DEG,
        "contact_s_mm": contact_s,
        "ball_r_mm": BALL_R,
        "push_depth_mm": PUSH_DEPTH,
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
    cleanup(job_name, inp_name, sets_name)

try:
    os.remove(centerline_path)
except OSError:
    pass

print(f"[{tag}] 완료: {len(results)}/{total} 성공", flush=True)
