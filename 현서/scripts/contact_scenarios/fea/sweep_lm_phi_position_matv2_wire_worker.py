"""
2026-08-18: sweep_lm_phi_position_matv2_worker.py(재료값-검증용, 와이어 없음)의 별도 사본.
K1 구간(베이스~MOM)에 니티놀 와이어(실리콘 벽 두께 안쪽 임베드, include_wire=True)를 넣어서
같은 L_M/phi/s 그리드를 돌린다. 기존 재료값-검증 스윕과 완전히 분리된 워커/출력 파일을 써서
(fea_lm_phi_pos_matv2_wire_*), 실행 중인 다른 스윕에 영향을 주지 않는다.
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
PUSH_DEPTH = 0.10  # mm, 기존 스윕과 동일
BALL_R = 0.4       # mm

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--beta", type=float, required=True, help="0 또는 180 (원주방향 접촉각)")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True, help="파일명용 접미사 (예: LM25_phiN90_b0)")
args = parser.parse_args()

L_M, phi, beta, tag = args.L_M, args.phi, args.beta, args.tag

centerline_path = os.path.join(HERE, f"matv2wire_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_matv2wire_{tag}.json")
out_path = os.path.abspath(out_path)
# 주의: 파일명이 "fea_lm_phi_pos_matv2_"로 시작하면 안 됨 - 지금 동시에 돌고 있는
# run_lm_phi_position_matv2_sweep.sh(와이어 없는 재료값-검증 스윕)의 병합 단계가
# glob('fea_lm_phi_pos_matv2_*.json')으로 결과를 모으는데, 그 패턴에 걸리면 이 와이어
# 스윕 결과가 섞여 들어가 버림(접두사를 완전히 다르게 함으로써 방지).


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
    inp_name = f"matv2wire_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"matv2wire_node_sets_{tag}_s{contact_s:.0f}.inp"
    job_name = f"matv2wiresweep_{tag}_{n:02d}"
    t0 = time.time()
    print(f"[{tag} {n}/{total}] L_M={L_M}, phi={phi}, beta={beta}, s={contact_s}mm, "
          f"push_depth={PUSH_DEPTH}mm (wire) ...", flush=True)
    try:
        scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                       centerline_path=centerline_path, beta_deg=beta,
                                       inp_name=inp_name, sets_name=sets_name,
                                       include_wire=True)
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
        "beta_deg": beta,
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
