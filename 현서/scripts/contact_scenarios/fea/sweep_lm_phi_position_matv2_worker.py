"""
2026-08-18: K1/K2/Br 상수 변경(교수님 MATLAB 코드 방식, E=850kPa 직접계산) 이후
material_convert.py도 같은 E를 쓰도록 고쳤는데, 기존 FEA 664개는 전부 옛날 K2(9.8872e-07
기반 재료값)로 계산된 거라 force_model.py의 자유단 형상(새 K2)과 어긋남. 이 워커는 그 어긋남을
바로잡는 새 데이터를 만든다 - sweep_lm_phi_position_worker.py(옛 664개짜리 스윕의 핵심
부분)와 거의 동일하지만:
  1) beta를 고정하지 않고 인자로 받음(--beta) - 이번엔 0/180 둘 다 필요해서.
  2) material_convert.py가 이미 새 E를 쓰도록 고쳐졌으므로 이 워커 자체는 그대로 두면
     자동으로 새 재료값으로 계산됨(따로 손댈 곳 없음, run_contact.py가 mat.C10_MM/D1_MM을
     실행 시점에 읽어옴).
결과 파일명 접두사를 fea_lm_phi_pos_sweep_*(옛날 파일)와 다르게 fea_lm_phi_pos_matv2_*로
분리 - 기존 664개 데이터와 섞이지 않게.
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
PUSH_DEPTH = 0.10  # mm, 고정(기존 스윕과 동일 - 힘/깊이는 덜 중요하다는 전제 유지)
BALL_R = 0.4       # mm

parser = argparse.ArgumentParser()
parser.add_argument("--L_M", type=float, required=True)
parser.add_argument("--phi", type=float, required=True)
parser.add_argument("--beta", type=float, required=True, help="0 또는 180 (원주방향 접촉각)")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--tag", type=str, required=True, help="파일명용 접미사 (예: LM25_phiN90_b0)")
parser.add_argument("--s_list", type=str, default=None,
                     help="쉼표구분 s(mm) 목록으로 기본 S_LIST 덮어쓰기 (예: '15,25,35') - "
                          "2026-08-21 phi=90~150 구간 s 격자 조밀화용으로 추가")
parser.add_argument("--stabilize_value", type=float, default=None,
                     help="stabilize=True(자동감쇠) 대신 명시적 감쇠계수 지정 (*STATIC,STABILIZE=값) - "
                          "2026-08-21 STABILIZE 감쇠값 튜닝 파일럿용으로 추가")
args = parser.parse_args()

L_M, phi, beta, tag = args.L_M, args.phi, args.beta, args.tag
if args.s_list is not None:
    S_LIST = [float(v) for v in args.s_list.split(",")]
stabilize_arg = args.stabilize_value if args.stabilize_value is not None else True

centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
    check=True, capture_output=True, text=True,
)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                         f"fea_lm_phi_pos_matv2_{tag}.json")
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
    inp_name = f"matv2_mesh_{tag}_s{contact_s:.0f}.inp"
    sets_name = f"matv2_node_sets_{tag}_s{contact_s:.0f}.inp"
    job_name = f"matv2sweep_{tag}_{n:02d}"
    t0 = time.time()
    print(f"[{tag} {n}/{total}] L_M={L_M}, phi={phi}, beta={beta}, s={contact_s}mm, "
          f"push_depth={PUSH_DEPTH}mm ...", flush=True)
    try:
        scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                       centerline_path=centerline_path, beta_deg=beta,
                                       inp_name=inp_name, sets_name=sets_name)
        normal = scene_info["normal"]
        res = rc.run_case(
            PUSH_DEPTH, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
            timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=args.threads,
            print_tip=True, stabilize=stabilize_arg,
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
