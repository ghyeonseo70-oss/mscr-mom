"""
2026-08-19: PROJECT_STATUS.md "1-C" 계획의 두번째 옵션 - phi=+-30 실패 케이스 12개(min_inc
파일럿과 동일 세트)에 *STATIC,STABILIZE(자동 감쇠)를 줘서 재시도. min_inc 파일럿과 동시에
돌려서 어느 레버가(혹은 둘 다) 효과 있는지 비교.

--case-index N으로 케이스 하나만 처리(병렬 실행용).
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

PILOT_CASES = [
    (0.0, 30.0, 70.0), (0.0, 30.0, 90.0), (0.0, 30.0, 100.0),
    (25.0, 30.0, 20.0), (25.0, 30.0, 40.0), (25.0, 30.0, 60.0),
    (50.0, 30.0, 30.0), (50.0, 30.0, 40.0), (50.0, 30.0, 50.0),
    (75.0, 30.0, 40.0), (75.0, 30.0, 50.0), (75.0, 30.0, 60.0),
]
BALL_R = 0.4
PUSH_DEPTH = 0.10
FEA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea"))


def cleanup(job_name, inp_name, sets_name):
    for pat in [f"{job_name}.*", inp_name, sets_name]:
        for f in glob.glob(os.path.join(HERE, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


def run_one(n):
    import make_bent_contact_scene as scene
    import run_contact as rc

    L_M, phi, s = PILOT_CASES[n]
    tag = f"stabpilot{n:02d}_LM{int(L_M)}_phi{'N' if phi < 0 else 'P'}{int(abs(phi))}_s{int(s)}"
    centerline_path = os.path.join(HERE, f"pilot_centerline_{tag}.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
        check=True, capture_output=True, text=True,
    )
    inp_name = f"pilot_mesh_{tag}.inp"
    sets_name = f"pilot_sets_{tag}.inp"
    job_name = f"pilotstab_{tag}"
    t0 = time.time()
    print(f"[{tag}] L_M={L_M}, phi={phi}, s={s} (stabilize=True) ...", flush=True)
    try:
        scene_info = scene.build_mesh(contact_s=s, ball_r=BALL_R, verbose=False,
                                       centerline_path=centerline_path, beta_deg=0.0,
                                       inp_name=inp_name, sets_name=sets_name)
        normal = scene_info["normal"]
        res = rc.run_case(
            PUSH_DEPTH, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
            timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=4,
            print_tip=True, stabilize=True,  # inc/initial_inc 기본값 그대로, min_inc는 안 씀
        )
    except Exception as e:
        print(f"  [{tag}] 실패(예외): {e}", flush=True)
        res = None
        normal = None
        scene_info = None
    dt = time.time() - t0
    if res is None:
        row = {"L_M_mm": L_M, "phi_deg": phi, "contact_s_mm": s, "success": False, "wall_time_s": dt}
        print(f"  [{tag}] [실패] ({dt:.1f}s)", flush=True)
    else:
        row = {"L_M_mm": L_M, "phi_deg": phi, "beta_deg": 0.0, "contact_s_mm": s,
               "ball_r_mm": BALL_R, "push_depth_mm": PUSH_DEPTH, "normal": normal,
               "ball_center": scene_info["ball_center"],
               "Fx_total_N": res["Fx_total_N"], "Fy_total_N": res["Fy_total_N"],
               "Fz_total_N": res["Fz_total_N"], "F_mag_N": res["F_mag_N"],
               "ux_avg_mm": res["ux_avg_mm"], "tip_ux_avg_mm": res["tip_ux_avg_mm"],
               "tip_uy_avg_mm": res["tip_uy_avg_mm"], "tip_uz_avg_mm": res["tip_uz_avg_mm"],
               "tip_theta_deg_board": res["tip_theta_deg"],
               "tip_rotation_rmse_mm": res["tip_rotation_rmse_mm"],
               "wall_time_s": dt, "success": True, "stabilize_pilot": True}
        print(f"  [{tag}] [성공] F_mag={res['F_mag_N']*1000:.4f}mN ({dt:.1f}s)", flush=True)

    with open(os.path.join(FEA_DIR, f"fea_pilot_stabilize_{n:02d}.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    cleanup(job_name, inp_name, sets_name)
    try:
        os.remove(centerline_path)
    except OSError:
        pass


def merge():
    rows = []
    for f in sorted(glob.glob(os.path.join(FEA_DIR, "fea_pilot_stabilize_*.json"))):
        rows.append(json.load(open(f, encoding="utf-8")))
    out = os.path.join(FEA_DIR, "fea_pilot_stabilize_all.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    n_success = sum(1 for r in rows if r["success"])
    avg_time = sum(r["wall_time_s"] for r in rows) / len(rows) if rows else 0
    print(f"=== 파일럿(STABILIZE) 결과: {n_success}/{len(rows)} 성공, 평균 {avg_time:.1f}s/케이스 ===")
    print("판정:", "성공률 상승 -> 새 레버 유효, phi=+-30 전체(53개)로 확장 권장" if n_success >= 6
          else "개선 미미 -> 폐기 권장")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge()
    else:
        run_one(args.case_index)
