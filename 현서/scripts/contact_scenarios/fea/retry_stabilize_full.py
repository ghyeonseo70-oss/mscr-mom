"""
2026-08-19: PROJECT_STATUS.md "1-C" 파일럿(min_inc 17%, STABILIZE 42%, 둘다 33%) 결과
STABILIZE가 가장 나아서, 이걸로 phi 전체(0,+-30,+-60,+-90,+-120,+-150)에 걸친 실패
케이스(218개) + phi=+-60 신규(80개) = 298개를 전부 재시도.

phi=90/120/150(토크-제로 특이점 구간)도 제외하지 않고 포함 - STABILIZE가 원래 이런 불안정
평형 근처 문제에 쓰는 옵션이라 거기서도 통할지 실제로 확인해보는 게 목적(사용자 요청).

결과는 fea_matv2_stab_*.json으로 저장(기존 all.json과 안 섞임, retry_failed_matv2_cases.py와
동일 패턴), --merge로 병합.
"""
import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

BALL_R = 0.4
PUSH_DEPTH = 0.10


def cleanup(job_name, inp_name, sets_name):
    for pat in [f"{job_name}.*", inp_name, sets_name]:
        for f in glob.glob(os.path.join(HERE, pat)):
            try:
                os.remove(f)
            except OSError:
                pass


def run_combo(L_M, phi, beta, s_list, tag):
    centerline_path = os.path.join(HERE, f"stab_centerline_{tag}.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(L_M), "--phi", str(phi), "--out", centerline_path],
        check=True, capture_output=True, text=True,
    )
    out_path = os.path.abspath(os.path.join(
        HERE, "..", "..", "..", "data", "contact_scenarios", "fea", f"fea_matv2_stab_{tag}.json"))
    results = []
    for n, contact_s in enumerate(s_list, 1):
        inp_name = f"stab_mesh_{tag}_s{contact_s:.0f}.inp"
        sets_name = f"stab_sets_{tag}_s{contact_s:.0f}.inp"
        job_name = f"stabsweep_{tag}_{n:02d}"
        t0 = time.time()
        print(f"[{tag} {n}/{len(s_list)}] L_M={L_M}, phi={phi}, beta={beta}, s={contact_s}mm "
              f"(stabilize=True) ...", flush=True)
        try:
            scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False,
                                           centerline_path=centerline_path, beta_deg=beta,
                                           inp_name=inp_name, sets_name=sets_name)
            normal = scene_info["normal"]
            res = rc.run_case(
                PUSH_DEPTH, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
                timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=4,
                print_tip=True, stabilize=True,
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
            "L_M_mm": L_M, "phi_deg": phi, "beta_deg": beta, "contact_s_mm": contact_s,
            "ball_r_mm": BALL_R, "push_depth_mm": PUSH_DEPTH, "normal": normal,
            "ball_center": scene_info["ball_center"],
            "Fx_total_N": res["Fx_total_N"], "Fy_total_N": res["Fy_total_N"],
            "Fz_total_N": res["Fz_total_N"], "F_mag_N": res["F_mag_N"],
            "ux_avg_mm": res["ux_avg_mm"],
            "tip_ux_avg_mm": res["tip_ux_avg_mm"], "tip_uy_avg_mm": res["tip_uy_avg_mm"],
            "tip_uz_avg_mm": res["tip_uz_avg_mm"],
            "tip_theta_deg_board": res["tip_theta_deg"],
            "tip_rotation_rmse_mm": res["tip_rotation_rmse_mm"],
            "wall_time_s": dt, "stabilize": True,
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


def merge():
    fea_dir = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea"))
    out = os.path.join(fea_dir, "fea_lm_phi_pos_matv2_all.json")
    rows = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else []
    seen = {(r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"]) for r in rows}
    added = 0
    for f in sorted(glob.glob(os.path.join(fea_dir, "fea_matv2_stab_*.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            key = (r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"])
            if key not in seen:
                rows.append(r)
                seen.add(key)
                added += 1
    rows.sort(key=lambda r: (r["beta_deg"], r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]))
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"병합 완료: 기존 {len(rows) - added}개 + STABILIZE 신규 {added}개 = 총 {len(rows)}개 -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo-index", type=int)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge()
        sys.exit(0)

    targets = pickle.load(open("/tmp/stabilize_full_targets.pkl", "rb"))
    grouped = defaultdict(list)
    for lm, phi, beta, s in targets:
        grouped[(lm, phi, beta)].append(s)
    combos = sorted(grouped.items())

    if args.combo_index >= len(combos):
        sys.exit(0)
    (lm, phi, beta), s_list = combos[args.combo_index]
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    tag = f"LM{int(lm)}_phi{sign}{int(abs(phi))}_b{int(beta)}"
    run_combo(lm, phi, beta, sorted(s_list), tag)
