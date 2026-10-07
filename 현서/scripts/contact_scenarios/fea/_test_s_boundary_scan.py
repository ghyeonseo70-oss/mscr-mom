"""2026-10-06: s=30은 성공, s=80은 4전 4패 - 정확히 어디서부터 실패하기 시작하는지
경계를 찾기 위해 중간값(s=50, s=65)을 같은 조건(depth=0.20mm, ball_r=0.4mm, 기존 레시피)
으로 테스트. "접촉점 뒤 남은 길이" 가설이 맞는지 vs 다른 이유(예: 팁 자석 근접)인지
데이터로 구분하기 위함.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

L_M, PHI, DEPTH = 12.5, -120.0, 0.20
BALL_R = 0.4
MIN_INC = 1e-8
CONTACT_MESH_SIZE = 0.08

S_VALUES = [50.0, 65.0]

for s in S_VALUES:
    tag = f"SBOUND_s{int(s)}"
    centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    inp_name = f"matv2_mesh_{tag}.inp"
    sets_name = f"matv2_node_sets_{tag}.inp"
    job_name = f"matv2test_{tag}"

    print(f"\n=== {tag}: L_M={L_M}, phi={PHI}, s={s}, depth={DEPTH}mm, ball_r={BALL_R}mm "
          f"(팁까지 남은 길이 {100-s:.0f}mm) ===")
    t0 = time.time()
    try:
        scene_info = scene.build_mesh(
            contact_s=s, ball_r=BALL_R, verbose=False, centerline_path=centerline_path,
            beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
            contact_mesh_size=CONTACT_MESH_SIZE, contact_refine_radius=BALL_R + 0.3,
        )
        normal = scene_info["normal"]
        res = rc.run_case(
            DEPTH, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
            timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=4,
            print_tip=True, print_mom=True, stabilize=True, min_inc=MIN_INC,
        )
    except Exception as e:
        res = None
        print(f"  예외 발생: {e}")
    dt = time.time() - t0
    if res is None:
        print(f"  [{tag}] 실패 ({dt:.1f}s)")
    else:
        tip_mag = (res["tip_ux_avg_mm"]**2 + res["tip_uy_avg_mm"]**2) ** 0.5
        print(f"  [{tag}] 성공! F_mag={res['F_mag_N']*1000:.4f}mN, "
              f"tip_ux={res['tip_ux_avg_mm']:.4f}mm, tip_uy={res['tip_uy_avg_mm']:.4f}mm, "
              f"팁변위={tip_mag:.4f}mm, 팁회전={res['tip_theta_deg']:.3f}deg ({dt:.1f}s)")
    try:
        os.remove(centerline_path)
    except OSError:
        pass

print("\n=== 테스트 완료 ===")
