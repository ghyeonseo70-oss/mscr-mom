"""2026-10-06: "넓은 볼이면 벽 두께(0.5mm) 한계를 넘어도 안 뚫리고 버틸 수 있나?" 테스트.
같은 조건(L_M=12.5,phi=-120,s=30)에서 큰 볼(0.8mm)로 depth=0.5mm(벽 두께와 같음),
0.6mm(벽 두께 초과)를 시도 - 수렴하는지, 결과가 물리적으로 말이 되는지 확인.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

L_M, PHI, S = 12.5, -120.0, 30.0
BALL_R = 0.8
DEPTHS = [0.50, 0.60]  # mm - 벽 두께(0.5mm)와 그 초과
MIN_INC = 1e-8
CONTACT_MESH_SIZE = 0.08

for depth in DEPTHS:
    tag = f"WALLTEST_d{int(depth*100)}"
    centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    inp_name = f"matv2_mesh_{tag}.inp"
    sets_name = f"matv2_node_sets_{tag}.inp"
    job_name = f"matv2test_{tag}"

    print(f"\n=== {tag}: ball_r={BALL_R}mm, depth={depth}mm (벽두께 0.5mm 대비 {depth/0.5*100:.0f}%) ===")
    t0 = time.time()
    try:
        scene_info = scene.build_mesh(
            contact_s=S, ball_r=BALL_R, verbose=False, centerline_path=centerline_path,
            beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
            contact_mesh_size=CONTACT_MESH_SIZE, contact_refine_radius=BALL_R + 0.3,
        )
        normal = scene_info["normal"]
        res = rc.run_case(
            depth, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
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
              f"팁변위={tip_mag:.4f}mm, 팁회전={res['tip_theta_deg']:.3f}deg ({dt:.1f}s)")
    try:
        os.remove(centerline_path)
    except OSError:
        pass

print("\n=== 테스트 완료 ===")
