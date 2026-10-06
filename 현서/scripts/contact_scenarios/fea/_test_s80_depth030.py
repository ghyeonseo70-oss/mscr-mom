"""2026-10-06: 사용자 요청 - s=80mm(팁 근처)에서 depth=0.30mm로 밀면 어떻게 되는지
실제 FEA로 확인(그림용). depth=0.30mm 성공 레시피(ball_r=0.4, 메쉬 세밀화, min_inc=1e-8)
그대로 쓰되 접촉 위치만 s=30 -> s=80으로 바꿈. L_M/phi는 기존 비교 케이스와 동일하게
L_M=12.5, phi=-120 유지(비교 가능하게).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

L_M, PHI, S = 12.5, -120.0, 80.0
BALL_R = 0.4
DEPTH = 0.30
MIN_INC = 1e-8
CONTACT_MESH_SIZE = 0.08

tag = "S80_DEPTH030"
centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
    check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
)
inp_name = f"matv2_mesh_{tag}.inp"
sets_name = f"matv2_node_sets_{tag}.inp"
job_name = f"matv2test_{tag}"

print(f"\n=== {tag}: L_M={L_M}, phi={PHI}, s={S}, depth={DEPTH}mm, ball_r={BALL_R}mm ===")
t0 = time.time()
try:
    scene_info = scene.build_mesh(
        contact_s=S, ball_r=BALL_R, verbose=False, centerline_path=centerline_path,
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
