"""2026-10-06: s=80mm은 depth=0.20mm(제일 안전한 깊이)에서도, 볼 크기를 바꿔도 전부 실패함
(3전 3패) - 볼 크기나 깊이가 아니라 위치 자체의 수렴 문제로 보임. 메쉬를 더 세밀화해서
(접촉부 0.08->0.04mm, 세밀화 반경도 ball_r+0.3->ball_r+0.5로 확대) 제일 쉬운 케이스
(작은 볼, depth=0.20mm)부터 다시 시도.
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
DEPTH = 0.20
MIN_INC = 1e-8
CONTACT_MESH_SIZE = 0.04  # 기존 0.08 -> 0.04로 세밀화
CONTACT_REFINE_RADIUS = BALL_R + 0.5  # 기존 +0.3 -> +0.5로 확대 (남은 길이가 짧으므로)

tag = "S80_FINEMESH_d20"
centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
    check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
)
inp_name = f"matv2_mesh_{tag}.inp"
sets_name = f"matv2_node_sets_{tag}.inp"
job_name = f"matv2test_{tag}"

print(f"\n=== {tag}: L_M={L_M}, phi={PHI}, s={S}, depth={DEPTH}mm, ball_r={BALL_R}mm, "
      f"mesh_size={CONTACT_MESH_SIZE}mm, refine_r={CONTACT_REFINE_RADIUS}mm ===")
t0 = time.time()
try:
    scene_info = scene.build_mesh(
        contact_s=S, ball_r=BALL_R, verbose=False, centerline_path=centerline_path,
        beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
        contact_mesh_size=CONTACT_MESH_SIZE, contact_refine_radius=CONTACT_REFINE_RADIUS,
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
