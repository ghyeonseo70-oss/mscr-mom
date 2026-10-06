"""2026-10-06: 사용자 요청 - 볼 면적을 더 넓혀서 전체적으로 더 큰 변위를 만들 수 있는지 탐색.
지금까지 가장 큰 충돌은 ball_r=0.8mm, depth=0.6mm (팁변위 2.053mm, _test_wall_limit.py).
여기서 더 밀어붙여봄: 볼을 1.0mm로 키우고, depth도 0.6/0.8mm로 시도.
같은 조건(L_M=12.5,phi=-120,s=30)으로 비교 가능하게 유지.
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
MIN_INC = 1e-8
CONTACT_MESH_SIZE = 0.08

CASES = [
    dict(ball_r=1.0, depth=0.6, tag="BIGBALL_r100_d60"),
    dict(ball_r=1.0, depth=0.8, tag="BIGBALL_r100_d80"),
]

for case in CASES:
    tag = case["tag"]
    centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    inp_name = f"matv2_mesh_{tag}.inp"
    sets_name = f"matv2_node_sets_{tag}.inp"
    job_name = f"matv2test_{tag}"

    print(f"\n=== {tag}: L_M={L_M}, phi={PHI}, s={S}, depth={case['depth']}mm, ball_r={case['ball_r']}mm ===")
    t0 = time.time()
    try:
        scene_info = scene.build_mesh(
            contact_s=S, ball_r=case["ball_r"], verbose=False, centerline_path=centerline_path,
            beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
            contact_mesh_size=CONTACT_MESH_SIZE, contact_refine_radius=case["ball_r"] + 0.3,
        )
        normal = scene_info["normal"]
        res = rc.run_case(
            case["depth"], inp_name=inp_name, sets_name=sets_name, job_name=job_name,
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
